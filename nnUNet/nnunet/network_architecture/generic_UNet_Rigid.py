"""
Generic_UNet_Rigid - Modified UNet with Bayesian Rigid Layers in Deep Encoder

Uses RigidConvDropoutNormNonlin for the two deepest encoder stages (bottleneck and
penultimate stage), while keeping the rest of the network standard.

This allows learning weight rigidity (precision) for the most abstract features
without the computational cost of full Bayesian inference.

Key differences from Generic_UNet:
- conv_blocks_context[-1] (bottleneck): Uses RigidStackedConvLayers
- conv_blocks_context[-2] (penultimate): Uses RigidStackedConvLayers
- All other layers: Standard ConvDropoutNormNonlin
- Added get_kl_divergence() and get_rigidity_stats() methods
"""

from copy import deepcopy
import numpy as np
import torch
from torch import nn
import torch.nn.functional

from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.network_architecture.generic_UNet import (
    ConvDropoutNormNonlin, StackedConvLayers, Upsample
)
from nnunet.network_architecture.custom_modules.rigid_blocks import (
    RigidConvDropoutNormNonlin, RigidStackedConvLayers
)


class Generic_UNet_Rigid(SegmentationNetwork):
    """
    Generic UNet with Rigid (Bayesian) layers in the two deepest encoder stages.
    
    Architecture:
    - Encoder stages 0 to (num_pool-2): Standard convolutions
    - Encoder stage (num_pool-1): Rigid convolutions (penultimate)
    - Bottleneck: Rigid convolutions
    - Decoder: Standard convolutions
    """
    
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000
    
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, seg_output_use_bias=False,
                 num_rigid_stages=2):
        """
        Initialize Generic_UNet_Rigid.
        
        Args:
            num_rigid_stages: Number of deepest encoder stages to use Rigid convolutions.
                             Default 2 = bottleneck + penultimate stage.
        """
        super().__init__()
        
        self.num_rigid_stages = num_rigid_stages
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        # Set up pooling and upsampling
        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError(f"Unknown conv op: {conv_op}")

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        # Lists to hold network components
        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []
        
        # Track which stages are rigid (for KL computation)
        self.rigid_stage_indices = []

        output_features = base_num_features
        input_features = input_channels

        # Build encoder stages
        for d in range(num_pool):
            # Determine first stride for convolutional pooling
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            
            # Determine if this stage should use Rigid convolutions
            # We use Rigid for the last (num_rigid_stages - 1) stages before bottleneck
            # i.e., if num_rigid_stages=2 and num_pool=5, we use Rigid for d=4 (penultimate)
            use_rigid = (d >= num_pool - (self.num_rigid_stages - 1)) if self.num_rigid_stages > 1 else False
            
            if use_rigid:
                # Use Rigid convolutions
                self.rigid_stage_indices.append(d)
                self.conv_blocks_context.append(RigidStackedConvLayers(
                    input_features, output_features, num_conv_per_stage,
                    self.conv_op, self.conv_kwargs, self.norm_op,
                    self.norm_op_kwargs, self.dropout_op,
                    self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                    first_stride
                ))
            else:
                # Use standard convolutions
                self.conv_blocks_context.append(StackedConvLayers(
                    input_features, output_features, num_conv_per_stage,
                    self.conv_op, self.conv_kwargs, self.norm_op,
                    self.norm_op_kwargs, self.dropout_op,
                    self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                    first_stride, basic_block=ConvDropoutNormNonlin
                ))
            
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))
            output_features = min(output_features, self.max_num_features)

        # Build bottleneck (always Rigid)
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        
        # Bottleneck uses Rigid convolutions
        self.rigid_stage_indices.append('bottleneck')
        self.conv_blocks_context.append(nn.Sequential(
            RigidStackedConvLayers(
                input_features, output_features, num_conv_per_stage - 1,
                self.conv_op, self.conv_kwargs, self.norm_op,
                self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                self.nonlin, self.nonlin_kwargs, first_stride
            ),
            RigidStackedConvLayers(
                output_features, final_num_features, 1,
                self.conv_op, self.conv_kwargs, self.norm_op,
                self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                self.nonlin, self.nonlin_kwargs
            )
        ))

        # Handle dropout in localization
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # Build decoder (standard convolutions)
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[-(2 + u)].output_channels
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling:
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[-(u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[-(u + 1)]
            
            # Decoder always uses standard convolutions
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=ConvDropoutNormNonlin),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=ConvDropoutNormNonlin)
            ))

        # Segmentation outputs
        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes,
                                            1, 1, 0, 1, 1, seg_output_use_bias))

        # Upscale logits
        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        # Restore dropout
        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # Register modules
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(self.upscale_logits_ops)

        # Initialize weights (standard layers only - Rigid layers have their own init)
        if self.weightInitializer is not None:
            # Only apply to non-Rigid modules
            for module in self.modules():
                if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                    self.weightInitializer(module)
    
    def forward(self, x, sample=True):
        """
        Forward pass.
        
        Args:
            x: Input tensor
            sample: Whether to sample weights for Rigid layers (True during training)
        """
        skips = []
        seg_outputs = []
        
        # Encoder
        for d in range(len(self.conv_blocks_context) - 1):
            block = self.conv_blocks_context[d]
            if isinstance(block, RigidStackedConvLayers):
                x = block(x, sample=sample)
            else:
                x = block(x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

        # Bottleneck
        bottleneck = self.conv_blocks_context[-1]
        for sub_block in bottleneck:
            if isinstance(sub_block, RigidStackedConvLayers):
                x = sub_block(x, sample=sample)
            else:
                x = sub_block(x)

        # Decoder
        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]
    
    def get_kl_divergence(self):
        """
        Compute total KL divergence from all Rigid layers.
        
        Returns:
            Scalar tensor with total KL divergence.
        """
        kl_total = 0.0
        
        for block in self.conv_blocks_context:
            if isinstance(block, RigidStackedConvLayers):
                kl_total = kl_total + block.kl_divergence()
            elif isinstance(block, nn.Sequential):
                for sub_block in block:
                    if isinstance(sub_block, RigidStackedConvLayers):
                        kl_total = kl_total + sub_block.kl_divergence()
        
        return kl_total
    
    def get_rigidity_stats(self):
        """
        Get rigidity statistics from all Rigid layers for logging/visualization.
        
        Returns:
            Dict with per-layer rigidity statistics.
        """
        stats = {}
        
        for i, block in enumerate(self.conv_blocks_context):
            if isinstance(block, RigidStackedConvLayers):
                stats[f'encoder_stage_{i}'] = block.get_rigidity_stats()
            elif isinstance(block, nn.Sequential):
                bottleneck_stats = []
                for j, sub_block in enumerate(block):
                    if isinstance(sub_block, RigidStackedConvLayers):
                        sub_stats = sub_block.get_rigidity_stats()
                        for s in sub_stats:
                            s['sub_block'] = j
                        bottleneck_stats.extend(sub_stats)
                if bottleneck_stats:
                    stats['bottleneck'] = bottleneck_stats
        
        return stats
    
    def get_rigidity_summary(self):
        """
        Get a simplified rigidity summary for logging.
        
        Returns:
            Dict with overall mean and std rigidity.
        """
        all_rigidities = []
        
        for block in self.conv_blocks_context:
            if isinstance(block, RigidStackedConvLayers):
                for sub_block in block.blocks:
                    rig = sub_block.conv.rigidity.detach().flatten()
                    all_rigidities.append(rig)
            elif isinstance(block, nn.Sequential):
                for sub_block in block:
                    if isinstance(sub_block, RigidStackedConvLayers):
                        for b in sub_block.blocks:
                            rig = b.conv.rigidity.detach().flatten()
                            all_rigidities.append(rig)
        
        if all_rigidities:
            all_rig = torch.cat(all_rigidities)
            return {
                'mean': all_rig.mean().item(),
                'std': all_rig.std().item(),
                'min': all_rig.min().item(),
                'max': all_rig.max().item(),
                'num_params': all_rig.numel()
            }
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'num_params': 0}

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        Approximate VRAM consumption (same as Generic_UNet but with slight increase for Rigid layers).
        
        Rigid layers use ~2x parameters (mu + rho), so we add a factor for the deepest stages.
        """
        # Get base estimate from parent
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage
            
            # Add 50% extra for Rigid stages (last 2)
            rigid_factor = 1.5 if p >= npool - 2 else 1.0
            tmp += int(num_blocks * np.prod(map_size, dtype=np.int64) * num_feat * rigid_factor)
            
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
        
        return tmp
