"""
Generic_UNet_SENet - Symmetry-Enhanced UNet for Stroke Lesion Segmentation

Implements the core SE-Net ideas from:
"SE-Net: Symmetry-Enhanced Deep Learning for Advanced Stroke-Lesion 
Segmentation Using Multi-Level Brain-Symmetry Priors"

Architecture:
- Dual-stream encoder with shared weights (original + sagittally-flipped input)
- Cross-hemisphere feature aggregation (learnable α, β, γ per level)
- Symmetry Difference Module (SDM) at each encoder level
- Enhanced skip connections with gated fusion
- Standard nnUNet decoder with deep supervision

Excluded (not applicable to ISLES 2022 with DWI+ADC only):
- Perfusion-Diffusion Mismatch Attention (PDMA) - requires T2

Key equations:
  F_agg = α·F_orig + β·F_mir + γ·|F_orig - F_mir|        (Eq. 1)
  F_diff = |F_orig - F_mir|                                 (Eq. 2)
  F_filtered = σ(W2·δ(W1·F_diff)) ⊙ F_diff                (Eq. 3)
  F_out = F_orig + λ·F_filtered                             (Eq. 4)
  F_skip = γ_skip·F_enc + (1-γ_skip)·F_sdm                 (Eq. 7)

Symmetry transform: T(x)[h,w,d] = x[h, W-w-1, d]  (sagittal reflection)
"""

from copy import deepcopy
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.network_architecture.generic_UNet import (
    ConvDropoutNormNonlin, StackedConvLayers, Upsample
)
import torch.nn.functional


class SymmetryDifferenceModule(nn.Module):
    """
    Symmetry Difference Module (SDM).
    
    Filters pathological asymmetries from the difference between original
    and mirrored feature maps, suppressing normal anatomical variations.
    
    F_diff = |F_orig - F_mir|
    F_filtered = σ(W2·δ(W1·F_diff)) ⊙ F_diff     (gating)
    F_out = F_orig + λ·F_filtered                   (residual)
    """
    
    def __init__(self, num_features, conv_op, reduction=4):
        """
        Args:
            num_features: Number of input/output feature channels
            conv_op: nn.Conv2d or nn.Conv3d 
            reduction: Channel reduction factor for bottleneck
        """
        super().__init__()
        
        reduced = max(num_features // reduction, 8)
        
        # Two-layer bottleneck for gating: reduce channels → expand back
        self.gate = nn.Sequential(
            conv_op(num_features, reduced, kernel_size=1, bias=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            conv_op(reduced, num_features, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        
        # Learnable scaling for residual connection (Eq. 4: λ)
        self.lambda_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, f_orig, f_mir):
        """
        Args:
            f_orig: Features from original stream [B, C, ...]
            f_mir: Features from mirrored stream [B, C, ...]
        
        Returns:
            f_out: Enhanced features with symmetry information
            f_sdm: Filtered symmetry difference (for skip connections)
        """
        # Eq. 2: compute difference
        f_diff = torch.abs(f_orig - f_mir)
        
        # Eq. 3: gated filtering
        gate_weights = self.gate(f_diff)
        f_filtered = gate_weights * f_diff
        
        # Eq. 4: residual addition
        f_out = f_orig + self.lambda_scale * f_filtered
        
        return f_out, f_filtered


class Generic_UNet_SENet(SegmentationNetwork):
    """
    Symmetry-Enhanced Generic UNet.
    
    Architecture overview:
    1. Input x → also compute T(x) (sagittal flip)
    2. Encoder: process both x and T(x) through SHARED conv blocks
    3. At each level: aggregate features + apply SDM
    4. Enhanced skip connections: gated fusion of encoder + SDM features
    5. Decoder: standard nnUNet decoder with deep supervision
    
    VRAM note: ~2x encoder activation cost vs standard UNet due to dual stream.
    Parameter overhead is minimal (shared weights + small SDM modules + scalars).
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
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False):
        """
        Same signature as Generic_UNet for drop-in compatibility.
        Additional symmetry modules are created automatically.
        """
        super(Generic_UNet_SENet, self).__init__()
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
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

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

        # =====================================================================
        # ENCODER (shared weights for dual-stream processing)
        # =====================================================================
        self.conv_blocks_context = []
        self.td = []

        output_features = base_num_features
        input_features = input_channels

        for d in range(num_pool):
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]

            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))
            output_features = min(output_features, self.max_num_features)

        # Bottleneck
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
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        # =====================================================================
        # SYMMETRY MODULES
        # =====================================================================
        
        # Feature aggregation parameters (Eq. 1): α, β, γ per encoder level
        # Initialised to emphasise original features + difference
        # num_pool encoder stages + 1 bottleneck
        num_encoder_levels = num_pool + 1
        self.agg_alpha = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(num_encoder_levels)])
        self.agg_beta = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for _ in range(num_encoder_levels)])
        self.agg_gamma = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(num_encoder_levels)])
        
        # SDM modules - one per encoder level BEFORE bottleneck only
        # (bottleneck has no skip connection, so SDM output would be unused)
        self.sdm_modules = nn.ModuleList()
        enc_features = base_num_features
        for d in range(num_pool):
            self.sdm_modules.append(SymmetryDifferenceModule(enc_features, conv_op))
            enc_features = int(np.round(enc_features * feat_map_mul_on_downscale))
            enc_features = min(enc_features, self.max_num_features)
        
        # Enhanced skip connection gating (Eq. 7): γ_skip per skip level 
        # num_pool skip connections (one per encoder level, excluding bottleneck)
        self.skip_gate = nn.ParameterList([nn.Parameter(torch.tensor(0.7)) for _ in range(num_pool)])

        # =====================================================================
        # DECODER (standard nnUNet decoder)
        # =====================================================================
        self.conv_blocks_localization = []
        self.tu = []
        self.seg_outputs = []

        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels
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

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes,
                                            1, 1, 0, 1, 1, seg_output_use_bias))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # Register all modules
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(self.upscale_logits_ops)

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)

    def _mirror_sagittal(self, x):
        """
        Reflect input across the sagittal (W) axis.
        T(x)[h, w, d] = x[h, W-w-1, d]
        
        For 3D: x is [B, C, H, W, D] → flip along dim=3 (W axis)
        For 2D: x is [B, C, H, W] → flip along dim=3 (W axis)  
        """
        # W is always the second spatial dimension (dim index 3)
        return torch.flip(x, dims=[3])
    
    def forward(self, x):
        """
        Dual-stream forward pass with symmetry enhancement.
        
        1. Compute mirrored input T(x)
        2. Process both through shared encoder
        3. Aggregate features + apply SDM at each level
        4. Enhanced skip connections
        5. Standard decoder with deep supervision
        """
        # Step 1: Create mirrored input
        x_mir = self._mirror_sagittal(x)
        
        # Step 2-3: Dual-stream encoder with aggregation and SDM
        skips = []  # Enhanced skip connections for decoder
        seg_outputs = []
        
        x_orig = x
        x_flip = x_mir
        
        for d in range(len(self.conv_blocks_context) - 1):
            # Process both streams through SHARED encoder block
            f_orig = self.conv_blocks_context[d](x_orig)
            f_mir = self.conv_blocks_context[d](x_flip)
            
            # Feature aggregation (Eq. 1)
            f_agg = (self.agg_alpha[d] * f_orig + 
                     self.agg_beta[d] * f_mir + 
                     self.agg_gamma[d] * torch.abs(f_orig - f_mir))
            
            # SDM (Eqs. 2-4)
            f_enhanced, f_sdm = self.sdm_modules[d](f_orig, f_mir)
            
            # Enhanced skip connection (Eq. 7)
            gamma = torch.sigmoid(self.skip_gate[d])
            f_skip = gamma * f_enhanced + (1 - gamma) * f_sdm
            skips.append(f_skip)
            
            # Pool aggregated features for next level
            if not self.convolutional_pooling:
                x_orig = self.td[d](f_agg)
                x_flip = self.td[d](self._mirror_sagittal(f_agg))  
            else:
                x_orig = f_agg
                x_flip = self._mirror_sagittal(f_agg)
        
        # Bottleneck (last encoder block)
        d_bn = len(self.conv_blocks_context) - 1
        f_orig = self.conv_blocks_context[-1](x_orig)
        f_mir = self.conv_blocks_context[-1](x_flip)
        
        # Aggregate bottleneck features
        x = (self.agg_alpha[d_bn] * f_orig + 
             self.agg_beta[d_bn] * f_mir + 
             self.agg_gamma[d_bn] * torch.abs(f_orig - f_mir))
        
        # Step 5: Standard decoder with enhanced skips
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
    
    def get_symmetry_params_summary(self):
        """Get a summary of all learned symmetry parameters for logging."""
        summary = {}
        for d in range(len(self.agg_alpha)):
            summary[f'level_{d}_alpha'] = self.agg_alpha[d].item()
            summary[f'level_{d}_beta'] = self.agg_beta[d].item()
            summary[f'level_{d}_gamma'] = self.agg_gamma[d].item()
        
        for d in range(len(self.sdm_modules)):
            summary[f'level_{d}_sdm_lambda'] = self.sdm_modules[d].lambda_scale.item()
        
        for d in range(len(self.skip_gate)):
            summary[f'level_{d}_skip_gate'] = torch.sigmoid(self.skip_gate[d]).item()
        
        return summary

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        Approximate VRAM consumption.
        
        ~2x encoder cost vs standard UNet due to dual-stream processing.
        Decoder cost is unchanged.
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        # Encoder cost: 2x due to dual streams
        tmp = np.int64(2 * (conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            # 2x for encoder (dual stream) + 1x for decoder
            num_blocks_enc = conv_per_stage * 2  # dual stream encoder
            num_blocks_dec = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage
            tmp += (num_blocks_enc + num_blocks_dec) * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
        return tmp
