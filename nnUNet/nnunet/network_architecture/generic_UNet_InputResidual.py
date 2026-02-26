#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
Input-Residual UNet Variants
=============================

Two alternative UNet architectures where the original input image is added
(with adaptive pooling + 1x1 channel projection) to feature maps BEFORE
the activation function:

- PerConv:  input residual injected at EVERY conv block
- PerStage: input residual injected at the END of each encoder/decoder stage

The residual is:
    1. Adaptively average-pooled to match the current spatial resolution
    2. Projected via a 1x1 conv to match the current channel count
    3. Added to the feature maps after InstanceNorm, before LeakyReLU
"""

from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.network_architecture.generic_UNet import (
    Generic_UNet, ConvDropoutNormNonlin, StackedConvLayers
)
import torch.nn.functional as F


class _InputResidualBase(Generic_UNet):
    """
    Base class providing shared input-residual projection infrastructure.
    
    After the parent Generic_UNet builds the standard encoder/decoder,
    this class adds:
      - 1x1 projection convs (one per unique channel count) to map
        input_channels -> feature_channels
      - Helpers to pool the input and project it for residual addition
    
    Subclasses override _run_stage() to control WHERE the injection happens.
    """

    def __init__(self, input_channels, base_num_features, num_classes, num_pool,
                 num_conv_per_stage=2, feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None,
                 deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2),
                 pool_op_kernel_sizes=None, conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False,
                 convolutional_upsampling=False, max_num_features=None,
                 basic_block=ConvDropoutNormNonlin, seg_output_use_bias=False):

        # Build the standard UNet first
        super().__init__(
            input_channels, base_num_features, num_classes, num_pool,
            num_conv_per_stage, feat_map_mul_on_downscale, conv_op,
            norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
            nonlin, nonlin_kwargs, deep_supervision, dropout_in_localization,
            final_nonlin, weightInitializer, pool_op_kernel_sizes, conv_kernel_sizes,
            upscale_logits, convolutional_pooling, convolutional_upsampling,
            max_num_features, basic_block, seg_output_use_bias
        )

        self._is_3d = (conv_op == nn.Conv3d)
        self._input_channels_orig = input_channels

        # Collect all unique output channel counts across all blocks
        unique_channels = set()
        for stage in self.conv_blocks_context:
            for block in self._iter_basic_blocks(stage):
                unique_channels.add(block.conv.out_channels)
        for stage in self.conv_blocks_localization:
            for block in self._iter_basic_blocks(stage):
                unique_channels.add(block.conv.out_channels)

        # Create a 1x1 projection conv for each unique channel count
        # Key = str(num_channels), maps input_channels -> num_channels
        self.input_projections = nn.ModuleDict({
            str(ch): conv_op(input_channels, ch, kernel_size=1, stride=1, padding=0, bias=False)
            for ch in sorted(unique_channels)
        })

        # Apply weight initialization to the new projection layers
        if self.weightInitializer is not None:
            self.input_projections.apply(self.weightInitializer)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _iter_basic_blocks(module):
        """Yield all ConvDropoutNormNonlin blocks from a stage module.
        
        Handles both:
          - StackedConvLayers (encoder stages)
          - nn.Sequential(StackedConvLayers, StackedConvLayers) (bottleneck/decoder)
        """
        if isinstance(module, StackedConvLayers):
            yield from module.blocks
        elif isinstance(module, nn.Sequential):
            for sub in module:
                if isinstance(sub, StackedConvLayers):
                    yield from sub.blocks
                elif isinstance(sub, ConvDropoutNormNonlin):
                    yield sub

    def _pool_input(self, x_input, target_spatial):
        """Adaptively pool the original input to match target spatial dims."""
        if self._is_3d:
            return F.adaptive_avg_pool3d(x_input, target_spatial)
        else:
            return F.adaptive_avg_pool2d(x_input, target_spatial)

    def _get_projected_input(self, x_input, target_spatial, target_channels):
        """Pool + project input to match a block's output shape."""
        x_pooled = self._pool_input(x_input, target_spatial)
        return self.input_projections[str(target_channels)](x_pooled)

    # ------------------------------------------------------------------ #
    #  Subclass hook                                                       #
    # ------------------------------------------------------------------ #

    def _run_stage(self, stage_module, x, x_input):
        """Run a stage with input-residual injection. Override in subclasses."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    #  Forward (shared by both variants)                                   #
    # ------------------------------------------------------------------ #

    def forward(self, x):
        x_input = x  # store original input for residual injection
        skips = []
        seg_outputs = []

        # Encoder
        for d in range(len(self.conv_blocks_context) - 1):
            x = self._run_stage(self.conv_blocks_context[d], x, x_input)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

        # Bottleneck
        x = self._run_stage(self.conv_blocks_context[-1], x, x_input)

        # Decoder
        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self._run_stage(self.conv_blocks_localization[u], x, x_input)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self._deep_supervision and self.do_ds:
            return tuple(
                [seg_outputs[-1]] +
                [i(j) for i, j in zip(
                    list(self.upscale_logits_ops)[::-1],
                    seg_outputs[:-1][::-1]
                )]
            )
        else:
            return seg_outputs[-1]


# ====================================================================== #
#  Variant A — Per-Conv injection                                         #
# ====================================================================== #

class Generic_UNet_InputResidual_PerConv(_InputResidualBase):
    """
    Input residual is injected at EVERY ConvDropoutNormNonlin block:
    
        x = Conv(x)
        x = Dropout(x)        (if enabled)
        x = InstanceNorm(x)
        x = x + pool_project(input)   <-- injected here
        x = LeakyReLU(x)
    
    This creates a strong gradient highway from every block back to the
    input, acting as an implicit regulariser.
    """

    def _run_stage(self, stage_module, x, x_input):
        for block in self._iter_basic_blocks(stage_module):
            # Manually run block internals to inject residual before nonlin
            x = block.conv(x)
            if block.dropout is not None:
                x = block.dropout(x)
            x = block.instnorm(x)

            # Add projected input (matched to this block's output shape)
            spatial = tuple(x.shape[2:])
            channels = x.shape[1]
            input_proj = self._get_projected_input(x_input, spatial, channels)
            x = x + input_proj  # use + not += to avoid in-place on instnorm output

            x = block.lrelu(x)
        return x


# ====================================================================== #
#  Variant B — Per-Stage injection                                        #
# ====================================================================== #

class Generic_UNet_InputResidual_PerStage(_InputResidualBase):
    """
    Input residual is injected only at the LAST block of each stage:
    
        blocks[0..n-2]:  normal  Conv -> Dropout -> Norm -> Nonlin
        blocks[n-1]:     Conv -> Dropout -> Norm -> (+input_residual) -> Nonlin
    
    Lighter than PerConv: fewer residual additions, less overhead.
    """

    def _run_stage(self, stage_module, x, x_input):
        blocks = list(self._iter_basic_blocks(stage_module))

        # Run all blocks except the last one with normal forward
        for block in blocks[:-1]:
            x = block(x)  # standard ConvDropoutNormNonlin.forward

        # Last block: inject residual between norm and nonlin
        last = blocks[-1]
        x = last.conv(x)
        if last.dropout is not None:
            x = last.dropout(x)
        x = last.instnorm(x)

        spatial = tuple(x.shape[2:])
        channels = x.shape[1]
        input_proj = self._get_projected_input(x_input, spatial, channels)
        x = x + input_proj

        x = last.lrelu(x)
        return x


# ====================================================================== #
#  New Variants (Unique Weights per Insertion Point)                     #
# ====================================================================== #

class _InputResidualBase_Unique(Generic_UNet):
    """
    Base class for unique-weight variants. (No shared ModuleDict).
    """
    def __init__(self, input_channels, base_num_features, num_classes, num_pool,
                 num_conv_per_stage=2, feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None,
                 deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2),
                 pool_op_kernel_sizes=None, conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False,
                 convolutional_upsampling=False, max_num_features=None,
                 basic_block=ConvDropoutNormNonlin, seg_output_use_bias=False):

        super().__init__(
            input_channels, base_num_features, num_classes, num_pool,
            num_conv_per_stage, feat_map_mul_on_downscale, conv_op,
            norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
            nonlin, nonlin_kwargs, deep_supervision, dropout_in_localization,
            final_nonlin, weightInitializer, pool_op_kernel_sizes, conv_kernel_sizes,
            upscale_logits, convolutional_pooling, convolutional_upsampling,
            max_num_features, basic_block, seg_output_use_bias
        )
        self._is_3d = (conv_op == nn.Conv3d)
        self._input_channels_orig = input_channels

    # Helper methods duplicated from _InputResidualBase to avoid tying weight init in __init__
    def _pool_input(self, x_input, target_spatial):
        if self._is_3d:
            return F.adaptive_avg_pool3d(x_input, target_spatial)
        else:
            return F.adaptive_avg_pool2d(x_input, target_spatial)

    @staticmethod
    def _iter_basic_blocks(module):
        if isinstance(module, StackedConvLayers):
            yield from module.blocks
        elif isinstance(module, nn.Sequential):
            for sub in module:
                if isinstance(sub, StackedConvLayers):
                    yield from sub.blocks
                elif isinstance(sub, ConvDropoutNormNonlin):
                    yield sub


class Generic_UNet_InputResidual_PerStage_Unique(_InputResidualBase_Unique):
    """
    Input residual injected at END of each stage.
    Each insertion point has its own unique 1x1 projection weights.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Create unique projections for each stage
        # Encoder (0..N-1) + Bottleneck (1) + Decoder (N..0)
        
        self.projections_encoder = nn.ModuleList()
        # Iterate stages and create a unique projection for each
        for stage in self.conv_blocks_context[:-1]:
            # Get output channels of the last block in this stage
            last_block = list(self._iter_basic_blocks(stage))[-1]
            ch = last_block.conv.out_channels
            proj = self.conv_op(self._input_channels_orig, ch, 1, 1, 0, bias=False)
            self.projections_encoder.append(proj)
            
        # Bottleneck
        bn_stage = self.conv_blocks_context[-1]
        last_block = list(self._iter_basic_blocks(bn_stage))[-1]
        ch = last_block.conv.out_channels
        self.projection_bottleneck = self.conv_op(self._input_channels_orig, ch, 1, 1, 0, bias=False)
        
        # Decoder (0..N-1)
        self.projections_decoder = nn.ModuleList()
        for stage in self.conv_blocks_localization:
            last_block = list(self._iter_basic_blocks(stage))[-1]
            ch = last_block.conv.out_channels
            proj = self.conv_op(self._input_channels_orig, ch, 1, 1, 0, bias=False)
            self.projections_decoder.append(proj)
            
        if self.weightInitializer is not None:
             self.projections_encoder.apply(self.weightInitializer)
             self.projection_bottleneck.apply(self.weightInitializer)
             self.projections_decoder.apply(self.weightInitializer)

    def _run_stage_unique(self, stage_module, x, x_input, projection_module):
        blocks = list(self._iter_basic_blocks(stage_module))
        for block in blocks[:-1]:
            x = block(x)
        
        last = blocks[-1]
        x = last.conv(x)
        if last.dropout is not None:
            x = last.dropout(x)
        x = last.instnorm(x)
        
        spatial = tuple(x.shape[2:])
        x_pooled = self._pool_input(x_input, spatial)
        x = x + projection_module(x_pooled)
        
        x = last.lrelu(x)
        return x

    def forward(self, x):
        x_input = x
        skips = []
        seg_outputs = []
        
        # Encoder
        for d in range(len(self.conv_blocks_context) - 1):
            x = self._run_stage_unique(self.conv_blocks_context[d], x, x_input, self.projections_encoder[d])
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)
                
        # Bottleneck
        x = self._run_stage_unique(self.conv_blocks_context[-1], x, x_input, self.projection_bottleneck)
        
        # Decoder
        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self._run_stage_unique(self.conv_blocks_localization[u], x, x_input, self.projections_decoder[u])
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))
            
        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]


class Generic_UNet_InputResidual_PerConv_Unique(_InputResidualBase_Unique):
    """
    Input residual injected at EVERY conv block.
    Each insertion point has its own unique 1x1 projection weights.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Encoder projections (including bottleneck as last element)
        self.projections_encoder = nn.ModuleList()
        for stage in self.conv_blocks_context:
            stage_projs = nn.ModuleList()
            for block in self._iter_basic_blocks(stage):
                ch = block.conv.out_channels
                stage_projs.append(self.conv_op(self._input_channels_orig, ch, 1, 1, 0, bias=False))
            self.projections_encoder.append(stage_projs)
            
        # Decoder projections
        self.projections_decoder = nn.ModuleList()
        for stage in self.conv_blocks_localization:
            stage_projs = nn.ModuleList()
            for block in self._iter_basic_blocks(stage):
                ch = block.conv.out_channels
                stage_projs.append(self.conv_op(self._input_channels_orig, ch, 1, 1, 0, bias=False))
            self.projections_decoder.append(stage_projs)
            
        if self.weightInitializer is not None:
             self.projections_encoder.apply(self.weightInitializer)
             self.projections_decoder.apply(self.weightInitializer)

    def _run_stage_unique(self, stage_module, x, x_input, stage_projections):
        # Iterate blocks and matching projections
        for i, block in enumerate(self._iter_basic_blocks(stage_module)):
            x = block.conv(x)
            if block.dropout is not None:
                x = block.dropout(x)
            x = block.instnorm(x)
            
            spatial = tuple(x.shape[2:])
            x_pooled = self._pool_input(x_input, spatial)
            x = x + stage_projections[i](x_pooled)
            
            x = block.lrelu(x)
        return x

    def forward(self, x):
        x_input = x
        skips = []
        seg_outputs = []
        
        # Encoder context blocks
        # Context includes bottleneck as last element.
        # We process len(context)-1 encoder stages, store skips, then bottleneck.
        
        for d in range(len(self.conv_blocks_context) - 1):
            x = self._run_stage_unique(self.conv_blocks_context[d], x, x_input, self.projections_encoder[d])
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)
        
        # Bottleneck (last context block)
        x = self._run_stage_unique(self.conv_blocks_context[-1], x, x_input, self.projections_encoder[-1])
        
        # Decoder
        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self._run_stage_unique(self.conv_blocks_localization[u], x, x_input, self.projections_decoder[u])
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))
            
        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]
