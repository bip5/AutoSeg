from collections import defaultdict

import torch
import torch.nn as nn

from nnunet.network_architecture.generic_UNet_InputResidual import (
    _InputResidualSplitDDSBase,
    StackedConvLayers,
    ConvDropoutNormNonlin,
)
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.utilities.nd_softmax import softmax_helper
import torch.nn.functional as F


class Generic_UNet_AccumulativeResidual_SplitDDS(_InputResidualSplitDDSBase):
    """
    Projected Accumulative Residuals (PAR) UNet.

    Every block receives a softmax-weighted sum of all preceding block outputs
    at the same spatial resolution, each projected to the current channel count
    via a dedicated 1x1 conv. This is injected after InstanceNorm, before the
    existing split input residual and LeakyReLU.

    Attention weights are separate learnable parameters of shape [N, C]:
    C independent sets of N weights (one per preceding block), softmax-normalised
    over the N dimension. Zero-initialised so the network starts with uniform
    averaging across history.

    Encoder->decoder cross-injection at the same resolution is intentional:
    decoder blocks receive accumulated encoder history via the shared res_memory.

    Inherits from _InputResidualSplitDDSBase:
      - split input residual (_split_add_input) applied after PAR
      - encoder deep supervision heads (seg_outputs_enc)
    """

    def __init__(self, input_channels, base_num_features, num_classes, num_pool,
                 num_conv_per_stage=2, feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.InstanceNorm2d, norm_op_kwargs=None,
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

        # Enumerate all blocks in forward execution order to determine N per block k
        all_block_channels = []
        for stage in self.conv_blocks_context:
            for block in self._iter_basic_blocks(stage):
                all_block_channels.append(block.conv.out_channels)
        for stage in self.conv_blocks_localization:
            for block in self._iter_basic_blocks(stage):
                all_block_channels.append(block.conv.out_channels)

        # par_projections[k] = ModuleList of k convs, one per preceding block
        # par_weights[k]     = Parameter [k, target_ch], zero-init
        #   For k=0: shape [0, C] — valid empty tensor, never accessed (history is empty)
        self.par_projections = nn.ModuleList()
        self.par_weights = nn.ParameterList()

        for k, target_ch in enumerate(all_block_channels):
            projs = nn.ModuleList([
                conv_op(all_block_channels[i], target_ch, 1, 1, 0, bias=False)
                for i in range(k)
            ])
            self.par_projections.append(projs)
            self.par_weights.append(nn.Parameter(torch.zeros(k, target_ch)))

        if weightInitializer is not None:
            self.par_projections.apply(weightInitializer)
        # par_weights are zero-init intentionally — do not apply He init

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(self, x):
        # Reset state at the top of every forward call
        self._res_memory = defaultdict(list)
        self._block_counter = 0

        x_input = x
        skips = []
        seg_outputs = []
        seg_outputs_enc_list = []

        # Encoder
        for d in range(len(self.conv_blocks_context) - 1):
            x = self._run_stage(self.conv_blocks_context[d], x, x_input)
            skips.append(x)
            seg_outputs_enc_list.append(self.final_nonlin(self.seg_outputs_enc[d](x)))
            if not self.convolutional_pooling:
                x = self.td[d](x)

        # Bottleneck
        x = self._run_stage(self.conv_blocks_context[-1], x, x_input)
        seg_outputs_enc_list.append(self.final_nonlin(self.seg_outputs_enc[-1](x)))

        # Decoder
        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self._run_stage(self.conv_blocks_localization[u], x, x_input)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self._deep_supervision and self.do_ds:
            dec_outputs = [seg_outputs[-1]] + [
                i(j) for i, j in zip(
                    list(self.upscale_logits_ops)[::-1],
                    seg_outputs[:-1][::-1]
                )
            ]
            return tuple(dec_outputs + seg_outputs_enc_list)
        else:
            return seg_outputs[-1]

    # ------------------------------------------------------------------ #
    #  Stage runner with PAR injection                                     #
    # ------------------------------------------------------------------ #

    def _run_stage(self, stage_module, x, x_input):
        for block in self._iter_basic_blocks(stage_module):
            x = block.conv(x)
            if block.dropout is not None:
                x = block.dropout(x)
            x = block.instnorm(x)

            # --- PAR injection ---
            res = tuple(x.shape[2:])
            history = self._res_memory[res]
            k = self._block_counter

            if history:
                # history entries are (global_block_idx, tensor) pairs
                # par_projections[k][prev_k] maps prev_k's channels -> current channels
                N = len(history)
                projected = [
                    self.par_projections[k][prev_k](tensor)
                    for prev_k, tensor in history
                ]                                                       # N x [B, C, ...]
                stacked = torch.stack(projected, dim=1)                 # [B, N, C, ...]
                # par_weights[k] has shape [k, C] (all global predecessors)
                # select only the weights for the N resolution-matched predecessors
                res_indices = torch.tensor(
                    [prev_k for prev_k, _ in history],
                    device=x.device
                )
                w_selected = self.par_weights[k][res_indices]           # [N, C]
                weights = torch.softmax(w_selected, dim=0)              # [N, C]
                # reshape for broadcasting over batch and spatial dims
                extra_dims = x.dim() - 2                                # 2 for 3D, 1 for 2D
                weights = weights.view(1, N, x.shape[1], *([1] * extra_dims))
                par = (stacked * weights).sum(dim=1)                    # [B, C, ...]
                x = x + par

            # --- Original split input residual (unchanged) ---
            x = self._split_add_input(x, x_input)

            x = block.lrelu(x)

            # Store (global_index, output) — index needed to look up correct projection conv
            # No detach: keeps tensor in computation graph for gradient flow through PAR history
            self._res_memory[res].append((self._block_counter, x))
            self._block_counter += 1

        return x


# ====================================================================== #
#  Vanilla per-block residual + split input residual                      #
# ====================================================================== #

class Generic_UNet_VanillaRes_SplitDDS(_InputResidualSplitDDSBase):
    """
    Vanilla residual + split input residual, per conv block.

    At each block:
        identity = x                        (before conv)
        x = conv -> dropout -> instnorm
        x = x + identity                    (vanilla residual, no projection needed
                                             because in_channels == out_channels
                                             within every stage)
        x = _split_add_input(x, x_input)   (original network input residual)
        x = lrelu(x)

    No new parameters beyond the parent class. This is a pre-activation
    residual block fused with the existing split input residual.
    """

    def _run_stage(self, stage_module, x, x_input):
        for block in self._iter_basic_blocks(stage_module):
            identity = x
            x = block.conv(x)
            if block.dropout is not None:
                x = block.dropout(x)
            x = block.instnorm(x)
            # Only add identity when shape matches — strided convs (convolutional
            # pooling) change spatial size so no residual is possible without projection
            if x.shape == identity.shape:
                x = x + identity
            x = self._split_add_input(x, x_input)  # split input residual
            x = block.lrelu(x)
        return x


# ====================================================================== #
#  PAR v2 — per-stage, cross-resolution (downsample only),               #
#  injected before skip concat in decoder only                            #
# ====================================================================== #

class Generic_UNet_PAR_Decoder_SplitDDS(_InputResidualSplitDDSBase):
    """
    PAR v2: Projected Accumulative Residuals, decoder injection variant.

    Design:
    - Encoder stages run unchanged (split input residual from parent).
    - At each decoder stage, BEFORE the skip concatenation:
        1. Collect all prior stage outputs (encoder + earlier decoder stages).
        2. Filter to those at >= current spatial resolution (downsample only,
           no upsampling to avoid prohibitive temp-buffer costs).
        3. Adaptive-pool each to current spatial size, project via 1x1 conv.
        4. Softmax-weight across N valid predecessors (per-channel, [N,C]).
        5. Add weighted sum to x, then proceed to cat(skip) and blocks.
    - Only the propagated path is accumulated (skip-side features are never
      stored and never contribute to future PAR).
    - Per-stage granularity: one memory entry per stage, stored after blocks run.

    Inherits split input residual + encoder DDS from _InputResidualSplitDDSBase.
    """

    def __init__(self, input_channels, base_num_features, num_classes, num_pool,
                 num_conv_per_stage=2, feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.InstanceNorm2d, norm_op_kwargs=None,
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

        # Channel count of each stage output, in execution order:
        # [enc_stage_0, ..., enc_stage_M-1 (bottleneck), dec_stage_0, ..., dec_stage_N-1]
        enc_ch = []
        for stage in self.conv_blocks_context:
            last = list(self._iter_basic_blocks(stage))[-1]
            enc_ch.append(last.conv.out_channels)

        dec_ch = []
        for stage in self.conv_blocks_localization:
            last = list(self._iter_basic_blocks(stage))[-1]
            dec_ch.append(last.conv.out_channels)

        M = len(enc_ch)   # encoder stages including bottleneck
        N = len(dec_ch)   # decoder stages
        all_ch = enc_ch + dec_ch

        # For decoder stage u, prior stages are indices 0..M+u-1.
        # par_projections[u][k]: maps all_ch[k] -> dec_ch[u]  (1x1 conv)
        # par_weights[u]:        shape [M+u, dec_ch[u]], zero-init -> uniform softmax
        self.par_projections = nn.ModuleList()
        self.par_weights = nn.ParameterList()

        for u in range(N):
            target_ch = dec_ch[u]
            n_prior = M + u
            projs = nn.ModuleList([
                conv_op(all_ch[k], target_ch, 1, 1, 0, bias=False)
                for k in range(n_prior)
            ])
            self.par_projections.append(projs)
            self.par_weights.append(nn.Parameter(torch.zeros(n_prior, target_ch)))

        if weightInitializer is not None:
            self.par_projections.apply(weightInitializer)
        # par_weights zero-init intentionally (-> uniform softmax at start)

    # ------------------------------------------------------------------ #
    #  Stage runner — split residual at every conv block (encoder path)   #
    # ------------------------------------------------------------------ #

    def _run_stage(self, stage_module, x, x_input):
        """Split input residual injected at every conv block (PerConv behaviour)."""
        for block in self._iter_basic_blocks(stage_module):
            x = block.conv(x)
            if block.dropout is not None:
                x = block.dropout(x)
            x = block.instnorm(x)
            x = self._split_add_input(x, x_input)
            x = block.lrelu(x)
        return x

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(self, x):
        x_input = x
        skips = []
        seg_outputs = []
        seg_outputs_enc_list = []

        # stage_memory: list of (global_stage_idx, tensor, spatial_tuple)
        # populated after each stage completes
        stage_memory = []

        # ---- Encoder ----
        for d in range(len(self.conv_blocks_context) - 1):
            x = self._run_stage(self.conv_blocks_context[d], x, x_input)
            skips.append(x)
            seg_outputs_enc_list.append(self.final_nonlin(self.seg_outputs_enc[d](x)))
            stage_memory.append((len(stage_memory), x, tuple(x.shape[2:])))
            if not self.convolutional_pooling:
                x = self.td[d](x)

        # ---- Bottleneck ----
        x = self._run_stage(self.conv_blocks_context[-1], x, x_input)
        seg_outputs_enc_list.append(self.final_nonlin(self.seg_outputs_enc[-1](x)))
        stage_memory.append((len(stage_memory), x, tuple(x.shape[2:])))

        M = len(self.conv_blocks_context)   # encoder stages incl. bottleneck

        # ---- Decoder ----
        for u in range(len(self.tu)):
            x = self.tu[u](x)   # upsample

            # PAR: inject before skip concat
            current_sp = tuple(x.shape[2:])

            # Filter to predecessors at same or larger spatial (downsample only)
            valid = [
                (k, t) for k, t, sp in stage_memory
                if all(s >= c for s, c in zip(sp, current_sp))
            ]

            if valid:
                N_valid = len(valid)
                projected = [
                    self.par_projections[u][k](self._pool_input(t, current_sp))
                    for k, t in valid
                ]                                                       # N_valid x [B, C, ...]
                stacked = torch.stack(projected, dim=1)                 # [B, N, C, ...]
                valid_idx = torch.tensor([k for k, _ in valid], device=x.device)
                w = torch.softmax(self.par_weights[u][valid_idx], dim=0)  # [N, C]
                extra = x.dim() - 2
                w = w.view(1, N_valid, x.shape[1], *([1] * extra))
                x = x + (stacked * w).sum(dim=1)

            # Skip concatenation
            x = torch.cat((x, skips[-(u + 1)]), dim=1)

            # Decoder blocks (split residual from parent _run_stage)
            x = self._run_stage(self.conv_blocks_localization[u], x, x_input)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

            # Store propagated path output (not the concatenated input)
            stage_memory.append((M + u, x, tuple(x.shape[2:])))

        if self._deep_supervision and self.do_ds:
            dec_outputs = [seg_outputs[-1]] + [
                i(j) for i, j in zip(
                    list(self.upscale_logits_ops)[::-1],
                    seg_outputs[:-1][::-1]
                )
            ]
            return tuple(dec_outputs + seg_outputs_enc_list)
        else:
            return seg_outputs[-1]


# ====================================================================== #
#  PAR Full — per-block, cross-resolution, encoder + decoder             #
# ====================================================================== #

class Generic_UNet_PAR_Full_SplitDDS(_InputResidualSplitDDSBase):
    """
    Full Projected Accumulative Residuals (PAR) UNet.

    Every conv block (encoder and decoder) receives a softmax-weighted sum
    of ALL preceding block outputs, projected to the current channel count
    and adaptive-pooled to the current spatial resolution. This is injected
    after InstanceNorm, before the existing split input residual and LeakyReLU.

    Differences from Generic_UNet_AccumulativeResidual_SplitDDS:
    - Resolution scope: ALL prior blocks contribute (not same-resolution-only).
      Adaptive average pooling handles cross-resolution mismatches.
    - par_weights[k] is used in full (shape [k, C]) — no subset selection
      needed because _block_memory always contains exactly k entries.

    Skip connections are never stored in _block_memory and never contribute
    to any PAR computation.

    Inherits from _InputResidualSplitDDSBase:
      - split input residual (_split_add_input) applied after PAR
      - encoder deep supervision heads (seg_outputs_enc)
    """

    def __init__(self, input_channels, base_num_features, num_classes, num_pool,
                 num_conv_per_stage=2, feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.InstanceNorm2d, norm_op_kwargs=None,
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

        # Enumerate all blocks in forward execution order to determine channel count per block
        all_block_channels = []
        for stage in self.conv_blocks_context:
            for block in self._iter_basic_blocks(stage):
                all_block_channels.append(block.conv.out_channels)
        for stage in self.conv_blocks_localization:
            for block in self._iter_basic_blocks(stage):
                all_block_channels.append(block.conv.out_channels)

        # par_projections[k] = ModuleList of k convs, one per preceding block
        #   par_projections[k][i]: maps all_block_channels[i] -> all_block_channels[k]
        # par_weights[k]     = Parameter [k, target_ch], zero-init -> uniform softmax
        #   shape [0, C] for k=0 — valid empty tensor, never accessed (history empty)
        self.par_projections = nn.ModuleList()
        self.par_weights = nn.ParameterList()

        for k, target_ch in enumerate(all_block_channels):
            projs = nn.ModuleList([
                conv_op(all_block_channels[i], target_ch, 1, 1, 0, bias=False)
                for i in range(k)
            ])
            self.par_projections.append(projs)
            self.par_weights.append(nn.Parameter(torch.zeros(k, target_ch)))

        if weightInitializer is not None:
            self.par_projections.apply(weightInitializer)
        # par_weights zero-init intentionally — do not apply He init

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(self, x):
        # Reset state at the top of every forward call
        self._block_memory = []   # flat list of (global_k, tensor) for ALL prior blocks
        self._block_counter = 0

        x_input = x
        skips = []
        seg_outputs = []
        seg_outputs_enc_list = []

        # Encoder
        for d in range(len(self.conv_blocks_context) - 1):
            x = self._run_stage(self.conv_blocks_context[d], x, x_input)
            skips.append(x)
            seg_outputs_enc_list.append(self.final_nonlin(self.seg_outputs_enc[d](x)))
            if not self.convolutional_pooling:
                x = self.td[d](x)

        # Bottleneck
        x = self._run_stage(self.conv_blocks_context[-1], x, x_input)
        seg_outputs_enc_list.append(self.final_nonlin(self.seg_outputs_enc[-1](x)))

        # Decoder — skip cat happens here, BEFORE _run_stage, so skip
        # features are never seen by _block_memory
        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self._run_stage(self.conv_blocks_localization[u], x, x_input)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self._deep_supervision and self.do_ds:
            dec_outputs = [seg_outputs[-1]] + [
                i(j) for i, j in zip(
                    list(self.upscale_logits_ops)[::-1],
                    seg_outputs[:-1][::-1]
                )
            ]
            return tuple(dec_outputs + seg_outputs_enc_list)
        else:
            return seg_outputs[-1]

    # ------------------------------------------------------------------ #
    #  Stage runner with full cross-resolution PAR injection               #
    # ------------------------------------------------------------------ #

    def _run_stage(self, stage_module, x, x_input):
        for block in self._iter_basic_blocks(stage_module):
            x = block.conv(x)
            if block.dropout is not None:
                x = block.dropout(x)
            x = block.instnorm(x)

            # --- Full PAR injection: ALL prior blocks, any resolution ---
            k = self._block_counter
            if self._block_memory:
                current_sp = tuple(x.shape[2:])
                N = len(self._block_memory)
                projected = [
                    self.par_projections[k][prev_k](self._pool_input(tensor, current_sp))
                    for prev_k, tensor in self._block_memory
                ]                                                       # N x [B, C, ...]
                stacked = torch.stack(projected, dim=1)                 # [B, N, C, ...]
                # par_weights[k] shape [k, C] == [N, C] since _block_memory has exactly k entries
                weights = torch.softmax(self.par_weights[k], dim=0)    # [N, C]
                extra_dims = x.dim() - 2
                weights = weights.view(1, N, x.shape[1], *([1] * extra_dims))
                par = (stacked * weights).sum(dim=1)                    # [B, C, ...]
                x = x + par

            # --- Original split input residual (unchanged) ---
            x = self._split_add_input(x, x_input)

            x = block.lrelu(x)

            # Store output in flat history — no detach: full gradient flow
            self._block_memory.append((k, x))
            self._block_counter += 1

        return x
