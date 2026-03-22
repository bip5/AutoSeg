"""
nnUNetTrainerV2_SeqWT  -  Sequential Weight Transfer Network (corrected)

Key corrections over original:
  1. Alpha trained in FEATURE-MAP space, not weight space.
     A FeatureMapFuser module computes Σ αᵢ·fᵢ as a residual added to the
     active network's encoder/decoder features at every spatial level.
     Gradients flow through this addition into log_alphas every iteration.

  2. Soft probability mask replaces argmax mask.
     The second input channel is the softmax probability of the foreground
     class from the previous network, preserving all uncertainty information.

  3. Alpha initialisation is cumulative, not discarded.
     Each new run inherits the final softmax alphas from the previous run,
     extended with a new entry for run K-1, rather than reinitialising to
     uniform.

  4. Weight initialisation still uses alpha-weighted combination of frozen
     state_dicts, but this is a one-shot operation (no gradient needed) and
     is now correctly separated from the trainable fuser alphas.

Architecture per run K (K >= 1):
  - Input:   [image, softmax_prob_from_net_{K-1}]   shape: (B, 2C, *spatial)
  - Frozen nets 0..K-1 run in eval() with no_grad, exposing feature maps
    via forward hooks at each encoder/decoder level.
  - FeatureMapFuser adds Σ softmax(log_alpha)_i * frozen_feat_i  to each
    level of the active network's feature maps via registered forward hooks.
  - Active network backpropagates normally; gradients reach log_alphas
    through the fusion additions.

Run 0:
  - No frozen networks yet.
  - Input: [image, Gaussian noise]
  - No fusion (nothing to fuse with).

Schedule (default):
  - 32 runs × 50 epochs, polyLR reset per run
  - base_num_features = 1
  - No data augmentation
  - Batch size 2

Usage:
    nnUNet_train 3d_fullres nnUNetTrainerV2_SeqWT TASK_ID FOLD
"""

import copy
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict
from datetime import datetime
from time import time
from torch import nn
from typing import List, Optional

from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.data_augmentation_noDA import get_no_augmentation
from nnunet.training.dataloading.dataset_loading import unpack_dataset, DataLoader3D, DataLoader2D
from nnunet.training.learning_rate.poly_lr import poly_lr
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.network_training.network_trainer import NetworkTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.to_torch import maybe_to_torch
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, save_json, isfile
import os


# =============================================================================
#  RAM-Cached DataLoader
# =============================================================================

class DataLoader3D_RAMCached(DataLoader3D):
    """
    DataLoader3D subclass that preloads ALL .npy files into RAM at init.
    Eliminates per-iteration disk I/O for small datasets (e.g. ISLES 2022).
    """

    def __init__(self, data, patch_size, final_patch_size, batch_size,
                 has_prev_stage=False, oversample_foreground_percent=0.0,
                 memmap_mode="r", pad_mode="edge", pad_kwargs_data=None,
                 pad_sides=None):
        super().__init__(data, patch_size, final_patch_size, batch_size,
                         has_prev_stage, oversample_foreground_percent,
                         memmap_mode, pad_mode, pad_kwargs_data, pad_sides)

        self._ram_cache = {}
        total_bytes = 0
        for key in self.list_of_keys:
            npy_path = self._data[key]['data_file'][:-4] + ".npy"
            if os.path.isfile(npy_path):
                arr = np.load(npy_path)
            else:
                arr = np.load(self._data[key]['data_file'])['data']
            self._ram_cache[key] = arr
            total_bytes += arr.nbytes

        print(f"[RAMCached] Preloaded {len(self._ram_cache)} cases "
              f"({total_bytes / 1e9:.2f} GB) into RAM")

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)
        case_properties = []

        for j, i in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)

            if 'properties' in self._data[i].keys():
                properties = self._data[i]['properties']
            else:
                from batchgenerators.utilities.file_and_folder_operations import load_pickle
                properties = load_pickle(self._data[i]['properties_file'])
            case_properties.append(properties)

            case_all_data = self._ram_cache[i]

            if self.has_prev_stage:
                if os.path.isfile(self._data[i]['seg_from_prev_stage_file'][:-4] + ".npy"):
                    segs_from_previous_stage = np.load(
                        self._data[i]['seg_from_prev_stage_file'][:-4] + ".npy",
                        mmap_mode=self.memmap_mode)[None]
                else:
                    segs_from_previous_stage = np.load(
                        self._data[i]['seg_from_prev_stage_file'])['data'][None]
                seg_key = np.random.choice(segs_from_previous_stage.shape[0])
                seg_from_previous_stage = segs_from_previous_stage[seg_key:seg_key + 1]
            else:
                seg_from_previous_stage = None

            need_to_pad = self.need_to_pad.copy()
            for d in range(3):
                if need_to_pad[d] + case_all_data.shape[d + 1] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - case_all_data.shape[d + 1]

            shape = case_all_data.shape[1:]
            lb_x = - need_to_pad[0] // 2
            ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
            lb_y = - need_to_pad[1] // 2
            ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]
            lb_z = - need_to_pad[2] // 2
            ub_z = shape[2] + need_to_pad[2] // 2 + need_to_pad[2] % 2 - self.patch_size[2]

            if not force_fg:
                bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                bbox_z_lb = np.random.randint(lb_z, ub_z + 1)
            else:
                if 'class_locations' not in properties.keys():
                    raise RuntimeError("Please rerun the preprocessing with the newest version of nnU-Net!")

                foreground_classes = np.array(
                    [i for i in properties['class_locations'].keys()
                     if len(properties['class_locations'][i]) != 0])
                foreground_classes = foreground_classes[foreground_classes > 0]

                if len(foreground_classes) == 0:
                    selected_class = None
                    voxels_of_that_class = None
                    print('case does not contain any foreground classes', i)
                else:
                    selected_class = np.random.choice(foreground_classes)
                    voxels_of_that_class = properties['class_locations'][selected_class]

                if voxels_of_that_class is not None:
                    selected_voxel = voxels_of_that_class[
                        np.random.choice(len(voxels_of_that_class))]
                    bbox_x_lb = max(lb_x, selected_voxel[0] - self.patch_size[0] // 2)
                    bbox_y_lb = max(lb_y, selected_voxel[1] - self.patch_size[1] // 2)
                    bbox_z_lb = max(lb_z, selected_voxel[2] - self.patch_size[2] // 2)
                else:
                    bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                    bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                    bbox_z_lb = np.random.randint(lb_z, ub_z + 1)

            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]
            bbox_z_ub = bbox_z_lb + self.patch_size[2]

            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)
            valid_bbox_z_lb = max(0, bbox_z_lb)
            valid_bbox_z_ub = min(shape[2], bbox_z_ub)

            case_all_data = np.copy(case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                                    valid_bbox_y_lb:valid_bbox_y_ub,
                                    valid_bbox_z_lb:valid_bbox_z_ub])

            if seg_from_previous_stage is not None:
                seg_from_previous_stage = seg_from_previous_stage[
                    :, valid_bbox_x_lb:valid_bbox_x_ub,
                    valid_bbox_y_lb:valid_bbox_y_ub,
                    valid_bbox_z_lb:valid_bbox_z_ub]

            data[j] = np.pad(case_all_data[:-1], ((0, 0),
                              (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                              (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                              (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                             self.pad_mode, **self.pad_kwargs_data)

            seg[j, 0] = np.pad(case_all_data[-1:], ((0, 0),
                                (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                               'constant', **{'constant_values': -1})

            if seg_from_previous_stage is not None:
                seg[j, 1] = np.pad(seg_from_previous_stage, ((0, 0),
                                    (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                    (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                    (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                                   'constant', **{'constant_values': 0})

        return {'data': data, 'seg': seg, 'properties': case_properties, 'keys': selected_keys}


# =============================================================================
#  FeatureMapFuser — the core new component
# =============================================================================

class FeatureMapFuser(nn.Module):
    """
    Learns a weighted sum of feature maps from K frozen networks and adds it
    as a residual to the active network's feature maps at every spatial level.

    Parameters
    ----------
    num_frozen : int
        Number of frozen networks whose feature maps are being fused (= K for
        run K, where K >= 1).

    The module holds a single nn.Parameter vector `log_alphas` of shape
    (num_frozen,). At forward time these are softmax-normalised so they always
    sum to 1 and are all positive.

    Gradient flow
    -------------
    During run K's forward pass:
        fused_level_l = Σᵢ softmax(log_alpha)ᵢ · frozen_feat[i][l]
        active_feat[l] = active_feat[l] + fused_level_l

    The addition keeps the gradient path alive:  dL/d(log_alphas) is non-zero
    as long as the loss depends on the final prediction, which it always does.

    When num_frozen == 0 (run 0), forward() is a no-op — alphas don't exist
    yet and there is nothing to fuse.

    Level alignment
    ---------------
    Generic_UNet produces feature maps at `num_pool` encoder stages and
    `num_pool` decoder stages (plus the bottleneck).  All frozen networks share
    the same architecture, so spatial sizes match exactly.  The fuser is hooked
    into the active network at the same stages via _register_hooks().
    """

    def __init__(self, num_frozen: int):
        super().__init__()
        self.num_frozen = num_frozen
        if num_frozen > 0:
            # Size K+1: entries 0..K-1 gate frozen nets, entry K gates the active net.
            # Initialised to zeros → softmax gives uniform 1/(K+1) over all slots.
            self.log_alphas = nn.Parameter(torch.zeros(num_frozen + 1))

    def forward(self,
                frozen_feats: List[List[torch.Tensor]],
                active_feat: torch.Tensor,
                level: int) -> torch.Tensor:
        """
        Parameters
        ----------
        frozen_feats : list[list[Tensor]]
            Outer list: one entry per frozen network (length = num_frozen).
            Inner list: feature maps collected by hooks, indexed by level.
        active_feat : Tensor
            The active network's feature map at this level, shape (B, C, *spatial).
        level : int
            Which spatial level is being processed (used to index frozen_feats[i]).

        Returns
        -------
        Tensor : α_K·active_feat + Σᵢ αᵢ·frozen_feat_i  (fully alpha-gated sum).

        alphas has size num_frozen + 1:
          alphas[0..K-1]  weight the K frozen networks
          alphas[K]       weights the active network's own features
        """
        if self.num_frozen == 0:
            return active_feat

        alphas = F.softmax(self.log_alphas, dim=0)  # (num_frozen + 1,)

        contrib = torch.zeros_like(active_feat)
        for i in range(self.num_frozen):
            frozen_f = frozen_feats[i][level]
            # Frozen nets are on the same device; shape must match active_feat
            # (guaranteed by shared architecture + same input patch size)
            contrib = contrib + alphas[i] * frozen_f

        # Gate the active network's features with its own learned alpha weight.
        return alphas[-1] * active_feat + contrib


# =============================================================================
#  Hook manager for extracting intermediate feature maps
# =============================================================================

class FeatureHookManager:
    """
    Registers forward hooks on a Generic_UNet to capture encoder/decoder
    feature maps at every spatial resolution level.

    Generic_UNet stores its encoder blocks in `self.conv_blocks_context` and
    decoder blocks in `self.conv_blocks_localization`.  We hook the output of
    every block in both lists, giving us one feature map per resolution level
    on both encoder and decoder paths.

    Usage
    -----
        mgr = FeatureHookManager(network)
        mgr.register()
        _ = network(x)          # forward pass populates mgr.features
        feats = mgr.features    # list of tensors, one per hooked block
        mgr.clear()             # reset between batches
        mgr.remove()            # remove hooks when done
    """

    def __init__(self, network: nn.Module):
        self.network = network
        self.features: List[torch.Tensor] = []
        self._handles = []

    def _hook_fn(self, module, input, output):
        # output may be a tuple (e.g. when do_ds=True at top level) —
        # but block outputs are plain tensors at the block level
        if isinstance(output, (tuple, list)):
            output = output[0]
        self.features.append(output.detach())  # detach: frozen net, no grad

    def register(self):
        """Attach hooks to encoder and decoder conv blocks."""
        for block in self.network.conv_blocks_context:
            self._handles.append(block.register_forward_hook(self._hook_fn))
        for block in self.network.conv_blocks_localization:
            self._handles.append(block.register_forward_hook(self._hook_fn))

    def clear(self):
        """Clear captured features between iterations."""
        self.features = []

    def remove(self):
        """Remove all hooks (call when done with this frozen network)."""
        for h in self._handles:
            h.remove()
        self._handles = []


class ActiveNetHookManager:
    """
    Registers forward hooks on the ACTIVE network to intercept its feature
    maps and add the fused frozen contribution in-place.

    Unlike FeatureHookManager (which just records), this manager modifies
    the output tensor of each hooked block to include the frozen residual.
    This keeps the gradient path intact — the modification happens inside
    the forward graph.

    Because PyTorch hooks cannot modify the output tensor in-place without
    breaking autograd, we return a new tensor from the hook (PyTorch supports
    returning a replacement output from forward hooks).
    """

    def __init__(self, network: nn.Module, fuser: FeatureMapFuser,
                 frozen_feats_provider):
        """
        Parameters
        ----------
        network : nn.Module
            The active (trainable) network.
        fuser : FeatureMapFuser
            The fusion module whose log_alphas will be optimised.
        frozen_feats_provider : callable
            A callable returning List[List[Tensor]] — the current batch's
            frozen feature maps, indexed [frozen_net_idx][level].
        """
        self.network = network
        self.fuser = fuser
        self.frozen_feats_provider = frozen_feats_provider
        self._handles = []
        self._level_counter = 0

    def _hook_fn(self, module, input, output):
        level = self._level_counter
        self._level_counter += 1
        frozen_feats = self.frozen_feats_provider()
        if isinstance(output, (tuple, list)):
            # Should not happen at block level, but be safe
            fused = self.fuser(frozen_feats, output[0], level)
            return (fused,) + output[1:]
        return self.fuser(frozen_feats, output, level)

    def register(self):
        self._level_counter = 0
        for block in self.network.conv_blocks_context:
            self._handles.append(block.register_forward_hook(self._hook_fn))
        for block in self.network.conv_blocks_localization:
            self._handles.append(block.register_forward_hook(self._hook_fn))

    def reset_counter(self):
        """Call before each forward pass so levels are counted from zero."""
        self._level_counter = 0

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


# =============================================================================
#  Main trainer
# =============================================================================

class nnUNetTrainerV2_SeqWT(nnUNetTrainerV2):
    """
    Sequential Weight Transfer trainer (corrected).

    Trains `num_runs` sequential networks, each a 1-filter-base Generic_UNet.
    Every run after the first:
      - Receives [image, softmax_prob_from_prev_net] as input.
      - Has its feature maps augmented at each level by a learned weighted sum
        of the corresponding frozen network feature maps (FeatureMapFuser).
      - Is initialised with a weighted average of all frozen network weights.
      - Trains log_alphas jointly with its own weights via SGD on the task loss.
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16=False)

        # ---- run schedule ----
        self.num_runs = 32
        self.epochs_per_run = 20
        self.seqwt_base_num_features = 1
        self.seqwt_batch_size = 8

        self.max_num_epochs = self.epochs_per_run
        self.initial_lr = 1e-2
        self.num_batches_per_epoch = 50
        self.num_val_batches_per_epoch = 10
        self.pin_memory = False

        # ---- sequential state ----
        self.frozen_state_dicts: List[OrderedDict] = []
        # Frozen network instances kept alive for hook-based feature extraction
        self.frozen_networks_eval: List[nn.Module] = []
        # One FeatureHookManager per frozen network
        self.frozen_hook_managers: List[FeatureHookManager] = []

        # The fuser module (None for run 0)
        self.fuser: Optional[FeatureMapFuser] = None
        # Hook manager on the active network
        self.active_hook_manager: Optional[ActiveNetHookManager] = None

        # Snapshot of converged softmax alphas from previous run (for init)
        self.prev_alpha_snapshot: Optional[torch.Tensor] = None

        self.current_run = 0
        self.all_run_losses = []
        self.all_run_alphas = []

    # =========================================================================
    #  Initialisation
    # =========================================================================

    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params["do_mirror"] = False
        self.data_aug_params["mirror_axes"] = tuple()

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr = DataLoader3D_RAMCached(
                self.dataset_tr, self.patch_size, self.patch_size,
                self.seqwt_batch_size, False,
                oversample_foreground_percent=self.oversample_foreground_percent,
                pad_mode="constant", pad_sides=self.pad_all_sides)
            dl_val = DataLoader3D_RAMCached(
                self.dataset_val, self.patch_size, self.patch_size,
                self.seqwt_batch_size, False,
                oversample_foreground_percent=self.oversample_foreground_percent,
                pad_mode="constant", pad_sides=self.pad_all_sides)
        else:
            dl_tr = DataLoader2D(self.dataset_tr, self.patch_size, self.patch_size,
                                 self.seqwt_batch_size,
                                 transpose=self.plans.get('transpose_forward'),
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides)
            dl_val = DataLoader2D(self.dataset_val, self.patch_size, self.patch_size,
                                  self.seqwt_batch_size,
                                  transpose=self.plans.get('transpose_forward'),
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides)
        return dl_tr, dl_val

    def initialize(self, training=True, force_load_plans=False):
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.base_num_features = self.seqwt_base_num_features

            # Second channel: softmax prob from previous net (same C as image)
            self.original_num_input_channels = self.num_input_channels
            self.num_input_channels = 2 * self.num_input_channels

            self.setup_DA_params()

            net_numpool = len(self.net_num_pool_op_kernel_sizes)
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])
            mask = np.array([True] + [True if i < net_numpool - 1 else False
                                       for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)

            if training:
                self.folder_with_preprocessed_data = join(
                    self.dataset_directory,
                    self.plans['data_identifier'] + "_stage%d" % self.stage
                )
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")

                self.tr_gen, self.val_gen = get_no_augmentation(
                    self.dl_tr, self.dl_val,
                    params=self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory
                )

                self.print_to_log_file("TRAINING KEYS:\n %s" % str(self.dataset_tr.keys()),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % str(self.dataset_val.keys()),
                                       also_print_to_console=False)

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            self.print_to_log_file(f"[SeqWT] Initialized — "
                                   f"base_num_features={self.seqwt_base_num_features}, "
                                   f"input_channels={self.num_input_channels}, "
                                   f"params={sum(p.numel() for p in self.network.parameters()):,}")
        else:
            self.print_to_log_file('already initialized, skipping')
        self.was_initialized = True

    def initialize_network(self):
        """Build a 1-filter-base Generic_UNet."""
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d
        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}

        self.network = Generic_UNet(
            self.num_input_channels,
            self.seqwt_base_num_features,
            self.num_classes,
            len(self.net_num_pool_op_kernel_sizes),
            self.conv_per_stage,
            2,
            conv_op, norm_op, norm_op_kwargs,
            dropout_op, dropout_op_kwargs,
            net_nonlin, net_nonlin_kwargs,
            True,   # deep_supervision
            False,  # dropout_in_localization
            lambda x: x,
            InitWeights_He(1e-2),
            self.net_num_pool_op_kernel_sizes,
            self.net_conv_kernel_sizes,
            False,  # upscale_logits
            True,   # convolutional_pooling
            True,   # convolutional_upsampling
            max_num_features=None
        )
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        """SGD including fuser.log_alphas when a fuser is present."""
        assert self.network is not None
        params = list(self.network.parameters())
        if self.fuser is not None:
            params += list(self.fuser.parameters())

        self.optimizer = torch.optim.SGD(
            params,
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True
        )
        self.lr_scheduler = None

    # =========================================================================
    #  LR schedule
    # =========================================================================

    def maybe_update_lr(self, epoch=None):
        ep = (epoch if epoch is not None else self.epoch + 1)
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.epochs_per_run, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        super().on_epoch_end()
        return self.epoch < self.max_num_epochs

    # =========================================================================
    #  Weight combination (initialisation only, no gradient needed)
    # =========================================================================

    def _combine_frozen_weights_for_init(self) -> OrderedDict:
        """
        Compute a weighted average of all frozen state_dicts using the
        converged alpha snapshot from the previous run.

        With the K+1 alpha scheme:
          - After Run K finishes, prev_alpha_snapshot has K+1 entries
            (K frozen-net alphas + 1 active-net alpha).
          - At the start of Run K+1 we have K+1 frozen state-dicts (the
            previous active net was just frozen as net K).
          - The K+1 snapshot entries map directly onto those K+1 frozen nets
            — no extension arithmetic needed.

        For run 1 there is only one frozen network so the result is just a
        copy of sd_0 (prev_alpha_snapshot is None after run 0 which had no
        fuser).
        """
        n = len(self.frozen_state_dicts)
        assert n > 0

        if n == 1:
            # Only one frozen network — trivial copy
            return OrderedDict({k: v.clone() for k, v in self.frozen_state_dicts[0].items()})

        # With K+1 alphas, prev_alpha_snapshot has exactly n entries:
        # the last entry was the active-net alpha in the previous run, and
        # that network has just been frozen as frozen_net[n-1].
        if self.prev_alpha_snapshot is not None and len(self.prev_alpha_snapshot) == n:
            alphas = self.prev_alpha_snapshot          # already on CPU, sums to 1
        else:
            alphas = torch.ones(n) / n                 # fallback: uniform

        # Pin to CPU — frozen state dicts are always stored on CPU, and the
        # combined result is loaded via load_state_dict which handles device placement.
        alphas_cpu = alphas.cpu()
        combined = OrderedDict()
        for key in self.frozen_state_dicts[0]:
            stacked = torch.stack([sd[key].cpu().float() for sd in self.frozen_state_dicts])
            alpha_view = alphas_cpu.view(-1, *([1] * (stacked.dim() - 1)))
            combined[key] = (stacked * alpha_view).sum(dim=0)

        return combined

    # =========================================================================
    #  Fuser setup and hook wiring
    # =========================================================================

    def _setup_fuser_for_run(self, run_idx: int):
        """
        Create a new FeatureMapFuser for run_idx, initialise its log_alphas
        from the previous run's converged snapshot, and wire hooks on all
        frozen networks and the active network.
        """
        num_frozen = len(self.frozen_networks_eval)
        assert num_frozen == run_idx, \
            f"Expected {run_idx} frozen nets, found {num_frozen}"

        # Remove any existing hooks
        self._teardown_hooks()

        # Build fuser
        self.fuser = FeatureMapFuser(num_frozen)
        device = next(self.network.parameters()).device
        self.fuser = self.fuser.to(device)

        # Alpha vector is always initialised to zeros → softmax gives uniform
        # 1/(K+1) over all K+1 slots (K frozen nets + 1 active net).
        # Network weight initialisation (done before this call) uses the converged
        # K-element alpha snapshot from the previous run — kept separate.
        init_log_alphas = torch.zeros(num_frozen + 1)

        with torch.no_grad():
            self.fuser.log_alphas.copy_(init_log_alphas.to(device))

        # Register read-only hooks on every frozen network
        self.frozen_hook_managers = []
        for fn in self.frozen_networks_eval:
            mgr = FeatureHookManager(fn)
            mgr.register()
            self.frozen_hook_managers.append(mgr)

        # Register fusion hooks on the active network
        def _get_frozen_feats():
            return [mgr.features for mgr in self.frozen_hook_managers]

        self.active_hook_manager = ActiveNetHookManager(
            self.network, self.fuser, _get_frozen_feats
        )
        self.active_hook_manager.register()

        init_softmax = [f'{v:.4f}' for v in
                        torch.softmax(init_log_alphas, dim=0).tolist()]
        self.print_to_log_file(
            f"  Fuser created: {num_frozen} frozen nets + 1 active, "
            f"alpha init (softmax) = {init_softmax}  "
            f"[uniform 1/{num_frozen + 1} each]"
        )

    def _teardown_hooks(self):
        """Remove all forward hooks cleanly before rebuilding."""
        if self.active_hook_manager is not None:
            self.active_hook_manager.remove()
            self.active_hook_manager = None
        for mgr in self.frozen_hook_managers:
            mgr.remove()
        self.frozen_hook_managers = []

    # =========================================================================
    #  Input channel: softmax probability mask
    # =========================================================================

    def _build_input(self, data: torch.Tensor) -> torch.Tensor:
        """
        Construct the 2C-channel input for the current run.

        Run 0  →  [image, Gaussian noise]  (no previous network)
        Run K  →  [image, softmax_foreground_prob from cascaded frozen nets]

        For run K, we cascade through all K frozen networks sequentially:
          - net_0 receives [image, noise]  → produces softmax prob p_0
          - net_i receives [image, p_{i-1}] → produces softmax prob p_i
          - Final second channel = p_{K-1}

        Using softmax probabilities (not argmax) preserves all uncertainty
        information and gives the active network a richer gradient signal.

        For multi-class (num_classes > 2) we use the max-class probability
        collapsed to a single channel so the channel count stays at 2C.
        For binary (num_classes == 2) we use the foreground class (index 1).
        """
        device = data.device

        if self.current_run == 0:
            noise = torch.randn_like(data)
            return torch.cat([data, noise], dim=1)

        with torch.no_grad():
            # Cascade through frozen networks to build the mask
            prev_prob = torch.randn_like(data)  # start: noise for net_0
            for i, frozen_net in enumerate(self.frozen_networks_eval):
                net_input = torch.cat([data, prev_prob], dim=1)
                ds_was = frozen_net.do_ds
                frozen_net.do_ds = False
                logits = frozen_net(net_input)
                frozen_net.do_ds = ds_was

                soft = torch.softmax(logits, dim=1)  # (B, num_classes, *spatial)

                if self.num_classes == 2:
                    # Binary: foreground probability, kept as (B, 1, *spatial)
                    prev_prob = soft[:, 1:2]
                else:
                    # Multi-class: max probability across classes → (B, 1, *spatial)
                    prev_prob, _ = soft.max(dim=1, keepdim=True)

                # Expand to match image channel count C for consistent 2C input
                orig_ch = data.shape[1]
                if prev_prob.shape[1] != orig_ch:
                    prev_prob = prev_prob.expand(-1, orig_ch, *[-1] * (data.dim() - 2))

        return torch.cat([data, prev_prob], dim=1)

    # =========================================================================
    #  Training iteration
    # =========================================================================

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        Training iteration with feature-map fusion.

        Before the active network's forward pass we must:
          1. Run all frozen networks on the same batch to populate their hook
             managers with current-batch feature maps.
          2. Reset the active hook manager's level counter.
          3. Run the active network (hooks fire and add fused features automatically).
          4. Backpropagate — gradient reaches log_alphas through the additions.
        """
        data_dict = next(data_generator)
        data = maybe_to_torch(data_dict['data']).float()
        target = maybe_to_torch(data_dict['target'])

        device = next(self.network.parameters()).device
        data = data.to(device)
        if isinstance(target, list):
            target = [t.to(device) for t in target]
        else:
            target = target.to(device)

        # Build 2C input (runs frozen cascade inside no_grad)
        data = self._build_input(data)

        # --- Populate frozen feature maps for this batch ---
        if self.fuser is not None and self.fuser.num_frozen > 0:
            for mgr in self.frozen_hook_managers:
                mgr.clear()
            with torch.no_grad():
                for i, frozen_net in enumerate(self.frozen_networks_eval):
                    # Use the same data (already 2C from _build_input) for
                    # the frozen forward pass so spatial sizes match exactly.
                    ds_was = frozen_net.do_ds
                    frozen_net.do_ds = False
                    _ = frozen_net(data)
                    frozen_net.do_ds = ds_was
                    # frozen_hook_managers[i].features is now populated

            # Reset active hook counter so levels are numbered from 0
            self.active_hook_manager.reset_counter()

        # --- Active network forward + loss ---
        self.optimizer.zero_grad()
        output = self.network(data)
        del data
        l = self.loss(output, target)

        if do_backprop:
            l.backward()
            # Clip gradients for both network and fuser parameters
            all_params = list(self.network.parameters())
            if self.fuser is not None:
                all_params += list(self.fuser.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, 12)
            self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target
        return l.detach().cpu().numpy()

    # =========================================================================
    #  Online evaluation (CPU-compatible)
    # =========================================================================

    def run_online_evaluation(self, output, target):
        with torch.no_grad():
            if isinstance(output, (tuple, list)):
                output = output[0]
            if isinstance(target, (tuple, list)):
                target = target[0]

            num_classes = output.shape[1]
            output_softmax = softmax_helper(output)
            output_seg = output_softmax.argmax(1)
            target = target[:, 0]

            tp_hard = torch.zeros((target.shape[0], num_classes - 1))
            fp_hard = torch.zeros((target.shape[0], num_classes - 1))
            fn_hard = torch.zeros((target.shape[0], num_classes - 1))

            for c in range(1, num_classes):
                tp_hard[:, c - 1] = torch.sum(
                    (output_seg == c).float() * (target == c).float(),
                    dim=tuple(range(1, len(target.shape))))
                fp_hard[:, c - 1] = torch.sum(
                    (output_seg == c).float() * (target != c).float(),
                    dim=tuple(range(1, len(target.shape))))
                fn_hard[:, c - 1] = torch.sum(
                    (output_seg != c).float() * (target == c).float(),
                    dim=tuple(range(1, len(target.shape))))

            self.online_eval_foreground_dc.append(
                list((2 * tp_hard.sum(0)) / (2 * tp_hard.sum(0) + fp_hard.sum(0) + fn_hard.sum(0) + 1e-8)))
            self.online_eval_tp.append(list(tp_hard.sum(0).detach().cpu().numpy()))
            self.online_eval_fp.append(list(fp_hard.sum(0).detach().cpu().numpy()))
            self.online_eval_fn.append(list(fn_hard.sum(0).detach().cpu().numpy()))

    # =========================================================================
    #  Main training loop
    # =========================================================================

    def run_training(self):
        """
        Outer loop: num_runs sequential training runs.

        Run 0:
          - He-initialised network, no fuser.
          - Input: [image, noise].

        Run K (K >= 1):
          - Network initialised with alpha-weighted average of frozen weights.
          - Fuser created with K log_alphas; hooks wired.
          - Input: [image, softmax_prob from cascaded frozen nets].
          - log_alphas jointly optimised with network weights.

        After each run:
          - Snapshot converged softmax alphas (for next run's initialisation).
          - Freeze network; store state_dict and eval copy.
        """
        self.print_to_log_file(
            f"\n{'='*70}\n"
            f"SEQUENTIAL WEIGHT TRANSFER (corrected — feature-map fusion)\n"
            f"  Runs: {self.num_runs}, Epochs/run: {self.epochs_per_run}\n"
            f"  Base features: {self.seqwt_base_num_features}\n"
            f"{'='*70}\n"
        )

        total_start = time()

        for run_idx in range(self.num_runs):
            self.current_run = run_idx
            run_start = time()

            self.print_to_log_file(
                f"\n{'='*60}\n"
                f"RUN {run_idx + 1}/{self.num_runs}\n"
                f"  Frozen weight history: {len(self.frozen_state_dicts)} nets\n"
                f"{'='*60}"
            )

            if run_idx == 0:
                # --- Run 0: fresh He-init, no fuser ---
                self.print_to_log_file("  Input: image + Gaussian noise. No fusion.")
                self.fuser = None
                self.active_hook_manager = None
                self.frozen_hook_managers = []

            else:
                # --- Run K: initialise from frozen weights, setup fuser ---

                # Step 1: freeze the previous run's network.
                # IMPORTANT: tear down hooks BEFORE deepcopy.
                # copy.deepcopy copies the module's _forward_hooks dict, so
                # without this the frozen net inherits the ActiveNetHookManager
                # hook. That hook then fires when _build_input runs the frozen
                # cascade, tries to index frozen_feats[i][level] on empty
                # lists, and crashes with IndexError.
                self._teardown_hooks()

                frozen_sd = OrderedDict(
                    {k: v.clone().detach() for k, v in self.network.state_dict().items()})
                self.frozen_state_dicts.append(frozen_sd)

                frozen_net = copy.deepcopy(self.network)   # now hook-free
                frozen_net.eval()
                for p in frozen_net.parameters():
                    p.requires_grad = False
                self.frozen_networks_eval.append(frozen_net)

                self.print_to_log_file(
                    f"  Frozen {len(self.frozen_state_dicts)} network(s) stored.")

                # Step 2: initialise new network from weighted average of frozen weights
                combined_sd = self._combine_frozen_weights_for_init()
                self.initialize_network()
                self.network.do_ds = True
                self.network.load_state_dict(combined_sd)
                self.print_to_log_file("  Network initialised from frozen weight combination.")

                # Step 3: create fuser and wire hooks
                self._setup_fuser_for_run(run_idx)

                # Step 4: fresh optimiser that includes fuser.log_alphas
                self.initialize_optimizer_and_scheduler()

                self.print_to_log_file(
                    f"  Input: image + softmax_prob (cascaded {run_idx} frozen nets).")

            # Reset per-run state
            self.epoch = 0
            self.max_num_epochs = self.epochs_per_run
            self.train_loss_MA = None
            self.best_MA_tr_loss_for_patience = None
            self.best_epoch_based_on_MA_tr_loss = None
            self.val_eval_criterion_MA = None
            self.best_val_eval_criterion_MA = None
            self.all_tr_losses = []
            self.all_val_losses = []
            self.all_val_losses_tr_mode = []
            self.all_val_eval_metrics = []
            self.network.do_ds = True
            self.maybe_update_lr(self.epoch)

            # Run standard nnUNet training loop
            NetworkTrainer.run_training(self)

            # --- Post-run bookkeeping ---

            # Snapshot converged alphas for next run's init
            if self.fuser is not None and self.fuser.num_frozen > 0:
                self.prev_alpha_snapshot = F.softmax(
                    self.fuser.log_alphas.detach().cpu(), dim=0)
                self.all_run_alphas.append(self.prev_alpha_snapshot.tolist())
                self.print_to_log_file(
                    f"  Converged alphas: {[f'{a:.4f}' for a in self.prev_alpha_snapshot.tolist()]}")
            else:
                self.prev_alpha_snapshot = None

            self.all_run_losses.append({
                'run': run_idx,
                'train_losses': [float(x) for x in self.all_tr_losses],
                'val_losses': [float(x) for x in self.all_val_losses],
            })

            run_time = time() - run_start
            self.print_to_log_file(
                f"  Run {run_idx + 1} complete in {run_time / 60:.1f} min")

            run_ckpt = join(self.output_folder, f"model_run{run_idx:02d}.model")
            self.save_checkpoint(run_ckpt)

        # Final: freeze last network
        self._teardown_hooks()
        frozen_sd = OrderedDict(
            {k: v.clone().detach() for k, v in self.network.state_dict().items()})
        self.frozen_state_dicts.append(frozen_sd)
        frozen_net = copy.deepcopy(self.network)
        frozen_net.eval()
        for p in frozen_net.parameters():
            p.requires_grad = False
        self.frozen_networks_eval.append(frozen_net)

        total_time = time() - total_start
        self.print_to_log_file(
            f"\n{'='*70}\n"
            f"TRAINING COMPLETE — {total_time / 60:.1f} min\n"
            f"  Total frozen networks: {len(self.frozen_state_dicts)}\n"
            f"{'='*70}\n"
        )
        self._save_experiment_summary(total_time)

    # =========================================================================
    #  Checkpoint
    # =========================================================================

    def save_checkpoint(self, fname, save_optimizer=True):
        super().save_checkpoint(fname, save_optimizer)

        if len(self.frozen_state_dicts) > 0:
            weights_archive = {
                'num_networks': len(self.frozen_state_dicts),
                'frozen_state_dicts': {str(i): sd for i, sd in enumerate(self.frozen_state_dicts)},
            }
            if self.prev_alpha_snapshot is not None:
                weights_archive['prev_alpha_snapshot'] = self.prev_alpha_snapshot
            if self.fuser is not None:
                weights_archive['fuser_log_alphas'] = self.fuser.log_alphas.detach().cpu()

            weights_file = fname + ".seqwt_weights.pth"
            torch.save(weights_archive, weights_file)
            self.print_to_log_file(
                f"  Saved {len(self.frozen_state_dicts)} frozen nets to {weights_file}")

        seqwt_state = {
            'current_run': self.current_run,
            'num_runs': self.num_runs,
            'epochs_per_run': self.epochs_per_run,
            'all_run_losses': self.all_run_losses,
            'all_run_alphas': self.all_run_alphas,
            'num_frozen_weights': len(self.frozen_state_dicts),
            'prev_alpha_snapshot': (self.prev_alpha_snapshot.tolist()
                                    if self.prev_alpha_snapshot is not None else None),
        }
        seqwt_file = fname.replace('.model', '_seqwt_state.json')
        try:
            with open(seqwt_file, 'w') as f:
                json.dump(seqwt_state, f, indent=2)
        except Exception as e:
            self.print_to_log_file(f"Failed to save SeqWT state: {e}")

    def load_seqwt_weights(self, fname):
        weights_file = fname + ".seqwt_weights.pth"
        if not isfile(weights_file):
            weights_file = join(self.output_folder,
                                "model_final_checkpoint.model.seqwt_weights.pth")
        if not isfile(weights_file):
            raise FileNotFoundError(f"No frozen weights archive found at {weights_file}.")

        archive = torch.load(weights_file, map_location='cpu')
        n = archive['num_networks']
        frozen = [archive['frozen_state_dicts'][str(i)] for i in range(n)]
        self.print_to_log_file(f"Loaded {n} frozen network(s) from {weights_file}")
        return frozen

    def _save_experiment_summary(self, total_time):
        if self.output_folder is None:
            return
        summary = {
            "timestamp": datetime.now().isoformat(),
            "trainer_class": self.__class__.__name__,
            "num_runs": self.num_runs,
            "epochs_per_run": self.epochs_per_run,
            "base_num_features": self.seqwt_base_num_features,
            "batch_size": self.seqwt_batch_size,
            "total_time_minutes": total_time / 60,
            "all_run_losses": self.all_run_losses,
            "all_run_alphas": self.all_run_alphas,
            "network_params": sum(p.numel() for p in self.network.parameters()),
            "num_frozen_networks": len(self.frozen_state_dicts),
            "fusion": "feature_map_residual",
            "mask_channel": "softmax_probability",
        }
        summary_file = join(self.output_folder, "seqwt_experiment_summary.json")
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            self.print_to_log_file(f"Failed to save summary: {e}")

    # =========================================================================
    #  Validation & inference
    # =========================================================================

    def _restore_inference_state(self):
        """
        Restore frozen networks, fuser, and hooks from saved checkpoints so
        that inference exactly replicates the trained forward pass.

        Called lazily on first inference if the trainer was freshly loaded
        (i.e. frozen_networks_eval is empty).

        After this method returns:
          - self.frozen_networks_eval contains N eval-mode frozen nets.
          - self.network holds the final (run N) weights.
          - self.fuser holds a FeatureMapFuser with the final log_alphas.
          - Hooks are wired: frozen nets expose features, active net fuses them.
        """
        ckpt_path = join(self.output_folder, "model_final_checkpoint.model")
        weights_file = ckpt_path + ".seqwt_weights.pth"
        if not isfile(weights_file):
            raise FileNotFoundError(
                f"No frozen weights archive at {weights_file}. Run training first.")

        archive = torch.load(weights_file, map_location='cpu')
        n = archive['num_networks']
        device = next(self.network.parameters()).device

        self.print_to_log_file(f"[SeqWT] Restoring inference state: {n} frozen networks.")

        # Rebuild frozen network instances from saved state dicts.
        # state_dicts[0..n-2] are the frozen historical nets;
        # state_dict[n-1] is the final trained network.
        self.frozen_state_dicts = [archive['frozen_state_dicts'][str(i)] for i in range(n)]
        self.frozen_networks_eval = []

        # All historical nets except the last are the "frozen context" nets
        # that feed the fuser. The last state dict is the active (final) net.
        num_context_nets = n - 1

        for i in range(num_context_nets):
            self.initialize_network()
            self.network.load_state_dict(self.frozen_state_dicts[i])
            fn = copy.deepcopy(self.network)
            fn.eval()
            fn = fn.to(device)
            for p in fn.parameters():
                p.requires_grad = False
            self.frozen_networks_eval.append(fn)

        # Load the final network weights into self.network
        self.initialize_network()
        self.network.load_state_dict(self.frozen_state_dicts[n - 1])
        self.network.eval()

        # Restore fuser with saved log_alphas
        if num_context_nets > 0:
            self.fuser = FeatureMapFuser(num_context_nets).to(device)
            if 'fuser_log_alphas' in archive:
                with torch.no_grad():
                    self.fuser.log_alphas.copy_(archive['fuser_log_alphas'].to(device))
                self.print_to_log_file(
                    f"  Restored log_alphas: softmax = "
                    f"{F.softmax(self.fuser.log_alphas.detach().cpu(), dim=0).tolist()}")
            else:
                self.print_to_log_file(
                    "  WARNING: fuser_log_alphas not found in archive, using uniform.")

            # Wire hooks exactly as during training
            self.frozen_hook_managers = []
            for fn in self.frozen_networks_eval:
                mgr = FeatureHookManager(fn)
                mgr.register()
                self.frozen_hook_managers.append(mgr)

            def _get_frozen_feats():
                return [mgr.features for mgr in self.frozen_hook_managers]

            self.active_hook_manager = ActiveNetHookManager(
                self.network, self.fuser, _get_frozen_feats)
            self.active_hook_manager.register()
        else:
            # Run 0 only — no fuser
            self.fuser = None
            self.frozen_hook_managers = []
            self.active_hook_manager = None

        self.print_to_log_file("[SeqWT] Inference state restored. Hooks active.")

    def _run_fused_patch(self, data_input_np: np.ndarray,
                         mirror_axes, use_sliding_window, step_size,
                         use_gaussian, pad_border_mode, pad_kwargs,
                         all_in_gpu, verbose, mixed_precision) -> tuple:
        """
        Run a single forward pass (or sliding window) on the final network
        with the fuser hooks active.

        Before calling the parent sliding-window inference, we must populate
        the frozen networks' hook managers with features for the current patch.
        The parent's sliding-window loop calls our network many times (once per
        patch position), so we need the hooks to fire on both the frozen nets
        and the active net for every window position.

        We achieve this by wrapping self.network in a thin nn.Module whose
        forward() first runs the frozen nets (populating hooks + feature
        buffers), resets the active hook counter, then runs self.network.
        The parent's sliding-window code calls this wrapper transparently.
        """
        if self.fuser is None or self.fuser.num_frozen == 0:
            # No fusion — straightforward single-network inference
            return super(nnUNetTrainerV2, self).predict_preprocessed_data_return_seg_and_softmax(
                data_input_np, do_mirroring=False, mirror_axes=mirror_axes,
                use_sliding_window=use_sliding_window, step_size=step_size,
                use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                mixed_precision=False)

        # Temporarily replace self.network with a wrapper that runs frozen
        # nets first (to populate hook managers) then runs the real network.
        real_network = self.network
        frozen_nets = self.frozen_networks_eval
        hook_managers = self.frozen_hook_managers
        active_mgr = self.active_hook_manager

        class _FusedForwardWrapper(nn.Module):
            """
            Wraps the active network so that every forward call:
              1. Runs each frozen net (no_grad) to fill its hook manager.
              2. Resets the active hook counter.
              3. Runs the active network (fuser hooks fire automatically).
            """
            def __init__(self):
                super().__init__()
                # Expose the real network's attributes so nnUNet internals
                # (do_ds, inference_apply_nonlin, etc.) work unchanged.
                self._real = real_network

            def __getattr__(self, name):
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(real_network, name)

            def __setattr__(self, name, value):
                if name == '_real':
                    super().__setattr__(name, value)
                else:
                    setattr(real_network, name, value)

            def forward(self, x):
                # Step 1: populate frozen feature maps
                for i, fn in enumerate(frozen_nets):
                    hook_managers[i].clear()
                    ds_was = fn.do_ds
                    fn.do_ds = False
                    with torch.no_grad():
                        fn(x)
                    fn.do_ds = ds_was

                # Step 2: reset active hook level counter
                active_mgr.reset_counter()

                # Step 3: active network forward (fuser fires via hooks)
                return real_network(x)

        wrapper = _FusedForwardWrapper()
        original_network = self.network
        self.network = wrapper

        try:
            ret = super(nnUNetTrainerV2, self).predict_preprocessed_data_return_seg_and_softmax(
                data_input_np, do_mirroring=False, mirror_axes=mirror_axes,
                use_sliding_window=use_sliding_window, step_size=step_size,
                use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                mixed_precision=False)
        finally:
            self.network = original_network  # always restore

        return ret

    def validate(self, do_mirroring=True, use_sliding_window=True,
                 step_size=0.5, save_softmax=True, use_gaussian=True,
                 overwrite=True, validation_folder_name='validation_raw',
                 debug=False, all_in_gpu=False,
                 segmentation_export_kwargs=None,
                 run_postprocessing_on_folds=True):
        do_mirroring = False
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super(nnUNetTrainerV2, self).validate(
            do_mirroring=do_mirroring, use_sliding_window=use_sliding_window,
            step_size=step_size, save_softmax=save_softmax, use_gaussian=use_gaussian,
            overwrite=overwrite, validation_folder_name=validation_folder_name,
            debug=debug, all_in_gpu=all_in_gpu,
            segmentation_export_kwargs=segmentation_export_kwargs,
            run_postprocessing_on_folds=run_postprocessing_on_folds)
        self.network.do_ds = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data, do_mirroring=True,
                                                          mirror_axes=None,
                                                          use_sliding_window=True,
                                                          step_size=0.5,
                                                          use_gaussian=True,
                                                          pad_border_mode='constant',
                                                          pad_kwargs=None,
                                                          all_in_gpu=False,
                                                          verbose=True,
                                                          mixed_precision=False):
        """
        Correct inference procedure that exactly replicates the training forward pass.

        Structure
        ---------
        The final model is network N trained with:
          - Input channel: [image, softmax_prob from cascade of nets 0..N-1]
          - Feature maps: augmented at every level by FeatureMapFuser using
            nets 0..N-1 as frozen context, with learned log_alphas

        Inference must reproduce both of these:

        Step 1 — Build the second input channel via the frozen cascade:
          net_0([image, noise])       → prob_0
          net_1([image, prob_0])      → prob_1
          ...
          net_{N-1}([image, prob_{N-2}]) → prob_{N-1}
          second_channel = prob_{N-1}

        Step 2 — Run the final network with fuser hooks active:
          final_input = [image, prob_{N-1}]
          Frozen nets 0..N-1 run in no_grad to populate hook managers.
          Active net (N) runs with hooks firing → fuser adds Σ αᵢ·fᵢ at each level.
          Output logits → softmax → final prediction.

        Sliding window: for each window position the wrapper in _run_fused_patch
        ensures frozen nets fire first, then the active net, so fusion is
        consistent across all patch positions.

        Lazy restoration: if this is called on a freshly loaded trainer
        (frozen_networks_eval empty), _restore_inference_state() rebuilds
        everything from the checkpoint archive before proceeding.
        """
        # Lazy restore from checkpoint if needed
        if len(self.frozen_networks_eval) == 0:
            try:
                self._restore_inference_state()
            except FileNotFoundError:
                self.print_to_log_file(
                    "WARNING: no frozen archive. Falling back to single-network inference.")
                noise = np.random.randn(*data.shape).astype(data.dtype)
                data_dual = np.concatenate([data, noise], axis=0)
                ds = self.network.do_ds
                self.network.do_ds = False
                ret = super(nnUNetTrainerV2, self).predict_preprocessed_data_return_seg_and_softmax(
                    data_dual, do_mirroring=False, mirror_axes=mirror_axes,
                    use_sliding_window=use_sliding_window, step_size=step_size,
                    use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                    pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                    mixed_precision=False)
                self.network.do_ds = ds
                return ret

        current_mode = self.network.training
        self.network.eval()
        self.network.do_ds = False

        # ---- Step 1: build second input channel via frozen cascade ----
        # We run this on the full image (numpy), mirroring _build_input but
        # without gradients and entirely in numpy/torch CPU for memory safety.
        device = next(self.network.parameters()).device
        num_context_nets = len(self.frozen_networks_eval)

        prev_prob_np = None
        for i, frozen_net in enumerate(self.frozen_networks_eval):
            if i == 0:
                noise = np.random.randn(*data.shape).astype(data.dtype)
                net_input_np = np.concatenate([data, noise], axis=0)
            else:
                orig_ch = data.shape[0]
                prob_tiled = np.tile(prev_prob_np, (orig_ch, *([1] * (data.ndim - 1))))
                net_input_np = np.concatenate([data, prob_tiled], axis=0)

            # Run this frozen net's full sliding-window prediction
            ds_was = frozen_net.do_ds
            frozen_net.do_ds = False
            # Temporarily swap self.network to run parent's sliding window on frozen net
            real_net = self.network
            self.network = frozen_net
            cascade_ret = super(nnUNetTrainerV2, self).predict_preprocessed_data_return_seg_and_softmax(
                net_input_np, do_mirroring=False, mirror_axes=mirror_axes,
                use_sliding_window=use_sliding_window, step_size=step_size,
                use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                mixed_precision=False)
            self.network = real_net
            frozen_net.do_ds = ds_was

            softmax_vol = cascade_ret[1]  # (n_classes, *spatial) numpy
            if self.num_classes == 2:
                prev_prob_np = softmax_vol[1:2]       # foreground prob
            else:
                prev_prob_np = softmax_vol.max(axis=0, keepdims=True)  # max class prob

            if verbose:
                self.print_to_log_file(
                    f"  Cascade net {i + 1}/{num_context_nets}: "
                    f"prob range [{prev_prob_np.min():.3f}, {prev_prob_np.max():.3f}]")

        # ---- Step 2: run final network with fuser hooks active ----
        orig_ch = data.shape[0]
        if prev_prob_np is not None:
            prob_tiled = np.tile(prev_prob_np, (orig_ch, *([1] * (data.ndim - 1))))
            final_input_np = np.concatenate([data, prob_tiled], axis=0)
        else:
            # num_context_nets == 0, run 0 case
            noise = np.random.randn(*data.shape).astype(data.dtype)
            final_input_np = np.concatenate([data, noise], axis=0)

        # _run_fused_patch installs the wrapper that fires frozen hooks then
        # active hooks for every sliding-window position, then restores self.network.
        ret = self._run_fused_patch(
            final_input_np, mirror_axes, use_sliding_window, step_size,
            use_gaussian, pad_border_mode, pad_kwargs, all_in_gpu, verbose, mixed_precision)

        if verbose:
            self.print_to_log_file(
                f"  Final fused prediction: output shape={ret[1].shape}")

        self.network.train(current_mode)
        return ret