"""
nnUNetTrainerV2_SeqWT  -  Sequential Weight Transfer Network

CPU-only trainer that replaces nnUNet's 32-filter parallel capacity with
32 sequential 1-filter training runs. Each run's weights are frozen and
combined via a learned softmax-normalised scalar vector to initialise the
next run.

Schedule (default):
  - 32 runs × 50 epochs, polyLR reset per run
  - base_num_features = 1 (channel progression: 1→2→4→8→16→...)
  - No data augmentation
  - Batch size 8
  - Run 1 input: image + Gaussian noise
  - Run 2+ input: image + predicted mask from prior run
  - Weight combination: softmax(α) · frozen_weights, α jointly optimised

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


# =========================================================================
#  RAM-Cached DataLoader — eliminates per-iteration disk I/O
# =========================================================================

class DataLoader3D_RAMCached(DataLoader3D):
    """
    DataLoader3D subclass that preloads ALL .npy files into RAM at init.

    Standard DataLoader3D uses memmap_mode='r' which re-reads from disk
    on every sample access. For small datasets (like ISLES 2022, ~2GB),
    preloading into RAM eliminates I/O and drops per-iteration time from
    ~4s to ~10ms.
    """

    def __init__(self, data, patch_size, final_patch_size, batch_size,
                 has_prev_stage=False, oversample_foreground_percent=0.0,
                 memmap_mode="r", pad_mode="edge", pad_kwargs_data=None,
                 pad_sides=None):
        # Init parent (this will set up list_of_keys, etc.)
        super().__init__(data, patch_size, final_patch_size, batch_size,
                         has_prev_stage, oversample_foreground_percent,
                         memmap_mode, pad_mode, pad_kwargs_data, pad_sides)

        # Preload ALL cases into RAM
        self._ram_cache = {}
        total_bytes = 0
        for key in self.list_of_keys:
            npy_path = self._data[key]['data_file'][:-4] + ".npy"
            if os.path.isfile(npy_path):
                arr = np.load(npy_path)  # Full load, NOT memmap
            else:
                arr = np.load(self._data[key]['data_file'])['data']
            self._ram_cache[key] = arr
            total_bytes += arr.nbytes

        print(f"[RAMCached] Preloaded {len(self._ram_cache)} cases "
              f"({total_bytes / 1e9:.2f} GB) into RAM")

    def generate_train_batch(self):
        """
        Same as DataLoader3D.generate_train_batch but reads from
        self._ram_cache instead of disk.
        """
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)
        case_properties = []

        for j, i in enumerate(selected_keys):
            if self.get_do_oversample(j):
                force_fg = True
            else:
                force_fg = False

            if 'properties' in self._data[i].keys():
                properties = self._data[i]['properties']
            else:
                from batchgenerators.utilities.file_and_folder_operations import load_pickle
                properties = load_pickle(self._data[i]['properties_file'])
            case_properties.append(properties)

            # === KEY CHANGE: read from RAM cache instead of disk ===
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


class nnUNetTrainerV2_SeqWT(nnUNetTrainerV2):
    """
    Sequential Weight Transfer trainer.

    Trains 32 sequential runs of a 1-filter-base UNet on CPU.
    Each run's frozen weights are combined via a learned scalar vector
    (softmax-normalised) to initialise the next run's weights.
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        # Force fp16 off — CPU doesn't support AMP
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16=False)

        # ============== SEQUENTIAL RUN SCHEDULE ==============
        self.num_runs = 32
        self.epochs_per_run = 50
        self.seqwt_base_num_features = 1
        self.seqwt_batch_size = 2

        # Override nnUNetTrainerV2 defaults
        self.max_num_epochs = self.epochs_per_run
        self.initial_lr = 1e-2
        self.num_batches_per_epoch = 80
        self.num_val_batches_per_epoch = 10
        self.pin_memory = False  # CPU — no pinned memory

        # ============== SEQUENTIAL STATE ==============
        self.frozen_state_dicts = []   # list of OrderedDicts (frozen weight history)
        self.frozen_networks_eval = [] # list of frozen PyTorch models for cascading inference
        self.alpha_params = None       # nn.Parameter vector (learned scalars)
        self.current_run = 0
        self.predicted_masks = {}      # {sample_key: np.ndarray} for mask channel

        # Per-run loss tracking
        self.all_run_losses = []       # list of per-run loss histories
        self.all_run_alphas = []       # list of alpha snapshots

    # =====================================================================
    #  INITIALISATION OVERRIDES
    # =====================================================================

    def setup_DA_params(self):
        """Override to disable mirroring (no augmentation)."""
        super().setup_DA_params()
        self.data_aug_params["do_mirror"] = False
        self.data_aug_params["mirror_axes"] = tuple()

    def get_basic_generators(self):
        """Override to use RAM-cached dataloaders and custom batch size."""
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
        """
        Override to:
        - Use no-augmentation pipeline
        - Double input channels (image + noise/mask)
        - Set base_num_features = 1
        - Stay on CPU
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            # Override base_num_features from plans
            self.base_num_features = self.seqwt_base_num_features

            # Double input channels: image + noise/mask
            self.original_num_input_channels = self.num_input_channels
            self.num_input_channels = 2 * self.num_input_channels

            self.setup_DA_params()

            # ===== Deep supervision loss =====
            net_numpool = len(self.net_num_pool_op_kernel_sizes)
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])
            mask = np.array([True] + [True if i < net_numpool - 1 else False
                                       for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)

            # ===== Data loaders =====
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

                # No-augmentation pipeline (just basic transforms + deep supervision)
                self.tr_gen, self.val_gen = get_no_augmentation(
                    self.dl_tr, self.dl_val,
                    params=self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory
                )

                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))

            self.print_to_log_file(f"[SeqWT] Initialized:")
            self.print_to_log_file(f"  base_num_features = {self.seqwt_base_num_features}")
            self.print_to_log_file(f"  input_channels = {self.num_input_channels} "
                                   f"(2 × {self.original_num_input_channels})")
            self.print_to_log_file(f"  batch_size = {self.seqwt_batch_size}")
            self.print_to_log_file(f"  num_runs = {self.num_runs}, epochs_per_run = {self.epochs_per_run}")
            self.print_to_log_file(f"  params = {sum(p.numel() for p in self.network.parameters()):,}")
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """Build a 1-filter-base Generic_UNet, CPU only, no max_features cap."""
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
            self.seqwt_base_num_features,  # 1 filter
            self.num_classes,
            len(self.net_num_pool_op_kernel_sizes),
            self.conv_per_stage,
            2,  # feat_map_mul_on_downscale
            conv_op, norm_op, norm_op_kwargs,
            dropout_op, dropout_op_kwargs,
            net_nonlin, net_nonlin_kwargs,
            True,   # deep_supervision
            False,  # dropout_in_localization
            lambda x: x,  # final_nonlin
            InitWeights_He(1e-2),
            self.net_num_pool_op_kernel_sizes,
            self.net_conv_kernel_sizes,
            False,  # upscale_logits
            True,   # convolutional_pooling
            True,   # convolutional_upsampling
            max_num_features=None  # No cap — let channels grow naturally
        )
        # Move to GPU if available
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

        self.print_to_log_file(
            f"[SeqWT] Network built: base={self.seqwt_base_num_features}, "
            f"params={sum(p.numel() for p in self.network.parameters()):,}"
        )

    def initialize_optimizer_and_scheduler(self):
        """SGD with polyLR. Includes alpha_params if present."""
        assert self.network is not None, "self.initialize_network must be called first"

        params_to_optimise = list(self.network.parameters())
        if self.alpha_params is not None:
            params_to_optimise.append(self.alpha_params)

        self.optimizer = torch.optim.SGD(
            params_to_optimise,
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True
        )
        self.lr_scheduler = None

    # =====================================================================
    #  LR SCHEDULE
    # =====================================================================

    def maybe_update_lr(self, epoch=None):
        """PolyLR that resets each run (epoch is relative to current run)."""
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.epochs_per_run, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """Override to disable early stopping — always run to epochs_per_run."""
        super().on_epoch_end()
        return self.epoch < self.max_num_epochs

    # =====================================================================
    #  WEIGHT COMBINATION
    # =====================================================================

    def _combine_frozen_weights(self):
        """
        Combine all frozen historical weights using softmax-normalised alphas.

        Returns a state_dict with combined weights for network initialisation.
        The alpha_params gradients are maintained for joint optimisation.
        """
        assert len(self.frozen_state_dicts) > 0, "No frozen weights to combine"
        assert self.alpha_params is not None, "alpha_params not initialised"

        softmax_alphas = F.softmax(self.alpha_params, dim=0)
        combined = OrderedDict()

        for key in self.frozen_state_dicts[0]:
            # Stack all historical versions of this parameter: (num_history, *param_shape)
            stacked = torch.stack([sd[key].float() for sd in self.frozen_state_dicts])
            # Move to same device as the alpha vector (CPU or GPU)
            stacked = stacked.to(softmax_alphas.device)
            # Broadcast alpha over all dims of the weight tensor
            alpha_view = softmax_alphas.view(-1, *([1] * (stacked.dim() - 1)))
            combined[key] = (stacked * alpha_view).sum(dim=0)

        return combined

    # =====================================================================
    #  DATA INPUT: NOISE / MASK INJECTION
    # =====================================================================

    def _inject_second_channel(self, data_dict):
        """
        Modify data_dict in-place to concatenate noise or predicted mask
        as additional channels.

        data_dict['data']: (B, C, *spatial) — already a torch.Tensor
                           from the NumpyToTensor transform.

        After injection: (B, 2*C, *spatial)
        """
        data = data_dict['data']  # torch.Tensor from batchgenerators pipeline

        # Move to same device as the network
        device = next(self.network.parameters()).device
        data = data.to(device)

        if self.current_run == 0:
            # Run 1: concatenate Gaussian noise
            noise = torch.randn_like(data)
            data_dict['data'] = torch.cat([data, noise], dim=1)
        else:
            # Run 2+: cascade through all frozen networks to generate prior mask
            with torch.no_grad():
                # Start with image + noise
                curr_input = torch.cat([data, torch.randn_like(data)], dim=1)
                
                # Cascade through all frozen previous models
                for frozen_net in self.frozen_networks_eval:
                    ds_was = frozen_net.do_ds
                    frozen_net.do_ds = False
                    output = frozen_net(curr_input)
                    frozen_net.do_ds = ds_was

                    output_softmax = torch.softmax(output, dim=1)
                    output_seg = output_softmax.argmax(1, keepdim=True).float()

                    orig_ch = data.shape[1]
                    mask_channels = output_seg.expand(-1, orig_ch, *[-1] * (data.dim() - 2))
                    curr_input = torch.cat([data, mask_channels], dim=1)

            # Assign the cascaded result (image + final mask of run K-1) as input for Run K
            data_dict['data'] = curr_input

    # =====================================================================
    #  TRAINING ITERATION (GPU-enabled)
    # =====================================================================

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        Training iteration with GPU support.
        Injects noise/mask as second channel, moves data to device.
        """
        data_dict = next(data_generator)

        # Inject noise or mask as second channel (also moves to device)
        self._inject_second_channel(data_dict)

        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data).float()
        target = maybe_to_torch(target)

        # Move to GPU if available
        device = next(self.network.parameters()).device
        if device.type == 'cuda':
            data = data.cuda(non_blocking=True)
            if isinstance(target, list):
                target = [i.cuda(non_blocking=True) for i in target]
            else:
                target = target.cuda(non_blocking=True)

        self.optimizer.zero_grad()

        output = self.network(data)
        del data
        l = self.loss(output, target)

        if do_backprop:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    # =====================================================================
    #  ONLINE EVALUATION (CPU-compatible)
    # =====================================================================

    def run_online_evaluation(self, output, target):
        """Override to work on CPU (no .to(device.index) calls)."""
        with torch.no_grad():
            # Deep supervision: take full-resolution output
            if isinstance(output, (tuple, list)):
                output = output[0]
            if isinstance(target, (tuple, list)):
                target = target[0]

            num_classes = output.shape[1]
            output_softmax = softmax_helper(output)
            output_seg = output_softmax.argmax(1)
            target = target[:, 0]
            axes = tuple(range(1, len(target.shape)))

            tp_hard = torch.zeros((target.shape[0], num_classes - 1))
            fp_hard = torch.zeros((target.shape[0], num_classes - 1))
            fn_hard = torch.zeros((target.shape[0], num_classes - 1))

            for c in range(1, num_classes):
                tp_hard[:, c - 1] = torch.sum(
                    (output_seg == c).float() * (target == c).float(),
                    dim=tuple(range(1, len(target.shape)))
                )
                fp_hard[:, c - 1] = torch.sum(
                    (output_seg == c).float() * (target != c).float(),
                    dim=tuple(range(1, len(target.shape)))
                )
                fn_hard[:, c - 1] = torch.sum(
                    (output_seg != c).float() * (target == c).float(),
                    dim=tuple(range(1, len(target.shape)))
                )

            tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

            self.online_eval_foreground_dc.append(
                list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8))
            )
            self.online_eval_tp.append(list(tp_hard))
            self.online_eval_fp.append(list(fp_hard))
            self.online_eval_fn.append(list(fn_hard))

    # =====================================================================
    #  MAIN TRAINING LOOP
    # =====================================================================

    def run_training(self):
        """
        Outer loop controlling 32 sequential runs.

        For each run:
          1. Combine frozen weights (if not run 0)
          2. Initialise network with combined weights
          3. Run 50 epochs of standard training
          4. Freeze weights and add to history
        """
        self.print_to_log_file(
            f"\n{'='*70}\n"
            f"SEQUENTIAL WEIGHT TRANSFER TRAINING\n"
            f"  Runs: {self.num_runs}, Epochs/run: {self.epochs_per_run}\n"
            f"  Base features: {self.seqwt_base_num_features}, Batch size: {self.seqwt_batch_size}\n"
            f"  Device: CPU\n"
            f"{'='*70}\n"
        )

        total_start = time()

        for run_idx in range(self.num_runs):
            self.current_run = run_idx
            run_start = time()

            self.print_to_log_file(
                f"\n{'='*60}\n"
                f"RUN {run_idx + 1}/{self.num_runs}\n"
                f"  Frozen weight history: {len(self.frozen_state_dicts)} runs\n"
                f"{'='*60}"
            )

            if run_idx == 0:
                # First run: network already initialised with He init
                self.print_to_log_file("  Input mode: image + Gaussian noise")
            else:
                # Freeze current weights and add to history
                frozen_sd = OrderedDict()
                for k, v in self.network.state_dict().items():
                    frozen_sd[k] = v.clone().detach()
                self.frozen_state_dicts.append(frozen_sd)

                # Store an eval-mode copy of the network for cascading inference
                import copy
                frozen_net = copy.deepcopy(self.network)
                frozen_net.eval()
                for p in frozen_net.parameters():
                    p.requires_grad = False
                self.frozen_networks_eval.append(frozen_net)

                self.print_to_log_file(
                    f"  Frozen {len(self.frozen_state_dicts)} historical weight sets"
                )

                # Create new alpha_params (one scalar per historical run)
                self.alpha_params = nn.Parameter(
                    torch.ones(len(self.frozen_state_dicts)) / len(self.frozen_state_dicts)
                )

                self.print_to_log_file(
                    f"  Alpha vector: {len(self.frozen_state_dicts)} scalars "
                    f"(softmax-normalised, jointly optimised)"
                )

                # Combine frozen weights → new initialisation
                combined_sd = self._combine_frozen_weights()

                # Reinitialise network and load combined weights
                self.initialize_network()
                self.network.do_ds = True

                # Detach combined weights for loading (they were computed with
                # alpha gradients, but we only want them as initial values)
                detached_sd = OrderedDict()
                for k, v in combined_sd.items():
                    detached_sd[k] = v.detach().clone()
                self.network.load_state_dict(detached_sd)

                self.print_to_log_file("  Combined weights loaded. Input mode: image + predicted mask")

                # Fresh optimizer (includes alpha_params)
                self.initialize_optimizer_and_scheduler()

            # Reset epoch counter for this run
            self.epoch = 0
            self.max_num_epochs = self.epochs_per_run

            # Reset training MA to avoid false early stopping from previous run
            self.train_loss_MA = None
            self.best_MA_tr_loss_for_patience = None
            self.best_epoch_based_on_MA_tr_loss = None
            self.val_eval_criterion_MA = None
            self.best_val_eval_criterion_MA = None

            # Reset loss histories for this run
            self.all_tr_losses = []
            self.all_val_losses = []
            self.all_val_losses_tr_mode = []
            self.all_val_eval_metrics = []

            # Reset LR to initial
            self.maybe_update_lr(self.epoch)

            # Enable deep supervision
            self.network.do_ds = True

            # Run the standard nnUNet training loop for this run
            # Call NetworkTrainer.run_training directly (skip nnUNetTrainerV2 wrapper)
            NetworkTrainer.run_training(self)

            # Log run completion
            run_time = time() - run_start
            self.print_to_log_file(
                f"\n  Run {run_idx + 1} complete in {run_time:.1f}s "
                f"({run_time/60:.1f} min)"
            )

            # Store per-run loss history
            self.all_run_losses.append({
                'run': run_idx,
                'train_losses': [float(x) for x in self.all_tr_losses],
                'val_losses': [float(x) for x in self.all_val_losses],
            })

            # Store alpha values
            if self.alpha_params is not None:
                alphas_snapshot = F.softmax(self.alpha_params.detach(), dim=0).numpy().tolist()
                self.all_run_alphas.append(alphas_snapshot)
                self.print_to_log_file(
                    f"  Learned alphas (softmax): {[f'{a:.4f}' for a in alphas_snapshot]}"
                )

            # Save run checkpoint
            run_ckpt = join(self.output_folder, f"model_run{run_idx:02d}.model")
            self.save_checkpoint(run_ckpt)
            self.print_to_log_file(f"  Checkpoint: {run_ckpt}")

        # ===== FINAL =====
        total_time = time() - total_start

        # Freeze the final run's weights
        frozen_sd = OrderedDict()
        for k, v in self.network.state_dict().items():
            frozen_sd[k] = v.clone().detach()
        self.frozen_state_dicts.append(frozen_sd)

        self.print_to_log_file(
            f"\n{'='*70}\n"
            f"SEQUENTIAL WEIGHT TRANSFER COMPLETE\n"
            f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)\n"
            f"  Total frozen weight sets: {len(self.frozen_state_dicts)}\n"
            f"{'='*70}\n"
        )

        # Save final experiment summary
        self._save_experiment_summary(total_time)

    # =====================================================================
    #  CHECKPOINT & LOGGING
    # =====================================================================

    def save_checkpoint(self, fname, save_optimizer=True):
        """Override to include SeqWT-specific state.

        Saves:
          - Standard nnUNet checkpoint (.model) — current network weights
          - Frozen weights archive (.model.seqwt_weights.pth) — all 32 frozen
            state_dicts in a single dict, keyed by run index
          - SeqWT metadata (.seqwt_state.json) — alphas, losses, run info
        """
        super().save_checkpoint(fname, save_optimizer)

        # ===== Save all frozen state_dicts in a single .pth file =====
        if len(self.frozen_state_dicts) > 0:
            weights_archive = {
                'num_networks': len(self.frozen_state_dicts),
                'frozen_state_dicts': {
                    str(i): sd for i, sd in enumerate(self.frozen_state_dicts)
                },
            }
            if self.alpha_params is not None:
                weights_archive['alpha_params'] = self.alpha_params.detach()

            weights_file = fname + ".seqwt_weights.pth"
            torch.save(weights_archive, weights_file)
            self.print_to_log_file(
                f"  Saved {len(self.frozen_state_dicts)} frozen network(s) "
                f"to {weights_file}"
            )

        # ===== Save SeqWT metadata as JSON =====
        seqwt_state = {
            'current_run': self.current_run,
            'num_runs': self.num_runs,
            'epochs_per_run': self.epochs_per_run,
            'all_run_losses': self.all_run_losses,
            'all_run_alphas': self.all_run_alphas,
            'num_frozen_weights': len(self.frozen_state_dicts),
        }
        if self.alpha_params is not None:
            seqwt_state['alpha_params_raw'] = self.alpha_params.detach().numpy().tolist()
            seqwt_state['alpha_params_softmax'] = F.softmax(
                self.alpha_params.detach(), dim=0
            ).numpy().tolist()

        seqwt_file = fname.replace('.model', '_seqwt_state.json')
        try:
            with open(seqwt_file, 'w') as f:
                json.dump(seqwt_state, f, indent=2)
        except Exception as e:
            self.print_to_log_file(f"Failed to save SeqWT state: {e}")

    def load_seqwt_weights(self, fname):
        """Load frozen weight archive from a checkpoint path.

        Args:
            fname: path to .model file (the .seqwt_weights.pth is derived)

        Returns:
            list of OrderedDicts — one per frozen network
        """
        weights_file = fname + ".seqwt_weights.pth"
        if not isfile(weights_file):
            # Try the final checkpoint path
            weights_file = join(self.output_folder,
                                "model_final_checkpoint.model.seqwt_weights.pth")
        if not isfile(weights_file):
            raise FileNotFoundError(
                f"No frozen weights archive found at {weights_file}. "
                f"Run training first."
            )

        archive = torch.load(weights_file, map_location='cpu')
        n = archive['num_networks']
        frozen = []
        for i in range(n):
            frozen.append(archive['frozen_state_dicts'][str(i)])
        self.print_to_log_file(f"Loaded {n} frozen network(s) from {weights_file}")
        return frozen

    def _save_experiment_summary(self, total_time):
        """Save a comprehensive experiment summary JSON."""
        if self.output_folder is None:
            return

        summary = {
            "timestamp": datetime.now().isoformat(),
            "trainer_class": self.__class__.__name__,
            "num_runs": self.num_runs,
            "epochs_per_run": self.epochs_per_run,
            "base_num_features": self.seqwt_base_num_features,
            "batch_size": self.seqwt_batch_size,
            "total_time_seconds": total_time,
            "total_time_minutes": total_time / 60,
            "device": "GPU",
            "augmentation": "None",
            "lr_schedule": "poly_lr per run",
            "initial_lr": self.initial_lr,
            "weight_combination": "softmax(alpha) . frozen_weights",
            "all_run_losses": self.all_run_losses,
            "all_run_alphas": self.all_run_alphas,
            "network_params": sum(p.numel() for p in self.network.parameters()),
            "num_frozen_networks": len(self.frozen_state_dicts),
        }

        summary_file = join(self.output_folder, "seqwt_experiment_summary.json")
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            self.print_to_log_file(f"Experiment summary saved: {summary_file}")
        except Exception as e:
            self.print_to_log_file(f"Failed to save experiment summary: {e}")

    # =====================================================================
    #  VALIDATION & INFERENCE
    # =====================================================================

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True,
                 overwrite: bool = True, validation_folder_name: str = 'validation_raw',
                 debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None,
                 run_postprocessing_on_folds: bool = True):
        """Override to disable mirroring and deep supervision for validation."""
        if do_mirroring:
            self.print_to_log_file("WARNING: Mirroring disabled (trained without augmentation)")
        do_mirroring = False

        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super(nnUNetTrainerV2, self).validate(
            do_mirroring=do_mirroring, use_sliding_window=use_sliding_window,
            step_size=step_size, save_softmax=save_softmax, use_gaussian=use_gaussian,
            overwrite=overwrite, validation_folder_name=validation_folder_name,
            debug=debug, all_in_gpu=all_in_gpu,
            segmentation_export_kwargs=segmentation_export_kwargs,
            run_postprocessing_on_folds=run_postprocessing_on_folds
        )
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
        Sequential inference through all frozen networks.

        Mirrors the training data flow:
          - Network 0: input = [image, noise]     -> seg_0
          - Network 1: input = [image, seg_0]     -> seg_1
          - Network k: input = [image, seg_{k-1}] -> seg_k
          - Final output = seg_{N-1}  (from the last network)

        All 32 networks share the same architecture (1-filter-base UNet);
        only the weights differ. They are loaded sequentially from the
        frozen_state_dicts archive.
        """
        # Load frozen weights if not already in memory
        if len(self.frozen_state_dicts) == 0:
            try:
                # Try loading from the latest checkpoint
                ckpt_base = join(self.output_folder, "model_final_checkpoint.model")
                self.frozen_state_dicts = self.load_seqwt_weights(ckpt_base)
            except FileNotFoundError:
                # Fall back: we only have the current network
                self.print_to_log_file(
                    "WARNING: No frozen weights archive found. "
                    "Running single-network inference with noise input."
                )
                noise = np.random.randn(*data.shape).astype(data.dtype)
                data_dual = np.concatenate([data, noise], axis=0)

                ds = self.network.do_ds
                self.network.do_ds = False
                ret = super(nnUNetTrainerV2, self).predict_preprocessed_data_return_seg_and_softmax(
                    data_dual, do_mirroring=False, mirror_axes=mirror_axes,
                    use_sliding_window=use_sliding_window, step_size=step_size,
                    use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                    pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                    mixed_precision=False
                )
                self.network.do_ds = ds
                return ret

        num_networks = len(self.frozen_state_dicts)
        if verbose:
            self.print_to_log_file(
                f"[SeqWT] Sequential inference through {num_networks} networks"
            )

        current_mode = self.network.training
        self.network.eval()
        self.network.do_ds = False

        ret = None
        for net_idx in range(num_networks):
            # Load this run's frozen weights
            self.network.load_state_dict(self.frozen_state_dicts[net_idx])

            if net_idx == 0:
                # First network: image + Gaussian noise
                noise = np.random.randn(*data.shape).astype(data.dtype)
                data_input = np.concatenate([data, noise], axis=0)
            else:
                # Subsequent networks: image + previous seg mask
                # prev_seg is (C_classes, *spatial) softmax probabilities
                # Take argmax to get (1, *spatial) mask, then tile to match
                # the original number of input channels
                prev_seg_softmax = ret[1]  # (n_classes, *spatial) from prior run
                prev_mask = prev_seg_softmax.argmax(0)[np.newaxis]  # (1, *spatial)
                prev_mask = prev_mask.astype(data.dtype)
                # Tile to match original channel count
                orig_ch = data.shape[0]  # original C
                prev_mask_tiled = np.tile(prev_mask, (orig_ch, *([1] * (data.ndim - 1))))
                data_input = np.concatenate([data, prev_mask_tiled], axis=0)

            # Run inference for this network
            ret = super(nnUNetTrainerV2, self).predict_preprocessed_data_return_seg_and_softmax(
                data_input, do_mirroring=False, mirror_axes=mirror_axes,
                use_sliding_window=use_sliding_window, step_size=step_size,
                use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                mixed_precision=False
            )

            if verbose:
                self.print_to_log_file(
                    f"  Network {net_idx + 1}/{num_networks}: "
                    f"input={'noise' if net_idx == 0 else 'mask'}, "
                    f"output shape={ret[1].shape}"
                )

        self.network.train(current_mode)
        return ret
