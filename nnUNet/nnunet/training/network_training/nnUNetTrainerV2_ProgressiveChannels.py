"""
nnUNetTrainerV2_ProgressiveChannels  -  "Explode Finetuning"

Progressive channel-doubling trainer that starts with a narrow network
and expands capacity in phases, preserving learned weights via Net2Net
tiling at each transition.

Schedule (default):
  Phase 1: base_num_features =  2, train 20 epochs, poly LR cycle
  Phase 2: base_num_features =  8, train 20 epochs, poly LR cycle (4x expansion)
  Phase 3: base_num_features = 32, train 20 epochs, poly LR cycle (4x expansion)

Usage:
    nnUNet_train 3d_fullres nnUNetTrainerV2_ProgressiveChannels TASK_ID FOLD
"""

import numpy as np
import torch
from torch import nn

from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.network_training.expand_state_dict import expand_state_dict
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import join


class nnUNetTrainerV2_ProgressiveChannels(nnUNetTrainerV2):
    """
    Progressive channel-doubling trainer.

    Trains in multiple phases, each with a narrow-to-wide expansion of
    base_num_features.  Between phases, learned weights are tiled into
    the wider network (Net2Net-style) and the LR schedule resets.
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)

        # ============== PROGRESSIVE SCHEDULE ==============
        self.channel_phases = [2, 8, 32]   # base_num_features per phase
        self.epochs_per_phase = 20         # epochs per phase

        # Internal state
        self.current_phase = 0
        self.current_base_features = self.channel_phases[0]

        # Total epochs across all phases (for output folder naming)
        self.total_planned_epochs = len(self.channel_phases) * self.epochs_per_phase

        # Override max_num_epochs to first phase length
        self.max_num_epochs = self.epochs_per_phase

        # Append suffix to output folder
        phase_str = "_".join(str(c) for c in self.channel_phases)
        if self.output_folder is not None:
            self.output_folder = self.output_folder + f"_progressive_{phase_str}_ep{self.epochs_per_phase}"

    def initialize_network(self):
        """
        Build Generic_UNet with current phase's base_num_features.
        """
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
            self.num_input_channels, self.current_base_features, self.num_classes,
            len(self.net_num_pool_op_kernel_sizes),
            self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
            dropout_op_kwargs,
            net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
            self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True
        )
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

        self.print_to_log_file(
            f"Network initialized: base_num_features={self.current_base_features}, "
            f"params={sum(p.numel() for p in self.network.parameters()):,}"
        )

    def _build_network_for_base(self, base_features):
        """Build a throwaway network with given base_features (for shape reference)."""
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

        net = Generic_UNet(
            self.num_input_channels, base_features, self.num_classes,
            len(self.net_num_pool_op_kernel_sizes),
            self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
            dropout_op_kwargs,
            net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, None,
            self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True
        )
        return net

    def maybe_update_lr(self, epoch=None):
        """
        Poly LR that resets each phase.
        epoch is relative to the current phase (0-based within phase).
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.epochs_per_phase, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """Override to always run to epochs_per_phase (no early stopping)."""
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs
        return continue_training

    def run_training(self):
        """
        Outer loop over channel phases.

        For each phase:
          1. Build network with current base_features
          2. If not phase 0, expand weights from previous phase
          3. Initialize fresh optimizer
          4. Run standard nnUNet training loop for epochs_per_phase
          5. Save phase checkpoint
        """
        self.maybe_update_lr(self.epoch)
        ds = self.network.do_ds
        self.network.do_ds = True

        for phase_idx in range(len(self.channel_phases)):
            self.current_phase = phase_idx
            new_base = self.channel_phases[phase_idx]

            self.print_to_log_file(
                f"\n{'='*60}\n"
                f"PHASE {phase_idx + 1}/{len(self.channel_phases)}: "
                f"base_num_features = {new_base}\n"
                f"{'='*60}"
            )

            if phase_idx == 0:
                # First phase: network already initialized by initialize()
                pass
            else:
                # Expand weights from previous phase
                old_base = self.channel_phases[phase_idx - 1]
                old_sd = self.network.state_dict()

                # Build reference networks for shape comparison
                old_ref = self._build_network_for_base(old_base)
                new_ref = self._build_network_for_base(new_base)

                self.print_to_log_file(
                    f"Expanding weights: base {old_base} -> {new_base} "
                    f"({new_base // old_base}x expansion)"
                )

                # Expand state dict
                expanded_sd = expand_state_dict(old_sd, old_ref, new_ref)

                # Clean up references
                del old_ref, new_ref

                # Rebuild network with new base
                self.current_base_features = new_base
                self.initialize_network()
                self.network.do_ds = True

                # Load expanded weights
                self.network.load_state_dict(expanded_sd)
                if torch.cuda.is_available():
                    self.network.cuda()

                self.print_to_log_file("Expanded weights loaded successfully.")

                # Fresh optimizer for this phase
                self.initialize_optimizer_and_scheduler()

            # Reset epoch counter for this phase
            self.epoch = 0
            self.max_num_epochs = self.epochs_per_phase

            # Reset training loss MA (otherwise first epoch may trigger
            # early stopping from previous phase's loss values)
            self.train_loss_MA = None
            self.best_MA_tr_loss_for_patience = None
            self.best_epoch_based_on_MA_tr_loss = None

            # Reset LR to initial
            self.maybe_update_lr(self.epoch)

            # Run the standard nnUNet training loop for this phase
            # We call the grandparent run_training (NetworkTrainer) to avoid
            # nnUNetTrainerV2 re-setting do_ds
            from nnunet.training.network_training.network_trainer import NetworkTrainer
            NetworkTrainer.run_training(self)

            # Save phase checkpoint
            phase_ckpt = join(self.output_folder, f"model_phase{phase_idx + 1}_base{new_base}.model")
            self.save_checkpoint(phase_ckpt)
            self.print_to_log_file(
                f"Phase {phase_idx + 1} complete. Checkpoint: {phase_ckpt}"
            )

        self.network.do_ds = ds

    def save_checkpoint(self, fname, save_optimizer=True):
        """Override to include progressive state."""
        super().save_checkpoint(fname, save_optimizer)

        # Save progressive-specific state
        prog_state = {
            'current_phase': self.current_phase,
            'current_base_features': self.current_base_features,
            'channel_phases': self.channel_phases,
            'epochs_per_phase': self.epochs_per_phase,
        }
        prog_file = fname.replace('.model', '_progressive_state.pkl')
        torch.save(prog_state, prog_file)
