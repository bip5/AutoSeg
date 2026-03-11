"""
nnUNetTrainer_SENet - Trainer for SE-Net Symmetry-Enhanced Segmentation

Extends nnUNetTrainerV2 to use the SE-Net architecture with bilateral
symmetry priors for stroke lesion segmentation.

Key features:
- Uses Generic_UNet_SENet instead of Generic_UNet
- Compound loss: Dice + Focal + Symmetry-weighted BCE
- Computes symmetry maps from input data for loss weighting
- Logs learned symmetry parameters (α, β, γ, λ, gate values)

Usage:
    nnUNet_train 3d_fullres nnUNetTrainer_SENet TASK_ID FOLD
"""

import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast

from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.network_architecture.generic_UNet_SENet import Generic_UNet_SENet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.loss_functions.symmetry_loss import (
    DC_Focal_SymBCE_loss, MultipleOutputSymmetryLoss
)
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.utilities.nd_softmax import softmax_helper
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, save_json


class nnUNetTrainer_SENet(nnUNetTrainerV2):
    """
    Trainer for SE-Net symmetry-enhanced segmentation.
    
    Extends nnUNetTrainerV2 with:
    - Generic_UNet_SENet network architecture
    - Compound loss (Dice + Focal + Symmetry BCE)
    - Symmetry map computation during training
    - Symmetry parameter logging
    """
    
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)
        
        # Loss weights (Eq. 8)
        self.weight_dice = 1.0
        self.weight_focal = 0.5
        self.weight_sym = 0.5
        self.sym_beta = 1.0  # β for symmetry weighting (Eq. 15)
        
        # Tracking
        self.symmetry_param_history = []
    
    def initialize(self, training=True, force_load_plans=False):
        """Override to set up symmetry stats folder."""
        super().initialize(training, force_load_plans)
        
        if training:
            self.symmetry_stats_folder = join(self.output_folder, "symmetry_stats")
            maybe_mkdir_p(self.symmetry_stats_folder)
    
    def initialize_network(self):
        """
        Override to create Generic_UNet_SENet instead of Generic_UNet.
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
        
        self.network = Generic_UNet_SENet(
            self.num_input_channels, 
            self.base_num_features, 
            self.num_classes,
            len(self.net_num_pool_op_kernel_sizes),
            self.conv_per_stage, 
            2,  # feat_map_mul_on_downscale
            conv_op, 
            norm_op, 
            norm_op_kwargs, 
            dropout_op,
            dropout_op_kwargs,
            net_nonlin, 
            net_nonlin_kwargs, 
            True,  # deep_supervision
            False,  # dropout_in_localization
            lambda x: x,  # final_nonlin placeholder
            InitWeights_He(1e-2),
            self.net_num_pool_op_kernel_sizes, 
            self.net_conv_kernel_sizes, 
            False,  # upscale_logits
            True,   # convolutional_pooling
            True,   # convolutional_upsampling
        )
        
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
        
        # Log network info
        sym_summary = self.network.get_symmetry_params_summary()
        self.print_to_log_file("Generic_UNet_SENet created:")
        self.print_to_log_file(f"  Symmetry params: {len(sym_summary)} parameters")
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        self.print_to_log_file(f"  Total params: {total_params:,}")
        self.print_to_log_file(f"  Trainable params: {trainable_params:,}")
    
    def _setup_loss(self):
        """Set up the symmetry-aware compound loss function."""
        base_loss = DC_Focal_SymBCE_loss(
            soft_dice_kwargs={'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False},
            focal_kwargs={'gamma': 2, 'smooth': 1e-5},
            weight_dice=self.weight_dice,
            weight_focal=self.weight_focal,
            weight_sym=self.weight_sym,
            sym_beta=self.sym_beta
        )
        
        self.loss = MultipleOutputSymmetryLoss(
            base_loss,
            self.ds_loss_weights
        )
    
    def initialize_optimizer_and_scheduler(self):
        """Override to set up loss after network is created."""
        super().initialize_optimizer_and_scheduler()
        self._setup_loss()
    
    def _compute_symmetry_maps(self, data, target_list):
        """
        Compute symmetry maps from raw input data for the symmetry loss.
        
        For each voxel: w_sym = norm(|x - T(x)|) where T is sagittal reflection.
        Maps are normalised to [0, 1] and downscaled to match each deep
        supervision resolution.
        
        Args:
            data: Input tensor [B, C, H, W, D] or [B, C, H, W]
            target_list: List of targets at different DS resolutions (for shape info)
        
        Returns:
            List of symmetry maps at each DS resolution [B, 1, ...]
        """
        # Compute full-resolution symmetry map
        # Flip along W axis (dim=3 for both 2D and 3D)
        data_flipped = torch.flip(data, dims=[3])
        
        # Per-channel absolute difference, averaged across channels
        sym_diff = torch.abs(data - data_flipped).mean(dim=1, keepdim=True)  # [B, 1, ...]
        
        # Normalise to [0, 1] per sample
        B = sym_diff.shape[0]
        for b in range(B):
            s = sym_diff[b]
            s_min = s.min()
            s_max = s.max()
            if s_max - s_min > 1e-8:
                sym_diff[b] = (s - s_min) / (s_max - s_min)
            else:
                sym_diff[b] = torch.zeros_like(s)
        
        # Create downscaled versions for each DS resolution
        symmetry_maps = []
        for i, tgt in enumerate(target_list):
            if i == 0:
                # Full resolution
                symmetry_maps.append(sym_diff)
            else:
                # Downscale to match target resolution
                target_shape = tgt.shape[2:]  # spatial dimensions
                if len(target_shape) == 3:
                    mode = 'trilinear'
                else:
                    mode = 'bilinear'
                
                sym_ds = torch.nn.functional.interpolate(
                    sym_diff, 
                    size=target_shape,
                    mode=mode, 
                    align_corners=False
                )
                symmetry_maps.append(sym_ds)
        
        return symmetry_maps
    
    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        Override to compute symmetry maps and use compound loss.
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del_data = False
                
                # Compute symmetry maps from input before deleting
                if isinstance(output, (tuple, list)):
                    symmetry_maps = self._compute_symmetry_maps(data, target)
                    del data
                    del_data = True
                    l = self.loss(output, target, symmetry_maps=symmetry_maps)
                else:
                    del data
                    del_data = True
                    l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            
            if isinstance(output, (tuple, list)):
                symmetry_maps = self._compute_symmetry_maps(data, target)
                del data
                l = self.loss(output, target, symmetry_maps=symmetry_maps)
            else:
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
    
    def on_epoch_end(self):
        """Override to log symmetry parameters."""
        # Log symmetry params
        sym_summary = self.network.get_symmetry_params_summary()
        self.symmetry_param_history.append(sym_summary)
        
        # Log to file every 50 epochs
        if self.epoch % 50 == 0 or self.epoch == self.max_num_epochs - 1:
            self.print_to_log_file(f"\n--- Symmetry Parameters (epoch {self.epoch}) ---")
            for key, val in sorted(sym_summary.items()):
                self.print_to_log_file(f"  {key}: {val:.4f}")
            self.print_to_log_file("---")
            
            # Also log loss breakdown if available
            if hasattr(self.loss, 'get_loss_breakdown'):
                breakdown = self.loss.get_loss_breakdown()
                self.print_to_log_file(f"  Loss breakdown: {breakdown}")
        
        # Call parent
        ret = super().on_epoch_end()
        return ret
    
    def finish(self):
        """Override to save symmetry parameter evolution at end of training."""
        # Save symmetry parameter history
        if self.symmetry_param_history:
            save_json(
                self.symmetry_param_history,
                join(self.symmetry_stats_folder, "symmetry_params_history.json")
            )
            self.print_to_log_file(f"Symmetry parameter history saved to {self.symmetry_stats_folder}")
        
        super().finish()
    
    def save_checkpoint(self, fname, save_optimizer=True):
        """Override to save symmetry history with checkpoint."""
        super().save_checkpoint(fname, save_optimizer)
        # The symmetry params (α, β, γ, λ, gate) are part of the network 
        # state_dict, so they're automatically saved
    
    def load_checkpoint_ram(self, checkpoint, train=True):
        """Override to handle SE-Net checkpoint loading."""
        # Load the network state dict which includes symmetry params
        super().load_checkpoint_ram(checkpoint, train)
        
        # Re-setup loss if training (in case ds_weights changed)
        if train and hasattr(self, 'ds_loss_weights') and self.ds_loss_weights is not None:
            self._setup_loss()
