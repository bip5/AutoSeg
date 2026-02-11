"""
nnUNetTrainer_Rigidity - Trainer for Bayesian Rigidity Score Experiment

Extends nnUNetTrainerV2 to use Generic_UNet_Rigid with learned weight rigidity
in the two deepest encoder stages.

Key features:
- Uses Generic_UNet_Rigid instead of Generic_UNet
- Adds KL divergence to the loss function
- Logs rigidity statistics per epoch
- Saves rigidity evolution visualization

The "Rigidity Score" (weight precision = 1/σ²) is learned during training:
- High rigidity: Weight is fundamental to the task
- Low rigidity: Weight is uncertain/can vary without affecting accuracy
"""

import os
import json
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast

from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.network_architecture.generic_UNet_Rigid import Generic_UNet_Rigid
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet.training.loss_functions.rigidity_loss import MultipleOutputRigidityLoss
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.utilities.nd_softmax import softmax_helper
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, save_json


class nnUNetTrainer_Rigidity(nnUNetTrainerV2):
    """
    Trainer for Bayesian Rigidity Score experiment.
    
    Extends nnUNetTrainerV2 with:
    - Generic_UNet_Rigid network architecture
    - KL divergence loss component
    - Rigidity statistics logging
    - Per-epoch rigidity visualization
    """
    
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        
        # Rigidity-specific settings
        self.kl_weight = 1e-3  # Weight for KL divergence term
        self.kl_warmup_epochs = 10  # Linear warmup for KL weight
        self.num_rigid_stages = 2  # Number of deepest stages to use Rigid convolutions
        
        # Rigidity tracking
        self.rigidity_history = []
        self.kl_loss_history = []
        
        # Directory for rigidity stats
        self.rigidity_stats_folder = None
    
    def initialize(self, training=True, force_load_plans=False):
        """
        Override to set up rigidity stats folder.
        """
        super().initialize(training, force_load_plans)
        
        if training:
            # Create folder for rigidity statistics
            self.rigidity_stats_folder = join(self.output_folder, 'rigidity_stats')
            maybe_mkdir_p(self.rigidity_stats_folder)
            
            self.print_to_log_file(f"Rigidity experiment initialized:")
            self.print_to_log_file(f"  KL weight: {self.kl_weight}")
            self.print_to_log_file(f"  KL warmup epochs: {self.kl_warmup_epochs}")
            self.print_to_log_file(f"  Rigid stages: {self.num_rigid_stages}")
    
    def initialize_network(self):
        """
        Override to create Generic_UNet_Rigid instead of Generic_UNet.
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
        
        self.network = Generic_UNet_Rigid(
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
            num_rigid_stages=self.num_rigid_stages
        )
        
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
        
        # Log network info
        rigid_stats = self.network.get_rigidity_summary()
        self.print_to_log_file(f"Generic_UNet_Rigid created:")
        self.print_to_log_file(f"  Rigid stages: {self.network.rigid_stage_indices}")
        self.print_to_log_file(f"  Total rigid params: {rigid_stats['num_params']}")
        self.print_to_log_file(f"  Initial rigidity: mean={rigid_stats['mean']:.4f}, std={rigid_stats['std']:.4f}")
    
    def _setup_loss(self):
        """Set up the rigidity-aware loss function."""
        # Base loss (Dice + CE)
        base_loss = DC_and_CE_loss(
            {'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False},
            {}
        )
        
        # Wrap with KL divergence
        self.loss = MultipleOutputRigidityLoss(
            base_loss,
            self.ds_loss_weights,
            kl_weight=self.kl_weight,
            kl_warmup_epochs=self.kl_warmup_epochs
        )
    
    def initialize_optimizer_and_scheduler(self):
        """Override to set up loss after network is created."""
        super().initialize_optimizer_and_scheduler()
        
        # Set up rigidity loss (needs ds_loss_weights which is set in initialize)
        self._setup_loss()
    
    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        Override to add KL divergence to loss computation.
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
                output = self.network(data, sample=True)
                del data
                
                # Get KL divergence from network
                kl_divergence = self.network.get_kl_divergence()
                
                # Compute loss with KL term
                l = self.loss(output, target, kl_divergence)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data, sample=True)
            del data
            
            # Get KL divergence from network
            kl_divergence = self.network.get_kl_divergence()
            
            # Compute loss with KL term
            l = self.loss(output, target, kl_divergence)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()
    
    def on_epoch_end(self):
        """
        Override to log rigidity statistics and update KL weight.
        """
        # Update loss epoch for warmup
        self.loss.set_epoch(self.epoch)
        
        # Get rigidity statistics
        rigidity_summary = self.network.get_rigidity_summary()
        rigidity_details = self.network.get_rigidity_stats()
        
        # Get loss breakdown
        loss_breakdown = self.loss.get_loss_breakdown()
        
        # Store history
        self.rigidity_history.append({
            'epoch': self.epoch,
            'summary': rigidity_summary,
            'details': rigidity_details
        })
        self.kl_loss_history.append({
            'epoch': self.epoch,
            **loss_breakdown
        })
        
        # Log to console
        self.print_to_log_file(
            f"Rigidity @ epoch {self.epoch}: "
            f"mean={rigidity_summary['mean']:.4f}, std={rigidity_summary['std']:.4f}, "
            f"range=[{rigidity_summary['min']:.4f}, {rigidity_summary['max']:.4f}]"
        )
        self.print_to_log_file(
            f"Loss breakdown: task={loss_breakdown['task_loss']:.4f}, "
            f"kl={loss_breakdown['kl_loss']:.6f} (weight={loss_breakdown['effective_kl_weight']:.6f})"
        )
        
        # Save detailed stats to JSON
        if self.rigidity_stats_folder is not None:
            epoch_stats = {
                'epoch': self.epoch,
                'rigidity': rigidity_summary,
                'loss': loss_breakdown
            }
            save_json(epoch_stats, join(self.rigidity_stats_folder, f'epoch_{self.epoch:04d}.json'))
        
        # Call parent on_epoch_end
        return super().on_epoch_end()
    
    def finish(self):
        """
        Override to save rigidity evolution visualization at end of training.
        """
        # Save complete history
        if self.rigidity_stats_folder is not None:
            save_json({
                'rigidity_history': self.rigidity_history,
                'kl_loss_history': self.kl_loss_history
            }, join(self.rigidity_stats_folder, 'complete_history.json'))
            
            # Generate visualization
            self._save_rigidity_visualization()
        
        return super().finish()
    
    def _save_rigidity_visualization(self):
        """Generate and save rigidity evolution plot."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            if not self.rigidity_history:
                return
            
            epochs = [h['epoch'] for h in self.rigidity_history]
            means = [h['summary']['mean'] for h in self.rigidity_history]
            stds = [h['summary']['std'] for h in self.rigidity_history]
            mins = [h['summary']['min'] for h in self.rigidity_history]
            maxs = [h['summary']['max'] for h in self.rigidity_history]
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # Plot 1: Rigidity evolution
            ax1 = axes[0]
            ax1.fill_between(epochs, mins, maxs, alpha=0.2, color='blue', label='min-max range')
            ax1.fill_between(epochs, 
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            alpha=0.4, color='blue', label='±1 std')
            ax1.plot(epochs, means, 'b-', linewidth=2, label='mean')
            ax1.set_ylabel('Rigidity (Precision)')
            ax1.set_title('Rigidity Evolution During Training')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Loss components
            ax2 = axes[1]
            task_losses = [h['task_loss'] for h in self.kl_loss_history]
            kl_losses = [h['kl_loss'] for h in self.kl_loss_history]
            ax2.plot(epochs, task_losses, 'g-', linewidth=2, label='Task Loss')
            ax2.plot(epochs, kl_losses, 'r-', linewidth=2, label='KL Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_title('Loss Components')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(join(self.rigidity_stats_folder, 'rigidity_evolution.png'), dpi=150)
            plt.close()
            
            self.print_to_log_file(f"Saved rigidity evolution plot to {self.rigidity_stats_folder}")
            
        except ImportError:
            self.print_to_log_file("matplotlib not available, skipping rigidity visualization")
        except Exception as e:
            self.print_to_log_file(f"Error saving rigidity visualization: {e}")
    
    def save_checkpoint(self, fname, save_optimizer=True):
        """
        Override to save rigidity history with checkpoint.
        """
        super().save_checkpoint(fname, save_optimizer)
        
        # Save rigidity state separately
        rigidity_state = {
            'rigidity_history': self.rigidity_history,
            'kl_loss_history': self.kl_loss_history
        }
        rigidity_fname = fname.replace('.model', '_rigidity_state.json')
        save_json(rigidity_state, rigidity_fname)
    
    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        Override to handle rigidity network checkpoint.
        """
        super().load_checkpoint_ram(checkpoint, train)
        
        # Only load rigidity state if training (not during inference)
        # During inference, latest_checkpoint_file() doesn't exist
        if train and hasattr(self, 'output_folder') and self.output_folder is not None:
            try:
                # Try to find rigidity state file in output folder
                rigidity_fname = join(self.output_folder, 'model_final_checkpoint_rigidity_state.json')
                if not os.path.exists(rigidity_fname):
                    # Also try the best checkpoint
                    rigidity_fname = join(self.output_folder, 'model_best_rigidity_state.json')
                
                if os.path.exists(rigidity_fname):
                    with open(rigidity_fname, 'r') as f:
                        rigidity_state = json.load(f)
                    self.rigidity_history = rigidity_state.get('rigidity_history', [])
                    self.kl_loss_history = rigidity_state.get('kl_loss_history', [])
                    self.print_to_log_file(f"Loaded rigidity state from {rigidity_fname}")
            except Exception as e:
                self.print_to_log_file(f"Warning: Could not load rigidity state: {e}")
    
    def predict_preprocessed_data_return_seg_and_softmax(self, data, **kwargs):
        """
        Override to use deterministic inference (sample=False).
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        
        # Store original forward
        original_forward = self.network.forward
        
        # Wrap forward to use sample=False
        def deterministic_forward(x):
            return original_forward(x, sample=False)
        
        self.network.forward = deterministic_forward
        
        ret = super(nnUNetTrainerV2, self).predict_preprocessed_data_return_seg_and_softmax(
            data, **kwargs
        )
        
        # Restore
        self.network.forward = original_forward
        self.network.do_ds = ds
        
        return ret
