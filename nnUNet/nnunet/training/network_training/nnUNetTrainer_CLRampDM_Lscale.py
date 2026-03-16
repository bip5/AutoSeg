"""
nnUNetTrainer_CLRampDM_Lscale - Loss-Scaled Curriculum Learning Ramp Trainer

Extends the DualMask CLRamp architecture by linking augmentation intensity
directly to the training loss improvement ratio, rather than validation plateaus.

Key features:
- Dual-output architecture (2 x num_classes)
- Product Dice loss
- AdamW (default) with SGD variant
- Intensity is mathematically linked to: (init_loss - best_loss) / init_loss
- Uses multiprocessing.Value to safely pass dynamic intensity to worker threads
"""

import numpy as np
import torch
import multiprocessing
from batchgenerators.utilities.file_and_folder_operations import join, save_json, maybe_mkdir_p

from nnunet.training.network_training.nnUNetTrainer_CLRamp_DualMask import nnUNetTrainer_CLRamp_DualMask
from nnunet.training.data_augmentation.dualmask_augmentation import get_dualmask_generators


class nnUNetTrainer_CLRampDM_Lscale(nnUNetTrainer_CLRamp_DualMask):
    """
    Loss-Scaled CLRamp Trainer for DualMask.
    """
    
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        # Grandparent initialization handles standard nnU-Net folder creation
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        
        # ============== L-SCALE STATE ==============
        # We replace the plateau logic from standard CLRamp with loss tracking
        self.init_loss = None
        self.best_loss_ever = None
        
        # Shared memory for worker threads to read the current intensity.
        # It starts at 0.0 (identity/no intensity) and scales up to 1.0 (max intensity).
        self.shared_intensity = multiprocessing.Value('f', 0.0)
        self.use_adamw = False

    def _save_experiment_config(self):
        """Override to save Lscale tracking state."""
        super()._save_experiment_config()
        
        if self.output_folder is None:
            return
            
        # We save our custom state in a separate file so we don't interfere
        # with the parent's config saving loop when it searches for latest json.
        state_dict = {
            "init_loss": float(self.init_loss) if self.init_loss is not None else None,
            "best_loss_ever": float(self.best_loss_ever) if self.best_loss_ever is not None else None,
            "current_intensity": float(self.shared_intensity.value),
            "use_adamw": self.use_adamw
        }
        
        save_file = join(self.output_folder, "lscale_ramp_state.json")
        try:
            save_json(state_dict, save_file)
        except Exception as e:
            self.print_to_log_file("Failed to save lscale_ramp_state:", e)

    def initialize(self, training=True, force_load_plans=False):
        """
        Modified to pass our multiprocessing `shared_intensity` down into 
        the CLRamp generators via an intensity_getter lambda.
        """
        # Call grandparent initialize to set up folders, plans, etc.
        # (This bypasses the parent nnUNetTrainer_CLRamp_DualMask `initialize` 
        # so we can strictly control `get_dualmask_generators` ourselves here).
        super(nnUNetTrainer_CLRamp_DualMask, self).initialize(training, force_load_plans)
        
        if training:
            # The CLRamp transforms expect a callable that returns the current intensity float.
            # We simply wrap our multiprocessing value in a lambda.
            intensity_getter = lambda: float(self.shared_intensity.value)
            
            self.tr_gen, self.val_generators = get_dualmask_generators(
                self.dl_tr, self.dl_val,
                self.data_aug_params['patch_size_for_spatialtransform'],
                self.data_aug_params,
                intensity_getter=intensity_getter,
                deep_supervision_scales=self.deep_supervision_scales,
                pin_memory=self.pin_memory,
                validation_mode=self.validation_mode,
                validation_levels=self.validation_intensity_levels
            )
            self.val_gen = self.val_generators[1]
            
            self.print_to_log_file(f"CLRampDM_Lscale initialized:")
            self.print_to_log_file(f"  Optimizer: {'AdamW' if self.use_adamw else 'SGD'}")
            self.print_to_log_file(f"  Validation mode: {self.validation_mode}")
            self.print_to_log_file(f"  Initial intensity: {self.shared_intensity.value}")

    def on_epoch_end(self):
        """
        Override standard CLRamp plateau detection.
        Here we calculate epoch loss improvement and mathematically map it to intensity.
        """
        # Let grandparent do checkpointing and validation
        # We use super(nnUNetTrainer_CLRamp_DualMask, self) to bypass the plateau logic
        # entirely from the parent CLRamp class!
        continue_training = super(nnUNetTrainer_CLRamp_DualMask, self).on_epoch_end()
        
        # Calculate L-Scale Update
        if len(self.all_tr_losses) > 0:
            current_epoch_loss = self.all_tr_losses[-1]
            
            # Setup init loss on epoch 0
            if self.init_loss is None:
                self.init_loss = current_epoch_loss
                self.best_loss_ever = current_epoch_loss
                self.print_to_log_file(f"[CLRampDM Lscale] Initial Loss set to {self.init_loss:.4f}")
                
            # Check for new validation loss minimum
            if current_epoch_loss < self.best_loss_ever:
                old_intensity = float(self.shared_intensity.value)
                self.best_loss_ever = current_epoch_loss
                
                # Math: improvement_ratio = (init_loss - best_loss) / (init_loss - target_loss)
                # target_loss for DualMaskProductLoss (Dice product + CE) is -1.0.
                total_possible_improvement = self.init_loss - (-1.0)
                improvement_ratio = (self.init_loss - self.best_loss_ever) / total_possible_improvement
                
                # Clamp between 0.0 and 1.0
                new_intensity = max(0.0, min(1.0, float(improvement_ratio)))
                
                # Update shared memory for worker threads
                self.shared_intensity.value = new_intensity
                
                self.print_to_log_file(
                    f"[CLRampDM Lscale] NEW BEST LOSS! {current_epoch_loss:.4f} < {self.init_loss:.4f} (init)"
                )
                self.print_to_log_file(
                    f"INTENSITY UPDATE: Shared Intensity {old_intensity:.4f} -> {new_intensity:.4f}"
                )
                
        return continue_training

    def save_checkpoint(self, fname, save_optimizer=True):
        """Save network and our custom state files."""
        super(nnUNetTrainer_CLRamp_DualMask, self).save_checkpoint(fname, save_optimizer)
        self._save_experiment_config()


# =================================================================================
# ADAM AND SGD SUBCLASSES (For automatic output folder parsing)
# =================================================================================

class nnUNetTrainer_CLRampDM_Lscale_Adam(nnUNetTrainer_CLRampDM_Lscale):
    """Lscale DualMask Trainer using AdamW optimizer."""
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.use_adamw = True

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), 
            lr=1e-4, 
            weight_decay=3e-5
        )
        self.print_to_log_file("Optimizer: AdamW (lr=1e-4, weight_decay=3e-5)")
        self.lr_scheduler = None


class nnUNetTrainer_CLRampDM_Lscale_SGD(nnUNetTrainer_CLRampDM_Lscale):
    """Lscale DualMask Trainer using default SGD optimizer."""
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.use_adamw = False

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(
            self.network.parameters(), 
            self.initial_lr, 
            weight_decay=self.weight_decay,
            momentum=0.99, 
            nesterov=True
        )
        self.print_to_log_file(f"Optimizer: SGD (lr={self.initial_lr}, momentum=0.99)")
        self.lr_scheduler = None
