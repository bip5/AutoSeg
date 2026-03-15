"""
nnUNetTrainer_CLRamp_DualMask - Curriculum Learning with Dual Output

Extends NoCLRamp_DualMask with:
- AdamW optimizer (default, toggleable to SGD)
- Progressive intensity ramping based on validation plateau
- AND condition: ramp ONLY when BOTH Dice_aug AND Dice_clean don't improve
- Early stopping after extended plateau

Key features:
- Dual-output architecture (2 × num_classes output channels)
- Product Dice loss: -1 × (Dice_aug × Dice_clean)
- Curriculum learning: starts at intensity 0.1, ramps when both scores plateau
- AND ramping condition: more conservative than single-metric ramping

Ramping logic (Experiment 2 - with gain-guard):
- Track best_product separately
- Ramp only when product has not improved for plateau_patience epochs
- After ramp: PAUSE counter until at least one epoch shows gain relative to
  first post-ramp epoch (prevents premature re-ramping before adaptation)
"""

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, save_json

from nnunet.training.network_training.nnUNetTrainer_NoCLRamp_DualMask import nnUNetTrainer_NoCLRamp_DualMask
from nnunet.training.data_augmentation.dualmask_augmentation import get_dualmask_generators


class nnUNetTrainer_CLRamp_DualMask(nnUNetTrainer_NoCLRamp_DualMask):
    """
    Curriculum Learning Ramp trainer with Dual Mask output.
    
    Extends NoCLRamp_DualMask with:
    - AdamW optimizer (toggleable to SGD)
    - Dynamic intensity ramping with AND condition
    - Early stopping after extended plateau
    """
    
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        
        # ============== TOGGLES ==============
        self.validation_mode = "standard"  # "standard" or "identical"
        
        # ============== INTENSITY RAMP SETTINGS ==============
        self.initial_intensity = 0.1
        self.max_intensity = 1.0
        self.intensity_ramp_step = 0.05
        self.plateau_patience_epochs = 5
        self.early_stop_patience_epochs = 100
        
        # ============== RAMP STATE (saved in checkpoint) ==============
        self.current_intensity = self.initial_intensity
        
        # Product Dice tracking for ramping
        self.epochs_since_product_improvement = 0
        self.best_product = None
        
        # Overall improvement tracking for early stopping
        self.epochs_since_any_improvement = 0
        self.best_product_ever = None
        
        # Ramp history for logging
        self.ramp_history = []  # [(epoch, old_intensity, new_intensity), ...]
        
        # Experiment 2: Gain-guard state
        self.ramp_wait_for_gain = False      # True after ramp, False after first gain
        self.ramp_first_epoch_value = None   # Performance at first epoch after ramp
        
        self.ramp_first_epoch_value = None   # Performance at first epoch after ramp

    def initialize(self, training=True, force_load_plans=False):
        """
        Modified to use DualMask generators with dynamic intensity.
        """
        # Call grandparent initialize (skip NoCLRamp_DualMask's random intensity setup)
        super(nnUNetTrainer_NoCLRamp_DualMask, self).initialize(training, force_load_plans)
        
        if training:
            # Use dynamic intensity from trainer (for curriculum learning)
            self.tr_gen, self.val_generators = get_dualmask_generators(
                self.dl_tr, self.dl_val,
                self.data_aug_params['patch_size_for_spatialtransform'],
                self.data_aug_params,
                intensity_getter=lambda: self.current_intensity,  # Dynamic!
                deep_supervision_scales=self.deep_supervision_scales,
                pin_memory=self.pin_memory,
                validation_mode=self.validation_mode,
                validation_levels=self.validation_intensity_levels
            )
            
            # nnUNet expects self.val_gen for sanity checks
            self.val_gen = self.val_generators[1]
            
            # Set up DualMaskProductLoss (from parent)
            self._setup_dual_mask_loss()
            
            self.print_to_log_file(f"CLRamp_DualMask initialized:")
            self.print_to_log_file(f"  Validation mode: {self.validation_mode}")
            self.print_to_log_file(f"  Initial intensity: {self.current_intensity}")
            self.print_to_log_file(f"  Intensity ramp step: {self.intensity_ramp_step}")
            self.print_to_log_file(f"  Plateau patience: {self.plateau_patience_epochs} epochs")
            self.print_to_log_file(f"  Early stop patience: {self.early_stop_patience_epochs} epochs")
            self.print_to_log_file(f"  Ramping condition: Product Dice plateau")

    # initialize_optimizer_and_scheduler handled by subclasses
    # maybe_update_lr handled by subclasses
    def _save_experiment_config(self):
        """Override to add CLRamp-specific fields."""
        # Call parent to save base config
        super()._save_experiment_config()
        
        # Now append CLRamp fields to the latest config file
        if self.output_folder is None:
            return
        
        import glob
        config_files = sorted(glob.glob(join(self.output_folder, "experiment_config_*.json")))
        if not config_files:
            return
        
        import json
        latest = config_files[-1]
        with open(latest, 'r') as f:
            config = json.load(f)
        
        config.update({
            "initial_intensity": self.initial_intensity,
            "max_intensity": self.max_intensity,
            "intensity_ramp_step": self.intensity_ramp_step,
            "plateau_patience_epochs": self.plateau_patience_epochs,
            "early_stop_patience_epochs": self.early_stop_patience_epochs,
            "training_intensity": f"CLRamp starting at {self.initial_intensity}",
        })
        
        with open(latest, 'w') as f:
            json.dump(config, f, indent=2)

    def on_epoch_end(self):
        """
        Override to implement plateau detection and intensity ramping.
        
        Uses the product Dice from online evaluation (populated by finish_online_evaluation)
        to detect plateau and trigger intensity ramping.
        """
        # First call parent's on_epoch_end which:
        # 1. Calls finish_online_evaluation (populates all_val_eval_metrics)
        # 2. Calls maybe_update_lr
        # 3. Calls maybe_save_checkpoint
        # 4. Updates MA and manages patience
        continue_training = super().on_epoch_end()
        
        # Now check for ramping using the product Dice from online evaluation
        # all_val_eval_metrics is populated by our finish_online_evaluation (product Dice)
        if len(self.all_val_eval_metrics) == 0:
            return continue_training
        
        current_product = self.all_val_eval_metrics[-1]
        
        # Initialize baselines if first epoch
        if self.best_product is None:
            self.best_product = current_product
        if self.best_product_ever is None:
            self.best_product_ever = current_product
        
        # ============== CHECK IMPROVEMENT ON PRODUCT DICE ==============
        # Gain-guard: If we just ramped, wait for at least one improvement
        if self.ramp_wait_for_gain:
            if self.ramp_first_epoch_value is None:
                # First epoch after ramp - capture baseline
                self.ramp_first_epoch_value = current_product
                self.print_to_log_file(
                    f"  [Gain-guard] Post-ramp baseline: {current_product:.4f} "
                    f"(waiting for gain before resuming plateau counter)"
                )
            elif current_product > self.ramp_first_epoch_value:
                # Gained! Resume normal ramping
                old_baseline = self.ramp_first_epoch_value
                self.ramp_wait_for_gain = False
                self.ramp_first_epoch_value = None
                self.best_product = current_product
                self.epochs_since_product_improvement = 0
                self.print_to_log_file(
                    f"  [Gain-guard] Gain achieved ({current_product:.4f} > {old_baseline:.4f}). "
                    f"Resuming plateau counter."
                )
            else:
                # Still waiting for gain - do nothing
                self.print_to_log_file(
                    f"  [Gain-guard] Still waiting for gain (current: {current_product:.4f}, "
                    f"baseline: {self.ramp_first_epoch_value:.4f})"
                )
        else:
            # Normal improvement tracking (not in gain-guard)
            if current_product > self.best_product:
                self.best_product = current_product
                self.epochs_since_product_improvement = 0
            else:
                self.epochs_since_product_improvement += 1
            
        # Track best product ever (for early stopping)
        if current_product > self.best_product_ever:
            self.best_product_ever = current_product
            self.epochs_since_any_improvement = 0
        else:
            self.epochs_since_any_improvement += 1
        
        # ============== PRODUCT DICE RAMPING ==============
        # Only ramp if NOT in gain-guard mode
        if (not self.ramp_wait_for_gain and 
            self.epochs_since_product_improvement >= self.plateau_patience_epochs and 
            self.current_intensity < self.max_intensity):
            
            old_intensity = self.current_intensity
            self.current_intensity = min(
                self.current_intensity + self.intensity_ramp_step, 
                self.max_intensity
            )
            
            # Log the ramp event
            self.ramp_history.append((self.epoch, old_intensity, self.current_intensity))
            self.print_to_log_file(
                f">>> Intensity ramp: {old_intensity:.2f} → {self.current_intensity:.2f} "
                f"at epoch {self.epoch}"
            )
            self.print_to_log_file(
                f"    (Product plateau: {self.epochs_since_product_improvement} epochs, "
                f"Product={current_product:.4f})"
            )
            
            # Activate gain-guard
            self.ramp_wait_for_gain = True
            self.ramp_first_epoch_value = None
            self.epochs_since_product_improvement = 0  # Reset (will be ignored until gain)
        
        # ============== EARLY STOPPING ==============
        if self.epochs_since_any_improvement >= self.early_stop_patience_epochs:
            self.print_to_log_file(
                f">>> Early stopping triggered at epoch {self.epoch} "
                f"(no product improvement for {self.epochs_since_any_improvement} epochs)"
            )
            continue_training = False
        
        return continue_training

    def save_checkpoint(self, fname, save_optimizer=True):
        """
        Override to save CLRamp_DualMask state.
        """
        # Call parent save
        super().save_checkpoint(fname, save_optimizer)
        
        # Save CLRamp-specific state to a separate file
        clramp_state = {
            'current_intensity': self.current_intensity,
            'epochs_since_product_improvement': self.epochs_since_product_improvement,
            'epochs_since_any_improvement': self.epochs_since_any_improvement,
            'best_product': self.best_product,
            'best_product_ever': self.best_product_ever,
            'ramp_history': self.ramp_history,
            'validation_mode': self.validation_mode,
            # Experiment 2: Gain-guard state
            'ramp_wait_for_gain': self.ramp_wait_for_gain,
            'ramp_first_epoch_value': self.ramp_first_epoch_value,
        }
        
        clramp_state_file = fname.replace('.model', '_clramp_dualmask_state.pkl')
        torch.save(clramp_state, clramp_state_file)

    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        Override to load CLRamp state.
        """
        super().load_checkpoint_ram(checkpoint, train)

    def load_checkpoint(self, fname, train=True):
        """
        Override to load CLRamp_DualMask state from separate file.
        """
        super().load_checkpoint(fname, train)
        
        # Load CLRamp-specific state
        clramp_state_file = fname.replace('.model', '_clramp_dualmask_state.pkl')
        try:
            clramp_state = torch.load(clramp_state_file, map_location=torch.device('cpu'), weights_only=False)
            
            self.current_intensity = clramp_state.get('current_intensity', self.initial_intensity)
            self.epochs_since_product_improvement = clramp_state.get('epochs_since_product_improvement', 0)
            self.epochs_since_any_improvement = clramp_state.get('epochs_since_any_improvement', 0)
            self.best_product = clramp_state.get('best_product', None)
            self.best_product_ever = clramp_state.get('best_product_ever', None)
            self.ramp_history = clramp_state.get('ramp_history', [])
            # Experiment 2: Gain-guard state
            self.ramp_wait_for_gain = clramp_state.get('ramp_wait_for_gain', False)
            self.ramp_first_epoch_value = clramp_state.get('ramp_first_epoch_value', None)
            
            self.print_to_log_file(f"Loaded CLRamp_DualMask state:")
            self.print_to_log_file(f"  Current intensity: {self.current_intensity}")
            self.print_to_log_file(f"  Epochs since product improvement: {self.epochs_since_product_improvement}")
            self.print_to_log_file(f"  Best product: {self.best_product}")
            self.print_to_log_file(f"  Ramp history entries: {len(self.ramp_history)}")
            
        except FileNotFoundError:
            self.print_to_log_file(f"WARNING: CLRamp_DualMask state file not found. Using defaults.")
        except Exception as e:
            self.print_to_log_file(f"WARNING: Could not load CLRamp_DualMask state: {e}. Using defaults.")

class nnUNetTrainer_CLRamp_DualMask_Adam(nnUNetTrainer_CLRamp_DualMask):
    """
    DualMask Curriculum Learning Ramp Trainer.
    Uses AdamW optimizer and cosine decay. Provides a clean folder output out-of-the-box.
    """
    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), 
            lr=1e-4, 
            weight_decay=3e-5
        )
        self.print_to_log_file("Optimizer: AdamW (lr=1e-4, weight_decay=3e-5)")
        self.lr_scheduler = None
        
    def _save_experiment_config(self):
        super()._save_experiment_config()
        if self.output_folder is None: return
        import glob, json
        config_files = sorted(glob.glob(join(self.output_folder, "experiment_config_*.json")))
        if config_files:
            latest = config_files[-1]
            with open(latest, 'r') as f: config = json.load(f)
            config["optimizer"] = "AdamW"
            config["initial_lr"] = 1e-4
            with open(latest, 'w') as f: json.dump(config, f, indent=2)

    def maybe_update_lr(self, epoch=None):
        import math
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        new_lr = 1e-4 * 0.5 * (1.0 + math.cos(math.pi * ep / self.max_num_epochs))
        self.optimizer.param_groups[0]['lr'] = new_lr
        self.print_to_log_file("lr:", np.round(new_lr, decimals=6))


class nnUNetTrainer_CLRamp_DualMask_SGD(nnUNetTrainer_CLRamp_DualMask):
    """
    DualMask Curriculum Learning Ramp Trainer.
    Uses standard SGD with poly_lr. Provides a clean folder output out-of-the-box.
    """
    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        sgd_lr = 0.01
        self.optimizer = torch.optim.SGD(
            self.network.parameters(), 
            sgd_lr, 
            weight_decay=self.weight_decay,
            momentum=0.99, 
            nesterov=True
        )
        self.initial_lr = sgd_lr
        self.print_to_log_file(f"Optimizer: SGD (lr={sgd_lr}, momentum=0.99)")
        self.lr_scheduler = None

    def _save_experiment_config(self):
        super()._save_experiment_config()
        if self.output_folder is None: return
        import glob, json
        config_files = sorted(glob.glob(join(self.output_folder, "experiment_config_*.json")))
        if config_files:
            latest = config_files[-1]
            with open(latest, 'r') as f: config = json.load(f)
            config["optimizer"] = "SGD"
            with open(latest, 'w') as f: json.dump(config, f, indent=2)
