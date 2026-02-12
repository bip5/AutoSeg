"""
nnUNetTrainer_CLRamp - Curriculum Learning Ramp Trainer

Progressive augmentation intensity ramp based on validation plateau detection.

Key features:
- AdamW optimizer (default) with SGD toggle
- Intensity starts at 0.1, ramps up by 0.05 when validation plateaus for 5 epochs
- Early stopping after 100 epochs without improvement
- Validation mode: "standard" (conditioning=clean) or "identical" (conditioning=augmented)
- Checkpoint persistence for all ramp state

To switch to SGD:
    trainer.use_adamw = False

To switch validation mode:
    trainer.validation_mode = "identical"
"""

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, save_json

from nnunet.training.network_training.nnUNetTrainer_NoCLRamp import nnUNetTrainer_NoCLRamp
from nnunet.training.data_augmentation.clramp_augmentation import get_clramp_generators


class nnUNetTrainer_CLRamp(nnUNetTrainer_NoCLRamp):
    """
    Curriculum Learning Ramp trainer for iterative denoising.
    
    Extends NoCLRamp with:
    - AdamW optimizer (toggleable to SGD)
    - Dynamic intensity ramping based on validation plateau
    - Early stopping after extended plateau
    """
    
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        
        # ============== TOGGLES ==============
        self.use_adamw = False  # Set False for standard SGD
        self.validation_mode = "standard"  # "standard" or "identical"
        
        # ============== INTENSITY RAMP SETTINGS ==============
        self.initial_intensity = 0.1
        self.max_intensity = 1.0
        self.intensity_ramp_step = 0.05
        self.plateau_patience_epochs = 5
        self.early_stop_patience_epochs = 100
        
        # ============== RAMP STATE (saved in checkpoint) ==============
        self.current_intensity = self.initial_intensity
        self.epochs_since_improvement = 0
        self.epochs_since_any_improvement = 0
        self.baseline_epoch_score = None
        self.best_score_ever = None
        self.ramp_history = []  # [(epoch, old_intensity, new_intensity), ...]
        
        # ============== OUTPUT FOLDER SUFFIX (replaces NoCLRamp's base suffix) ==============
        opt = "adamw" if self.use_adamw else "sgd"
        if self.output_folder is not None:
            # Remove parent's validation_mode suffix (e.g. "_standard")
            parent_suffix = f"_{self.validation_mode}"
            if self.output_folder.endswith(parent_suffix):
                self.output_folder = self.output_folder[:-len(parent_suffix)]
            # Apply full CLRamp suffix
            self.output_folder = self.output_folder + f"_{self.validation_mode}_{opt}_i{self.initial_intensity}_p{self.plateau_patience_epochs}"
    
    def _save_experiment_config(self):
        """Override to add CLRamp-specific fields."""
        # Call parent to save base config
        super()._save_experiment_config()
        
        # Append CLRamp fields to the latest config file
        if self.output_folder is None:
            return
        
        import glob
        import json
        config_files = sorted(glob.glob(join(self.output_folder, "experiment_config_*.json")))
        if not config_files:
            return
        
        latest = config_files[-1]
        with open(latest, 'r') as f:
            config = json.load(f)
        
        config.update({
            "use_adamw": self.use_adamw,
            "optimizer": "AdamW" if self.use_adamw else "SGD",
            "initial_intensity": self.initial_intensity,
            "max_intensity": self.max_intensity,
            "intensity_ramp_step": self.intensity_ramp_step,
            "plateau_patience_epochs": self.plateau_patience_epochs,
            "early_stop_patience_epochs": self.early_stop_patience_epochs,
            "training_intensity": f"CLRamp starting at {self.initial_intensity}",
        })
        
        if self.use_adamw:
            config["initial_lr"] = 1e-4
        
        with open(latest, 'w') as f:
            json.dump(config, f, indent=2)

    def initialize(self, training=True, force_load_plans=False):
        """
        Modified to use CLRamp generators with dynamic intensity.
        """
        # Call grandparent initialize to set up folders, plans, etc.
        # We skip parent's initialize logic for generators because we override immediately
        super(nnUNetTrainer_NoCLRamp, self).initialize(training, force_load_plans)
        
        if training:
            # Use CLRamp generators with dynamic intensity getter
            self.tr_gen, self.val_generators = get_clramp_generators(
                self.dl_tr, self.dl_val,
                self.data_aug_params['patch_size_for_spatialtransform'],
                self.data_aug_params,
                intensity_getter=lambda: self.current_intensity,
                deep_supervision_scales=self.deep_supervision_scales,
                pin_memory=self.pin_memory,
                validation_mode=self.validation_mode,
                validation_levels=self.validation_intensity_levels
            )
            # nnUNet expects self.val_gen to exist for sanity checks
            self.val_gen = self.val_generators[1]
            
            self.print_to_log_file(f"CLRamp initialized:")
            self.print_to_log_file(f"  Optimizer: {'AdamW' if self.use_adamw else 'SGD'}")
            self.print_to_log_file(f"  Validation mode: {self.validation_mode}")
            self.print_to_log_file(f"  Initial intensity: {self.current_intensity}")
            self.print_to_log_file(f"  Intensity ramp step: {self.intensity_ramp_step}")
            self.print_to_log_file(f"  Plateau patience: {self.plateau_patience_epochs} epochs")
            self.print_to_log_file(f"  Early stop patience: {self.early_stop_patience_epochs} epochs")

    def initialize_optimizer_and_scheduler(self):
        """
        Override to use AdamW (default) or SGD (toggle).
        """
        assert self.network is not None, "self.initialize_network must be called first"
        
        if self.use_adamw:
            # AdamW optimizer
            self.optimizer = torch.optim.AdamW(
                self.network.parameters(), 
                lr=1e-4, 
                weight_decay=3e-5
            )
            self.print_to_log_file("Optimizer: AdamW (lr=1e-4, weight_decay=3e-5)")
        else:
            # Standard nnUNet SGD
            self.optimizer = torch.optim.SGD(
                self.network.parameters(), 
                self.initial_lr, 
                weight_decay=self.weight_decay,
                momentum=0.99, 
                nesterov=True
            )
            self.print_to_log_file(f"Optimizer: SGD (lr={self.initial_lr}, momentum=0.99)")
        
        self.lr_scheduler = None

    def on_epoch_end(self):
        """
        Override to implement plateau detection and intensity ramping.
        """
        # Get current validation score (mean robustness from spectrum validation)
        current_score = self.all_val_eval_metrics[-1] if len(self.all_val_eval_metrics) > 0 else 0
        
        # Initialize baseline if first epoch
        if self.baseline_epoch_score is None:
            self.baseline_epoch_score = current_score
            self.best_score_ever = current_score
        
        # Check for improvement vs baseline (for ramp trigger)
        improved_vs_baseline = current_score > self.baseline_epoch_score
        
        # Check for improvement vs best ever (for early stopping)
        improved_overall = current_score > self.best_score_ever
        if improved_overall:
            self.best_score_ever = current_score
            self.epochs_since_any_improvement = 0
        else:
            self.epochs_since_any_improvement += 1
        
        # Plateau detection and ramping
        if improved_vs_baseline:
            self.epochs_since_improvement = 0
            self.baseline_epoch_score = current_score
        else:
            self.epochs_since_improvement += 1
            
            # Trigger ramp if plateau reached and not at max intensity
            if (self.epochs_since_improvement >= self.plateau_patience_epochs and 
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
                    f"at epoch {self.epoch} (plateau for {self.epochs_since_improvement} epochs)"
                )
                
                # Reset counters after ramp
                self.epochs_since_improvement = 0
                self.baseline_epoch_score = current_score
        
        # Call parent on_epoch_end (handles validation, checkpointing, etc.)
        continue_training = super().on_epoch_end()
        
        # Early stopping check
        if self.epochs_since_any_improvement >= self.early_stop_patience_epochs:
            self.print_to_log_file(
                f">>> Early stopping triggered at epoch {self.epoch} "
                f"(no improvement for {self.epochs_since_any_improvement} epochs)"
            )
            continue_training = False
        
        return continue_training

    def save_checkpoint(self, fname, save_optimizer=True):
        """
        Override to save CLRamp state.
        """
        # Call parent save
        super().save_checkpoint(fname, save_optimizer)
        
        # Save CLRamp-specific state to a separate file
        clramp_state = {
            'current_intensity': self.current_intensity,
            'epochs_since_improvement': self.epochs_since_improvement,
            'epochs_since_any_improvement': self.epochs_since_any_improvement,
            'baseline_epoch_score': self.baseline_epoch_score,
            'best_score_ever': self.best_score_ever,
            'ramp_history': self.ramp_history,
            'validation_mode': self.validation_mode,
            'use_adamw': self.use_adamw,
        }
        
        clramp_state_file = fname.replace('.model', '_clramp_state.pkl')
        torch.save(clramp_state, clramp_state_file)

    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        Override to load CLRamp state.
        """
        super().load_checkpoint_ram(checkpoint, train)
        
        # Try to load CLRamp state
        # Note: checkpoint is already loaded, we need to find the clramp state file
        # This is called from load_checkpoint which has the filename

    def load_checkpoint(self, fname, train=True):
        """
        Override to load CLRamp state from separate file.
        """
        super().load_checkpoint(fname, train)
        
        # Load CLRamp-specific state
        clramp_state_file = fname.replace('.model', '_clramp_state.pkl')
        try:
            clramp_state = torch.load(clramp_state_file, map_location=torch.device('cpu'), weights_only=False)
            
            self.current_intensity = clramp_state.get('current_intensity', self.initial_intensity)
            self.epochs_since_improvement = clramp_state.get('epochs_since_improvement', 0)
            self.epochs_since_any_improvement = clramp_state.get('epochs_since_any_improvement', 0)
            self.baseline_epoch_score = clramp_state.get('baseline_epoch_score', None)
            self.best_score_ever = clramp_state.get('best_score_ever', None)
            self.ramp_history = clramp_state.get('ramp_history', [])
            
            self.print_to_log_file(f"Loaded CLRamp state:")
            self.print_to_log_file(f"  Current intensity: {self.current_intensity}")
            self.print_to_log_file(f"  Epochs since improvement: {self.epochs_since_improvement}")
            self.print_to_log_file(f"  Ramp history entries: {len(self.ramp_history)}")
            
        except FileNotFoundError:
            self.print_to_log_file(f"WARNING: CLRamp state file not found ({clramp_state_file}). Using defaults.")
        except Exception as e:
            self.print_to_log_file(f"WARNING: Could not load CLRamp state: {e}. Using defaults.")

    def maybe_update_lr(self, epoch=None):
        """
        Override LR scheduling for AdamW (constant LR) vs SGD (poly_lr).
        """
        if self.use_adamw:
            # AdamW: Keep constant LR (no poly decay)
            # You can add warmup or other schedules here if needed
            self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))
        else:
            # SGD: Use standard poly_lr from parent
            super().maybe_update_lr(epoch)
