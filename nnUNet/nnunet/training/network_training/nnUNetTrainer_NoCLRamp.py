import json
import os
import numpy as np
import torch
from datetime import datetime
from nnunet.training.network_training.nnUNetTrainer_IterativeDenoising import nnUNetTrainer_IterativeDenoising
from nnunet.training.data_augmentation.intensity_controlled_augmentation import get_intensity_controlled_generators
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from batchgenerators.utilities.file_and_folder_operations import join, save_json

class nnUNetTrainer_NoCLRamp(nnUNetTrainer_IterativeDenoising):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        
        # ============== TOGGLES ==============
        self.validation_mode = "standard"  # "standard" or "identical"
        
        # Spectrum Validation Settings
        self.validation_intensity_levels = 10
        self.min_training_intensity = 0.1
        self.max_training_intensity = 1.0
        
        # Track mean robustness across epochs
        self.all_val_eval_metrics_spectrum = []
        
        # Flag to track if we're in training loop (for validate() context detection)
        self._in_training = False
        
        # Best spectrum score for model selection
        self.best_spectrum_score = None
        
        # ============== OUTPUT FOLDER SUFFIX ==============
        if self.validation_mode != "standard":
            suffix = f"_{self.validation_mode}"
            if self.output_folder is not None:
                self.output_folder = self.output_folder + suffix

    def _save_experiment_config(self):
        """Save experiment configuration to a timestamped JSON file."""
        if self.output_folder is None:
            return
        
        config = {
            "timestamp": datetime.now().isoformat(),
            "trainer_class": self.__class__.__name__,
            "validation_mode": self.validation_mode,
            "training_intensity": f"random [{self.min_training_intensity}, {self.max_training_intensity}]",
            "validation_intensity_levels": self.validation_intensity_levels,
            "augmentation_limits": "NNUNET_LIMITS (v1/v2 defaults)",
            "optimizer": "SGD",
            "initial_lr": self.initial_lr,
            "max_num_epochs": self.max_num_epochs,
        }
        
        # Add plans-derived settings if available
        if hasattr(self, 'plans') and self.plans is not None:
            stage_plan = self.plans['plans_per_stage'][self.stage]
            config["batch_size"] = int(stage_plan['batch_size'])
            config["patch_size"] = [int(x) for x in stage_plan['patch_size']]
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_config_{timestamp_str}.json"
        filepath = join(self.output_folder, filename)
        
        os.makedirs(self.output_folder, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.print_to_log_file(f"Experiment config saved to: {filename}")

    def on_epoch_end(self):
        """Override to add spectrum validation at checkpoint saves AND best model saves."""
        # Track whether model_best.model will be saved (detected via MA improvement)
        prev_best_val_ma = self.best_val_eval_criterion_MA
        
        continue_training = super().on_epoch_end()
        
        # Determine if spectrum validation should run
        run_spectrum = False
        
        # Trigger 1: Checkpoint interval (every save_every epochs)
        if self.epoch % self.save_every == (self.save_every - 1):
            run_spectrum = True
        
        # Trigger 2: model_best.model was just saved (best val MA improved)
        current_best_val_ma = self.best_val_eval_criterion_MA
        if (current_best_val_ma is not None and 
            (prev_best_val_ma is None or current_best_val_ma > prev_best_val_ma)):
            run_spectrum = True
        
        if run_spectrum:
            self.print_to_log_file(f"\n=== Spectrum Validation (Epoch {self.epoch}) ===")
            spectrum_score = self.validate_spectrum()
            
            # Save best spectrum model separately
            if self.best_spectrum_score is None or spectrum_score > self.best_spectrum_score:
                self.best_spectrum_score = spectrum_score
                self.save_checkpoint(join(self.output_folder, "model_best_spectrum.model"))
                self.print_to_log_file(f"  New best spectrum model! Score: {spectrum_score:.4f}")
        
        return continue_training

    def initialize(self, training=True, force_load_plans=False):
        """
        Modified to use 'get_intensity_controlled_generators'
        """
        # Call grandparent initialize to set up folders, plans, etc.
        # We skip parent's initialize logic for generators because we replace it immediately
        super(nnUNetTrainer_IterativeDenoising, self).initialize(training, force_load_plans)
        
        if training:
            self.tr_gen, self.val_generators = get_intensity_controlled_generators(
                self.dl_tr, self.dl_val,
                self.data_aug_params['patch_size_for_spatialtransform'],
                self.data_aug_params,
                deep_supervision_scales=self.deep_supervision_scales,
                pin_memory=self.pin_memory,
                min_intensity=self.min_training_intensity,
                max_intensity=self.max_training_intensity,
                validation_levels=self.validation_intensity_levels
            )
            # nnUNet expects self.val_gen to exist for sanity checks, so assign the first level
            self.val_gen = self.val_generators[1]

    def run_training(self):
        """Override to track training context and save experiment config."""
        self._save_experiment_config()
        self._in_training = True
        try:
            return super().run_training()
        finally:
            self._in_training = False

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True, step_size: float = 0.5,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        Override standard validation to perform Spectrum Validation (10 levels) during training,
        or full validation (with nifti output) when called post-training for evaluation.
        
        Uses _in_training flag to detect context, fixing early stopping detection issue where
        epoch comparison would incorrectly route to spectrum validation after early stop.
        """
        # If in training loop, perform spectrum validation (fast, no nifti output)
        if self._in_training:
            return self.validate_spectrum()
        
        # If called after training (inference/evaluation), perform standard validation
        # This generates nifti files in validation_raw/ needed by nnUNet_find_best_configuration
        return super().validate(do_mirroring, use_sliding_window, step_size, save_softmax, use_gaussian,
                                overwrite, validation_folder_name, debug, all_in_gpu,
                                segmentation_export_kwargs, run_postprocessing_on_folds)

    def validate_spectrum(self):
        """
        Validate across all 10 intensity levels.
        """
        self.network.eval()
        level_scores = []
        
        self.print_to_log_file(f"\n--- Spectrum Validation (Epoch {self.epoch}) ---")
        
        for level in range(1, self.validation_intensity_levels + 1):
            intensity = level / self.validation_intensity_levels
            val_gen = self.val_generators[level]
            
            # Use `run_online_evaluation` logic but on validation generator
            # We can reuse _validate_with_generator helper from V2 if available, or write custom loop
            # nnUNetTrainerV2 doesn't expose a clean "validate generator" method that returns Dice
            # So we borrow logic from `run_iteration` but in eval mode
            
            val_dice_scores = []
            
            # Reset online evaluation metrics
            self.online_eval_foreground_dc = []
            self.online_eval_tp = []
            self.online_eval_fp = []
            self.online_eval_fn = []
            
            # Iterate through validation generator (one epoch equivalent)
            # Note: MultiThreadedAugmenter is infinite, need to know num updates
            num_val_batches = self.num_val_batches_per_epoch
            
            with torch.no_grad():
                for _ in range(num_val_batches):
                    batch = next(val_gen)
                    data = batch['data']
                    target = batch['target']
                    
                    data = maybe_to_torch(data)
                    target = maybe_to_torch(target)
                    if torch.cuda.is_available():
                        data = to_cuda(data)
                        target = to_cuda(target)
                    
                    output = self.network(data)
                    
                    # Deep supervision handling
                    if isinstance(output, tuple):
                        output = output[0]
                        target = target[0]
                        
                    self.run_online_evaluation(output, target)
            
            # Compute metric for this level
            self.finish_online_evaluation()
            level_dice = self.all_val_eval_metrics[-1] # Get the metric just added
            
            level_scores.append(level_dice)
            self.print_to_log_file(f"  Level {level} (intensity={intensity:.1f}): Dice={level_dice:.4f}")
            
        mean_robustness = np.mean(level_scores)
        self.print_to_log_file(f"Mean Robustness: {mean_robustness:.4f}")
        self.print_to_log_file("----------------------------------\n")
        
        # Add to history
        self.all_val_eval_metrics_spectrum.append(mean_robustness)
        
        # We also append to standard metric list so finish_online_evaluation logic in parent works
        # (Though we already appended individual level scores, mean robustness is what we care about for 'best')
        # Parent uses self.all_val_eval_metrics[-1] to check for best checkpoint
        self.all_val_eval_metrics[-1] = mean_robustness 
        
        self.network.train()
        return mean_robustness

    def predict_preprocessed_data_return_seg_and_softmax(self, data, **kwargs):
        """
        Inference Input Handling for dual-input network.
        
        data shape: (C, X, Y, Z) → need (2C, X, Y, Z)
        
        Noise modes (set via nnUNet_predict_noise --noise_mode):
            0 = Standard: [img, img]
            1 = Noise in augmented slot: [noise, img]
            2 = Noise in reference slot: [img, noise]
        """
        from nnunet.inference.predict_noise import get_noise_mode
        noise_mode = get_noise_mode()
        
        if noise_mode == 1:
            # Noise in augmented slot, real image in reference slot
            noise = np.random.randn(*data.shape).astype(data.dtype)
            data_dual = np.concatenate([noise, data], axis=0)
        elif noise_mode == 2:
            # Real image in augmented slot, noise in reference slot
            noise = np.random.randn(*data.shape).astype(data.dtype)
            data_dual = np.concatenate([data, noise], axis=0)
        else:
            # Standard: duplicate input
            data_dual = np.concatenate([data, data], axis=0)
        
        return super().predict_preprocessed_data_return_seg_and_softmax(data_dual, **kwargs)

    def load_checkpoint(self, fname, train=True):
        """
        Gap 2 Fix: Checkpoint Compatibility
        Prohibit loading checkpoints from standard nnUNet (channel mismatch)
        """
        print(f"Loading checkpoint {fname}...")
        try:
            super().load_checkpoint(fname, train)
        except RuntimeError as e:
            if "size mismatch" in str(e) and "conv_blocks_context.0.0.weight" in str(e):
                raise RuntimeError("Checkpoint incompatible! You cannot load a standard nnUNet checkpoint "
                                   "into this dual-input trainer. Please train from scratch.") from e
            raise e
