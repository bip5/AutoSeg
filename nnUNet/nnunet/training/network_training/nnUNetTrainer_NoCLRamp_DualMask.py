"""
nnUNetTrainer_NoCLRamp_DualMask - Dual Output with Product Dice Loss

Network predicts TWO masks:
- Mask for augmented input (evaluates ability to segment corrupted images)
- Mask for clean input (ensures network doesn't ignore clean reference)

Loss = -1 × (Dice_aug × Dice_clean) + CE_aug + CE_clean

Key features:
- Dual-output architecture (2 × num_classes output channels)
- Product Dice loss forces network to excel at BOTH tasks
- Random intensity [0.1, 1.0] per batch (same as NoCLRamp)
- Spectrum validation at 10 intensity levels
- Product Dice used as validation metric

At inference: Uses clean-path mask (since test input is duplicated into both channels)
"""

import json
import os
import numpy as np
import torch
from datetime import datetime
from torch import nn

from nnunet.training.network_training.nnUNetTrainer_IterativeDenoising import nnUNetTrainer_IterativeDenoising
from nnunet.training.data_augmentation.dualmask_augmentation import get_dualmask_generators
from nnunet.training.loss_functions.dual_mask_loss import DualMaskProductLoss, MultipleOutputDualMaskLoss
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from batchgenerators.utilities.file_and_folder_operations import join, save_json


class nnUNetTrainer_NoCLRamp_DualMask(nnUNetTrainer_IterativeDenoising):
    """
    No Curriculum Learning Ramp trainer with Dual Mask output.
    
    Extends IterativeDenoising with:
    - Random intensity [0.1, 1.0] per batch
    - Dual-output architecture (2 × num_classes)
    - Product Dice loss
    - Spectrum validation at 10 intensity levels
    - Mean product Dice as validation metric
    """
    
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        
        # ============== TOGGLES ==============
        self.validation_mode = "standard"  # "standard" or "identical"
        self.do_spectrum_validation = False  # Set True to enable spectrum validation during training
        
        # Spectrum Validation Settings (same as NoCLRamp)
        self.validation_intensity_levels = 10
        self.min_training_intensity = 0.1
        self.max_training_intensity = 1.0
        
        # Track metrics
        self.all_val_eval_metrics_spectrum = []
        
        # For separate Dice tracking
        self.last_dice_aug = []
        self.last_dice_clean = []
        
        # Flag to track training context (for validate() routing)
        self._in_training = False
        
        # Best spectrum score for model selection
        self.best_spectrum_score = None
        
        # ============== OUTPUT FOLDER SUFFIX ==============
        suffix = f"_{self.validation_mode}"
        if self.output_folder is not None:
            self.output_folder = self.output_folder + suffix

    def initialize_network(self):
        """
        Override to create network with:
        - DOUBLE input channels (for dual input: augmented + clean)
        - DOUBLE output channels (for dual mask: aug prediction + clean prediction)
        
        Input: (B, 2 × num_input_channels, X, Y, Z)
        Output: (B, 2 × num_classes, X, Y, Z)
            - [:, :num_classes] = prediction for augmented input
            - [:, num_classes:] = prediction for clean input
        """
        # FIRST: Double input channels like parent IterativeDenoising does
        self.num_input_channels = 2 * self.num_input_channels
        
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
        
        # SECOND: Double output channels for dual mask prediction
        output_channels = 2 * self.num_classes
        
        self.network = Generic_UNet(
            self.num_input_channels,  # Now doubled (2 × original)
            self.base_num_features, 
            output_channels,          # DOUBLED for dual mask
            len(self.net_num_pool_op_kernel_sizes),
            self.conv_per_stage, 
            2, 
            conv_op, 
            norm_op, 
            norm_op_kwargs, 
            dropout_op,
            dropout_op_kwargs,
            net_nonlin, 
            net_nonlin_kwargs, 
            True,  # deep_supervision
            False, 
            lambda x: x, 
            InitWeights_He(1e-2),
            self.net_num_pool_op_kernel_sizes, 
            self.net_conv_kernel_sizes, 
            False, 
            True, 
            True
        )
        
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
        
        self.print_to_log_file(f"DualMask network initialized:")
        self.print_to_log_file(f"  Input channels: {self.num_input_channels} (doubled for dual input)")
        self.print_to_log_file(f"  Output channels: {output_channels} (2 × {self.num_classes})")

    def initialize(self, training=True, force_load_plans=False):
        """
        Override to:
        1. Set up DualMask generators (keep both targets)
        2. Set up DualMaskProductLoss
        """
        # Call grandparent initialize (skip IterativeDenoising's generator setup)
        super(nnUNetTrainer_IterativeDenoising, self).initialize(training, force_load_plans)
        
        if training:
            # NoCLRamp: Always use full intensity (1.0)
            # Augmentation magnitudes are sampled uniformly from full NNUNET_LIMITS
            def fixed_intensity_getter():
                return 1.0
            
            self.tr_gen, self.val_generators = get_dualmask_generators(
                self.dl_tr, self.dl_val,
                self.data_aug_params['patch_size_for_spatialtransform'],
                self.data_aug_params,
                intensity_getter=fixed_intensity_getter,  # Fixed at 1.0 for NoCLRamp!
                deep_supervision_scales=self.deep_supervision_scales,
                pin_memory=self.pin_memory,
                validation_mode=self.validation_mode,
                validation_levels=self.validation_intensity_levels
            )
            
            # nnUNet expects self.val_gen for sanity checks
            self.val_gen = self.val_generators[1]
            
            # Set up DualMaskProductLoss
            self._setup_dual_mask_loss()
            
            self.print_to_log_file(f"NoCLRamp_DualMask initialized:")
            self.print_to_log_file(f"  Training intensity: random in [{self.min_training_intensity}, {self.max_training_intensity}]")
            self.print_to_log_file(f"  Validation levels: {self.validation_intensity_levels}")
            self.print_to_log_file(f"  Loss: Product Dice + CE")

    def _setup_dual_mask_loss(self):
        """Set up the DualMaskProductLoss with deep supervision wrapper."""
        # Create base loss
        base_loss = DualMaskProductLoss(
            num_classes=self.num_classes,
            soft_dice_kwargs={'batch_dice': self.batch_dice, 'do_bg': False, 'smooth': 1e-5},
            ce_kwargs={},
            weight_ce=1.0,
            weight_product_dice=1.0
        )
        
        # Wrap for deep supervision
        self.loss = MultipleOutputDualMaskLoss(base_loss, self.ds_loss_weights)
        self.base_dual_mask_loss = base_loss  # Keep reference for logging

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
        
        if self.do_spectrum_validation:
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
        Route to spectrum validation during training, full validation after.
        """
        if self._in_training and self.do_spectrum_validation:
            return self.validate_spectrum()
        
        # After training, generate NIfTI files for evaluation
        return super().validate(do_mirroring, use_sliding_window, step_size, save_softmax, use_gaussian,
                                overwrite, validation_folder_name, debug, all_in_gpu,
                                segmentation_export_kwargs, run_postprocessing_on_folds)

    def validate_spectrum(self):
        """
        Validate across all 10 intensity levels.
        
        Computes product Dice at each level and averages for mean robustness.
        """
        self.network.eval()
        level_scores = []
        level_dice_aug = []
        level_dice_clean = []
        
        self.print_to_log_file(f"\n--- Spectrum Validation (Epoch {self.epoch}) ---")
        
        for level in range(1, self.validation_intensity_levels + 1):
            intensity = level / self.validation_intensity_levels
            val_gen = self.val_generators[level]
            
            # Accumulate metrics for this level
            batch_products = []
            batch_dice_aug = []
            batch_dice_clean = []
            
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
                    if isinstance(output, (tuple, list)):
                        output = output[0]
                        target = target[0]
                    
                    # Compute product Dice using our loss function
                    # The loss tracks individual Dice scores internally
                    _ = self.base_dual_mask_loss(output, target)
                    
                    # Get individual scores from loss
                    dice_aug = self.base_dual_mask_loss.last_dice_aug
                    dice_clean = self.base_dual_mask_loss.last_dice_clean
                    product = dice_aug * dice_clean
                    
                    batch_products.append(product)
                    batch_dice_aug.append(dice_aug)
                    batch_dice_clean.append(dice_clean)
            
            # Average over batches
            avg_product = np.mean(batch_products)
            avg_dice_aug = np.mean(batch_dice_aug)
            avg_dice_clean = np.mean(batch_dice_clean)
            
            level_scores.append(avg_product)
            level_dice_aug.append(avg_dice_aug)
            level_dice_clean.append(avg_dice_clean)
            
            self.print_to_log_file(
                f"  Level {level} (i={intensity:.1f}): "
                f"Product={avg_product:.4f} (Aug={avg_dice_aug:.4f}, Clean={avg_dice_clean:.4f})"
            )
        
        # Mean robustness = average product Dice across levels
        mean_robustness = np.mean(level_scores)
        mean_dice_aug = np.mean(level_dice_aug)
        mean_dice_clean = np.mean(level_dice_clean)
        
        self.print_to_log_file(f"Mean Product Dice: {mean_robustness:.4f}")
        self.print_to_log_file(f"  Mean Dice Aug: {mean_dice_aug:.4f}")
        self.print_to_log_file(f"  Mean Dice Clean: {mean_dice_clean:.4f}")
        self.print_to_log_file("----------------------------------\n")
        
        # Store for history
        self.all_val_eval_metrics_spectrum.append(mean_robustness)
        self.last_dice_aug.append(mean_dice_aug)
        self.last_dice_clean.append(mean_dice_clean)
        
        # Update standard metric list for best model selection
        # We need to append to match parent's expected behavior
        self.all_val_eval_metrics.append(mean_robustness)
        
        self.network.train()
        return mean_robustness

    def run_online_evaluation(self, output, target):
        """
        Override to handle dual-mask output and report Product Dice.
        
        Computes Dice for both augmented and clean predictions, then reports
        the product to align with the loss function.
        
        Input format:
        - output: list of (B, 2*num_classes, X, Y, Z) for deep supervision
        - target: list of (B, 2, X, Y, Z) for deep supervision
        """
        import torch
        from nnunet.utilities.nd_softmax import softmax_helper
        from nnunet.utilities.tensor_utilities import sum_tensor
        
        with torch.no_grad():
            # Deep supervision handling - get full resolution only
            if isinstance(output, (tuple, list)):
                output = output[0]
            if isinstance(target, (tuple, list)):
                target = target[0]
            
            # Split output into augmented and clean predictions
            output_aug = output[:, :self.num_classes]
            output_clean = output[:, self.num_classes:]
            
            # Split target into augmented and clean
            target_aug = target[:, 0]    # (B, X, Y, Z)
            target_clean = target[:, 1]  # (B, X, Y, Z)
            
            # Helper function to compute Dice for a single output-target pair
            def compute_dice(output_tensor, target_tensor):
                num_classes = output_tensor.shape[1]
                output_softmax = softmax_helper(output_tensor)
                output_seg = output_softmax.argmax(1)
                axes = tuple(range(1, len(target_tensor.shape)))
                
                tp_hard = torch.zeros((target_tensor.shape[0], num_classes - 1)).to(output_seg.device)
                fp_hard = torch.zeros((target_tensor.shape[0], num_classes - 1)).to(output_seg.device)
                fn_hard = torch.zeros((target_tensor.shape[0], num_classes - 1)).to(output_seg.device)
                
                for c in range(1, num_classes):
                    tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target_tensor == c).float(), axes=axes)
                    fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target_tensor != c).float(), axes=axes)
                    fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target_tensor == c).float(), axes=axes)
                
                tp = tp_hard.sum(0).cpu().numpy()
                fp = fp_hard.sum(0).cpu().numpy()
                fn = fn_hard.sum(0).cpu().numpy()
                
                return tp, fp, fn
            
            # Compute Dice for both
            tp_aug, fp_aug, fn_aug = compute_dice(output_aug, target_aug)
            tp_clean, fp_clean, fn_clean = compute_dice(output_clean, target_clean)
            
            # Compute individual Dice scores
            dice_aug = (2 * tp_aug) / (2 * tp_aug + fp_aug + fn_aug + 1e-8)
            dice_clean = (2 * tp_clean) / (2 * tp_clean + fp_clean + fn_clean + 1e-8)
            
            # Store for epoch-end reporting
            self.online_eval_foreground_dc.append(list(dice_aug * dice_clean))  # Product!
            self.online_eval_tp.append(list(tp_aug))  # Use aug for TP/FP/FN tracking
            self.online_eval_fp.append(list(fp_aug))
            self.online_eval_fn.append(list(fn_aug))
            
            # Also store individual scores for detailed logging
            if not hasattr(self, 'online_eval_dice_aug'):
                self.online_eval_dice_aug = []
                self.online_eval_dice_clean = []
            self.online_eval_dice_aug.append(list(dice_aug))
            self.online_eval_dice_clean.append(list(dice_clean))

    def finish_online_evaluation(self):
        """
        Override to report Product Dice, Aug Dice, and Clean Dice.
        """
        import numpy as np
        
        # Compute mean scores across all batches
        mean_dice_aug = np.mean([np.mean(x) for x in self.online_eval_dice_aug]) if self.online_eval_dice_aug else 0
        mean_dice_clean = np.mean([np.mean(x) for x in self.online_eval_dice_clean]) if self.online_eval_dice_clean else 0
        mean_product = mean_dice_aug * mean_dice_clean
        
        # Store for plateau detection (using parent's mechanism)
        self.all_val_eval_metrics.append(mean_product)
        
        # Report all three scores
        self.print_to_log_file(f"Online eval - Product Dice: {mean_product:.4f} (Aug: {mean_dice_aug:.4f}, Clean: {mean_dice_clean:.4f})")
        
        # Reset for next epoch
        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []
        self.online_eval_dice_aug = []
        self.online_eval_dice_clean = []

    def predict_preprocessed_data_return_seg_and_softmax(self, data, **kwargs):
        """
        At inference: handle dual input with optional noise injection.
        
        data: (C, X, Y, Z) - single modality input
        
        Noise modes (set via predict_noise --noise_mode):
            0 = Standard: [img, img] → clean-path output
            1 = [noise, img] → clean-path output
            2 = [img, noise] → clean-path output
            3 = [noise, img] → aug-path output
            4 = [img, noise] → aug-path output
            5 = Standard: [img, img] → aug-path output
        """
        from nnunet.inference.predict_noise import get_noise_mode
        noise_mode = get_noise_mode()
        
        # Build dual input based on noise mode
        if noise_mode in (1, 3):
            # Noise in augmented slot, real image in reference slot
            noise = np.random.randn(*data.shape).astype(data.dtype)
            data_dual = np.concatenate([noise, data], axis=0)
        elif noise_mode in (2, 4):
            # Real image in augmented slot, noise in reference slot
            noise = np.random.randn(*data.shape).astype(data.dtype)
            data_dual = np.concatenate([data, noise], axis=0)
        else:
            # Standard: duplicate input (modes 0 and 5)
            data_dual = np.concatenate([data, data], axis=0)
        
        # Temporarily disable deep supervision for inference
        ds = self.network.do_ds
        self.network.do_ds = False
        
        # Run prediction through grandparent (handles sliding window correctly)
        ret = super(nnUNetTrainer_IterativeDenoising, self).predict_preprocessed_data_return_seg_and_softmax(
            data_dual, **kwargs
        )
        
        self.network.do_ds = ds
        
        # ret = (segmentation, softmax)
        # softmax shape: (2*num_classes, X, Y, Z)
        segmentation, softmax_probs = ret
        
        # Select output path based on mode
        if noise_mode in (3, 4, 5):
            # Aug-path: first half of output channels
            softmax_out = softmax_probs[:self.num_classes]
        else:
            # Clean-path: second half of output channels (default)
            softmax_out = softmax_probs[self.num_classes:]
        
        segmentation_out = softmax_out.argmax(0)
        
        return segmentation_out, softmax_out

    def load_checkpoint(self, fname, train=True):
        """
        Check for checkpoint compatibility (must be dual-mask checkpoint).
        """
        print(f"Loading checkpoint {fname}...")
        try:
            super().load_checkpoint(fname, train)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                raise RuntimeError(
                    "Checkpoint incompatible! You cannot load a standard nnUNet or single-output "
                    "checkpoint into this dual-mask trainer. Please train from scratch."
                ) from e
            raise e
