"""
nnUNetTrainer_MaskDenoise - Mask-Denoising Curriculum Trainer

Inspired by diffusion-based segmentation. The GT mask is concatenated with the
image as a second input channel, progressively noised during training.

Input: [image, noised_mask]  (2 channels)
Output: segmentation         (num_classes channels)
Loss: standard Dice + CE

Noise schedule:
  noised_mask = (1 - t) * GT_mask + t * gaussian_noise

Two variants:
  - nnUNetTrainer_MaskDenoise:       t = epoch / max_epochs  (proportional)
  - nnUNetTrainer_MaskDenoiseRandom: t ~ Uniform(0, 1) per batch (stochastic)

At inference: mask channel = pure Gaussian noise (t=1)
"""

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast

from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from typing import Tuple


class nnUNetTrainer_MaskDenoise(nnUNetTrainerV2):
    """
    Curriculum mask-denoising trainer.
    
    Concatenates a progressively noised GT mask with the image as a second
    input channel. Noise level t increases linearly from 0 to 1 over training.
    
    At t=0 the network receives a clean GT mask hint.
    At t=1 the network receives pure Gaussian noise (no hint).
    """
    
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice,
                         stage, unpack_data, deterministic, fp16)
        self.noise_t = 0.0  # Current noise level [0, 1]
    
    def initialize_network(self):
        """
        Build the standard nnUNet but with +1 input channel for the mask.
        """
        # Save original for reference
        original_channels = self.num_input_channels
        self.num_input_channels = original_channels + 1  # image + binary mask
        
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
        
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x,
                                    InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes,
                                    False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
        
        # Restore for any downstream use
        self.num_input_channels = original_channels
    
    def get_noise_level(self):
        """
        Returns the current noise level t in [0, 1].
        Proportional variant: t = epoch / max_epochs.
        """
        return min(1.0, self.epoch / max(1, self.max_num_epochs))
    
    def _create_noised_mask(self, target, noise_t):
        """
        Create a noised mask from the GT segmentation.
        
        target: (B, 1, X, Y, Z) - GT segmentation labels
        noise_t: float in [0, 1] - noise level
        
        Returns: (B, 1, X, Y, Z) float tensor - noised binary mask
        """
        # Convert GT to binary float mask (foreground = 1, background = 0)
        gt_mask = (target > 0).float()
        
        # Generate Gaussian noise with same shape
        noise = torch.randn_like(gt_mask)
        
        # Blend: clean at t=0, pure noise at t=1
        noised_mask = (1.0 - noise_t) * gt_mask + noise_t * noise
        
        return noised_mask
    
    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        Override to concatenate noised mask with image before forward pass.
        """
        data_dict = next(data_generator)
        data = data_dict['data']      # (B, C, X, Y, Z)
        target = data_dict['target']   # (B, 1, X, Y, Z) or deep supervision list
        
        data = maybe_to_torch(data)
        target = maybe_to_torch(target)
        
        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
        
        # Get noise level for this iteration
        noise_t = self.get_noise_level()
        
        # Create noised mask from GT (use first target if deep supervision)
        if isinstance(target, (list, tuple)):
            gt_for_mask = target[0]
        else:
            gt_for_mask = target
        
        noised_mask = self._create_noised_mask(gt_for_mask, noise_t)
        
        # Concatenate: [image, noised_mask] → (B, C+1, X, Y, Z)
        data = torch.cat([data, noised_mask], dim=1)
        
        # Standard forward pass
        self.optimizer.zero_grad()
        
        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target)
            
            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
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
    
    def on_epoch_end(self):
        """Update and log the noise level."""
        self.noise_t = self.get_noise_level()
        self.print_to_log_file(f"Mask denoise noise_t = {self.noise_t:.4f} "
                               f"(epoch {self.epoch}/{self.max_num_epochs})")
        return super().on_epoch_end()
    
    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, 
                                                         do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True,
                                                         step_size: float = 0.5,
                                                         use_gaussian: bool = True,
                                                         pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None,
                                                         all_in_gpu: bool = False,
                                                         verbose: bool = True,
                                                         mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        At inference: concatenate pure Gaussian noise as the mask channel.
        The network has learned to segment without mask hints by the end of training.
        """
        # data shape: (C, X, Y, Z) — single modality
        # Create noise channel with same spatial dims
        mask_shape = (1, *data.shape[1:])  # (1, X, Y, Z)
        noise_mask = np.random.randn(*mask_shape).astype(data.dtype)
        
        # Concatenate: [image, noise] → (C+1, X, Y, Z)
        data_with_mask = np.concatenate([data, noise_mask], axis=0)
        
        # Disable deep supervision for inference
        ds = self.network.do_ds
        self.network.do_ds = False
        
        ret = super(nnUNetTrainerV2, self).predict_preprocessed_data_return_seg_and_softmax(
            data_with_mask,
            do_mirroring=do_mirroring, mirror_axes=mirror_axes,
            use_sliding_window=use_sliding_window, step_size=step_size,
            use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
            pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
            verbose=verbose, mixed_precision=mixed_precision
        )
        
        self.network.do_ds = ds
        return ret


class nnUNetTrainer_MaskDenoiseRandom(nnUNetTrainer_MaskDenoise):
    """
    Stochastic mask-denoising trainer.
    
    Same as MaskDenoise but noise level t is sampled uniformly per batch
    instead of tied to epoch progress. This means the network sees all
    noise levels throughout training, not just the current curriculum step.
    """
    
    def get_noise_level(self):
        """
        Returns a random noise level t ~ Uniform(0, 1) for each batch.
        """
        return float(np.random.uniform(0.0, 1.0))
