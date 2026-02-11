"""
CLRamp Augmentation Pipeline

Curriculum Learning Ramp augmentation with:
- Dynamic intensity from trainer (starts at 0.1, ramps up to 1.0)
- nnUNetv1/v2 default augmentation limits (reverted from doubled for Experiment 2)
- nnUNet default probabilities
- validation_mode toggle: "standard" (conditioning=clean) or "identical"

This file is separate from intensity_controlled_augmentation.py to preserve 
existing NoCLRamp functionality.
"""

import numpy as np
from copy import deepcopy
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform, ContrastAugmentationTransform,
    BrightnessTransform, GammaTransform
)
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from nnunet.training.data_augmentation.custom_transforms import ConvertSegmentationToRegionsTransform
from nnunet.training.data_augmentation.downsampling import DownsampleSegForDSTransform2


# ============== LIMITS (nnUNetv1/v2 defaults - Experiment 2) ==============
# Verified from: v1 nnUNetTrainerV2.setup_DA_params + data_augmentation_moreDA.py
#                v2 nnUNetTrainer.get_training_transforms() on GitHub master
NNUNET_LIMITS = {
    'rotation_max_deg': 30,           # v1/v2: ±30°
    'scale_range': (0.7, 1.4),        # v1/v2: (0.7, 1.4)
    'gamma_range': (0.7, 1.5),        # v1/v2: (0.7, 1.5)
    'brightness_range': (0.75, 1.25), # v1/v2: (0.75, 1.25)
    'contrast_range': (0.75, 1.25),   # v1/v2: (0.75, 1.25)
    'blur_sigma_max': 1.0,            # v1/v2: (0.5, 1.0)
    'noise_sigma_max': 0.316,         # v1/v2: variance (0, 0.1) → σ_max = √0.1
    'low_res_zoom_min': 0.5,          # v1/v2: (0.5, 1.0)
    # Additive brightness: NOT used by v1 or v2 default pipelines - disabled
}

# ============== CLRamp PROBABILITIES (nnUNet defaults) ==============
# Reverted to standard nnUNet probabilities - aggressive probs showed no improvement
CLRAMP_PROBS = {
    # Spatial transforms: nnUNet default p=0.2
    'p_rot_per_sample': 0.2,
    'p_scale_per_sample': 0.2,
    # Intensity transforms: nnUNet defaults
    'p_noise': 0.1,
    'p_blur': 0.2,
    'p_brightness': 0.15,
    'p_contrast': 0.15,
    'p_low_res': 0.25,
    'p_gamma_inverted': 0.1,
    'p_gamma_normal': 0.3,
    # Additive brightness: DISABLED (not used by v1/v2 defaults)
}


def get_scaled_params(intensity: float) -> dict:
    """
    Interpolate augmentation params from identity (intensity=0) 
    to full augmentation (intensity=1) using nnUNet v1/v2 limits.
    """
    i = intensity
    
    # Rotation: 0° at i=0, ±30° at i=1
    max_angle = NNUNET_LIMITS['rotation_max_deg'] / 360 * 2 * np.pi
    rotation = (-max_angle * i, max_angle * i)
    
    # Scale: (1.0, 1.0) → (0.7, 1.4)
    scale_low = 1.0 + (NNUNET_LIMITS['scale_range'][0] - 1.0) * i
    scale_high = 1.0 + (NNUNET_LIMITS['scale_range'][1] - 1.0) * i
    
    # Gamma: (1.0, 1.0) → (0.7, 1.5)
    gamma_low = 1.0 + (NNUNET_LIMITS['gamma_range'][0] - 1.0) * i
    gamma_high = 1.0 + (NNUNET_LIMITS['gamma_range'][1] - 1.0) * i
    
    # Brightness multiplicative: (1.0, 1.0) → (0.75, 1.25)
    bright_low = 1.0 + (NNUNET_LIMITS['brightness_range'][0] - 1.0) * i
    bright_high = 1.0 + (NNUNET_LIMITS['brightness_range'][1] - 1.0) * i
    
    # Contrast: (1.0, 1.0) → (0.75, 1.25)
    contrast_low = 1.0 + (NNUNET_LIMITS['contrast_range'][0] - 1.0) * i
    contrast_high = 1.0 + (NNUNET_LIMITS['contrast_range'][1] - 1.0) * i
    
    # Blur sigma: (0.5*i, 1.0*i)
    blur_sigma = (0.5 * i, NNUNET_LIMITS['blur_sigma_max'] * i)
    
    # Noise variance: 0 → 0.1 (σ_max = √0.1 ≈ 0.316)
    noise_variance = (0, (NNUNET_LIMITS['noise_sigma_max'] * i) ** 2)
    
    # Low res zoom: (1.0, 1.0) → (0.5, 1.0)
    low_res_zoom = (1.0 + (NNUNET_LIMITS['low_res_zoom_min'] - 1.0) * i, 1.0)
    
    return {
        'rotation': rotation,
        'scale_range': (scale_low, scale_high),
        'gamma_range': (gamma_low, gamma_high),
        'brightness_range': (bright_low, bright_high),
        'contrast_range': (contrast_low, contrast_high),
        'blur_sigma': blur_sigma,
        'noise_variance': noise_variance,
        'low_res_zoom': low_res_zoom,
    }


# ============== CLRamp DUAL-INPUT TRANSFORM ==============
class CLRampDualInputTransform(AbstractTransform):
    """
    CLRamp version of dual-input transform.
    
    Key differences from NoCLRamp:
    - Uses balanced CLRAMP_PROBS (0.5 spatial, 0.3 intensity)
    - Accepts intensity_getter callable (for dynamic intensity from trainer)
    """
    
    def __init__(self, base_transform, patch_size, intensity_getter, validation_mode="standard"):
        """
        Args:
            base_transform: SpatialTransform for cropping (no rotation/scale)
            patch_size: tuple, final patch dimensions
            intensity_getter: callable returning current intensity (e.g., lambda: trainer.current_intensity)
            validation_mode: "standard" (conditioning=clean) or "identical" (conditioning=augmented)
        """
        self.base_transform = base_transform
        self.patch_size = patch_size
        self.intensity_getter = intensity_getter
        self.validation_mode = validation_mode
    
    def __call__(self, **data_dict):
        # 1. Apply base transform (crop only)
        data_dict = self.base_transform(**data_dict)
        
        # 2. Clone clean data BEFORE applying noise
        data_clean = np.copy(data_dict['data'])
        seg_clean = np.copy(data_dict['seg'])
        
        # 3. Get current intensity from trainer
        intensity = self.intensity_getter()
        
        # 4. Build and apply noise transforms at this intensity
        noise_transforms = self._build_noise_pipeline(intensity)
        data_dict = noise_transforms(**data_dict)
        
        # 5. Handle Validation Modes
        if self.validation_mode == "identical":
            # Conditioning channel is copy of AUGMENTED image
            data_dict['data_clean'] = np.copy(data_dict['data'])
            data_dict['seg_clean'] = np.copy(data_dict['seg'])
        else:
            # Standard mode: Conditioning channel is CLEAN image
            data_dict['data_clean'] = data_clean
            data_dict['seg_clean'] = seg_clean
            
        return data_dict
    
    def _build_noise_pipeline(self, intensity):
        """Build noise transform pipeline with CLRamp probabilities."""
        p = get_scaled_params(intensity)
        probs = CLRAMP_PROBS
        transforms = []
        
        # 1. Spatial (rotation + scale) - p=0.5
        transforms.append(SpatialTransform(
            self.patch_size,
            do_elastic_deform=False,
            do_rotation=True,
            angle_x=p['rotation'], angle_y=p['rotation'], angle_z=p['rotation'],
            do_scale=True,
            scale=p['scale_range'],
            random_crop=False, 
            p_rot_per_sample=probs['p_rot_per_sample'],
            p_scale_per_sample=probs['p_scale_per_sample'],
        ))
        
        # Only apply intensity noise if intensity is meaningful
        if intensity > 0.01:
            # 2. Gaussian Noise - p=0.3
            transforms.append(GaussianNoiseTransform(
                noise_variance=p['noise_variance'], 
                p_per_sample=probs['p_noise']
            ))
            
            # 3. Gaussian Blur - p=0.3
            transforms.append(GaussianBlurTransform(
                blur_sigma=p['blur_sigma'], 
                different_sigma_per_channel=True, 
                p_per_sample=probs['p_blur'], 
                p_per_channel=0.5
            ))
            
            # 4. Brightness Multiplicative - p=0.3
            transforms.append(BrightnessMultiplicativeTransform(
                multiplier_range=p['brightness_range'], 
                p_per_sample=probs['p_brightness']
            ))
            
            # 5. Contrast - p=0.3
            transforms.append(ContrastAugmentationTransform(
                contrast_range=p['contrast_range'], 
                p_per_sample=probs['p_contrast']
            ))
            
            # 6. Low Resolution - p=0.3
            transforms.append(SimulateLowResolutionTransform(
                zoom_range=p['low_res_zoom'], 
                per_channel=True, 
                p_per_channel=0.5, 
                order_downsample=0, 
                order_upsample=3, 
                p_per_sample=probs['p_low_res']
            ))
            
            # 7. Gamma (Inverted) - p=0.3
            transforms.append(GammaTransform(
                gamma_range=p['gamma_range'], 
                invert_image=True, 
                per_channel=True, 
                retain_stats=True, 
                p_per_sample=probs['p_gamma_inverted']
            ))
            
            # 8. Gamma (Normal) - p=0.3
            transforms.append(GammaTransform(
                gamma_range=p['gamma_range'], 
                invert_image=False, 
                per_channel=True, 
                retain_stats=True, 
                p_per_sample=probs['p_gamma_normal']
            ))
            
            # Additive Brightness: DISABLED (not used by v1/v2 defaults)
        
        # 9. Mirror (standard p=0.5)
        transforms.append(MirrorTransform(axes=(0, 1, 2)))
        
        return Compose(transforms)


# ============== CONCATENATE TRANSFORM ==============
class ConcatenateDualInputTransform(AbstractTransform):
    """Concatenate augmented + clean along channel dimension."""
    
    def __call__(self, **data_dict):
        if 'data' in data_dict and 'data_clean' in data_dict:
            # Result: (B, 2*C, X, Y, Z)
            data_dict['data'] = np.concatenate(
                [data_dict['data'], data_dict['data_clean']], 
                axis=1
            )
            del data_dict['data_clean']
            if 'seg_clean' in data_dict:
                del data_dict['seg_clean']
        return data_dict


# ============== MAIN GENERATOR FUNCTION ==============
def get_clramp_generators(
    dl_train, dl_val, patch_size, params, 
    intensity_getter,
    deep_supervision_scales=None, 
    pin_memory=True,
    validation_mode="standard",
    validation_levels=10
):
    """
    Create training and validation generators for CLRamp.
    
    Args:
        dl_train: Training dataloader
        dl_val: Validation dataloader
        patch_size: Patch size tuple
        params: Data augmentation params dict
        intensity_getter: Callable returning current intensity (e.g., lambda: trainer.current_intensity)
        deep_supervision_scales: Deep supervision scales
        pin_memory: Pin memory for CUDA
        validation_mode: "standard" (conditioning=clean) or "identical" (conditioning=augmented)
        validation_levels: Number of validation intensity levels (default 10)
    
    Returns:
        tr_gen: Training generator
        val_generators: Dict of validation generators {level: generator}
    """
    
    # Base transform: crop only, no rotation/scale
    base_transform = SpatialTransform(
        patch_size,
        do_elastic_deform=False,
        do_rotation=False,
        do_scale=False,
        random_crop=True,
        border_mode_data=params.get("border_mode_data"),
        border_cval_data=0,
        order_data=3,
        border_mode_seg="constant",
        border_cval_seg=-1,
        order_seg=1
    )
    
    # Training transforms
    tr_transforms = [
        CLRampDualInputTransform(base_transform, patch_size, intensity_getter, validation_mode=validation_mode),
        ConcatenateDualInputTransform(),
        RemoveLabelTransform(-1, 0),
        RenameTransform('seg', 'target', True),
    ]
    
    if deep_supervision_scales is not None:
        tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target', output_key='target'))
        
    tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    
    tr_gen = MultiThreadedAugmenter(
        dl_train, Compose(tr_transforms),
        params.get('num_threads'), params.get('num_cached_per_thread'),
        pin_memory=pin_memory
    )
    
    # Validation generators: one per intensity level (for spectrum validation)
    val_generators = {}
    for level in range(1, validation_levels + 1):
        fixed_intensity = level / validation_levels  # 0.1, 0.2, ..., 1.0
        
        def make_fixed_intensity_getter(intensity):
            return lambda: intensity
        
        val_transforms = [
            CLRampDualInputTransform(
                base_transform, patch_size, 
                make_fixed_intensity_getter(fixed_intensity), 
                validation_mode=validation_mode
            ),
            ConcatenateDualInputTransform(),
            RemoveLabelTransform(-1, 0),
            RenameTransform('seg', 'target', True),
        ]
        
        if deep_supervision_scales is not None:
            val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target', output_key='target'))
             
        val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        
        val_generators[level] = MultiThreadedAugmenter(
            dl_val, Compose(val_transforms),
            max(params.get('num_threads') // 2, 1),
            params.get('num_cached_per_thread'),
            pin_memory=pin_memory
        )
    
    return tr_gen, val_generators
