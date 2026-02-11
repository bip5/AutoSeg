"""
Dual Mask Augmentation Pipeline

Modified version of CLRamp augmentation that:
1. Keeps BOTH targets (seg for augmented, seg_clean for clean)
2. Stacks them as (B, 2, X, Y, Z) for the DualMaskProductLoss

This enables the network to predict two masks and be evaluated on both.
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

# Import the same limits and scaling from clramp_augmentation
from nnunet.training.data_augmentation.clramp_augmentation import (
    NNUNET_LIMITS, CLRAMP_PROBS, get_scaled_params
)


# ============== DUAL MASK DUAL-INPUT TRANSFORM ==============
class DualMaskDualInputTransform(AbstractTransform):
    """
    Dual-input transform that KEEPS BOTH TARGETS.
    
    Unlike CLRampDualInputTransform which only keeps the augmented seg,
    this transform keeps both:
    - seg: spatially transformed segmentation (for augmented image)
    - seg_clean: original segmentation (for clean image)
    
    These are later concatenated as (B, 2, X, Y, Z).
    """
    
    def __init__(self, base_transform, patch_size, intensity_getter, validation_mode="standard"):
        """
        Args:
            base_transform: SpatialTransform for cropping (no rotation/scale)
            patch_size: tuple, final patch dimensions
            intensity_getter: callable returning current intensity
            validation_mode: "standard" (clean reference) or "identical" (both augmented)
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
        
        # 5. Store BOTH versions
        # After noise pipeline: 
        #   data_dict['data'] = augmented image
        #   data_dict['seg'] = augmented/transformed segmentation
        
        if self.validation_mode == "identical":
            # Both channels get augmented version
            data_dict['data_clean'] = np.copy(data_dict['data'])
            data_dict['seg_clean'] = np.copy(data_dict['seg'])
        else:
            # Standard mode: clean channel gets original
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
            
            # 9. Additive Brightness - DISABLED (not used by v1/v2 defaults)
        
        # 10. Mirror (standard p=0.5)
        transforms.append(MirrorTransform(axes=(0, 1, 2)))
        
        return Compose(transforms)


# ============== CONCATENATE TRANSFORM (MODIFIED FOR DUAL MASK) ==============
class ConcatenateDualMaskTransform(AbstractTransform):
    """
    Concatenate augmented + clean along channel dimension for BOTH data and seg.
    
    Result:
    - data: (B, 2*C, X, Y, Z) - [augmented_image, clean_image]
    - seg:  (B, 2, X, Y, Z) - [augmented_seg, clean_seg]
    """
    
    def __call__(self, **data_dict):
        if 'data' in data_dict and 'data_clean' in data_dict:
            # Concatenate image data: (B, 2*C, X, Y, Z)
            data_dict['data'] = np.concatenate(
                [data_dict['data'], data_dict['data_clean']], 
                axis=1
            )
            del data_dict['data_clean']
            
        if 'seg' in data_dict and 'seg_clean' in data_dict:
            # Concatenate segmentation targets: (B, 2, X, Y, Z)
            data_dict['seg'] = np.concatenate(
                [data_dict['seg'], data_dict['seg_clean']], 
                axis=1
            )
            del data_dict['seg_clean']
            
        return data_dict


# ============== DUAL TARGET DOWNSAMPLE TRANSFORM ==============
class DownsampleDualSegForDSTransform(AbstractTransform):
    """
    Downsample dual-target segmentation for deep supervision.
    
    Input target: (B, 2, X, Y, Z) where:
        - [:, 0] = augmented seg
        - [:, 1] = clean seg
    
    Output: List of targets at each scale, each (B, 2, X_s, Y_s, Z_s)
    """
    
    def __init__(self, ds_scales, order=0, input_key='target', output_key='target'):
        self.ds_scales = ds_scales
        self.order = order
        self.input_key = input_key
        self.output_key = output_key
        
    def __call__(self, **data_dict):
        from scipy.ndimage import zoom
        
        target = data_dict[self.input_key]  # (B, 2, X, Y, Z)
        
        # For each scale, downsample both channels
        downsampled_targets = []
        
        for scale in self.ds_scales:
            if np.all(np.array(scale) == 1):
                # Full resolution
                downsampled_targets.append(target)
            else:
                # Downsample: scale applies to spatial dims only
                # target shape: (B, 2, X, Y, Z)
                # zoom_factors: (1, 1, scale[0], scale[1], scale[2])
                zoom_factors = [1, 1] + list(scale)
                downsampled = zoom(target, zoom_factors, order=self.order)
                downsampled_targets.append(downsampled)
        
        data_dict[self.output_key] = downsampled_targets
        return data_dict


# ============== MAIN GENERATOR FUNCTION ==============
def get_dualmask_generators(
    dl_train, dl_val, patch_size, params, 
    intensity_getter,
    deep_supervision_scales=None, 
    pin_memory=True,
    validation_mode="standard",
    validation_levels=10
):
    """
    Create training and validation generators for DualMask training.
    
    Args:
        dl_train: Training dataloader
        dl_val: Validation dataloader
        patch_size: Patch size tuple
        params: Data augmentation params dict
        intensity_getter: Callable returning current intensity
        deep_supervision_scales: Deep supervision scales
        pin_memory: Pin memory for CUDA
        validation_mode: "standard" (conditioning=clean) or "identical"
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
        DualMaskDualInputTransform(base_transform, patch_size, intensity_getter, validation_mode=validation_mode),
        ConcatenateDualMaskTransform(),
        RemoveLabelTransform(-1, 0),
        RenameTransform('seg', 'target', True),
    ]
    
    if deep_supervision_scales is not None:
        tr_transforms.append(DownsampleDualSegForDSTransform(deep_supervision_scales, 0, input_key='target', output_key='target'))
        
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
            DualMaskDualInputTransform(
                base_transform, patch_size, 
                make_fixed_intensity_getter(fixed_intensity), 
                validation_mode=validation_mode
            ),
            ConcatenateDualMaskTransform(),
            RemoveLabelTransform(-1, 0),
            RenameTransform('seg', 'target', True),
        ]
        
        if deep_supervision_scales is not None:
            val_transforms.append(DownsampleDualSegForDSTransform(deep_supervision_scales, 0, input_key='target', output_key='target'))
             
        val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        
        val_generators[level] = MultiThreadedAugmenter(
            dl_val, Compose(val_transforms),
            max(params.get('num_threads') // 2, 1),
            params.get('num_cached_per_thread'),
            pin_memory=pin_memory
        )
    
    return tr_gen, val_generators
