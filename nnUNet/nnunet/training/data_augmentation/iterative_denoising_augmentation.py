
import numpy as np
from copy import deepcopy
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, BrightnessTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from nnunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params
from nnunet.training.data_augmentation.custom_transforms import DualInputGenericTransform

class ConcatenateDualInputTransform:
    """
    Concatenates 'data' and 'data_clean' along the channel dimension.
    Assumes data is (B, C, X, Y, Z).
    Result is (B, 2*C, X, Y, Z).
    """
    def __call__(self, **data_dict):
        if 'data' in data_dict and 'data_clean' in data_dict:
            data_dict['data'] = np.concatenate((data_dict['data'], data_dict['data_clean']), axis=1)
            # del data_dict['data_clean'] # Optional: keep it or delete it. Keeping it might be useful for debugging but we usually delete to save memory? 
            # Actually, standard pipelines expect 'data'.
        return data_dict

def get_iterative_denoising_augmentation(dataloader_train, dataloader_val, patch_size, params=default_3D_augmentation_params,
                                         border_val_seg=-1, seeds_train=None, seeds_val=None, order_seg=1, order_data=3,
                                         deep_supervision_scales=None, soft_ds=False, classes=None, pin_memory=True, regions=None,
                                         noise_alias="all"):
    """
    :param noise_alias: "all", "geometric", "intensity", "structure", or a comma-separated string of these.
    """
    
    # Override params to match nnUNetTrainerV2 defaults (Experiment 2 consistency)
    # TrainerV2 overrides rotation to ±30° and scale to (0.7, 1.4)
    params = deepcopy(params)
    params["rotation_x"] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
    params["rotation_y"] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
    params["rotation_z"] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
    params["scale_range"] = (0.7, 1.4)
    
    # --- 1. Define Base Transform (Geometric Alignment) ---
    # This transform crops the large patch to the final patch size and handles mirroring.
    # It does NOT apply rotation/scaling/deformation effectively, or rather, we will separate those.
    # Actually, standard nnU-Net typically does "SpatialTransform" which does EVERYTHING (crop + rot + scale + deform).
    #
    # DESIGN CHOICE:
    # To have "clean" anchor be aligned with "augmented" input, they must share the same CROP center.
    # But if we apply Rotation/Scale/Deformation to the "clean" one, it becomes "noisy".
    # So the "Base" transform should be:
    #   - Random Crop (if enabled / implied by SpatialTransform)
    #   - Mirroring? (Mirroring is a valid data augmentation that preserves semantics, but maybe we consider it 'clean'?)
    #   - Usually we want the anchor to be rigidly aligned.
    #
    # IMPLEMENTATION:
    # We will use SpatialTransform for the BASE, but with rotation/deformation/scaling DISABLED (or set to identity).
    # Then we use ANOTHER SpatialTransform for the NOISE, which applies the deformations.
    #
    # HOWEVER: SpatialTransform does the cropping. If we have two SpatialTransforms, the second one would need to work on the ALREADY CROPPED patch.
    # The standard SpatialTransform expects input LARGER than patch_size to crop from.
    # If we pass an already cropped patch, it can't crop freely.
    #
    # SOLUTION:
    # The `DualInputGenericTransform` takes `base_transform` and `noise_transform`.
    # `base_transform`: Should do the random crop to `patch_size`. 
    # `noise_transform`: Should take the `patch_size` input and apply deformations.
    # 
    # But `SpatialTransform` does both.
    # We will configure `base_transform` to ONLY do cropping (and maybe mirroring).
    # We will configure `noise_transform` to do the augmentations.
    # But `noise_transform` receiving a (X,Y,Z) patch and needing to rotate it might introduce black borders if we don't have margins.
    #
    # BETTER APPROACH from Implementation Plan:
    # "Assumption on Alignment: ... [Crop+Scale(Image)] as clean anchor and [Rot+Elastic(Crop+Scale(Image))] as augmented."
    # Wait, if we crop first, then rotate, we lose content at corners.
    # Standard nnU-Net does Rot+Scale+Crop in one go to avoid border artifacts.
    # 
    # If we want the anchor to be "unrotated", but the input "rotated", and they represent the SAME anatomy...
    # Then the anchor must be the unrotated view of that anatomy.
    # 
    # If we Crop first, we get a box. If we then Rotate that box, we get valid pixels only in the center.
    #
    # Alternative:
    # We need to sample the SAME center coordinate for both.
    # Base: Sample Center C, Crop Size S. -> Patch A.
    # Noise: Sample Center C (same), Crop Size S, BUT with Rotation R applied during the sampling (resampling). -> Patch B.
    #
    # This "Coupled Sampling" is hard to do with standard transforms unless we write a custom `DualSpatialTransform`.
    #
    # Let's stick to the simplest interpretation of "Denoising": the task is to recover the signal from a corrupted version.
    # If the "corruption" is rotation, the network sees a rotated image and must output... what? The mask for the rotated image?
    # NO. The prompt says: "place the GT mask ... with the image as conditioning input ... apply the augmentation to mask - i.e. rotate the GT ... GT will not be accessible ... provide the original image as a conditioning input alongside the augmented input."
    # "This approach solves the loss of performance due to forgetting ... similarity ... will be too high."
    #
    # Re-reading: "Iterative denoising ... requires the model to incrementally remove noise while there is a conditioning anchor ... image being segmented ... definitive target."
    # "Target ... teach distortion distribution."
    #
    # Interpretation:
    # Input 0: Augmented Image (Rotated).
    # Input 1: Clean Anchor Image (Unrotated).
    # Target: Segmentation of the Augmented Image (Rotated Mask).
    #
    # Wait, if the target is the Rotated Mask, and Input 0 is Rotated Image, this is standard segmentation.
    # Input 1 (Unrotated) provides context? But it's spatially Misaligned!
    # A unrotated brain and a rotated brain are not pixel-aligned.
    #
    # Prompt says: "Assumption on Alignment: The 'original' image (conditioning) will be spatially aligned with the 'augmented' image BEFORE the specific 'noise' augmentations ... are applied."
    #
    # Ah! Correct reading:
    # We take an image.
    # Branch A (Anchor): Just Crop. Result: Unrotated Patch.
    # Branch B (Augmented): Crop + Rotate. Result: Rotated Patch.
    #
    # BUT if they are misaligned, a convnet can't easily use Branch A to help segment Branch B pixel-wise, unless it learns the transformation.
    #
    # Let's look at "Denoising Diffusion" literature (Wolleb et al).
    # The conditioning is usually the "Image". The "Noise" is added to the "Mask". 
    # Here: "augmentation as noise". "apply augmentation to mask - i.e. rotate the GT ... GT will not be accessible ... Another option ... provide the original image as conditioning input alongside the augmented input."
    #
    # "Similarity to conditioning target and augmented sample will be too high."
    #
    # If I interpret "noise" as "augmentation of the IMAGE", then:
    # We want to recover the "clean" state? Or we want to segment the "noisy" state?
    # "Evaluated segmentation for augmented mask." -> We are segmenting the AUGMENTED image.
    # So Target = Mask_Augmented.
    # Input = Image_Augmented + Image_Clean.
    #
    # If Image_Augmented is Rotated, and Image_Clean is Unrotated, they are NOT pixel aligned.
    # Standard U-Nets assume channel-wise alignment.
    #
    # IF the User confirms "The 'original' image (conditioning) will be spatially aligned with the 'augmented' image *before* the specific 'noise' augmentations ... applied",
    # This implies:
    # Step 1: Alignment (e.g. Scaling, maybe minimal cropping to a large region).
    # Step 2: "Noise" (Augmentation).
    # 
    # If "Noise" includes Rotation, then:
    # Clean = Aligned.
    # Augmented = Aligned + Rotated.
    #
    # They are NOT pixel aligned after the noise step.
    #
    # UNLESS: The "Noise" is something that preserves alignment (like intensity noise, blur, etc.).
    # OR: The User *accepts* misalignment for spatial transforms and expects the network to handle it (unlikely for U-Net concatenation).
    # OR: The "Noise" includes DEFORMATIONS/ROTATIONS that are *spatial*, and the user accepts that "Conditioning" is just a hint "this is a brain" but not a pixel-map.
    #
    # BUT, the prompt mentions "similarity ... too high".
    #
    # Let's consider the "Denoising" metaphor.
    # Signal = Clean Image.
    # Noisy Observation = Clean + Noise.
    # We usually output Signal.
    # Here we output Segmentation of... Input?
    #
    # If I rotate the image, I must rotate the mask.
    # Input: Rotated Image. Target: Rotated Mask.
    # Conditioning: Unrotated Image?
    #
    # If I use Unrotated Image as conditioning, the network has (Rotated Image, Unrotated Image).
    # It predicts Rotated Mask.
    # The Unrotated Image helps how? It has the "true" anatomy.
    #
    # If the user says "Aliasing mode ... choose which noises to apply".
    # If I choose "Intensity" (Gaussian Noise), Clean and Augmented ARE aligned.
    # If I choose "Geometric" (Rotation), Clean and Augmented are NOT aligned.
    #
    # If they are not aligned, simple concatenation `(B, 2C, X, Y, Z)` is questionable for a standard U-Net.
    #
    # HOWEVER, this is an Agentic task. I must implement what I planned.
    # My plan (which the user saw) said:
    # "Assumption on Alignment: The 'original' image (conditioning) will be spatially aligned with the 'augmented' image BEFORE the specific 'noise' augmentations are applied."
    #
    # This implies that `Clean` is the "Pre-Noise" state.
    # So if Noise = Rotation, Clean = Unrotated.
    #
    # I will stick to this implementation. If the user intends something else, they would have corrected the plan (which was explicit about this).
    #
    # So:
    # Base Transform: Initial Crop (and maybe Mirroring if it's considered 'part of the object').
    # Noise Transform: The 'corruptions' (Rotation, Deformations, Intensity Noise).
    
    tr_transforms = []

    # 1. Base Transform (Crop & Initial processing)
    # We use SpatialTransform for this, but disable all "noise" aspects.
    # Note: SpatialTransform in nnU-Net handles cropping from the data_generator (which yields large patches).
    # So this MUST be the first step.
    
    base_params = deepcopy(params)
    # Disable rotations/deformations in base
    base_params["do_elastic"] = False
    base_params["do_rotation"] = False
    base_params["do_scaling"] = False # Scale is tricky, maybe we want Scale in base? 
    # If we Scale in noise, the Clean is Unscaled. 
    # Usually Scale is considered a "View", not "Noise". But prompted said "deformation and basic transforms we will use rotation...".
    # Let's put Scale in Noise too to be safe? Or Base?
    # Let's put Scale in Base so sizes match "on average"? 
    # Actually, if we scale the clean image, we change resolution.
    # Let's keep Base minimal: Just Crop.
    
    # We need to ensure we crop to `patch_size`.
    
    # The `SpatialTransform` usage in default_data_augmentation uses `patch_size_spatial`.
    # And it uses `params.get("random_crop")`.
    
    # Base Spatial Transform
    base_transform = SpatialTransform(
        patch_size, 
        patch_center_dist_from_border=None, 
        do_elastic_deform=False, 
        do_rotation=False, 
        do_scale=False,
        border_mode_data=params.get("border_mode_data"), 
        border_cval_data=0, 
        order_data=order_data, 
        border_mode_seg="constant", 
        border_cval_seg=border_val_seg,
        order_seg=order_seg, 
        random_crop=True, # Always generic crop at start
        p_el_per_sample=0, p_scale_per_sample=0, p_rot_per_sample=0
    )
    
    # 2. Noise Pipeline
    # We select transforms based on noise_alias
    noise_transforms = []
    
    alias_list = [x.strip().lower() for x in noise_alias.split(',')]
    use_all = "all" in alias_list
    
    # Geometric Noise
    if use_all or "geometric" in alias_list:
        # Rotation, Scaling, Elastic
        # We invoke SpatialTransform AGAIN, but this time on the ALREADY CROPPED patch.
        # Note: SpatialTransform handles existing patch size if provided.
        # But we need to be careful about borders. 'border_mode_data' handles padding.
        
        noise_transforms.append(SpatialTransform(
            patch_size, 
            patch_center_dist_from_border=None, 
            do_elastic_deform=params.get("do_elastic"), 
            alpha=params.get("elastic_deform_alpha"), 
            sigma=params.get("elastic_deform_sigma"), 
            do_rotation=params.get("do_rotation"), 
            angle_x=params.get("rotation_x"), 
            angle_y=params.get("rotation_y"), 
            angle_z=params.get("rotation_z"), 
            p_rot_per_axis=params.get("rotation_p_per_axis"), 
            do_scale=params.get("do_scaling"), 
            scale=params.get("scale_range"), 
            border_mode_data=params.get("border_mode_data"), 
            border_cval_data=0, 
            order_data=order_data, 
            border_mode_seg="constant", 
            border_cval_seg=border_val_seg, 
            order_seg=order_seg, 
            random_crop=False, # Already cropped
            p_el_per_sample=params.get("p_eldef"), 
            p_scale_per_sample=params.get("p_scale"), 
            p_rot_per_sample=params.get("p_rot"),
            independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
        ))
        
        # Mirroring? Usually in Base, but can be here. 
        # If Mirror is Noise, Clean is unmirrored.
        # Let's put Mirror in Base? No, default puts it generic.
        # Let's treat Mirror as "Noise" for now (as it changes the image).
        if params.get("do_mirror") or params.get("mirror"):
            noise_transforms.append(MirrorTransform(params.get("mirror_axes")))
            
    # Intensity Noise
    if use_all or "intensity" in alias_list:
        noise_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
        noise_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2, p_per_channel=0.5))
        noise_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
        if params.get("do_additive_brightness"):
            noise_transforms.append(BrightnessTransform(params.get("additive_brightness_mu"),
                                                     params.get("additive_brightness_sigma"),
                                                     True, p_per_sample=params.get("additive_brightness_p_per_sample"),
                                                     p_per_channel=params.get("additive_brightness_p_per_channel")))
        noise_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
        noise_transforms.append(GammaTransform(params.get("gamma_range"), True, True, retain_stats=params.get("gamma_retain_stats"), p_per_sample=0.1))
        if params.get("do_gamma"):
            noise_transforms.append(GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"), p_per_sample=params["p_gamma"]))
            
    # Structure/Resample Noise
    if use_all or "structure" in alias_list:
        noise_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                            p_per_channel=0.5,
                                                            order_downsample=0, order_upsample=3, p_per_sample=0.25))

    # Compose Noise
    noise_pipeline = Compose(noise_transforms)
    
    # 3. Dual Input Wrapper
    # Wraps the Base and Noise phases
    dual_transform = DualInputGenericTransform(base_transform, noise_pipeline)
    tr_transforms.append(dual_transform)
    
    # 4. Finalizing Transforms (ToTensor, etc.)
    # These mostly work on 'data' and 'target'. 
    # 'data_clean' will just pass through unless we rename it.
    
    # We need to Concatenate data and data_clean
    tr_transforms.append(ConcatenateDualInputTransform())
    
    # Rename 'seg' to 'target' (Standard nnU-Net)
    tr_transforms.append(RenameTransform('seg', 'target', True))
    
    if regions is not None:
        tr_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))
        
    # ToTensor
    tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    
    tr_transforms = Compose(tr_transforms)

    batchgenerator_train = MultiThreadedAugmenter(dataloader_train, tr_transforms, params.get('num_threads'),
                                                  params.get("num_cached_per_thread"),
                                                  seeds=seeds_train, pin_memory=pin_memory)
                                                  
    # Validation Generator
    # Validation usually has NO augmentation, just Center Crop (Base).
    # But for "Iterative Denoising" validation, we probably want to evaluate the Denoising capability?
    # Or do we just want to evaluate standard segmentation?
    # "Iterative denoising ... requires the model to incrementally remove noise..."
    # The Model EXPECTS 2 inputs.
    # So validation MUST provide 2 inputs.
    #
    # Input 0: Augmented (or Noisy).
    # Input 1: Clean.
    #
    # If standard validation is "Best Case" (No noise), then Input 0 = Input 1 = Clean.
    # Or checking if it handles moderate noise.
    # 
    # Let's duplicate Clean for Input 0 and Input 1 in validation if no noise is desired?
    # Or apply random noise in validation too?
    # Usually validation in nnU-Net is deterministic.
    # 
    # Let's create a validation pipeline that produces (Clean, Clean) or (Clean, Clean) -> Concatenated.
    # The network inputs 2 channels.
    # 
    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1, 0)) # Standard
    
    # Base Val Transform (Center Crop usually implicit in loader or we add it)
    # Actually validation loader yields full patches? 
    # See default_data_augmentation.py: it uses `ResizeTransform` or just `ValidationTransform`?
    # No, it just renamed/toTensor usually because patches are correct size.
    # BUT our network expects 2 inputs.
    
    # Let's verify standard val transforms.
    # They are minimal.
    
    # We need to duplicate 'data' to 'data_clean'.
    # And then concatenate.
    
    class CopyDataToCleanTransform(DualInputGenericTransform):
        def __init__(self):
            pass
        def __call__(self, **data_dict):
            if 'data' in data_dict:
                data_dict['data_clean'] = np.copy(data_dict['data'])
            return data_dict

    val_transforms.append(CopyDataToCleanTransform())
    val_transforms.append(ConcatenateDualInputTransform())
    val_transforms.append(RenameTransform('seg', 'target', True))
    
    if regions is not None:
        val_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    val_transforms = Compose(val_transforms)

    batchgenerator_val = MultiThreadedAugmenter(dataloader_val, val_transforms,
                                                max(params.get('num_threads') // 2, 1),
                                                params.get("num_cached_per_thread"),
                                                seeds=seeds_val, pin_memory=pin_memory)

    return batchgenerator_train, batchgenerator_val
