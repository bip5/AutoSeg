"""
nnUNetTrainerV2_SeqWT_DA
========================
Augmented training version of nnUNetTrainerV2_SeqWT.

The only behavioural difference from the base SeqWT trainer is that this
class uses the same data-augmentation pipeline that nnUNetTrainerV2 uses:
  ``get_moreDA_augmentation`` with the standard nnUNetTrainerV2 DA params.

Architecture invariant (preserved from base class)
---------------------------------------------------
Data augmentation is applied to the **raw image tensor only**, before
``_build_input`` injects the frozen-cascade softmax prior channel.

Design decisions
----------------
1. ``setup_DA_params`` inherits **directly** from ``nnUNetTrainerV2`` (via
   a direct call), then additionally sets standard nnUNetv2 parameters.
2. ``get_basic_generators`` feeds the augmented ``basic_generator_patch_size``
   into the dataloader so that the loader provides over-sized patches that
   survive SpatialTransform rotations without boundary artefacts.
3. ``_build_augmentation_generators`` swaps the base no-augmentation pipeline
   for the standard ``get_moreDA_augmentation``.
"""

from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.network_training.nnUNetTrainerV2_SeqWT import (
    nnUNetTrainerV2_SeqWT, DataLoader3D_RAMCached
)
from nnunet.training.dataloading.dataset_loading import DataLoader2D


class nnUNetTrainerV2_SeqWT_DA(nnUNetTrainerV2_SeqWT):
    """
    Augmented-training Sequential Weight Transfer trainer.

    Identical to nnUNetTrainerV2_SeqWT in every respect except the DA
    pipeline: this class activates the full ``get_moreDA_augmentation``
    stack (spatial deformations, intensity transforms, gamma, mirror) that
    nnUNetTrainerV2 uses by default.
    """

    def setup_DA_params(self):
        """
        Delegate fully to nnUNetTrainerV2.setup_DA_params() via direct call.
        """
        from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
        nnUNetTrainerV2.setup_DA_params(self)

    def get_basic_generators(self):
        """
        Override to use ``self.basic_generator_patch_size`` as the loader
        patch size.
        """
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr = DataLoader3D_RAMCached(
                self.dataset_tr,
                self.basic_generator_patch_size,   # enlarged for DA padding
                self.patch_size,                   # final patch for network
                self.seqwt_batch_size,
                False,
                oversample_foreground_percent=self.oversample_foreground_percent,
                pad_mode="constant",
                pad_sides=self.pad_all_sides,
            )
            dl_val = DataLoader3D_RAMCached(
                self.dataset_val,
                self.patch_size,
                self.patch_size,
                self.seqwt_batch_size,
                False,
                oversample_foreground_percent=self.oversample_foreground_percent,
                pad_mode="constant",
                pad_sides=self.pad_all_sides,
            )
        else:
            dl_tr = DataLoader2D(
                self.dataset_tr,
                self.basic_generator_patch_size,
                self.patch_size,
                self.seqwt_batch_size,
                transpose=self.plans.get('transpose_forward'),
                oversample_foreground_percent=self.oversample_foreground_percent,
                pad_mode="constant",
                pad_sides=self.pad_all_sides,
            )
            dl_val = DataLoader2D(
                self.dataset_val,
                self.patch_size,
                self.patch_size,
                self.seqwt_batch_size,
                transpose=self.plans.get('transpose_forward'),
                oversample_foreground_percent=self.oversample_foreground_percent,
                pad_mode="constant",
                pad_sides=self.pad_all_sides,
            )
        return dl_tr, dl_val

    def _build_augmentation_generators(self):
        """
        KEY CHANGE: standard nnUNetTrainerV2 augmentation
        """
        return get_moreDA_augmentation(
            self.dl_tr,
            self.dl_val,
            self.data_aug_params['patch_size_for_spatialtransform'],
            self.data_aug_params,
            deep_supervision_scales=self.deep_supervision_scales,
            pin_memory=self.pin_memory,
            use_nondetMultiThreadedAugmenter=False,
        )
