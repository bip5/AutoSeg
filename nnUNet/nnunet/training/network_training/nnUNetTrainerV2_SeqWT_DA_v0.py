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

Concretely, the pipeline is:

  DataLoader (raw image + seg) → batchgenerators augmentation pipeline
  → ``run_iteration`` → ``_build_input`` (injects frozen-cascade prob)
  → active network forward + loss + backward

Because the prior channel is computed inside ``run_iteration`` *after* the
augmenter has already applied spatial transforms to the image, each frozen
network receives the same spatially-augmented patch that the active network
will process.  This is the correct behaviour.

Design decisions
----------------
1. ``setup_DA_params`` inherits **directly** from ``nnUNetTrainerV2`` (via
   ``super()``), then additionally sets ``do_mirror=True`` and clears the
   SeqWT base-class mirror disable.  All other params are identical to
   nnUNetTrainerV2.
2. ``initialize`` replicates the ``nnUNetTrainerV2.initialize`` augmentation
   call exactly, using ``data_aug_params['patch_size_for_spatialtransform']``
   as the spatial patch size (respects rotation/scale padding).
3. ``get_basic_generators`` feeds the augmented ``basic_generator_patch_size``
   (set by ``nnUNetTrainerV2.setup_DA_params``) into the dataloader so that
   the loader provides over-sized patches that survive SpatialTransform
   rotations without boundary artefacts.
4. No other method is changed — SeqWT-specific logic (fuser, frozen
   networks, checkpointing) is fully inherited.

Usage
-----
    nnUNet_train 3d_fullres nnUNetTrainerV2_SeqWT_DA TASK_ID FOLD
"""

from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.dataloading.dataset_loading import unpack_dataset, DataLoader3D, DataLoader2D
from nnunet.training.network_training.nnUNetTrainerV2_SeqWT import (
    nnUNetTrainerV2_SeqWT, DataLoader3D_RAMCached
)
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p


class nnUNetTrainerV2_SeqWT_DA(nnUNetTrainerV2_SeqWT):
    """
    Augmented-training Sequential Weight Transfer trainer.

    Identical to nnUNetTrainerV2_SeqWT in every respect except the DA
    pipeline: this class activates the full ``get_moreDA_augmentation``
    stack (spatial deformations, intensity transforms, gamma, mirror) that
    nnUNetTrainerV2 uses by default.
    """

    # =========================================================================
    #  DA parameters — mirror ON (nnUNetTrainerV2 default)
    # =========================================================================

    def setup_DA_params(self):
        """
        Delegate fully to nnUNetTrainerV2.setup_DA_params() via the MRO.

        The base SeqWT trainer calls super().setup_DA_params() and then
        *disables* mirroring.  Here we call the same super() but do NOT
        suppress mirroring, giving us the full nnUNetTrainerV2 DA parameter
        set including:
          - Rotation ±30°  (was ±15° in default_3D_augmentation_params)
          - Scale (0.7, 1.4)  (was (0.85, 1.25))
          - No elastic deformation
          - Mirroring along all axes (axes 0, 1, 2)
          - Gaussian noise, blur, brightness, contrast, low-res simulation,
            gamma transforms (all inherited from get_moreDA_augmentation)
        """
        # nnUNetTrainerV2.setup_DA_params sets:
        #   self.deep_supervision_scales
        #   self.data_aug_params  (full 3D params with ±30° rotation)
        #   self.basic_generator_patch_size  (enlarged for rotation padding)
        #   data_aug_params['patch_size_for_spatialtransform'] = self.patch_size
        #   data_aug_params['do_elastic'] = False
        #   data_aug_params['scale_range'] = (0.7, 1.4)
        #   data_aug_params['selected_seg_channels'] = [0]
        #   data_aug_params['num_cached_per_thread'] = 2
        #
        # We call it via super() which follows the MRO:
        #   nnUNetTrainerV2_SeqWT_DA → nnUNetTrainerV2_SeqWT → nnUNetTrainerV2
        # But nnUNetTrainerV2_SeqWT.setup_DA_params() calls super() *then*
        # sets do_mirror=False.  We therefore skip the SeqWT override and call
        # nnUNetTrainerV2.setup_DA_params directly to get clean, unmodified
        # nnUNetTrainerV2 DA parameters.
        from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
        nnUNetTrainerV2.setup_DA_params(self)
        # do_mirror defaults to True in default_3D_augmentation_params — no
        # override needed.  mirror_axes defaults to (0, 1, 2) — also fine.

    # =========================================================================
    #  DataLoaders — use basic_generator_patch_size for rotation padding
    # =========================================================================

    def get_basic_generators(self):
        """
        Override to use ``self.basic_generator_patch_size`` as the loader
        patch size.

        The base SeqWT trainer always uses ``self.patch_size`` for both the
        loader and the final patch.  With DA active the SpatialTransform
        requires an over-sized patch (proportional to the maximum rotation
        magnitude and scale range) that is cropped to ``self.patch_size``
        *inside* the augmenter.  ``basic_generator_patch_size`` is computed
        by ``nnUNetTrainerV2.setup_DA_params`` specifically for this purpose.
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

    # =========================================================================
    #  Initialisation — swap get_no_augmentation → get_moreDA_augmentation
    # =========================================================================

    def initialize(self, training=True, force_load_plans=False):
        """
        Identical to ``nnUNetTrainerV2_SeqWT.initialize`` with one change:
        the augmenter call is replaced by ``get_moreDA_augmentation`` using
        the exact same signature as ``nnUNetTrainerV2.initialize``.

        All SeqWT-specific initialisation (base_num_features override,
        doubled input channels, network build, fuser setup, etc.) is
        performed by the base SeqWT initialise — we do not touch any of it.
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            # SeqWT-specific overrides (replicated from SeqWT.initialize)
            self.base_num_features = self.seqwt_base_num_features
            self.original_num_input_channels = self.num_input_channels
            self.num_input_channels = 2 * self.num_input_channels

            # ---- DA params (nnUNetTrainerV2 standard) ----
            self.setup_DA_params()

            # ---- Deep-supervision loss weights ----
            import numpy as np
            from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
            net_numpool = len(self.net_num_pool_op_kernel_sizes)
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])
            mask = np.array([True] + [True if i < net_numpool - 1 else False
                                      for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)

            if training:
                self.folder_with_preprocessed_data = join(
                    self.dataset_directory,
                    self.plans['data_identifier'] + "_stage%d" % self.stage,
                )
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")

                # ---- KEY CHANGE: standard nnUNetTrainerV2 augmentation ----
                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr,
                    self.dl_val,
                    self.data_aug_params['patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False,
                )

                self.print_to_log_file(
                    "TRAINING KEYS:\n %s" % str(self.dataset_tr.keys()),
                    also_print_to_console=False)
                self.print_to_log_file(
                    "VALIDATION KEYS:\n %s" % str(self.dataset_val.keys()),
                    also_print_to_console=False)

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            self.print_to_log_file(
                f"[SeqWT_DA] Initialized — "
                f"base_num_features={self.seqwt_base_num_features}, "
                f"input_channels={self.num_input_channels}, "
                f"params={sum(p.numel() for p in self.network.parameters()):,}, "
                f"DA=moreDA (nnUNetTrainerV2 standard)"
            )
        else:
            self.print_to_log_file('already initialized, skipping')
        self.was_initialized = True
