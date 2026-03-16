import torch
import numpy as np
import multiprocessing
from batchgenerators.utilities.file_and_folder_operations import join, save_json
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.data_augmentation.segresnet_aug_ramp import get_segresnet_ramp_augmentation
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2


class nnUNetTrainerV2_SegResNetAugRamp(nnUNetTrainerV2):
    """
    Implements the SegResNet Augmentation Probability Ramp logic:
    1. Initializes a shared multiprocessing.Value('f', 0.0) for ramp probability.
    2. Uses get_segresnet_ramp_augmentation which absolute-gates intensity augmentations.
    3. Tracks init_loss vs best_loss on epoch end, and dynamically scales the shared probability.
    """
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)

        # RAMP PROBABILITY SHARED MEMORY (starts at 0.0, goes up to 1.0)
        # Using multiprocessing.Value so C-level worker threads can read this safely.
        self.ramp_probability = multiprocessing.Value('f', 0.0)

        # STATE TRACKING
        self.init_loss = None
        self.best_loss_ever = None
        self.plateau_patience = 1  # Standard SegResNet patience scaler

        # Append suffix to clearly distinguish these models
        if self.output_folder is not None:
            self.output_folder += "_SegResNetAugRamp"

    def setup_DA_params(self):
        super().setup_DA_params()
        # Ensure our deep supervision logic matches expectations for get_moreDA_augmentation
        self.data_aug_params['deep_supervision_scales'] = self.deep_supervision_scales

    def initialize(self, training=True, force_load_plans=False):
        """
        We completely override the grandparent `initialize` (which calls get_moreDA_augmentation)
        with our custom generator pipeline that injects `self.ramp_probability`.
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            ################# Here we construct the dataloaders #################
            if training:
                self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                          "_stage%d" % self.stage)

                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                # =========================================================
                # INJECT SEGRESNET AUGMENTATION RAMP
                # =========================================================
                self.tr_gen, self.val_gen = get_segresnet_ramp_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.ramp_probability,  # <--- PASSING THE MULTIPROCESSING VALUE
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                self.print_to_log_file("SegResNet Augmentation Ramp Generator Instantiated!")
                self.print_to_log_file(f"Initial shared probe value: {self.ramp_probability.value}")
                
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def on_epoch_end(self):
        """
        Calculates epoch loss, calculates init vs best loss, and dynamically updates self.ramp_probability.
        """
        # 1. Let grandparent do checkpointing and validation
        res = super().on_epoch_end()

        # 2. Extract training loss for this epoch from the history list
        # We need the most recent average training loss
        if len(self.all_tr_losses) > 0:
            current_epoch_loss = self.all_tr_losses[-1]
            
            # Setup init_loss on epoch 0
            if self.init_loss is None:
                self.init_loss = current_epoch_loss
                self.best_loss_ever = current_epoch_loss
                self.print_to_log_file(f"[SegResNet Ramp] Initial Loss set to {self.init_loss:.4f}")
            
            # Check for a new best loss
            if current_epoch_loss < self.best_loss_ever:
                old_prob = self.ramp_probability.value
                self.best_loss_ever = current_epoch_loss
                
                # Math identically ported from localtransforms.py: DynamicProbabilityTransform.set_probability
                # improvement_ratio = (init_loss - best_loss) / (patience * init_loss)
                # self.current_prob = self.start_prob + (1 - self.start_prob) * improvement_ratio
                # Note: start_prob is 0.0
                # Math: improvement_ratio = (init_loss - best_loss) / (init_loss - target_loss)
                # For standard DC_and_CE, target_loss is -1.0.
                total_possible_improvement = self.init_loss - (-1.0)
                improvement_ratio = (self.init_loss - self.best_loss_ever) / (self.plateau_patience * total_possible_improvement)
                
                # Clamp between 0 and 1
                new_prob = max(0.0, min(1.0, float(improvement_ratio)))
                
                # Mutate shared memory specifically designed for this
                self.ramp_probability.value = new_prob
                
                self.print_to_log_file(
                    f"[SegResNet Ramp] NEW BEST LOSS! {current_epoch_loss:.4f} < {self.init_loss:.4f} (init)"
                )
                self.print_to_log_file(
                    f"AUGMENTATION UPDATE: Shared Probability {old_prob:.4f} -> {new_prob:.4f}"
                )

        return res

    def _save_experiment_config(self):
        # We also want to save the state of our probability value into an experiment config
        # Just creating a separate file next to the model is safest
        state_dict = {
            "init_loss": float(self.init_loss) if self.init_loss is not None else None,
            "best_loss_ever": float(self.best_loss_ever) if self.best_loss_ever is not None else None,
            "ramp_probability": float(self.ramp_probability.value)
        }
        
        save_file = join(self.output_folder, "segresnet_ramp_state.json")
        try:
            save_json(state_dict, save_file)
        except Exception as e:
            self.print_to_log_file("Failed to save segresnet_ramp_state:", e)

    def save_checkpoint(self, fname, save_optimizer=True):
        super().save_checkpoint(fname, save_optimizer)
        self._save_experiment_config()

# Helper imports used in grandparent override
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch.nn as nn
