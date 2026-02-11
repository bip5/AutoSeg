
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.data_augmentation.iterative_denoising_augmentation import get_iterative_denoising_augmentation
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from torch.cuda.amp import autocast
import torch
import numpy as np

class nnUNetTrainer_IterativeDenoising(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.noise_alias = "all" # Default alias, can be changed via args or subclassing
        self.current_noise_magnitude = 0.0 # Placeholder for ramping

    def initialize_network(self):
        # Double the input channels because we concat (Augmented, Clean)
        self.num_input_channels = 2 * self.num_input_channels
        super().initialize_network()

    def initialize(self, training=True, force_load_plans=False):
        super().initialize(training, force_load_plans)
        
        if training:
            # Overwrite the data generators with our Iterative Denoising one
            self.tr_gen, self.val_gen = get_iterative_denoising_augmentation(
                self.dl_tr, self.dl_val,
                self.data_aug_params['patch_size_for_spatialtransform'],
                self.data_aug_params,
                deep_supervision_scales=self.deep_supervision_scales,
                pin_memory=self.pin_memory,
                noise_alias=self.noise_alias
            )
            
    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        # Same as V2, but we might want to log something about noise?
        # Standard implementation works because data_generator yields 'data' with 2*channels.
        return super().run_iteration(data_generator, do_backprop, run_online_evaluation)

    def update_noise_magnitude(self, epoch):
        """
        Hook to update noise magnitude based on epoch (Ramping).
        Currently a placeholder. Implementing dynamic updates across multi-process workers 
        requires shared memory or generator re-initialization.
        """
        # Example logic:
        # magnitude = min(1.0, epoch / 100.0)
        # self.current_noise_magnitude = magnitude
        pass

    def on_epoch_end(self):
        self.update_noise_magnitude(self.epoch)
        return super().on_epoch_end()
