#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0

"""
150-epoch variants of all custom trainers.
Poly LR scheduling automatically adjusts to the shorter schedule.

Usage examples:
    nnUNet_train 3d_fullres nnUNetTrainer_IterativeDenoising_150epochs TASK_ID FOLD
    nnUNet_train 3d_fullres nnUNetTrainer_NoCLRamp_150epochs TASK_ID FOLD
    etc.
"""

from nnunet.training.network_training.nnUNetTrainer_IterativeDenoising import nnUNetTrainer_IterativeDenoising
from nnunet.training.network_training.nnUNetTrainer_NoCLRamp import nnUNetTrainer_NoCLRamp
from nnunet.training.network_training.nnUNetTrainer_CLRamp import nnUNetTrainer_CLRamp
from nnunet.training.network_training.nnUNetTrainer_NoCLRamp_DualMask import nnUNetTrainer_NoCLRamp_DualMask
from nnunet.training.network_training.nnUNetTrainer_CLRamp_DualMask import nnUNetTrainer_CLRamp_DualMask
from nnunet.training.network_training.nnUNetTrainer_Rigidity import nnUNetTrainer_Rigidity
from nnunet.training.network_training.nnUNetTrainer_SegResNet import nnUNetTrainer_SegResNet


class nnUNetTrainer_IterativeDenoising_150epochs(nnUNetTrainer_IterativeDenoising):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)
        self.max_num_epochs = 150


class nnUNetTrainer_NoCLRamp_150epochs(nnUNetTrainer_NoCLRamp):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)
        self.max_num_epochs = 150


class nnUNetTrainer_CLRamp_150epochs(nnUNetTrainer_CLRamp):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)
        self.max_num_epochs = 150


class nnUNetTrainer_NoCLRamp_DualMask_150epochs(nnUNetTrainer_NoCLRamp_DualMask):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)
        self.max_num_epochs = 150


class nnUNetTrainer_CLRamp_DualMask_150epochs(nnUNetTrainer_CLRamp_DualMask):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)
        self.max_num_epochs = 150


class nnUNetTrainer_Rigidity_150epochs(nnUNetTrainer_Rigidity):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)
        self.max_num_epochs = 150


class nnUNetTrainer_SegResNet_150epochs(nnUNetTrainer_SegResNet):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)
        self.max_num_epochs = 150
