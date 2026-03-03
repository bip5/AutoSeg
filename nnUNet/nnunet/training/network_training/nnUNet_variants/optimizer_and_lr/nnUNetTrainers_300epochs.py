#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0

"""
300-epoch variants of all custom trainers.
Poly LR scheduling automatically adjusts to the longer schedule.

Usage examples:
    nnUNet_train 3d_fullres nnUNetTrainerV2_300epochs TASK_ID FOLD
    nnUNet_train 3d_fullres nnUNetTrainer_NoCLRamp_300epochs TASK_ID FOLD
    nnUNet_train 3d_fullres nnUNetTrainer_Rigidity_300epochs TASK_ID FOLD
    etc.
"""

from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.network_training.nnUNetTrainer_NoCLRamp import nnUNetTrainer_NoCLRamp
from nnunet.training.network_training.nnUNetTrainer_CLRamp import nnUNetTrainer_CLRamp
from nnunet.training.network_training.nnUNetTrainer_NoCLRamp_DualMask import nnUNetTrainer_NoCLRamp_DualMask
from nnunet.training.network_training.nnUNetTrainer_CLRamp_DualMask import nnUNetTrainer_CLRamp_DualMask
from nnunet.training.network_training.nnUNetTrainer_Rigidity import nnUNetTrainer_Rigidity
from nnunet.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_InputResidualUNet_PerStage import nnUNetTrainerV2_InputResidualUNet_PerStage
from nnunet.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_InputResidualUNet_PerConv import nnUNetTrainerV2_InputResidualUNet_PerConv
from nnunet.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_InputResidualUNet_PerStage_Unique import nnUNetTrainerV2_InputResidualUNet_PerStage_Unique
from nnunet.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_InputResidualUNet_PerConv_Unique import nnUNetTrainerV2_InputResidualUNet_PerConv_Unique
from nnunet.training.network_training.nnUNetTrainer_SegResNet import nnUNetTrainer_SegResNet
from nnunet.training.network_training.nnUNetTrainer_SegResNet_IN import nnUNetTrainer_SegResNet_IN
from nnunet.training.network_training.nnUNetTrainer_SENet import nnUNetTrainer_SENet


class nnUNetTrainerV2_300epochs(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)
        self.max_num_epochs = 300


class nnUNetTrainer_NoCLRamp_300epochs(nnUNetTrainer_NoCLRamp):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)
        self.max_num_epochs = 300


class nnUNetTrainer_CLRamp_300epochs(nnUNetTrainer_CLRamp):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)
        self.max_num_epochs = 300


class nnUNetTrainer_NoCLRamp_DualMask_300epochs(nnUNetTrainer_NoCLRamp_DualMask):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)
        self.max_num_epochs = 300


class nnUNetTrainer_CLRamp_DualMask_300epochs(nnUNetTrainer_CLRamp_DualMask):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)
        self.max_num_epochs = 300


class nnUNetTrainer_Rigidity_300epochs(nnUNetTrainer_Rigidity):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)
        self.max_num_epochs = 300


class nnUNetTrainerV2_InputResidualUNet_PerStage_300epochs(nnUNetTrainerV2_InputResidualUNet_PerStage):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)
        self.max_num_epochs = 300


class nnUNetTrainerV2_InputResidualUNet_PerConv_300epochs(nnUNetTrainerV2_InputResidualUNet_PerConv):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)
        self.max_num_epochs = 300


class nnUNetTrainer_SegResNet_300epochs(nnUNetTrainer_SegResNet):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)
        self.max_num_epochs = 300


class nnUNetTrainer_SegResNet_IN_300epochs(nnUNetTrainer_SegResNet_IN):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)
        self.max_num_epochs = 300


class nnUNetTrainerV2_InputResidualUNet_PerStage_Unique_300epochs(nnUNetTrainerV2_InputResidualUNet_PerStage_Unique):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)
        self.max_num_epochs = 300


class nnUNetTrainerV2_InputResidualUNet_PerConv_Unique_300epochs(nnUNetTrainerV2_InputResidualUNet_PerConv_Unique):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)
        self.max_num_epochs = 300


class nnUNetTrainer_SENet_300epochs(nnUNetTrainer_SENet):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)
        self.max_num_epochs = 300

