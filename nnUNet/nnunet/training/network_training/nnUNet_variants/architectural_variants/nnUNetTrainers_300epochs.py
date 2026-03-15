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

# ── Defensive imports: missing files won't break unrelated trainers ──
def _try_import(module_path, class_name):
    """Import a class, returning None if the module is missing."""
    try:
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        import warnings
        warnings.warn(f"Could not import {class_name} from {module_path}: {e}")
        return None

nnUNetTrainer_NoCLRamp = _try_import(
    'nnunet.training.network_training.nnUNetTrainer_NoCLRamp', 'nnUNetTrainer_NoCLRamp')
nnUNetTrainer_CLRamp = _try_import(
    'nnunet.training.network_training.nnUNetTrainer_CLRamp', 'nnUNetTrainer_CLRamp')
nnUNetTrainer_NoCLRamp_DualMask = _try_import(
    'nnunet.training.network_training.nnUNetTrainer_NoCLRamp_DualMask', 'nnUNetTrainer_NoCLRamp_DualMask')
nnUNetTrainer_CLRamp_DualMask = _try_import(
    'nnunet.training.network_training.nnUNetTrainer_CLRamp_DualMask', 'nnUNetTrainer_CLRamp_DualMask')
nnUNetTrainer_NoCLRamp_DualMask_Adam = _try_import(
    'nnunet.training.network_training.nnUNetTrainer_NoCLRamp_DualMask', 'nnUNetTrainer_NoCLRamp_DualMask_Adam')
nnUNetTrainer_NoCLRamp_DualMask_SGD = _try_import(
    'nnunet.training.network_training.nnUNetTrainer_NoCLRamp_DualMask', 'nnUNetTrainer_NoCLRamp_DualMask_SGD')
nnUNetTrainer_CLRamp_DualMask_Adam = _try_import(
    'nnunet.training.network_training.nnUNetTrainer_CLRamp_DualMask', 'nnUNetTrainer_CLRamp_DualMask_Adam')
nnUNetTrainer_CLRamp_DualMask_SGD = _try_import(
    'nnunet.training.network_training.nnUNetTrainer_CLRamp_DualMask', 'nnUNetTrainer_CLRamp_DualMask_SGD')
nnUNetTrainer_Rigidity = _try_import(
    'nnunet.training.network_training.nnUNetTrainer_Rigidity', 'nnUNetTrainer_Rigidity')
nnUNetTrainerV2_InputResidualUNet_PerStage = _try_import(
    'nnunet.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_InputResidualUNet_PerStage',
    'nnUNetTrainerV2_InputResidualUNet_PerStage')
nnUNetTrainerV2_InputResidualUNet_PerConv = _try_import(
    'nnunet.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_InputResidualUNet_PerConv',
    'nnUNetTrainerV2_InputResidualUNet_PerConv')
nnUNetTrainerV2_InputResidualUNet_PerStage_Unique = _try_import(
    'nnunet.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_InputResidualUNet_PerStage_Unique',
    'nnUNetTrainerV2_InputResidualUNet_PerStage_Unique')
nnUNetTrainerV2_InputResidualUNet_PerConv_Unique = _try_import(
    'nnunet.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_InputResidualUNet_PerConv_Unique',
    'nnUNetTrainerV2_InputResidualUNet_PerConv_Unique')
nnUNetTrainerV2_InputResidualUNet_PerStage_Split = _try_import(
    'nnunet.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_InputResidualUNet_PerStage_Split',
    'nnUNetTrainerV2_InputResidualUNet_PerStage_Split')
nnUNetTrainerV2_InputResidualUNet_PerConv_Split = _try_import(
    'nnunet.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_InputResidualUNet_PerConv_Split',
    'nnUNetTrainerV2_InputResidualUNet_PerConv_Split')
nnUNetTrainerV2_InputResidualUNet_PerStage_SplitDDS = _try_import(
    'nnunet.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_InputResidualUNet_PerStage_SplitDDS',
    'nnUNetTrainerV2_InputResidualUNet_PerStage_SplitDDS')
nnUNetTrainerV2_InputResidualUNet_PerConv_SplitDDS = _try_import(
    'nnunet.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_InputResidualUNet_PerConv_SplitDDS',
    'nnUNetTrainerV2_InputResidualUNet_PerConv_SplitDDS')
nnUNetTrainer_SegResNet = _try_import(
    'nnunet.training.network_training.nnUNetTrainer_SegResNet', 'nnUNetTrainer_SegResNet')
nnUNetTrainer_SegResNet_IN = _try_import(
    'nnunet.training.network_training.nnUNetTrainer_SegResNet_IN', 'nnUNetTrainer_SegResNet_IN')
nnUNetTrainer_SENet = _try_import(
    'nnunet.training.network_training.nnUNetTrainer_SENet', 'nnUNetTrainer_SENet')
nnUNetTrainer_MaskDenoise = _try_import(
    'nnunet.training.network_training.nnUNetTrainer_MaskDenoise', 'nnUNetTrainer_MaskDenoise')
nnUNetTrainer_MaskDenoiseRandom = _try_import(
    'nnunet.training.network_training.nnUNetTrainer_MaskDenoise', 'nnUNetTrainer_MaskDenoiseRandom')
nnUNetTrainerV2_ProgressiveChannels = _try_import(
    'nnunet.training.network_training.nnUNetTrainerV2_ProgressiveChannels', 'nnUNetTrainerV2_ProgressiveChannels')


# ── Helper to create 300-epoch subclass ──────────────────────
def _make_300ep(base_cls, name):
    """Dynamically create a 300-epoch subclass if base class is available."""
    if base_cls is None:
        return None
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        base_cls.__init__(self, plans_file, fold, output_folder, dataset_directory,
                          batch_dice, stage, unpack_data, deterministic, fp16)
        self.max_num_epochs = 300
    cls = type(name, (base_cls,), {'__init__': __init__})
    return cls


class nnUNetTrainerV2_300epochs(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)
        self.max_num_epochs = 300


nnUNetTrainer_NoCLRamp_300epochs = _make_300ep(
    nnUNetTrainer_NoCLRamp, 'nnUNetTrainer_NoCLRamp_300epochs')
nnUNetTrainer_CLRamp_300epochs = _make_300ep(
    nnUNetTrainer_CLRamp, 'nnUNetTrainer_CLRamp_300epochs')
nnUNetTrainer_NoCLRamp_DualMask_300epochs = _make_300ep(
    nnUNetTrainer_NoCLRamp_DualMask, 'nnUNetTrainer_NoCLRamp_DualMask_300epochs')
nnUNetTrainer_CLRamp_DualMask_300epochs = _make_300ep(
    nnUNetTrainer_CLRamp_DualMask, 'nnUNetTrainer_CLRamp_DualMask_300epochs')
nnUNetTrainer_NoCLRamp_DualMask_Adam_300epochs = _make_300ep(
    nnUNetTrainer_NoCLRamp_DualMask_Adam, 'nnUNetTrainer_NoCLRamp_DualMask_Adam_300epochs')
nnUNetTrainer_NoCLRamp_DualMask_SGD_300epochs = _make_300ep(
    nnUNetTrainer_NoCLRamp_DualMask_SGD, 'nnUNetTrainer_NoCLRamp_DualMask_SGD_300epochs')
nnUNetTrainer_CLRamp_DualMask_Adam_300epochs = _make_300ep(
    nnUNetTrainer_CLRamp_DualMask_Adam, 'nnUNetTrainer_CLRamp_DualMask_Adam_300epochs')
nnUNetTrainer_CLRamp_DualMask_SGD_300epochs = _make_300ep(
    nnUNetTrainer_CLRamp_DualMask_SGD, 'nnUNetTrainer_CLRamp_DualMask_SGD_300epochs')
nnUNetTrainer_Rigidity_300epochs = _make_300ep(
    nnUNetTrainer_Rigidity, 'nnUNetTrainer_Rigidity_300epochs')
nnUNetTrainerV2_InputResidualUNet_PerStage_300epochs = _make_300ep(
    nnUNetTrainerV2_InputResidualUNet_PerStage, 'nnUNetTrainerV2_InputResidualUNet_PerStage_300epochs')
nnUNetTrainerV2_InputResidualUNet_PerConv_300epochs = _make_300ep(
    nnUNetTrainerV2_InputResidualUNet_PerConv, 'nnUNetTrainerV2_InputResidualUNet_PerConv_300epochs')
nnUNetTrainerV2_InputResidualUNet_PerStage_Unique_300epochs = _make_300ep(
    nnUNetTrainerV2_InputResidualUNet_PerStage_Unique, 'nnUNetTrainerV2_InputResidualUNet_PerStage_Unique_300epochs')
nnUNetTrainerV2_InputResidualUNet_PerConv_Unique_300epochs = _make_300ep(
    nnUNetTrainerV2_InputResidualUNet_PerConv_Unique, 'nnUNetTrainerV2_InputResidualUNet_PerConv_Unique_300epochs')
nnUNetTrainerV2_InputResidualUNet_PerStage_Split_300epochs = _make_300ep(
    nnUNetTrainerV2_InputResidualUNet_PerStage_Split, 'nnUNetTrainerV2_InputResidualUNet_PerStage_Split_300epochs')
nnUNetTrainerV2_InputResidualUNet_PerConv_Split_300epochs = _make_300ep(
    nnUNetTrainerV2_InputResidualUNet_PerConv_Split, 'nnUNetTrainerV2_InputResidualUNet_PerConv_Split_300epochs')
nnUNetTrainerV2_InputResidualUNet_PerStage_SplitDDS_300epochs = _make_300ep(
    nnUNetTrainerV2_InputResidualUNet_PerStage_SplitDDS, 'nnUNetTrainerV2_InputResidualUNet_PerStage_SplitDDS_300epochs')
nnUNetTrainerV2_InputResidualUNet_PerConv_SplitDDS_300epochs = _make_300ep(
    nnUNetTrainerV2_InputResidualUNet_PerConv_SplitDDS, 'nnUNetTrainerV2_InputResidualUNet_PerConv_SplitDDS_300epochs')
nnUNetTrainer_SegResNet_300epochs = _make_300ep(
    nnUNetTrainer_SegResNet, 'nnUNetTrainer_SegResNet_300epochs')
nnUNetTrainer_SegResNet_IN_300epochs = _make_300ep(
    nnUNetTrainer_SegResNet_IN, 'nnUNetTrainer_SegResNet_IN_300epochs')
nnUNetTrainer_SENet_300epochs = _make_300ep(
    nnUNetTrainer_SENet, 'nnUNetTrainer_SENet_300epochs')
nnUNetTrainer_MaskDenoise_300epochs = _make_300ep(
    nnUNetTrainer_MaskDenoise, 'nnUNetTrainer_MaskDenoise_300epochs')
nnUNetTrainer_MaskDenoiseRandom_300epochs = _make_300ep(
    nnUNetTrainer_MaskDenoiseRandom, 'nnUNetTrainer_MaskDenoiseRandom_300epochs')
nnUNetTrainerV2_ProgressiveChannels_300epochs = _make_300ep(
    nnUNetTrainerV2_ProgressiveChannels, 'nnUNetTrainerV2_ProgressiveChannels_300epochs')

