#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0

"""
Trainer using the Input-Residual UNet (Per-Conv SplitDDS variant).

This variant adds Deep Supervision to the encoder blocks.
Max epochs set to 150; poly LR scheduling adjusts automatically.
"""

import numpy as np
import torch
from torch import nn

from nnunet.network_architecture.generic_UNet_InputResidual import Generic_UNet_InputResidual_PerConv_SplitDDS
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2


class nnUNetTrainerV2_InputResidualUNet_PerConv_SplitDDS(nnUNetTrainerV2):

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)
        self.max_num_epochs = 150

    def initialize_network(self):
        """
        Replaces Generic_UNet with Generic_UNet_InputResidual_PerConv_SplitDDS.
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d
        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}

        self.network = Generic_UNet_InputResidual_PerConv_SplitDDS(
            self.num_input_channels, self.base_num_features, self.num_classes,
            len(self.net_num_pool_op_kernel_sizes),
            self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
            dropout_op_kwargs, net_nonlin, net_nonlin_kwargs,
            True, False, lambda x: x, InitWeights_He(1e-2),
            self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes,
            False, True, True
        )

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def setup_DA_params(self):
        """
        Override deep_supervision_scales to include both decoder and encoder outputs.
        Decoder targets: [highest_res, downsampled_1, downsampled_2, ...]
        Encoder targets: [highest_res, downsampled_1, ..., bottleneck]
        """
        super().setup_DA_params()
        
        dec_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]
        
        enc_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))
            
        self.deep_supervision_scales = dec_scales + enc_scales
        
    def initialize(self, training=True, force_load_plans=False):
        """
        Override to redefine the deep supervision loss weights.
        """
        original_loss = self.loss
        if isinstance(original_loss, MultipleOutputLoss2):
            original_loss = original_loss.loss
            
        super().initialize(training, force_load_plans)
        
        if training:
            net_numpool = len(self.net_num_pool_op_kernel_sizes)
            dec_weights = [1 / (2 ** i) for i in range(net_numpool)]
            enc_weights = [1 / (2 ** i) for i in range(net_numpool + 1)]
            
            all_weights = np.array(dec_weights + enc_weights)
            all_mask = np.ones(len(all_weights), dtype=bool)
            
            all_mask[net_numpool - 1] = False
            all_mask[-1] = False
            all_mask[-2] = False
            
            all_weights[~all_mask] = 0
            all_weights = all_weights / all_weights.sum()
            
            self.ds_loss_weights = all_weights
            self.loss = MultipleOutputLoss2(original_loss, self.ds_loss_weights)
