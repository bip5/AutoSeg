"""
SegResNet trainer with Instance Normalisation, Cosine Annealing LR, and Adam optimizer.

Differences from nnUNetTrainer_SegResNet:
    - Instance Norm instead of Group Norm (8 groups)
    - Adam optimizer (lr=2e-4, weight_decay=3e-5) instead of SGD
    - Cosine Annealing LR scheduler (T_max=max_num_epochs) instead of Poly LR
"""

from nnunet.training.network_training.nnUNetTrainer_SegResNet import nnUNetTrainer_SegResNet
from nnunet.network_architecture.segresnet import SegResNetDSWrapper
import torch
import numpy as np


class nnUNetTrainer_SegResNet_IN(nnUNetTrainer_SegResNet):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)
        self.initial_lr = 2e-4

    def initialize_network(self):
        """SegResNetDS with Instance Normalisation instead of Group Norm."""
        ds_depth = 4

        self.deep_supervision_scales = [[1, 1, 1], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]

        self.network = SegResNetDSWrapper(
            spatial_dims=3,
            in_channels=self.num_input_channels,
            out_channels=self.num_classes,
            init_filters=32,
            blocks_down=(1, 2, 2, 4),
            norm=("INSTANCE", {}),  # Instance Norm instead of GROUP
            dsdepth=ds_depth
        )

        if torch.cuda.is_available():
            self.network.cuda()

    def initialize_optimizer_and_scheduler(self):
        """Adam + Cosine Annealing instead of SGD + Poly LR."""
        assert self.network is not None, "initialize_network must be called first"

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay
        )

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_num_epochs,
            eta_min=0
        )

    def maybe_update_lr(self, epoch=None):
        """Step the cosine annealing scheduler instead of poly LR."""
        if epoch is not None:
            self.epoch = epoch

        self.lr_scheduler.step(self.epoch)

        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=8))
