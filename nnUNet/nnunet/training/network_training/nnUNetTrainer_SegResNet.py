from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.network_architecture.segresnet import SegResNetDSWrapper
from torch import nn
import numpy as np

class nnUNetTrainer_SegResNet(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)

    def initialize_network(self):
        """
        Modified from V2 to use SegResNetDSWrapper.
        We configure deep supervision scales here.
        By default SegResNetDS has `ds_depth=4`, meaning main output + 4 downsampled outputs?
        No, ds_depth refers to how many levels deep the supervision goes.
        Typically we align this with nnUNet's expectation.
        
        Let's assume default config:
        Main output (x)
        DS1 (0.5x)
        DS2 (0.25x)
        DS3 (0.125x)
        
        This corresponds to scales:
        [[1,1,1], [0.5,0.5,0.5], [0.25,0.25,0.25], [0.125,0.125,0.125]]
        """
        
        ds_depth = 4 # Default for SegResNetDS
        
        # Define scales for Deep Supervision Loss
        self.deep_supervision_scales = [[1, 1, 1]] + [[1 / (2 ** i)] * 3 for i in range(1, ds_depth)]
        
        # Instantiate the network wrapper
        # Input channels: self.num_input_channels
        # Output channels: self.num_classes
        self.network = SegResNetDSWrapper(
            spatial_dims=3, # Assuming 3D tasks for now
            in_channels=self.num_input_channels,
            out_channels=self.num_classes,
            init_filters=32, # Configurable
            blocks_down=(1, 2, 2, 4), # Default
            blocks_up=(1, 1, 1),      # Default
            ds_depth=ds_depth
        )
        
        if torch.cuda.is_available():
            self.network.cuda()
            
        # Inference nonlinearity
        # self.network.inference_apply_nonlin = softmax_helper # Inherited from SegmentationNetwork default
