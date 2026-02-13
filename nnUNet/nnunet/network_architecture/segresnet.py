from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch
import torch.nn as nn
from monai.networks.nets import SegResNetDS
import numpy as np

class SegResNetDSWrapper(SegmentationNetwork):
    def __init__(self, spatial_dims=3, init_filters=32, in_channels=1, out_channels=2, 
                 dropout_prob=None, norm_name='group', num_groups=8, use_conv_final=True, 
                 blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), ds_depth=4):
        super().__init__()
        
        # Standard nnU-Net attributes
        self.conv_op = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        self.num_classes = out_channels
        self._deep_supervision = True
        self.do_ds = True # Can be toggled by trainer

        self.segresnet_ds = SegResNetDS(
            spatial_dims=spatial_dims,
            init_filters=init_filters,
            in_channels=in_channels,
            out_channels=out_channels,
            norm=("GROUP", {"num_groups": num_groups}),
            blocks_down=blocks_down,
            dsdepth=ds_depth
        )

    def forward(self, x):
        # monai SegResNetDS forward returns:
        # - Training: x (original resolution) + intermediates (list)
        # - Inference: x (original resolution)
        
        # BUT, SegResNetDS.forward() signature in monai is simply forward(x)
        # It internally checks self.training to decide whether to return ds outputs.
        # nnU-Net controls validation behavior via self.do_ds.
        
        # We need to bridge this. 
        # MONAI's SegResNetDS forward behavior:
        # if self.training: return x, *upsamples (tuple)
        # else: return x
        
        # We must align wrapper's train/eval mode with do_ds?
        # Typically nnU-Net trainer sets model.train() and model.eval().
        # However, nnU-Net also toggles self.do_ds explicitly in validation.
        
        # Compatibility hack:
        # If self.do_ds is False, we only want the first output even if the model is in training mode (rare but possible).
        # OR if we are in eval mode, MONAI returns only one tensor, but do_ds might be True (unlikely).
        
        ret = self.segresnet_ds(x)
        
        if self.do_ds:
            # We expect a tuple/list. MONAI SegResNetDS returns it if in train mode.
            # nnU-Net expects a LIST of outputs [highest_res, lower_res, ...]
            if isinstance(ret, (tuple, list)):
                return list(ret)
            else:
                # Fallback if MONAI didn't return list (e.g. maybe it was in eval mode?)
                return [ret]
        else:
            # Inference mode: we want only single tensor
            if isinstance(ret, (tuple, list)):
                return ret[0]
            else:
                return ret

