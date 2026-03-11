"""
Symmetry Loss Functions for SE-Net

Compound loss: L_total = λ_dice·L_dice + λ_focal·L_focal + λ_sym·L_sym

Components:
  - L_dice: SoftDiceLoss (existing nnUNet)
  - L_focal: FocalLossV2 (existing nnUNet)
  - L_sym: Symmetry-weighted BCE loss (novel)

Symmetry-weighted BCE (Eqs. 14-15):
  w_sym = 1 + β · norm(|x - T(x)|)
  L_sym = -Σ w_sym · [y·log(p) + (1-y)·log(1-p)]

The symmetry weights emphasise regions where bilateral asymmetry is high,
focusing the network on pathologically asymmetric regions (likely lesions).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunet.training.loss_functions.dice_loss import SoftDiceLoss
from nnunet.training.loss_functions.focal_loss import FocalLossV2
from nnunet.utilities.nd_softmax import softmax_helper


class SymmetryWeightedBCELoss(nn.Module):
    """
    Symmetry-weighted Binary Cross-Entropy Loss.
    
    Per-voxel weights derived from bilateral brain asymmetry:
      w_sym = 1 + β · norm(|x - T(x)|)
    
    Applied to BCE:
      L_sym = -(1/|V|) · Σ w_sym · [y·log(p) + (1-y)·log(1-p)]
    
    Args:
        beta: Controls strength of symmetry weighting (default: 1.0)
        smooth: Smoothing for numerical stability (default: 1e-7)
    """
    
    def __init__(self, beta=1.0, smooth=1e-7):
        super().__init__()
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, net_output, target, symmetry_map):
        """
        Args:
            net_output: Network prediction [B, C, ...] (logits or softmax)
            target: Ground truth [B, 1, ...] (label map)
            symmetry_map: Pre-computed symmetry weights [B, 1, ...] 
                         already normalised to [0, 1]
        
        Returns:
            Weighted BCE loss scalar
        """
        # Apply softmax if needed
        net_output = softmax_helper(net_output)
        
        # Get the foreground (lesion) probability
        # For binary segmentation (background + lesion), use channel 1
        if net_output.shape[1] > 1:
            p = net_output[:, 1:2]  # Lesion probability [B, 1, ...]
        else:
            p = net_output
        
        # Clamp for numerical stability
        p = torch.clamp(p, self.smooth, 1.0 - self.smooth)
        
        # Ensure target shape matches
        if target.dim() < p.dim():
            target = target.unsqueeze(1)
        target = target.float()
        
        # Symmetry weight: w = 1 + β · symmetry_map (Eq. 15)
        w_sym = 1.0 + self.beta * symmetry_map
        
        # Weighted BCE (Eq. 14)
        bce = -(target * torch.log(p) + (1 - target) * torch.log(1 - p))
        weighted_bce = w_sym * bce
        
        return weighted_bce.mean()


class DC_Focal_SymBCE_loss(nn.Module):
    """
    Compound loss: L = λ_dice·L_dice + λ_focal·L_focal + λ_sym·L_sym
    
    Combines three complementary loss terms:
    - Dice: Handles class imbalance via overlap optimisation
    - Focal: Emphasises hard-to-classify voxels (boundaries, small lesions)
    - Symmetry BCE: Focuses on bilaterally asymmetric regions
    
    Args:
        soft_dice_kwargs: kwargs for SoftDiceLoss
        focal_kwargs: kwargs for FocalLossV2
        weight_dice: Weight for Dice loss component
        weight_focal: Weight for Focal loss component
        weight_sym: Weight for Symmetry BCE component
        sym_beta: β parameter for symmetry weighting strength
    """
    
    def __init__(self, soft_dice_kwargs, focal_kwargs=None, 
                 weight_dice=1.0, weight_focal=0.5, weight_sym=0.5,
                 sym_beta=1.0):
        super().__init__()
        
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.weight_sym = weight_sym
        
        # Dice loss (existing)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        
        # Focal loss (existing)
        if focal_kwargs is None:
            focal_kwargs = {}
        self.focal = FocalLossV2(apply_nonlin=softmax_helper, **focal_kwargs)
        
        # Symmetry-weighted BCE (novel)
        self.sym_bce = SymmetryWeightedBCELoss(beta=sym_beta)
        
        # Track components for logging
        self.last_dice = 0.0
        self.last_focal = 0.0
        self.last_sym = 0.0
    
    def forward(self, net_output, target, symmetry_map=None):
        """
        Args:
            net_output: Network prediction [B, C, ...]
            target: Ground truth [B, 1, ...]
            symmetry_map: Symmetry weights [B, 1, ...] or None
                         If None, symmetry loss is skipped
        
        Returns:
            Total weighted loss
        """
        # Dice loss
        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        
        # Focal loss
        focal_loss = self.focal(net_output, target) if self.weight_focal != 0 else 0
        
        # Symmetry BCE
        if self.weight_sym != 0 and symmetry_map is not None:
            sym_loss = self.sym_bce(net_output, target, symmetry_map)
        else:
            sym_loss = 0
        
        total = (self.weight_dice * dc_loss + 
                 self.weight_focal * focal_loss + 
                 self.weight_sym * sym_loss)
        
        # Track for logging
        self.last_dice = dc_loss.item() if torch.is_tensor(dc_loss) else dc_loss
        self.last_focal = focal_loss.item() if torch.is_tensor(focal_loss) else focal_loss
        self.last_sym = sym_loss.item() if torch.is_tensor(sym_loss) else sym_loss
        
        return total
    
    def get_loss_breakdown(self):
        """Get breakdown of last computed loss for logging."""
        return {
            'dice_loss': self.last_dice,
            'focal_loss': self.last_focal,
            'symmetry_loss': self.last_sym,
        }


class MultipleOutputSymmetryLoss(nn.Module):
    """
    Deep supervision wrapper for the SE-Net compound loss.
    
    Applies the compound loss at each deep supervision resolution, with
    symmetry maps downscaled to match each resolution.
    
    Follows the MultipleOutputRigidityLoss pattern.
    """
    
    def __init__(self, base_loss, ds_weights):
        """
        Args:
            base_loss: DC_Focal_SymBCE_loss instance
            ds_weights: Deep supervision weights (list of floats)
        """
        super().__init__()
        self.base_loss = base_loss
        self.ds_weights = ds_weights
        
        # Track for logging
        self.last_total_loss = 0.0
    
    def forward(self, outputs, targets, symmetry_maps=None):
        """
        Args:
            outputs: List of network outputs at different resolutions
            targets: List of ground truth at different resolutions
            symmetry_maps: List of symmetry maps at different resolutions,
                          or None to skip symmetry loss
        
        Returns:
            Total weighted loss across all resolutions
        """
        assert isinstance(outputs, (tuple, list)), "outputs must be tuple or list"
        assert isinstance(targets, (tuple, list)), "targets must be tuple or list"
        
        if self.ds_weights is None:
            weights = [1] * len(outputs)
        else:
            weights = self.ds_weights
        
        total_loss = 0.0
        for i in range(len(outputs)):
            if weights[i] > 0:
                sym_map = symmetry_maps[i] if symmetry_maps is not None else None
                total_loss = total_loss + weights[i] * self.base_loss(
                    outputs[i], targets[i], symmetry_map=sym_map
                )
        
        self.last_total_loss = total_loss.item() if torch.is_tensor(total_loss) else total_loss
        
        return total_loss
    
    def get_loss_breakdown(self):
        """Get breakdown from last call."""
        breakdown = self.base_loss.get_loss_breakdown()
        breakdown['total_ds_loss'] = self.last_total_loss
        return breakdown
