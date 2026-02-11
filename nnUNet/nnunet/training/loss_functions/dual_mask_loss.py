"""
Dual Mask Product Loss

Product Dice loss for dual-output segmentation where the network predicts
two masks (one for augmented input, one for clean input).

Loss = -1 × (Dice_augmented × Dice_clean)

This forces the network to perform well on BOTH masks, as the product
is sensitive to the "weaker" prediction (low × high = low).

The augmented mask naturally has lower Dice (harder task), so it
inherently receives more focus when multiplying.
"""

import torch
from torch import nn
import numpy as np
from nnunet.training.loss_functions.dice_loss import SoftDiceLoss, get_tp_fp_fn_tn
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor


class DualMaskSoftDiceLoss(nn.Module):
    """
    Soft Dice loss that computes Dice for a single output-target pair.
    Returns the Dice SCORE (not loss) for use in product calculation.
    """
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        super(DualMaskSoftDiceLoss, self).__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None, return_score=False):
        """
        Args:
            x: Network output (B, C, X, Y, Z)
            y: Ground truth (B, 1, X, Y, Z)
            loss_mask: Optional mask
            return_score: If True, return Dice score (higher=better), else return Dice loss (lower=better)
        """
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        if return_score:
            return dc  # Higher = better
        return -dc  # Lower = better (standard loss)


class DualMaskProductLoss(nn.Module):
    """
    Product Dice loss for dual-output segmentation.
    
    Network output shape: (B, 2*num_classes, X, Y, Z)
        - [:, :num_classes] = prediction for augmented input
        - [:, num_classes:] = prediction for clean input
    
    Target shape: (B, 2, X, Y, Z)
        - [:, 0:1] = ground truth for augmented (spatially transformed)
        - [:, 1:2] = ground truth for clean (original)
    
    Loss = -1 × (Dice_aug × Dice_clean) + CE_aug + CE_clean
    
    The product loss part uses Dice scores (higher=better), so we negate for minimization.
    CE loss is applied separately to both outputs for gradient stability.
    """
    
    def __init__(self, num_classes, soft_dice_kwargs=None, ce_kwargs=None, 
                 weight_ce=1.0, weight_product_dice=1.0):
        """
        Args:
            num_classes: Number of segmentation classes
            soft_dice_kwargs: kwargs for SoftDiceLoss
            ce_kwargs: kwargs for RobustCrossEntropyLoss
            weight_ce: Weight for combined CE loss
            weight_product_dice: Weight for product Dice loss
        """
        super(DualMaskProductLoss, self).__init__()
        
        self.num_classes = num_classes
        self.weight_ce = weight_ce
        self.weight_product_dice = weight_product_dice
        
        if soft_dice_kwargs is None:
            soft_dice_kwargs = {'batch_dice': False, 'do_bg': True, 'smooth': 1.}
        if ce_kwargs is None:
            ce_kwargs = {}
            
        self.dice_loss = DualMaskSoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        
        # For logging individual components
        self.last_dice_aug = None
        self.last_dice_clean = None
        self.last_product_loss = None
        
    def forward(self, net_output, target):
        """
        Args:
            net_output: (B, 2*num_classes, X, Y, Z) - two stacked predictions
            target: (B, 2, X, Y, Z) - [:, 0:1] = aug GT, [:, 1:2] = clean GT
            
        Returns:
            Combined loss (scalar)
        """
        # Split output into augmented and clean predictions
        pred_aug = net_output[:, :self.num_classes]     # (B, num_classes, X, Y, Z)
        pred_clean = net_output[:, self.num_classes:]   # (B, num_classes, X, Y, Z)
        
        # Split target into augmented and clean ground truths
        target_aug = target[:, 0:1]    # (B, 1, X, Y, Z)
        target_clean = target[:, 1:2]  # (B, 1, X, Y, Z)
        
        # Compute Dice SCORES (higher = better)
        dice_score_aug = self.dice_loss(pred_aug, target_aug, return_score=True)
        dice_score_clean = self.dice_loss(pred_clean, target_clean, return_score=True)
        
        # Store for logging
        self.last_dice_aug = dice_score_aug.detach().cpu().item()
        self.last_dice_clean = dice_score_clean.detach().cpu().item()
        
        # Product Dice loss (negate because higher score = better, but we minimize loss)
        product_dice_loss = -1.0 * dice_score_aug * dice_score_clean
        self.last_product_loss = product_dice_loss.detach().cpu().item()
        
        # CE losses for gradient stability
        ce_loss_aug = self.ce(pred_aug, target_aug[:, 0].long()) if self.weight_ce != 0 else 0
        ce_loss_clean = self.ce(pred_clean, target_clean[:, 0].long()) if self.weight_ce != 0 else 0
        ce_loss = ce_loss_aug + ce_loss_clean
        
        # Combined loss
        total_loss = self.weight_product_dice * product_dice_loss + self.weight_ce * ce_loss
        
        return total_loss


class MultipleOutputDualMaskLoss(nn.Module):
    """
    Wrapper for deep supervision with dual-mask product loss.
    
    Applies DualMaskProductLoss at each scale with exponentially decreasing weights.
    """
    
    def __init__(self, loss, weight_factors=None):
        """
        Args:
            loss: DualMaskProductLoss instance
            weight_factors: List of weights for each scale [1.0, 0.5, 0.25, ...]
        """
        super(MultipleOutputDualMaskLoss, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        """
        Args:
            x: List of outputs at each scale [(B, 2*C, X, Y, Z), (B, 2*C, X/2, Y/2, Z/2), ...]
            y: List of targets at each scale [(B, 2, X, Y, Z), (B, 2, X/2, Y/2, Z/2), ...]
        """
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i])
        return l
