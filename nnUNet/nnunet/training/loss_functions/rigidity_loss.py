"""
Rigidity Loss - Combined Task Loss + KL Divergence for Bayesian Neural Networks

This module provides loss wrappers that add KL divergence regularization to
the standard nnUNet loss (Dice + CrossEntropy).

The KL term implements the "viscosity force" that keeps weights uncertain
unless they are necessary for accuracy. Combined with the "accuracy force"
from backpropagation, this allows rigidity scores to be learned.

Total Loss = Task Loss + kl_weight * KL Divergence
"""

import torch
import torch.nn as nn
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss


class RigidityLoss(nn.Module):
    """
    Combined loss: Task Loss + weighted KL Divergence.
    
    Wraps any base loss and adds KL divergence from the network's Bayesian layers.
    """
    
    def __init__(self, base_loss, kl_weight=1e-3, kl_warmup_epochs=0):
        """
        Initialize RigidityLoss.
        
        Args:
            base_loss: Base loss function (e.g., DC_and_CE_loss)
            kl_weight: Weight for KL divergence term (default: 1e-3)
            kl_warmup_epochs: Number of epochs to linearly warm up KL weight from 0
        """
        super().__init__()
        self.base_loss = base_loss
        self.kl_weight = kl_weight
        self.kl_warmup_epochs = kl_warmup_epochs
        
        # Track for logging
        self.last_task_loss = 0.0
        self.last_kl_loss = 0.0
        self.last_total_loss = 0.0
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        """Set current epoch for warmup scheduling."""
        self.current_epoch = epoch
    
    def get_effective_kl_weight(self):
        """Get KL weight with optional warmup."""
        if self.kl_warmup_epochs <= 0:
            return self.kl_weight
        
        warmup_factor = min(1.0, self.current_epoch / self.kl_warmup_epochs)
        return self.kl_weight * warmup_factor
    
    def forward(self, output, target, kl_divergence):
        """
        Compute combined loss.
        
        Args:
            output: Network output
            target: Ground truth
            kl_divergence: KL divergence from network's get_kl_divergence()
        
        Returns:
            Total loss = task_loss + kl_weight * kl_divergence
        """
        # Compute task loss
        task_loss = self.base_loss(output, target)
        
        # Get effective KL weight (with warmup)
        effective_kl_weight = self.get_effective_kl_weight()
        
        # Compute total loss
        kl_loss = effective_kl_weight * kl_divergence
        total_loss = task_loss + kl_loss
        
        # Store for logging
        self.last_task_loss = task_loss.item() if torch.is_tensor(task_loss) else task_loss
        self.last_kl_loss = kl_loss.item() if torch.is_tensor(kl_loss) else kl_loss
        self.last_total_loss = total_loss.item() if torch.is_tensor(total_loss) else total_loss
        
        return total_loss
    
    def get_loss_breakdown(self):
        """Get breakdown of last computed loss for logging."""
        return {
            'task_loss': self.last_task_loss,
            'kl_loss': self.last_kl_loss,
            'total_loss': self.last_total_loss,
            'effective_kl_weight': self.get_effective_kl_weight()
        }


class MultipleOutputRigidityLoss(nn.Module):
    """
    Deep supervision wrapper for RigidityLoss.
    
    Applies weighted loss to each deep supervision output.
    """
    
    def __init__(self, base_loss, ds_weights, kl_weight=1e-3, kl_warmup_epochs=0):
        """
        Initialize MultipleOutputRigidityLoss.
        
        Args:
            base_loss: Base loss function (e.g., DC_and_CE_loss)
            ds_weights: Deep supervision weights (list of floats)
            kl_weight: Weight for KL divergence term
            kl_warmup_epochs: Number of epochs to warm up KL weight
        """
        super().__init__()
        self.base_loss = base_loss
        self.ds_weights = ds_weights
        self.kl_weight = kl_weight
        self.kl_warmup_epochs = kl_warmup_epochs
        
        # Track for logging
        self.last_task_loss = 0.0
        self.last_kl_loss = 0.0
        self.last_total_loss = 0.0
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        """Set current epoch for warmup scheduling."""
        self.current_epoch = epoch
    
    def get_effective_kl_weight(self):
        """Get KL weight with optional warmup."""
        if self.kl_warmup_epochs <= 0:
            return self.kl_weight
        
        warmup_factor = min(1.0, self.current_epoch / self.kl_warmup_epochs)
        return self.kl_weight * warmup_factor
    
    def forward(self, outputs, targets, kl_divergence):
        """
        Compute combined loss with deep supervision.
        
        Args:
            outputs: List of network outputs at different resolutions
            targets: List of ground truth at different resolutions
            kl_divergence: KL divergence from network's get_kl_divergence()
        
        Returns:
            Total loss = weighted_task_loss + kl_weight * kl_divergence
        """
        # Compute weighted task loss
        total_task_loss = 0.0
        for i, (output, target, weight) in enumerate(zip(outputs, targets, self.ds_weights)):
            if weight > 0:
                total_task_loss = total_task_loss + weight * self.base_loss(output, target)
        
        # Get effective KL weight
        effective_kl_weight = self.get_effective_kl_weight()
        
        # Compute total loss
        kl_loss = effective_kl_weight * kl_divergence
        total_loss = total_task_loss + kl_loss
        
        # Store for logging
        self.last_task_loss = total_task_loss.item() if torch.is_tensor(total_task_loss) else total_task_loss
        self.last_kl_loss = kl_loss.item() if torch.is_tensor(kl_loss) else kl_loss
        self.last_total_loss = total_loss.item() if torch.is_tensor(total_loss) else total_loss
        
        return total_loss
    
    def get_loss_breakdown(self):
        """Get breakdown of last computed loss for logging."""
        return {
            'task_loss': self.last_task_loss,
            'kl_loss': self.last_kl_loss,
            'total_loss': self.last_total_loss,
            'effective_kl_weight': self.get_effective_kl_weight()
        }
