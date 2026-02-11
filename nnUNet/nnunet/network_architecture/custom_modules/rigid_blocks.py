"""
Rigid Convolution Blocks - Drop-in replacements for standard nnUNet blocks

These blocks use RigidConv instead of standard Conv, enabling Bayesian inference
with learned rigidity scores per weight.

Compatible with nnUNet's architecture patterns.
"""

from copy import deepcopy
import torch
from torch import nn
import numpy as np

from nnunet.network_architecture.custom_modules.rigid_conv import RigidConv2d, RigidConv3d


class RigidConvDropoutNormNonlin(nn.Module):
    """
    Rigid (Bayesian) version of ConvDropoutNormNonlin.
    
    Uses RigidConv2d/3d instead of standard Conv2d/3d.
    Interface matches ConvDropoutNormNonlin for drop-in replacement.
    """
    
    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super().__init__()
        
        # Default kwargs
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
        
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        
        # Determine which RigidConv to use based on conv_op
        if conv_op == nn.Conv3d:
            RigidConv = RigidConv3d
        elif conv_op == nn.Conv2d:
            RigidConv = RigidConv2d
        else:
            raise ValueError(f"Unsupported conv_op: {conv_op}. Use nn.Conv2d or nn.Conv3d")
        
        # Extract conv kwargs for RigidConv
        kernel_size = conv_kwargs.get('kernel_size', 3)
        stride = conv_kwargs.get('stride', 1)
        padding = conv_kwargs.get('padding', 1)
        dilation = conv_kwargs.get('dilation', 1)
        bias = conv_kwargs.get('bias', True)
        
        # Create Rigid convolution
        self.conv = RigidConv(
            input_channels, output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias
        )
        
        # Dropout
        if dropout_op is not None and dropout_op_kwargs['p'] is not None and dropout_op_kwargs['p'] > 0:
            self.dropout = dropout_op(**dropout_op_kwargs)
        else:
            self.dropout = None
        
        # Normalization
        self.instnorm = norm_op(output_channels, **norm_op_kwargs)
        
        # Nonlinearity
        self.lrelu = nonlin(**nonlin_kwargs)
    
    def forward(self, x, sample=True):
        """
        Forward pass.
        
        Args:
            x: Input tensor
            sample: Whether to sample weights (True during training)
        """
        x = self.conv(x, sample=sample)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))
    
    def kl_divergence(self):
        """Get KL divergence from the rigid convolution."""
        return self.conv.kl_divergence()
    
    def get_rigidity_stats(self):
        """Get rigidity statistics from the rigid convolution."""
        return self.conv.get_rigidity_stats()


class RigidStackedConvLayers(nn.Module):
    """
    Rigid version of StackedConvLayers.
    
    Stacks RigidConvDropoutNormNonlin layers.
    """
    
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None):
        """
        Stack multiple RigidConvDropoutNormNonlin layers.
        
        first_stride applies only to the first layer in the stack.
        """
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels
        
        # Default kwargs
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
        
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        
        # Handle first stride
        if first_stride is not None:
            conv_kwargs_first = deepcopy(conv_kwargs)
            conv_kwargs_first['stride'] = first_stride
        else:
            conv_kwargs_first = conv_kwargs
        
        super().__init__()
        
        # Build layers
        layers = []
        
        # First layer (different in_channels, possibly different stride)
        layers.append(RigidConvDropoutNormNonlin(
            input_feature_channels, output_feature_channels,
            conv_op, conv_kwargs_first,
            norm_op, norm_op_kwargs,
            dropout_op, dropout_op_kwargs,
            nonlin, nonlin_kwargs
        ))
        
        # Remaining layers
        for _ in range(num_convs - 1):
            layers.append(RigidConvDropoutNormNonlin(
                output_feature_channels, output_feature_channels,
                conv_op, conv_kwargs,
                norm_op, norm_op_kwargs,
                dropout_op, dropout_op_kwargs,
                nonlin, nonlin_kwargs
            ))
        
        self.blocks = nn.ModuleList(layers)
    
    def forward(self, x, sample=True):
        """Forward through all blocks."""
        for block in self.blocks:
            x = block(x, sample=sample)
        return x
    
    def kl_divergence(self):
        """Sum KL divergence from all rigid convolutions."""
        kl = 0.0
        for block in self.blocks:
            kl = kl + block.kl_divergence()
        return kl
    
    def get_rigidity_stats(self):
        """Get rigidity statistics from all blocks."""
        all_stats = []
        for i, block in enumerate(self.blocks):
            stats = block.get_rigidity_stats()
            stats['block_idx'] = i
            all_stats.append(stats)
        return all_stats
