"""
Rigid Convolution Layers - Bayesian Neural Network Components

Implements variational convolution layers where each weight is a Gaussian distribution.
- mu (μ): Mean of the weight distribution (the "value")
- rho (ρ): Raw parameter controlling variance (the "rigidity control")
- sigma (σ): Standard deviation = softplus(ρ) = log(1 + exp(ρ))
- rigidity: Precision = 1/σ² (high = rigid/certain, low = viscous/uncertain)

During training, weights are sampled using the reparameterization trick:
    w = μ + σ * ε, where ε ~ N(0,1)

During inference (sample=False), weights are deterministic:
    w = μ

The KL divergence from the posterior q(w|μ,σ) to the prior p(w) = N(0,1) provides
regularization that encourages weights to remain uncertain unless necessary for accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RigidConvBase(nn.Module):
    """
    Base class for Rigid (Bayesian) convolution layers.
    
    Learns a Gaussian distribution per weight with mean μ and variance σ².
    The rigidity score (precision = 1/σ²) is learned during training.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, conv_fn=None, ndim=3):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Ensure kernel_size is a tuple of ints (nnUNet may pass lists)
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size,) * ndim
        elif isinstance(kernel_size, (list, tuple)):
            self.kernel_size = tuple(int(k) for k in kernel_size)
        else:
            self.kernel_size = (int(kernel_size),) * ndim
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        self.conv_fn = conv_fn
        self.ndim = ndim
        
        # Weight shape: (out_channels, in_channels/groups, *kernel_size)
        weight_shape = (out_channels, in_channels // groups) + self.kernel_size
        
        # Mean of weight distribution (μ)
        self.mu = nn.Parameter(torch.Tensor(*weight_shape))
        
        # Raw rigidity parameter (ρ) - transformed to σ via softplus
        # Initialize with negative values for higher initial variance (more exploration)
        self.rho = nn.Parameter(torch.Tensor(*weight_shape))
        
        # Bias parameters (optional)
        if bias:
            self.mu_bias = nn.Parameter(torch.Tensor(out_channels))
            self.rho_bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize μ with He init, ρ with values giving initial σ ≈ 0.1."""
        # He initialization for μ
        nn.init.kaiming_normal_(self.mu, mode='fan_in', nonlinearity='leaky_relu')
        
        # Initialize ρ to give σ ≈ 0.1 (softplus(-2.2) ≈ 0.1)
        # This starts with some uncertainty but not too much
        nn.init.constant_(self.rho, -2.2)
        
        if self.use_bias:
            # Bias initialized to zero mean
            nn.init.zeros_(self.mu_bias)
            nn.init.constant_(self.rho_bias, -2.2)
    
    @property
    def sigma(self):
        """Standard deviation σ = softplus(ρ) = log(1 + exp(ρ))."""
        return F.softplus(self.rho)
    
    @property
    def sigma_bias(self):
        """Standard deviation for bias."""
        if self.mu_bias is not None:
            return F.softplus(self.rho_bias)
        return None
    
    @property
    def rigidity(self):
        """Rigidity (precision) = 1/σ². Higher = more certain."""
        return 1.0 / (self.sigma ** 2 + 1e-8)
    
    @property
    def rigidity_bias(self):
        """Rigidity for bias."""
        if self.mu_bias is not None:
            return 1.0 / (self.sigma_bias ** 2 + 1e-8)
        return None
    
    def sample_weight(self, sample=True):
        """
        Sample weight using reparameterization trick.
        
        Args:
            sample: If True, sample w = μ + σ*ε. If False, return μ (deterministic).
        
        Returns:
            Sampled or deterministic weight tensor.
        """
        if sample and self.training:
            epsilon = torch.randn_like(self.sigma)
            weight = self.mu + self.sigma * epsilon
        else:
            weight = self.mu
        return weight
    
    def sample_bias(self, sample=True):
        """Sample bias using reparameterization trick."""
        if self.mu_bias is None:
            return None
        
        if sample and self.training:
            epsilon = torch.randn_like(self.sigma_bias)
            bias = self.mu_bias + self.sigma_bias * epsilon
        else:
            bias = self.mu_bias
        return bias
    
    def kl_divergence(self):
        """
        Compute KL divergence from posterior q(w|μ,σ) to prior p(w) = N(0,1).
        
        KL(q||p) = 0.5 * sum(σ² + μ² - 1 - log(σ²))
        
        Returns:
            Scalar tensor with total KL divergence.
        """
        sigma_sq = self.sigma ** 2
        kl = 0.5 * torch.sum(sigma_sq + self.mu ** 2 - 1 - torch.log(sigma_sq + 1e-8))
        
        if self.mu_bias is not None:
            sigma_bias_sq = self.sigma_bias ** 2
            kl = kl + 0.5 * torch.sum(sigma_bias_sq + self.mu_bias ** 2 - 1 - torch.log(sigma_bias_sq + 1e-8))
        
        return kl
    
    def get_rigidity_stats(self):
        """
        Get statistics about rigidity distribution for logging.
        
        Returns:
            Dict with mean, std, min, max rigidity values.
        """
        rig = self.rigidity.detach()
        stats = {
            'mean': rig.mean().item(),
            'std': rig.std().item(),
            'min': rig.min().item(),
            'max': rig.max().item(),
            'num_params': rig.numel()
        }
        return stats
    
    def forward(self, x, sample=True):
        """
        Forward pass with sampled or deterministic weights.
        
        Args:
            x: Input tensor
            sample: Whether to sample weights (True during training)
        
        Returns:
            Convolution output
        """
        weight = self.sample_weight(sample)
        bias = self.sample_bias(sample)
        return self.conv_fn(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)


class RigidConv2d(RigidConvBase):
    """2D Rigid (Bayesian) Convolution Layer."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, conv_fn=F.conv2d, ndim=2
        )


class RigidConv3d(RigidConvBase):
    """3D Rigid (Bayesian) Convolution Layer."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, conv_fn=F.conv3d, ndim=3
        )
