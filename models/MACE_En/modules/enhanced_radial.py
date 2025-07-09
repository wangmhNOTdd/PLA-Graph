#!/usr/bin/python
# -*- coding:utf-8 -*-
###########################################################################################
# Enhanced Radial Basis Functions with Improved Numerical Stability
# Based on MACE implementation with numerical improvements
# Authors: Enhanced for numerical stability
###########################################################################################

import numpy as np
import torch
from e3nn.util.jit import compile_mode


@compile_mode("script")
class EnhancedBesselBasis(torch.nn.Module):
    """
    Enhanced Bessel Basis with improved numerical stability
    Fixes division by zero and adds epsilon for small distances
    """

    def __init__(self, r_max: float, num_basis: int = 8, trainable: bool = False, eps: float = 1e-8):
        super().__init__()
        
        self.eps = eps  # Small epsilon to prevent division by zero
        
        bessel_weights = (
            np.pi
            / r_max
            * torch.linspace(
                start=1.0,
                end=num_basis,
                steps=num_basis,
                dtype=torch.get_default_dtype(),
            )
        )
        
        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "prefactor",
            torch.tensor(np.sqrt(2.0 / r_max), dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        # Ensure input is positive and add epsilon to prevent division by zero
        x_safe = torch.clamp(x, min=self.eps)
        
        # Ensure x_safe has the right shape for broadcasting
        if x_safe.dim() == 2 and x_safe.size(-1) == 1:
            x_safe = x_safe.squeeze(-1)  # [n_edges]
        
        # Compute arguments for sin function with proper broadcasting
        arguments = x_safe.unsqueeze(-1) * self.bessel_weights.unsqueeze(0)  # [n_edges, num_basis]
        
        # Compute sin(kx) with numerical stability check
        numerator = torch.sin(arguments)  # [n_edges, num_basis]
        
        # Use safe division with enhanced stability
        result = self.prefactor * (numerator / x_safe.unsqueeze(-1))
        
        # Handle very small distances with Taylor expansion: sin(kx)/x ≈ k - k³x²/6 + ...
        small_distance_mask = x.squeeze(-1) < self.eps if x.dim() == 2 else x < self.eps
        if small_distance_mask.any():
            # More accurate Taylor expansion: sin(kx)/x ≈ k - k³x²/6
            k = self.bessel_weights  # [num_basis]
            x_expanded = x_safe.unsqueeze(-1).expand_as(result)  # [n_edges, num_basis]
            taylor_order1 = self.prefactor * k.unsqueeze(0)  # Leading term
            taylor_order2 = -self.prefactor * (k.unsqueeze(0)**3 * x_expanded**2) / 6  # Correction term
            taylor_approx = taylor_order1 + taylor_order2
            
            result = torch.where(
                small_distance_mask.unsqueeze(-1).expand_as(result),
                taylor_approx,
                result
            )
        
        # Additional numerical stability: clamp result to reasonable range
        result = torch.clamp(result, min=-100.0, max=100.0)
        
        # Check for NaN and replace with zeros if found
        result = torch.where(torch.isnan(result), torch.zeros_like(result), result)
        
        return result

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={len(self.bessel_weights)}, "
            f"trainable={self.bessel_weights.requires_grad}, eps={self.eps})"
        )


@compile_mode("script") 
class HybridBasis(torch.nn.Module):
    """
    Hybrid basis combining Bessel and Gaussian for enhanced stability
    """
    
    def __init__(self, r_max: float, num_bessel: int = 16, num_gaussian: int = 16, 
                 trainable: bool = False, eps: float = 1e-8):
        super().__init__()
        
        self.num_bessel = num_bessel
        self.num_gaussian = num_gaussian
        self.total_basis = num_bessel + num_gaussian
        
        # Enhanced Bessel component
        self.bessel = EnhancedBesselBasis(r_max, num_bessel, trainable, eps)
        
        # Gaussian component for stability
        gaussian_centers = torch.linspace(0, r_max, num_gaussian)
        gaussian_widths = torch.ones(num_gaussian) * (r_max / num_gaussian) ** 2
        
        if trainable:
            self.gaussian_centers = torch.nn.Parameter(gaussian_centers)
            self.gaussian_widths = torch.nn.Parameter(gaussian_widths)
        else:
            self.register_buffer("gaussian_centers", gaussian_centers)
            self.register_buffer("gaussian_widths", gaussian_widths)
            
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.get_default_dtype()))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bessel component
        bessel_features = self.bessel(x)  # [..., num_bessel]
        
        # Gaussian component  
        diff = x - self.gaussian_centers  # [..., num_gaussian]
        gaussian_features = torch.exp(-diff ** 2 / (2 * self.gaussian_widths))
        
        # Concatenate both components
        return torch.cat([bessel_features, gaussian_features], dim=-1)
    
    @property
    def out_dim(self):
        return self.total_basis
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, "
            f"num_bessel={self.num_bessel}, num_gaussian={self.num_gaussian})"
        )


@compile_mode("script")
class AdaptiveBesselBasis(torch.nn.Module):
    """
    Adaptive Bessel basis with learnable frequency scaling
    """
    
    def __init__(self, r_max: float, num_basis: int = 8, eps: float = 1e-8):
        super().__init__()
        
        self.eps = eps
        self.num_basis = num_basis
        
        # Learnable frequency scaling
        base_frequencies = torch.arange(1.0, num_basis + 1.0) * np.pi / r_max
        self.frequency_scale = torch.nn.Parameter(torch.ones(num_basis))
        self.register_buffer("base_frequencies", base_frequencies)
        
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "prefactor", torch.tensor(np.sqrt(2.0 / r_max), dtype=torch.get_default_dtype())
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_safe = torch.clamp(x, min=self.eps)
        
        # Adaptive frequencies
        frequencies = self.base_frequencies * self.frequency_scale.abs()
        
        numerator = torch.sin(frequencies * x_safe)
        result = self.prefactor * (numerator / x_safe)
        
        # Taylor expansion for small distances
        small_mask = x < self.eps
        if small_mask.any():
            taylor_approx = self.prefactor * frequencies
            result = torch.where(
                small_mask.expand_as(result),
                taylor_approx.expand_as(result), 
                result
            )
            
        return result
    
    def __repr__(self):
        return f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={self.num_basis})"
