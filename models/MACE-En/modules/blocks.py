#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Enhanced blocks for MACE-En model
"""

import torch
from e3nn.util.jit import compile_mode

from .enhanced_radial import EnhancedBesselBasis, HybridBasis, AdaptiveBesselBasis
from ...MACE.modules.blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    ScaleShiftBlock,
    PolynomialCutoff,
)

# Re-export MACE blocks
from ...MACE.modules.blocks import (
    RealAgnosticResidualInteractionBlock,
    RealAgnosticInteractionBlock,
)

__all__ = [
    'RadialEmbeddingBlockEn',
    'AtomicEnergiesBlock',
    'EquivariantProductBasisBlock', 
    'InteractionBlock',
    'LinearNodeEmbeddingBlock',
    'ScaleShiftBlock',
    'RealAgnosticResidualInteractionBlock',
    'RealAgnosticInteractionBlock',
]


@compile_mode("script")
class RadialEmbeddingBlockEn(torch.nn.Module):
    """
    Enhanced Radial Embedding Block with improved numerical stability
    """
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        radial_basis: str = "enhanced",
    ):
        super().__init__()
        
        # Choose radial basis function
        if radial_basis == "enhanced":
            self.bessel_fn = EnhancedBesselBasis(r_max=r_max, num_basis=num_bessel)
        elif radial_basis == "hybrid":
            self.bessel_fn = HybridBasis(r_max=r_max, num_bessel=num_bessel//2, num_gaussian=num_bessel//2)
        elif radial_basis == "adaptive":
            self.bessel_fn = AdaptiveBesselBasis(r_max=r_max, num_basis=num_bessel)
        else:
            raise ValueError(f"Unknown radial_basis: {radial_basis}")
            
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        
        # Set output dimension
        if hasattr(self.bessel_fn, 'out_dim'):
            self.out_dim = self.bessel_fn.out_dim
        else:
            self.out_dim = num_bessel

    def forward(
        self,
        edge_lengths: torch.Tensor,  # [n_edges, 1]
    ):
        radial = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]
        return radial * cutoff  # [n_edges, n_basis]
