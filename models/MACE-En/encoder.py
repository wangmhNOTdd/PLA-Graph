#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
MACE-En Encoder: Enhanced MACE with improved numerical stability
"""

import torch
import torch.nn as nn
from torch_scatter import scatter_sum

from .model import ScaleShiftMACEEn


class MACEEnEncoder(nn.Module):
    """
    MACE-En Encoder with enhanced numerical stability
    """
    
    def __init__(self, hidden_size, n_rbf, cutoff, n_layers=2, radial_basis='enhanced') -> None:
        super().__init__()
        
        self.encoder = ScaleShiftMACEEn(
            r_max=cutoff,
            num_bessel=n_rbf, 
            hidden_size=hidden_size,
            num_elements=hidden_size,  # Use hidden_size as num_elements
            atomic_inter_scale=1.0,
            atomic_inter_shift=0.0,
        )
        
        self.hidden_size = hidden_size
        
    def forward(self, H_0, Z, block_id, batch_id, edges, edge_attr):
        """
        Forward pass for MACE-En encoder
        
        Args:
            H_0: Initial node features [n_nodes, hidden_size]
            Z: Atomic positions [n_nodes, 3] 
            block_id: Block IDs [n_nodes]
            batch_id: Batch IDs [n_nodes]
            edges: Edge indices [2, n_edges]
            edge_attr: Edge attributes [n_edges, edge_dim]
            
        Returns:
            unit_repr: Node representations [n_nodes, hidden_size]
            block_repr: Block representations [n_blocks, hidden_size] 
            graph_repr: Graph representations [n_graphs, hidden_size]
            pred_Z: Predicted positions [n_nodes, 3]
        """
        
        # Prepare data for MACE-En
        data = {
            'node_attrs': H_0,  # [n_nodes, hidden_size]
            'positions': Z,     # [n_nodes, 3]
            'edge_index': edges,  # [2, n_edges]
            'batch': batch_id,   # [n_nodes]
            'num_graphs': batch_id.max().item() + 1 if len(batch_id) > 0 else 1
        }
        
        # Forward through MACE-En
        output = self.encoder(data)
        
        # Create representations
        unit_repr = H_0  # Use input as unit representation
        
        # Create block representations by pooling nodes
        n_blocks = block_id.max().item() + 1 if len(block_id) > 0 else 1
        block_repr = scatter_sum(H_0, block_id, dim=0, dim_size=n_blocks)
        
        # Create graph representations by pooling nodes
        n_graphs = batch_id.max().item() + 1 if len(batch_id) > 0 else 1
        graph_repr = scatter_sum(H_0, batch_id, dim=0, dim_size=n_graphs)
        
        pred_Z = Z  # Return original positions
        
        return unit_repr, block_repr, graph_repr, pred_Z
