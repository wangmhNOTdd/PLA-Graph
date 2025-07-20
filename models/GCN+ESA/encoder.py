#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_batch

from .esa.masked_layers import ESA


class GCNESAEncoder(nn.Module):
    def __init__(self, hidden_size, edge_size=64, n_layers=3, gcn_layers=2, 
                 num_heads=8, sab_dropout=0.1, graph_dim=None) -> None:
        super().__init__()
        
        self.hidden_size = hidden_size
        self.gcn_layers = gcn_layers
        self.n_layers = n_layers
        self.graph_dim = graph_dim if graph_dim is not None else hidden_size
        
        # GCN layers for atom-level to block-level pooling
        self.gcn_convs = nn.ModuleList()
        for i in range(gcn_layers):
            if i == 0:
                self.gcn_convs.append(GCNConv(hidden_size, hidden_size))
            else:
                self.gcn_convs.append(GCNConv(hidden_size, hidden_size))
        
        # Projection layer after GCN
        self.gcn_projection = nn.Linear(hidden_size, hidden_size)
        
        # ESA (Edge Set Attention) for block-level interactions
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(sab_dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.esa = ESA(
            dim_hidden=[hidden_size] * n_layers,
            num_heads=[num_heads] * n_layers,
            dim_output=self.graph_dim,
            num_outputs=32,  # k for PMA
            layer_types=["SAB"] * (n_layers - 1) + ["PMA"],  # SAB layers + final PMA
            sab_dropout=sab_dropout,
            mab_dropout=sab_dropout,
            pma_dropout=sab_dropout,
            node_or_edge="edge",
            xformers_or_torch_attn="torch",
            use_bfloat16=False,
            set_max_items=512,  # max number of edges
            use_mlps=True,
            mlp_hidden_size=64,
            mlp_type="standard",
            norm_type="LN",
            residual_dropout=0.1,
            num_mlp_layers=2,
            pre_or_post="pre",
            pma_residual_dropout=0.1,
            use_mlp_ln=False,
            mlp_dropout=0.1
        )
        
        # Normalization
        self.norm = nn.LayerNorm(self.graph_dim)
        
    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None):
        """
        Args:
            H: [N_atoms, hidden_size] - atom features
            Z: [N_atoms, n_channel, 3] - atom coordinates  
            block_id: [N_atoms] - which block each atom belongs to
            batch_id: [N_blocks] - which graph each block belongs to
            edges: (edge_index_0, edge_index_1) - block-level edges
        """
        device = H.device
        n_atoms = H.shape[0]
        n_blocks = batch_id.shape[0]
        
        # Step 1: Build atom-level graph for GCN
        # Create atom-level edges within blocks
        atom_edges = self._create_atom_level_edges(block_id, device)
        
        # Step 2: Apply GCN on atom-level graph
        atom_features = H
        for i, gcn_conv in enumerate(self.gcn_convs):
            atom_features = gcn_conv(atom_features, atom_edges)
            if i < len(self.gcn_convs) - 1:
                atom_features = F.relu(atom_features)
                atom_features = F.dropout(atom_features, training=self.training)
        
        # Step 3: Pool atoms to blocks (residues)
        block_features = scatter_mean(atom_features, block_id, dim=0)  # [N_blocks, hidden_size]
        block_features = self.gcn_projection(block_features)
        block_features = F.relu(block_features)
        
        # Step 4: Apply ESA on block-level graph
        # Prepare edge features for ESA
        edge_index = torch.stack(edges, dim=0)  # [2, n_edges]
        
        # Create edge features by concatenating source and target block features
        if edge_index.shape[1] > 0:
            src_features = block_features[edge_index[0]]  # [n_edges, hidden_size]
            dst_features = block_features[edge_index[1]]  # [n_edges, hidden_size]
            edge_features = torch.cat([src_features, dst_features], dim=-1)  # [n_edges, 2*hidden_size]
            edge_features = self.edge_mlp(edge_features)  # [n_edges, hidden_size]
        else:
            edge_features = torch.zeros(0, self.hidden_size, device=device)
        
        # Convert to dense batch format for ESA
        batch_size = batch_id.max().item() + 1
        edge_batch_mapping = batch_id[edge_index[0]] if edge_index.shape[1] > 0 else torch.zeros(0, dtype=torch.long, device=device)
        max_edges_per_batch = max(1, edge_features.shape[0] // max(1, batch_size))
        
        if edge_features.shape[0] > 0:
            # Convert to dense batch for ESA
            dense_edge_features, _ = to_dense_batch(edge_features, edge_batch_mapping, 
                                                  fill_value=0, max_num_nodes=max_edges_per_batch)
            
            # Apply ESA
            graph_repr = self.esa(dense_edge_features, edge_index, edge_batch_mapping, max_edges_per_batch)
        else:
            # Handle case with no edges
            graph_repr = torch.zeros(batch_size, self.graph_dim, device=device)
        
        # Normalize output
        graph_repr = self.norm(graph_repr)
        
        # For compatibility with existing interface
        block_repr = scatter_sum(block_features, batch_id, dim=0)  # [batch_size, hidden_size] 
        block_repr = F.normalize(block_repr, dim=-1)
        graph_repr = F.normalize(graph_repr, dim=-1)
        
        return H, block_repr, graph_repr, None
    
    def _create_atom_level_edges(self, block_id, device):
        """Create fully connected edges within each block (residue)"""
        edges = []
        unique_blocks = torch.unique(block_id)
        
        for block in unique_blocks:
            atom_indices = torch.where(block_id == block)[0]
            if len(atom_indices) > 1:
                # Create all pairs within this block
                src = atom_indices.unsqueeze(1).repeat(1, len(atom_indices)).flatten()
                dst = atom_indices.unsqueeze(0).repeat(len(atom_indices), 1).flatten()
                
                # Remove self-loops
                mask = src != dst
                src = src[mask]
                dst = dst[mask]
                
                edges.append(torch.stack([src, dst], dim=0))
        
        if edges:
            return torch.cat(edges, dim=1)
        else:
            # Return empty edge index if no edges
            return torch.zeros(2, 0, dtype=torch.long, device=device)
