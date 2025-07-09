#!/usr/bin/python
# -*- coding:utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_add_pool, GINEConv, GINConv, GCNConv
from torch_scatter import scatter_mean, scatter_sum

from .appnp import APPNP
from .layer import DGNN, NodeWithEdgeUpdate


class EIGNEncoder(nn.Module):
    def __init__(self, hidden_size, n_channel, radial_size=16,
                 edge_size=16, k_neighbors=9, n_layers=3, dropout=0.1) -> None:
        super().__init__()

        # EIGN specific parameters
        self.hidden_size = hidden_size
        self.node_dim = hidden_size
        self.gin_dim = hidden_size
        self.dropout = dropout
        
        # Initialize EIGN components
        self.lin_node = nn.Sequential(Linear(hidden_size, hidden_size), nn.SiLU())
        self.encoder_inter = Encoder(hidden_size, hidden_size)

        self.mlp_encode = nn.Sequential(
            nn.Linear(hidden_size, self.gin_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.gin_dim))

        self.edge_inter_attr = EdgeAttr(self.gin_dim)
        self.edge_inter_update = NodeWithEdgeUpdate(self.gin_dim, self.gin_dim)
        self.edge_intra_attr = EdgeAttr(self.gin_dim)
        self.edge_intra_update = NodeWithEdgeUpdate(self.gin_dim, self.gin_dim)

        self.gin1 = GINEConv(nn.Sequential(
            nn.Linear(self.gin_dim, self.gin_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.gin_dim)))
        self.dgnn1 = DGNN([hidden_size, hidden_size, hidden_size, hidden_size])
        self.lin1 = nn.Sequential(Linear(hidden_size * 2, hidden_size), nn.SiLU())

        self.gin3 = GINEConv(nn.Sequential(
            nn.Linear(self.gin_dim, self.gin_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.gin_dim)))
        self.dgnn3 = DGNN([hidden_size, hidden_size, hidden_size, hidden_size])
        self.lin3 = nn.Sequential(Linear(hidden_size * 2, hidden_size), nn.SiLU())

        self.gin4 = GINConv(nn.Sequential(
            nn.Linear(self.gin_dim, self.gin_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.gin_dim)))

        hidden_list = [hidden_size * 2, hidden_size * 2, hidden_size]
        self.fc = FC(self.gin_dim, hidden_list, dropout)

    def forward(self, H_0, Z, block_id, batch_id, edges, edge_attr=None):
        """
        Forward pass following GET encoder interface
        Args:
            H: Node features [N, hidden_size] 
            Z: Node coordinates [N, n_channel, 3]
            block_id: Block assignments [N]
            batch_id: Batch assignments for blocks [Nb]
            edges: Edge indices [2, E] 
            edge_attr: Edge attributes [E, edge_size]
        Returns:
            H: Updated node features [N, hidden_size]
            block_repr: Block representations [Nb, hidden_size] 
            graph_repr: Graph representations [bs, hidden_size]
            pred_Z: Predicted coordinates [N, n_channel, 3]
        """
        device = H_0.device
        pos = Z.squeeze(-2)  # Remove channel dimension [N, 3]
        
        # Create batch assignments for atoms
        atom_batch_id = batch_id[block_id]
        
        # Extract different edge types from the concatenated edges
        # Assuming edges contains [intra_edges, inter_edges, ...]
        if edges.size(1) > 0:
            # For simplicity, we'll treat all edges as inter-molecular edges
            # In practice, you might want to separate intra/inter edges
            edge_index_inter = edges
            edge_index_intra = edges  # Using same edges for intra for now
            edge_index_aug = edges
        else:
            # Create minimal edge structure if no edges provided
            edge_index_inter = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_index_intra = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_index_aug = torch.empty((2, 0), dtype=torch.long, device=device)
        
        # Apply EIGN forward pass with H_0 as initial features
        x_raw = H_0
        
        if edge_index_inter.size(1) > 0:
            edge_weight = torch.norm((pos[edge_index_inter[0]] - pos[edge_index_inter[1]]), p=2, dim=1)
            x_inter, xg = self.encoder_inter(x_raw, edge_index_inter, atom_batch_id, edge_weight)
        else:
            x_inter = x_raw
            # Create global representation by pooling
            xg = scatter_sum(x_raw, atom_batch_id, dim=0)
            
        x_raw_processed = self.lin_node(x_raw)
        x = self.mlp_encode(x_inter + x_raw_processed)

        if edge_index_inter.size(1) > 0:
            edge_attr_inter = self.edge_inter_attr(pos, edge_index_inter)
            edge_attr_inter = self.edge_inter_update(x, edge_index_inter, edge_attr_inter)
            
            x_inter1 = self.gin1(x, edge_index_inter, edge_attr_inter)
            x_inter2 = self.dgnn1(x, edge_index_inter)
            x_inter = torch.concat([x_inter1, x_inter2], dim=-1)
            x_inter = self.lin1(x_inter)
        else:
            x_inter = torch.zeros_like(x)
        
        if edge_index_intra.size(1) > 0:
            edge_attr_intra = self.edge_intra_attr(pos, edge_index_intra)
            edge_attr_intra = self.edge_intra_update(x, edge_index_intra, edge_attr_intra)
            
            x_intra1 = self.gin3(x, edge_index_intra, edge_attr_intra)
            x_intra2 = self.dgnn3(x, edge_index_intra)
            x_intra = torch.concat([x_intra1, x_intra2], dim=-1)
            x_intra = self.lin3(x_intra)
        else:
            x_intra = torch.zeros_like(x)

        if edge_index_aug.size(1) > 0:
            x_mask = self.gin4(x, edge_index_aug)
        else:
            x_mask = torch.zeros_like(x)

        x_c = x_inter + x_intra + x_mask
        H_updated = self.fc(x_c)

        # Create block and graph representations following GET interface
        block_repr = scatter_sum(H_updated, block_id, dim=0)           # [Nb, hidden]
        block_repr = F.normalize(block_repr, dim=-1)
        graph_repr = scatter_sum(block_repr, batch_id, dim=0)  # [bs, hidden]
        graph_repr = F.normalize(graph_repr, dim=-1)
        
        # For pred_Z, we'll return the original coordinates (no coordinate prediction in EIGN)
        pred_Z = Z
        
        return H_updated, block_repr, graph_repr, pred_Z


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.SiLU()
        )
        self.propagate_inter = APPNP(K=1, alpha=0.1)
        self.pool = global_add_pool
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = self.linear(x)
        x_psc = F.normalize(x, p=2, dim=1) * 1.8
        if edge_index.size(1) > 0:
            x = self.propagate_inter(x_psc, edge_index, edge_weight=edge_weight)
        else:
            x = x_psc
        xg = self.pool(x, batch)
        x = self.dropout(x)
        return x, xg


class EdgeAttr(nn.Module):
    def __init__(self, hidden_dim):
        super(EdgeAttr, self).__init__()
        self.mlp_diff_pos = nn.Sequential(nn.Linear(16, hidden_dim), nn.Sigmoid())

    def forward(self, pos, edge_index):
        if edge_index.size(1) == 0:
            return torch.empty((0, self.mlp_diff_pos[0].out_features), device=pos.device)
        coord_diff = pos[edge_index[0]] - pos[edge_index[1]]
        diff_feat = self.mlp_diff_pos(
            _rbf(torch.norm(coord_diff, p=2, dim=1), D_min=0., D_max=6., D_count=16, device=pos.device))
        return diff_feat


class FC(nn.Module):
    def __init__(self, in_dim, hidden_list, dropout):
        super(FC, self).__init__()
        self.predict = nn.ModuleList()
        for hidden_dim in hidden_list:
            self.predict.append(nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(hidden_dim)))
            in_dim = hidden_dim
        self.predict.append(nn.Linear(in_dim, in_dim))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)
        return h


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count).to(device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF
