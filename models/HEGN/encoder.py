#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Hyperbolic Equivariant Graph Network (HEGN)
Combines EGNN's E(3) equivariance with hyperbolic geometry for hierarchical learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum, scatter_max, scatter_softmax
import math

from ..EGNN.egnn import EGNN


class HyperbolicLinear(nn.Module):
    """Hyperbolic linear transformation in the tangent space"""
    def __init__(self, in_features, out_features, curvature=-1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.curvature = nn.Parameter(torch.tensor(curvature))
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        # Transform in tangent space at origin
        x_tan = self.log_map(x)
        y_tan = F.linear(x_tan, self.weight, self.bias)
        return self.exp_map(y_tan)
    
    def exp_map(self, x):
        """Exponential map from origin tangent space to hyperbolic space"""
        k = torch.clamp(self.curvature, max=-1e-5)  # Ensure negative curvature
        sqrt_k = torch.sqrt(-k)
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        
        # Avoid division by zero
        x_norm = torch.clamp(x_norm, min=1e-5)
        
        return torch.tanh(sqrt_k * x_norm) * x / (sqrt_k * x_norm)
    
    def log_map(self, x):
        """Logarithmic map from hyperbolic space to origin tangent space"""
        k = torch.clamp(self.curvature, max=-1e-5)
        sqrt_k = torch.sqrt(-k)
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        
        # Avoid division by zero
        x_norm = torch.clamp(x_norm, min=1e-5)
        
        return torch.atanh(torch.clamp(x_norm, max=1-1e-5)) * x / (sqrt_k * x_norm)


class HyperbolicAttention(nn.Module):
    """Hyperbolic attention mechanism"""
    def __init__(self, hidden_size, curvature=-1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.curvature = nn.Parameter(torch.tensor(curvature))
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, h_i, h_neighbors, edge_index):
        # Convert to tangent space for attention computation
        h_i_tan = self.log_map(h_i)
        h_neighbors_tan = self.log_map(h_neighbors)
        
        # Compute attention weights
        row, col = edge_index
        h_i_expanded = h_i_tan[row]
        attention_input = torch.cat([h_i_expanded, h_neighbors_tan], dim=-1)
        attention_weights = self.attention_mlp(attention_input)
        
        # Apply softmax within each node's neighborhood
        try:
            attention_weights = scatter_softmax(attention_weights.squeeze(-1), row, dim=0)
        except:
            # Fallback implementation if scatter_softmax is not available
            attention_weights = self._manual_scatter_softmax(attention_weights.squeeze(-1), row)
        
        return attention_weights
    
    def _manual_scatter_softmax(self, src, index):
        """Manual implementation of scatter softmax"""
        # Get unique indices and their counts
        unique_indices = torch.unique(index)
        out = torch.zeros_like(src)
        
        for idx in unique_indices:
            mask = (index == idx)
            if mask.sum() > 0:
                subset = src[mask]
                subset_softmax = F.softmax(subset, dim=0)
                out[mask] = subset_softmax
                
        return out
    
    def log_map(self, x):
        """Logarithmic map from hyperbolic space to origin tangent space"""
        k = torch.clamp(self.curvature, max=-1e-5)
        sqrt_k = torch.sqrt(-k)
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=1e-5)
        
        return torch.atanh(torch.clamp(x_norm, max=1-1e-5)) * x / (sqrt_k * x_norm)


class HyperbolicGCNLayer(nn.Module):
    """Single Hyperbolic Graph Convolutional Layer with attention"""
    def __init__(self, hidden_size, curvature=-1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.curvature = nn.Parameter(torch.tensor(curvature))
        
        # Components
        self.hyperbolic_linear = HyperbolicLinear(hidden_size, hidden_size, curvature)
        self.attention = HyperbolicAttention(hidden_size, curvature)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Residual connection projection
        self.residual_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, h, edge_index):
        # Store input for residual connection
        h_residual = h
        
        # Apply hyperbolic transformation
        h_transformed = self.hyperbolic_linear(h)
        
        # Apply attention-based message passing
        row, col = edge_index
        h_neighbors = h_transformed[col]
        
        # Compute attention weights
        attention_weights = self.attention(h_transformed, h_neighbors, edge_index)
        
        # Aggregate messages with attention
        h_neighbors_weighted = h_neighbors * attention_weights.unsqueeze(-1)
        h_aggregated = scatter_mean(h_neighbors_weighted, row, dim=0, dim_size=h.size(0))
        
        # Convert back to tangent space for residual connection and layer norm
        h_agg_tan = self.attention.log_map(h_aggregated)
        h_res_tan = self.residual_proj(self.attention.log_map(h_residual))
        
        # Residual connection and layer normalization in tangent space
        h_out_tan = self.layer_norm(h_agg_tan + h_res_tan)
        
        # Map back to hyperbolic space
        return self.exp_map(h_out_tan)
    
    def exp_map(self, x):
        """Exponential map from origin tangent space to hyperbolic space"""
        k = torch.clamp(self.curvature, max=-1e-5)
        sqrt_k = torch.sqrt(-k)
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=1e-5)
        
        return torch.tanh(sqrt_k * x_norm) * x / (sqrt_k * x_norm)


class HEGNEncoder(nn.Module):
    """Hyperbolic Equivariant Graph Network Encoder"""
    def __init__(self, hidden_size, edge_size, n_layers=1, n_egnn_layers=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers  # Number of HGCN layers
        self.n_egnn_layers = n_egnn_layers  # Number of EGNN layers
        
        # Stage 1: EGNN layers for local geometric encoding
        self.egnn = EGNN(
            in_node_nf=hidden_size,
            hidden_nf=hidden_size,
            out_node_nf=hidden_size,
            in_edge_nf=edge_size,
            n_layers=n_egnn_layers
        )
        
        # Stage 2: Euclidean to Hyperbolic mapping
        self.euclidean_to_hyperbolic = HyperbolicLinear(hidden_size, hidden_size)
        
        # Stage 3: Hyperbolic GCN layers
        self.hyperbolic_layers = nn.ModuleList([
            HyperbolicGCNLayer(hidden_size, curvature=-1.0 * (i + 1))
            for i in range(n_layers)
        ])
        
        # Stage 4: Final projection back to Euclidean space with attention pooling
        self.final_projection = nn.Linear(hidden_size, hidden_size)
        
        # Attention pooling components
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.global_token = nn.Parameter(torch.randn(1, hidden_size))
        
        # Final MLP prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None):
        # Stage 1: Local geometric encoding with EGNN
        # Convert block-level to atom-level if needed
        if block_id is not None:
            H_atom, Z_atom = scatter_mean(H, block_id, dim=0), scatter_mean(Z, block_id, dim=0)
        else:
            H_atom, Z_atom = H, Z
        
        Z_atom = Z_atom.squeeze() if Z_atom.dim() > 2 else Z_atom
        
        # Apply EGNN for geometric feature learning
        H_egnn, Z_updated = self.egnn(H_atom, Z_atom, edges, edge_attr)
        
        # Stage 2: Map to hyperbolic space
        H_hyperbolic = self.euclidean_to_hyperbolic(H_egnn)
        
        # Stage 3: Hierarchical learning with hyperbolic GCN
        H_current = H_hyperbolic
        for layer in self.hyperbolic_layers:
            H_current = layer(H_current, edges)
        
        # Stage 4: Map back to Euclidean space for final processing
        H_final_tan = self.log_map_final(H_current)
        H_final = self.final_projection(H_final_tan)
        
        # Normalize features
        H_final = F.normalize(H_final, dim=-1)
        
        # Stage 5: Attention-based global pooling
        if batch_id is not None:
            # Create global tokens for each graph in the batch
            batch_size = batch_id.max().item() + 1
            global_tokens = self.global_token.expand(batch_size, -1).unsqueeze(1)  # [batch_size, 1, hidden_size]
            
            # Group node features by batch
            graph_representations = []
            for i in range(batch_size):
                mask = (batch_id == i)
                if mask.sum() > 0:
                    node_features = H_final[mask].unsqueeze(0)  # [1, num_nodes, hidden_size]
                    global_token = global_tokens[i:i+1]  # [1, 1, hidden_size]
                    
                    # Apply attention pooling
                    pooled_repr, attention_weights = self.attention_pooling(
                        query=global_token,
                        key=node_features,
                        value=node_features
                    )
                    graph_representations.append(pooled_repr.squeeze(1))  # [1, hidden_size]
            
            graph_repr = torch.cat(graph_representations, dim=0)  # [batch_size, hidden_size]
        else:
            # Single graph case
            global_token = self.global_token.unsqueeze(0)  # [1, 1, hidden_size]
            node_features = H_final.unsqueeze(0)  # [1, num_nodes, hidden_size]
            
            pooled_repr, attention_weights = self.attention_pooling(
                query=global_token,
                key=node_features,
                value=node_features
            )
            graph_repr = pooled_repr.squeeze(0)  # [1, hidden_size]
        
        # Final normalization
        graph_repr = F.normalize(graph_repr, dim=-1)
        
        return H_final, H_final, graph_repr, None
    
    def log_map_final(self, x):
        """Final logarithmic map for output"""
        # Use the curvature from the last hyperbolic layer
        last_curvature = self.hyperbolic_layers[-1].curvature
        k = torch.clamp(last_curvature, max=-1e-5)
        sqrt_k = torch.sqrt(-k)
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=1e-5)
        
        return torch.atanh(torch.clamp(x_norm, max=1-1e-5)) * x / (sqrt_k * x_norm)
