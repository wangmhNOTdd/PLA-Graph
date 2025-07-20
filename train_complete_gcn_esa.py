#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
完整的GCN+ESA模型实现
正确体现 "GCN从原子级池化到Blocks级，在Blocks级的图上使用ESA学习整个复合物的互作用关系来预测亲和力" 的思路
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, radius_graph
from torch_scatter import scatter_mean
import argparse
import os
import pickle
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr

# 假设VOCAB已定义
class VOCAB:
    @staticmethod
    def get_num_atom_type():
        return 100
    
    @staticmethod
    def get_num_atom_pos():
        return 100
    
    @staticmethod
    def __len__():
        return 21

class EdgeSetAttention(nn.Module):
    """Edge Set Attention (ESA) implementation"""
    
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Multi-head attention for edges
        self.edge_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Edge update MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Node update from edges
        self.node_update = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, node_features, edge_index, edge_features):
        """
        Args:
            node_features: [N_nodes, hidden_size]
            edge_index: [2, N_edges] 
            edge_features: [N_edges, hidden_size]
        Returns:
            updated_node_features: [N_nodes, hidden_size]
            updated_edge_features: [N_edges, hidden_size]
        """
        if edge_features.shape[0] == 0:
            return node_features, edge_features
            
        # Self-attention on edges
        attended_edges, _ = self.edge_attention(
            edge_features.unsqueeze(0),
            edge_features.unsqueeze(0),
            edge_features.unsqueeze(0)
        )
        attended_edges = attended_edges.squeeze(0)
        
        # Update edges with residual connection
        edge_features = self.norm1(edge_features + attended_edges)
        edge_features = self.norm2(edge_features + self.edge_mlp(edge_features))
        
        # Aggregate edge information to nodes
        if edge_index.shape[1] > 0:
            # For each node, aggregate information from incoming edges
            node_updates = scatter_mean(edge_features, edge_index[1], dim=0, dim_size=node_features.shape[0])
            node_features = node_features + self.node_update(node_updates)
        
        return node_features, edge_features


class GCNESAModel(nn.Module):
    """
    GCN+ESA模型: GCN从原子级池化到Blocks级，在Blocks级的图上使用ESA学习复合物互作用
    """
    
    def __init__(self, hidden_size=64, gcn_layers=2, esa_layers=2, num_heads=4, 
                 dropout=0.1, cutoff=6.0, k_neighbors=16):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.gcn_layers = gcn_layers
        self.esa_layers = esa_layers
        self.cutoff = cutoff
        self.k_neighbors = k_neighbors
        
        # Embeddings
        self.atom_embedding = nn.Embedding(VOCAB.get_num_atom_type(), hidden_size)
        self.position_embedding = nn.Embedding(VOCAB.get_num_atom_pos(), hidden_size)
        self.block_embedding = nn.Embedding(len(VOCAB), hidden_size)
        
        # Stage 1: GCN for atom-level processing
        self.gcn_convs = nn.ModuleList()
        for i in range(gcn_layers):
            self.gcn_convs.append(GCNConv(hidden_size, hidden_size))
        
        # Atom to block pooling projection
        self.atom_to_block_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Stage 2: ESA for block-level interactions
        self.esa_layers_list = nn.ModuleList()
        for _ in range(esa_layers):
            self.esa_layers_list.append(EdgeSetAttention(hidden_size, num_heads, dropout))
        
        # Edge feature initialization
        self.edge_init = nn.Sequential(
            nn.Linear(hidden_size * 2 + 1, hidden_size),  # node_i + node_j + distance
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Graph-level prediction
        self.graph_pool = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label=None):
        """Forward pass"""
        device = Z.device
        batch_size = lengths.shape[0]
        
        # Create block_id mapping (atom -> block)
        block_id = []
        for i, length in enumerate(block_lengths):
            block_id.extend([i] * length.item())
        block_id = torch.tensor(block_id, dtype=torch.long, device=device)
        
        # Create batch_id for atoms
        atom_batch_id = []
        atom_start = 0
        for i, block_length in enumerate(lengths):
            n_atoms_in_batch = sum(block_lengths[atom_start:atom_start + block_length]).item()
            atom_batch_id.extend([i] * n_atoms_in_batch)
            atom_start += block_length
        atom_batch_id = torch.tensor(atom_batch_id, dtype=torch.long, device=device)
        
        # ===== Stage 1: Atom-level GCN processing =====
        # Initial embeddings
        atom_features = self.atom_embedding(A) + self.position_embedding(atom_positions)
        block_features = self.block_embedding(B)
        
        # Add block information to atoms
        atom_features = atom_features + block_features[block_id]
        
        # Build atom-level graph
        coords = Z.squeeze(-2) if Z.dim() == 3 else Z  # [N_atoms, 3]
        atom_edges = radius_graph(coords, r=self.cutoff, batch=atom_batch_id, 
                                 max_num_neighbors=self.k_neighbors)
        
        # Apply GCN on atom-level graph
        x = atom_features
        for gcn_conv in self.gcn_convs:
            x = gcn_conv(x, atom_edges)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        
        # Pool atoms to blocks (key step: 原子级池化到Blocks级)
        pooled_block_features = scatter_mean(x, block_id, dim=0)  # [N_blocks, hidden_size]
        pooled_block_features = self.atom_to_block_proj(pooled_block_features)
        
        # ===== Stage 2: Block-level ESA processing =====
        # Create batch_id for blocks
        block_batch_id = []
        start = 0
        for i, length in enumerate(lengths):
            block_batch_id.extend([i] * length.item())
            start += length.item()
        block_batch_id = torch.tensor(block_batch_id, dtype=torch.long, device=device)
        
        # Get block center coordinates
        block_coords = scatter_mean(coords, block_id, dim=0)  # [N_blocks, 3]
        
        # Process each molecule in the batch
        graph_features = []
        for batch_idx in range(batch_size):
            batch_mask = block_batch_id == batch_idx
            batch_blocks = pooled_block_features[batch_mask]  # [n_blocks_in_batch, hidden_size]
            batch_coords = block_coords[batch_mask]  # [n_blocks_in_batch, 3]
            
            if batch_blocks.shape[0] > 1:
                # Build block-level edges (key step: 在Blocks级的图上)
                block_edges = radius_graph(batch_coords, r=self.cutoff * 1.5, 
                                         max_num_neighbors=min(16, batch_blocks.shape[0]-1))
                
                if block_edges.shape[1] > 0:
                    # Initialize edge features with node features + distance
                    src_features = batch_blocks[block_edges[0]]
                    dst_features = batch_blocks[block_edges[1]]
                    edge_distances = torch.norm(batch_coords[block_edges[0]] - batch_coords[block_edges[1]], dim=1, keepdim=True)
                    
                    edge_features = torch.cat([src_features, dst_features, edge_distances], dim=-1)
                    edge_features = self.edge_init(edge_features)
                    
                    # Apply ESA layers (key step: ESA学习整个复合物的互作用关系)
                    current_node_features = batch_blocks
                    current_edge_features = edge_features
                    
                    for esa_layer in self.esa_layers_list:
                        current_node_features, current_edge_features = esa_layer(
                            current_node_features, block_edges, current_edge_features
                        )
                    
                    # Pool to graph representation
                    graph_repr = current_node_features.mean(dim=0)
                else:
                    # No edges, just pool nodes
                    graph_repr = batch_blocks.mean(dim=0)
            elif batch_blocks.shape[0] == 1:
                graph_repr = batch_blocks.squeeze(0)
            else:
                graph_repr = torch.zeros(self.hidden_size, device=device)
            
            graph_features.append(graph_repr)
        
        graph_repr = torch.stack(graph_features, dim=0)  # [batch_size, hidden_size]
        graph_repr = self.norm(self.graph_pool(graph_repr))
        
        # Predict affinity
        output = self.final_mlp(graph_repr).squeeze(-1)  # [batch_size]
        
        if label is not None:
            return F.mse_loss(output, label)
        else:
            return output


# 数据加载和训练代码（与之前相同）
class PDBBindDataset:
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    Z_list, B_list, A_list, atom_positions_list, block_lengths_list = [], [], [], [], []
    segment_ids_list, affinity_list, lengths = [], [], []
    
    for item in batch:
        Z_list.append(torch.tensor(item['X'], dtype=torch.float32))
        B_list.append(torch.tensor(item['B'], dtype=torch.long))
        A_list.append(torch.tensor(item['A'], dtype=torch.long))
        atom_positions_list.append(torch.tensor(item['atom_positions'], dtype=torch.long))
        block_lengths_list.append(torch.tensor(item['block_lengths'], dtype=torch.long))
        segment_ids_list.append(torch.tensor(item['segment_ids'], dtype=torch.long))
        affinity_list.append(item['affinity'])
        lengths.append(len(item['B']))
    
    Z = torch.cat(Z_list, dim=0)
    B = torch.cat(B_list, dim=0)
    A = torch.cat(A_list, dim=0)
    atom_positions = torch.cat(atom_positions_list, dim=0)
    block_lengths = torch.cat(block_lengths_list, dim=0)
    segment_ids = torch.cat(segment_ids_list, dim=0)
    affinity = torch.tensor(affinity_list, dtype=torch.float32)
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    return {
        'Z': Z, 'B': B, 'A': A, 'atom_positions': atom_positions,
        'block_lengths': block_lengths, 'segment_ids': segment_ids,
        'affinity': affinity, 'lengths': lengths
    }

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    total_batches = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        try:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            
            loss = model(
                Z=batch['Z'],
                B=batch['B'], 
                A=batch['A'],
                atom_positions=batch['atom_positions'],
                block_lengths=batch['block_lengths'],
                lengths=batch['lengths'],
                segment_ids=batch['segment_ids'],
                label=batch['affinity']
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
            
        except Exception as e:
            logging.warning(f"Error in batch: {str(e)}")
            continue
    
    return total_loss / max(total_batches, 1)

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    total_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            try:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                loss = model(
                    Z=batch['Z'],
                    B=batch['B'],
                    A=batch['A'], 
                    atom_positions=batch['atom_positions'],
                    block_lengths=batch['block_lengths'],
                    lengths=batch['lengths'],
                    segment_ids=batch['segment_ids'],
                    label=batch['affinity']
                )
                
                pred = model(
                    Z=batch['Z'],
                    B=batch['B'],
                    A=batch['A'],
                    atom_positions=batch['atom_positions'], 
                    block_lengths=batch['block_lengths'],
                    lengths=batch['lengths'],
                    segment_ids=batch['segment_ids'],
                    label=None
                )
                
                total_loss += loss.item()
                predictions.extend(pred.cpu().numpy())
                targets.extend(batch['affinity'].cpu().numpy())
                total_batches += 1
                
            except Exception as e:
                logging.warning(f"Error in evaluation batch: {str(e)}")
                continue
    
    if len(predictions) > 0:
        pearson_corr = pearsonr(targets, predictions)[0]
        spearman_corr = spearmanr(targets, predictions)[0]
        rmse = np.sqrt(np.mean((np.array(targets) - np.array(predictions)) ** 2))
        avg_loss = total_loss / max(total_batches, 1)
        
        return avg_loss, pearson_corr, spearman_corr, rmse
    else:
        return float('inf'), 0.0, 0.0, float('inf')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_set', type=str, required=True)
    parser.add_argument('--valid_set', type=str, required=True)  
    parser.add_argument('--test_set', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--gcn_layers', type=int, default=2)
    parser.add_argument('--esa_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--cutoff', type=float, default=6.0)
    parser.add_argument('--k_neighbors', type=int, default=16)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s::%(levelname)s::%(message)s')
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    if args.gpus and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpus}')
    else:
        device = torch.device('cpu')
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    train_dataset = PDBBindDataset(args.train_set)
    valid_dataset = PDBBindDataset(args.valid_set)
    test_dataset = PDBBindDataset(args.test_set)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    model = GCNESAModel(
        hidden_size=args.hidden_size,
        gcn_layers=args.gcn_layers,
        esa_layers=args.esa_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        cutoff=args.cutoff,
        k_neighbors=args.k_neighbors
    ).to(device)
    
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_valid_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(args.max_epoch):
        logging.info(f"Epoch {epoch+1}/{args.max_epoch}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        logging.info(f"Train Loss: {train_loss:.4f}")
        
        valid_loss, valid_pearson, valid_spearman, valid_rmse = evaluate(model, valid_loader, device)
        logging.info(f"Valid Loss: {valid_loss:.4f}, Pearson: {valid_pearson:.4f}, Spearman: {valid_spearman:.4f}, RMSE: {valid_rmse:.4f}")
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            logging.info("Saved best model")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logging.info("Early stopping triggered")
                break
    
    # Test evaluation
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth')))
    test_loss, test_pearson, test_spearman, test_rmse = evaluate(model, test_loader, device)
    logging.info(f"Test Results - Loss: {test_loss:.4f}, Pearson: {test_pearson:.4f}, Spearman: {test_spearman:.4f}, RMSE: {test_rmse:.4f}")

if __name__ == "__main__":
    main()
