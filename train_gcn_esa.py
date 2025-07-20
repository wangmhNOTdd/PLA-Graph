#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import json
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.nn import GCNConv, radius_graph

from utils.logger import print_log
from utils.random_seed import setup_seed, SEED

from data.dataset import PDBBindBenchmark
import trainers
from utils.nn_utils import count_parameters
from data.pdb_utils import VOCAB

# Use simplified GCN+ESA implementation for now
USE_FULL_ESA = False
print_log("Using simplified GCN+ESA implementation")


class SimpleGCNESAEncoder(nn.Module):
    """Simplified GCN+ESA Encoder without complex dependencies"""
    
    def __init__(self, hidden_size, gcn_layers=2, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.gcn_layers = gcn_layers
        
        # GCN layers for atom-level processing
        self.gcn_convs = nn.ModuleList()
        for i in range(gcn_layers):
            self.gcn_convs.append(GCNConv(hidden_size, hidden_size))
        
        # Projection layer after GCN
        self.gcn_projection = nn.Linear(hidden_size, hidden_size)
        
        # Simple attention mechanism for block-level interactions
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Multi-head attention for edge features
        self.edge_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Final projection
        self.final_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_size)
        
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
        
        # Step 4: Apply simple attention on block-level edges
        edge_index = torch.stack(edges, dim=0)  # [2, n_edges]
        
        if edge_index.shape[1] > 0:
            # Create edge features
            src_features = block_features[edge_index[0]]  # [n_edges, hidden_size]
            dst_features = block_features[edge_index[1]]  # [n_edges, hidden_size]
            edge_features = torch.cat([src_features, dst_features], dim=-1)  # [n_edges, 2*hidden_size]
            edge_features = self.edge_mlp(edge_features)  # [n_edges, hidden_size]
            
            # Group edges by batch for attention
            batch_size = batch_id.max().item() + 1
            edge_batch_mapping = batch_id[edge_index[0]]
            
            # Simple pooling of edge features per batch
            graph_features = []
            for batch_idx in range(batch_size):
                batch_mask = edge_batch_mapping == batch_idx
                if batch_mask.sum() > 0:
                    batch_edges = edge_features[batch_mask]  # [n_batch_edges, hidden_size]
                    
                    # Apply self-attention to edge features
                    attended_edges, _ = self.edge_attention(
                        batch_edges.unsqueeze(0),  # [1, n_batch_edges, hidden_size]
                        batch_edges.unsqueeze(0),
                        batch_edges.unsqueeze(0)
                    )
                    attended_edges = attended_edges.squeeze(0)  # [n_batch_edges, hidden_size]
                    
                    # Pool to graph-level representation
                    graph_repr = attended_edges.mean(dim=0)  # [hidden_size]
                else:
                    # Handle case with no edges
                    graph_repr = torch.zeros(self.hidden_size, device=device)
                
                graph_features.append(graph_repr)
            
            graph_repr = torch.stack(graph_features, dim=0)  # [batch_size, hidden_size]
        else:
            # Handle case with no edges
            batch_size = batch_id.max().item() + 1
            graph_repr = torch.zeros(batch_size, self.hidden_size, device=device)
        
        # Final processing
        graph_repr = self.final_projection(graph_repr)
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


class GCNESAAffinityPredictor(torch.nn.Module):
    """GCN+ESA Affinity Predictor"""
    
    def __init__(self, hidden_size=128, n_channel=1, gcn_layers=2, esa_layers=3, 
                 num_heads=8, dropout=0.1, edge_size=64):
        super().__init__()
        
        # Block embedding (same as other models)
        from models.GET.modules.tools import BlockEmbedding
        self.block_embedding = BlockEmbedding(
            num_block_type=len(VOCAB),
            num_atom_type=VOCAB.get_num_atom_type(),
            num_atom_position=VOCAB.get_num_atom_pos(),
            embed_size=hidden_size,
            no_block_embedding=False
        )
        
        # Edge constructor for block-level edges
        from models.GET.modules.tools import KNNBatchEdgeConstructor
        self.edge_constructor = KNNBatchEdgeConstructor(
            k_neighbors=9,
            global_message_passing=True,
            global_node_id_vocab=[VOCAB.symbol_to_idx(VOCAB.GLB)],
            delete_self_loop=False
        )
        
        # Edge embedding
        self.edge_embedding = torch.nn.Embedding(4, edge_size)
        
        # GCN+ESA encoder
        self.encoder = SimpleGCNESAEncoder(
            hidden_size=hidden_size,
            gcn_layers=gcn_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Energy prediction head
        self.energy_ffn = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, 1, bias=False)
        )
        
    def forward(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label=None):
        """Forward pass"""
        device = Z.device
        batch_size = lengths.shape[0]
        
        # Create block_id mapping
        block_id = []
        for i, length in enumerate(block_lengths):
            block_id.extend([i] * length.item())
        block_id = torch.tensor(block_id, dtype=torch.long, device=device)
        
        # Create batch_id for blocks
        batch_id = []
        start = 0
        for i, length in enumerate(lengths):
            batch_id.extend([i] * length.item())
            start += length.item()
        batch_id = torch.tensor(batch_id, dtype=torch.long, device=device)
        
        # Block embedding
        H_0 = self.block_embedding(B, A, atom_positions, block_id)  # [N_atoms, hidden_size]
        
        # Construct edges
        S = torch.arange(len(B), device=device)  # block indices
        edge_result = self.edge_constructor(
            S, batch_id, segment_ids, 
            X=Z, block_id=block_id
        )
        edges = (edge_result[0], edge_result[1])  # Extract first two elements as edges
        edge_attr = edge_result[4] if len(edge_result) > 4 else None  # Extract edge attributes
        
        # Edge embedding
        if edge_attr is not None:
            edge_attr = self.edge_embedding(edge_attr)
        
        # Apply encoder
        unit_repr, block_repr, graph_repr, pred_Z = self.encoder(
            H_0, Z, block_id, batch_id, edges, edge_attr
        )
        
        # Predict energy
        energy = self.energy_ffn(graph_repr).squeeze(-1)  # [batch_size]
        
        if label is not None:
            # Training mode: return loss
            return torch.nn.functional.mse_loss(-energy, label)
        else:
            # Inference mode: return predictions
            return -energy


def parse():
    parser = argparse.ArgumentParser(description='GCN+ESA Training')
    
    # Data
    parser.add_argument('--train_set', type=str, required=True, help='path to train set')
    parser.add_argument('--valid_set', type=str, default=None, help='path to valid set')
    parser.add_argument('--test_set', type=str, default=None, help='path to test set')
    
    # Training
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--max_epoch', type=int, default=100, help='max training epoch')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--save_dir', type=str, required=True, help='directory to save model')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    
    # Model
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')
    parser.add_argument('--gcn_layers', type=int, default=2, help='number of GCN layers')
    parser.add_argument('--esa_layers', type=int, default=3, help='number of ESA layers')
    parser.add_argument('--num_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    
    # Device
    parser.add_argument('--gpus', type=int, nargs='+', required=True, help='gpu to use')
    parser.add_argument('--seed', type=int, default=SEED)
    
    return parser.parse_args()


def create_dataloader(data_path, batch_size, shuffle=True):
    """Create dataloader"""
    dataset = PDBBindBenchmark(data_path)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=PDBBindBenchmark.collate_fn,
        num_workers=4
    )


def train_epoch(model, dataloader, optimizer, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        loss = model(
            Z=batch['X'].squeeze(-2),  # Remove channel dimension for coordinates
            B=batch['B'],
            A=batch['A'], 
            atom_positions=batch['atom_positions'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
            label=batch['label']
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward pass
            loss = model(
                Z=batch['X'].squeeze(-2),
                B=batch['B'],
                A=batch['A'],
                atom_positions=batch['atom_positions'], 
                block_lengths=batch['block_lengths'],
                lengths=batch['lengths'],
                segment_ids=batch['segment_ids'],
                label=batch['label']
            )
            
            # Get predictions
            preds = model(
                Z=batch['X'].squeeze(-2),
                B=batch['B'], 
                A=batch['A'],
                atom_positions=batch['atom_positions'],
                block_lengths=batch['block_lengths'],
                lengths=batch['lengths'],
                segment_ids=batch['segment_ids']
            )
            
            total_loss += loss.item()
            num_batches += 1
            predictions.extend(preds.cpu().numpy())
            targets.extend(batch['label'].cpu().numpy())
    
    # Calculate metrics
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    pearson_r, _ = pearsonr(predictions, targets)
    spearman_r, _ = spearmanr(predictions, targets)
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mae = np.mean(np.abs(predictions - targets))
    
    return {
        'loss': total_loss / num_batches,
        'pearson': pearson_r,
        'spearman': spearman_r,
        'rmse': rmse,
        'mae': mae
    }


def main():
    args = parse()
    setup_seed(args.seed)
    VOCAB.load_tokenizer(None)
    
    # Setup device
    device = torch.device(f'cuda:{args.gpus[0]}' if args.gpus[0] >= 0 else 'cpu')
    
    # Create dataloaders
    train_loader = create_dataloader(args.train_set, args.batch_size, shuffle=True)
    valid_loader = create_dataloader(args.valid_set, args.batch_size, shuffle=False) if args.valid_set else None
    test_loader = create_dataloader(args.test_set, args.batch_size, shuffle=False) if args.test_set else None
    
    # Create model
    model = GCNESAAffinityPredictor(
        hidden_size=args.hidden_size,
        gcn_layers=args.gcn_layers,
        esa_layers=args.esa_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)
    
    print_log(f'Model parameters: {count_parameters(model)}')
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Training loop
    best_valid_loss = float('inf')
    patience_counter = 0
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    for epoch in range(args.max_epoch):
        print_log(f'Epoch {epoch + 1}/{args.max_epoch}')
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print_log(f'Train Loss: {train_loss:.4f}')
        
        # Validate
        if valid_loader:
            valid_metrics = evaluate(model, valid_loader, device)
            print_log(f'Valid Loss: {valid_metrics["loss"]:.4f}, '
                     f'Pearson: {valid_metrics["pearson"]:.4f}, '
                     f'Spearman: {valid_metrics["spearman"]:.4f}, '
                     f'RMSE: {valid_metrics["rmse"]:.4f}')
            
            scheduler.step(valid_metrics['loss'])
            
            # Early stopping and save best model
            if valid_metrics['loss'] < best_valid_loss:
                best_valid_loss = valid_metrics['loss']
                patience_counter = 0
                torch.save(model.state_dict(), 
                          os.path.join(args.save_dir, 'best_model.pth'))
                print_log('Saved best model')
            else:
                patience_counter += 1
                
            if patience_counter >= args.patience:
                print_log('Early stopping triggered')
                break
    
    # Test
    if test_loader:
        model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth')))
        test_metrics = evaluate(model, test_loader, device)
        print_log(f'Test Results - Loss: {test_metrics["loss"]:.4f}, '
                 f'Pearson: {test_metrics["pearson"]:.4f}, '
                 f'Spearman: {test_metrics["spearman"]:.4f}, '
                 f'RMSE: {test_metrics["rmse"]:.4f}')


if __name__ == '__main__':
    main()
