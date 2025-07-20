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


class SimpleGCNESAModel(nn.Module):
    """Simple GCN+ESA Model for Protein-Ligand Affinity Prediction"""
    
    def __init__(self, hidden_size=128, gcn_layers=2, num_heads=8, dropout=0.1, 
                 cutoff=8.0, k_neighbors=10):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.cutoff = cutoff
        self.k_neighbors = k_neighbors
        
        # Atom type embedding
        self.atom_embedding = nn.Embedding(VOCAB.get_num_atom_type(), hidden_size)
        self.position_embedding = nn.Embedding(VOCAB.get_num_atom_pos(), hidden_size)
        self.block_embedding = nn.Embedding(len(VOCAB), hidden_size)
        
        # GCN layers for atom-level processing
        self.gcn_convs = nn.ModuleList()
        for i in range(gcn_layers):
            self.gcn_convs.append(GCNConv(hidden_size, hidden_size))
        
        # Attention for block-level interactions
        self.block_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Final layers
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
        
        # Create block_id mapping
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
        
        # Initial embeddings
        atom_features = self.atom_embedding(A) + self.position_embedding(atom_positions)
        block_features = self.block_embedding(B)
        
        # Add block embedding to atoms
        atom_features = atom_features + block_features[block_id]
        
        # Build atom-level graph using radius
        coords = Z.squeeze(-2) if Z.dim() == 3 else Z  # [N_atoms, 3]
        atom_edges = radius_graph(coords, r=self.cutoff, batch=atom_batch_id, 
                                 max_num_neighbors=self.k_neighbors)
        
        # Apply GCN on atoms
        x = atom_features
        for gcn_conv in self.gcn_convs:
            x = gcn_conv(x, atom_edges)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        
        # Pool atoms to blocks
        block_features = scatter_mean(x, block_id, dim=0)  # [N_blocks, hidden_size]
        
        # Create batch_id for blocks
        block_batch_id = []
        start = 0
        for i, length in enumerate(lengths):
            block_batch_id.extend([i] * length.item())
            start += length.item()
        block_batch_id = torch.tensor(block_batch_id, dtype=torch.long, device=device)
        
        # Apply attention at block level
        graph_features = []
        for batch_idx in range(batch_size):
            batch_mask = block_batch_id == batch_idx
            batch_blocks = block_features[batch_mask]  # [n_blocks_in_batch, hidden_size]
            
            if batch_blocks.shape[0] > 0:
                # Self-attention on blocks
                attended_blocks, _ = self.block_attention(
                    batch_blocks.unsqueeze(0),
                    batch_blocks.unsqueeze(0), 
                    batch_blocks.unsqueeze(0)
                )
                attended_blocks = attended_blocks.squeeze(0)
                
                # Pool to graph representation
                graph_repr = attended_blocks.mean(dim=0)
            else:
                graph_repr = torch.zeros(self.hidden_size, device=device)
            
            graph_features.append(graph_repr)
        
        graph_repr = torch.stack(graph_features, dim=0)  # [batch_size, hidden_size]
        graph_repr = self.norm(graph_repr)
        
        # Predict affinity
        output = self.final_mlp(graph_repr).squeeze(-1)  # [batch_size]
        
        if label is not None:
            # Training mode: return loss
            return F.mse_loss(output, label)
        else:
            # Inference mode
            return output


def parse():
    parser = argparse.ArgumentParser(description='Simple GCN+ESA Training')
    
    # Data
    parser.add_argument('--train_set', type=str, required=True)
    parser.add_argument('--valid_set', type=str, default=None)
    parser.add_argument('--test_set', type=str, default=None)
    
    # Training
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--patience', type=int, default=20)
    
    # Model
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--gcn_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--cutoff', type=float, default=8.0)
    parser.add_argument('--k_neighbors', type=int, default=10)
    
    # Device
    parser.add_argument('--gpus', type=int, nargs='+', required=True)
    parser.add_argument('--seed', type=int, default=SEED)
    
    return parser.parse_args()


def create_dataloader(data_path, batch_size, shuffle=True):
    dataset = PDBBindBenchmark(data_path)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=PDBBindBenchmark.collate_fn,
        num_workers=2
    )


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        optimizer.zero_grad()
        
        try:
            # Forward pass
            loss = model(
                Z=batch['X'],
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
            
        except Exception as e:
            print_log(f"Error in batch: {e}")
            continue
    
    return total_loss / max(num_batches, 1)


def evaluate(model, dataloader, device):
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
            
            try:
                # Forward pass for loss
                loss = model(
                    Z=batch['X'],
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
                    Z=batch['X'],
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
                
            except Exception as e:
                print_log(f"Error in validation batch: {e}")
                continue
    
    if len(predictions) == 0:
        return {'loss': float('inf'), 'pearson': 0, 'spearman': 0, 'rmse': float('inf'), 'mae': float('inf')}
    
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
        'loss': total_loss / max(num_batches, 1),
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
    model = SimpleGCNESAModel(
        hidden_size=args.hidden_size,
        gcn_layers=args.gcn_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        cutoff=args.cutoff,
        k_neighbors=args.k_neighbors
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
