#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Test script for GCN+ESA model
"""
import sys
import os
sys.path.append('.')

import torch
from data.dataset import PDBBindBenchmark
from data.pdb_utils import VOCAB

# Load the simple test
def test_gcn_esa():
    print("Testing GCN+ESA model...")
    
    # Initialize vocab
    VOCAB.load_tokenizer(None)
    
    # Load test dataset (just a few samples)
    test_data_path = r"datasets\PDBBind\processed\identity30\test.pkl"
    if not os.path.exists(test_data_path):
        print(f"Test data not found at {test_data_path}")
        return False
    
    # Load dataset
    dataset = PDBBindBenchmark(test_data_path)
    print(f"Loaded dataset with {len(dataset)} samples")
    
    # Create dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, 
                           collate_fn=PDBBindBenchmark.collate_fn)
    
    # Get one batch
    batch = next(iter(dataloader))
    print("Batch keys:", list(batch.keys()))
    print("Batch shapes:")
    for key, value in batch.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {value}")
    
    # Test simplified encoder
    from train_gcn_esa import SimpleGCNESAEncoder
    
    encoder = SimpleGCNESAEncoder(
        hidden_size=128,
        gcn_layers=2,
        num_heads=4,
        dropout=0.1
    )
    
    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters())}")
    
    # Create dummy inputs
    device = torch.device('cpu')
    
    # Create fake edges for testing
    n_blocks = batch['B'].shape[0]
    batch_size = batch['lengths'].shape[0]
    
    # Create block_id
    block_id = []
    block_lengths = batch['block_lengths']
    for i, length in enumerate(block_lengths):
        block_id.extend([i] * length.item())
    block_id = torch.tensor(block_id, dtype=torch.long, device=device)
    
    # Create batch_id for blocks  
    batch_id = []
    lengths = batch['lengths']
    for i, length in enumerate(lengths):
        batch_id.extend([i] * length.item())
    batch_id = torch.tensor(batch_id, dtype=torch.long, device=device)
    
    # Create some dummy edges
    edges = (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0]))
    
    print("Testing encoder forward pass...")
    
    try:
        # Test forward pass
        H = torch.randn(len(block_id), 128)  # Atom features
        Z = batch['X'].squeeze(-2)  # Coordinates [N_atoms, 3]
        
        unit_repr, block_repr, graph_repr, pred_Z = encoder(
            H, Z, block_id, batch_id, edges, None
        )
        
        print("Forward pass successful!")
        print(f"Graph representation shape: {graph_repr.shape}")
        print(f"Block representation shape: {block_repr.shape}")
        
        return True
        
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gcn_esa()
    if success:
        print("✓ GCN+ESA model test passed!")
    else:
        print("✗ GCN+ESA model test failed!")
        sys.exit(1)
