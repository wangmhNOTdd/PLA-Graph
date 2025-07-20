#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import torch
import numpy as np
from data.dataset import PDBBindBenchmark
from data.pdb_utils import VOCAB

def test_gcn_esa_model():
    """Test GCN+ESA model with real data"""
    print("Testing GCN+ESA model...")
    
    # Load vocabulary
    VOCAB.load_tokenizer(None)
    
    # Load a small subset of data
    test_file = 'datasets/PDBBind/processed/identity30/test.pkl'
    if not os.path.exists(test_file):
        print(f"Test file {test_file} not found. Please check the path.")
        return
    
    dataset = PDBBindBenchmark(test_file)
    print(f"Loaded dataset with {len(dataset)} samples")
    
    # Get a single sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    
    # Create batch from single sample
    batch = dataset.collate_fn([sample])
    print(f"Batch keys: {batch.keys()}")
    print(f"Batch shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}, dtype: {value.dtype}")
        else:
            print(f"  {key}: {type(value)} = {value}")
    
    # Import and test simplified GCN+ESA encoder
    from train_gcn_esa import SimpleGCNESAEncoder
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    hidden_size = 128
    encoder = SimpleGCNESAEncoder(
        hidden_size=hidden_size, 
        gcn_layers=2,
        num_heads=8,
        dropout=0.1
    ).to(device)
    
    print(f"Created encoder with {sum(p.numel() for p in encoder.parameters())} parameters")
    
    # Move batch to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    
    # Test forward pass
    try:
        with torch.no_grad():
            # Prepare inputs for encoder
            H = torch.randn(batch['X'].shape[0], hidden_size, device=device)  # Mock atom features
            Z = batch['X']  # [N_atoms, 1, 3] 
            block_id = torch.arange(batch['B'].shape[0], device=device).repeat_interleave(batch['block_lengths'])
            batch_id = torch.arange(batch['lengths'].shape[0], device=device).repeat_interleave(batch['lengths'])
            
            print(f"Input shapes:")
            print(f"  H: {H.shape}")
            print(f"  Z: {Z.shape}")
            print(f"  block_id: {block_id.shape}")
            print(f"  batch_id: {batch_id.shape}")
            
            # Create dummy edges (fully connected within each graph)
            edges = []
            start_idx = 0
            for length in batch['lengths']:
                end_idx = start_idx + length
                block_indices = torch.arange(start_idx, end_idx, device=device)
                if len(block_indices) > 1:
                    src = block_indices.unsqueeze(1).repeat(1, len(block_indices)).flatten()
                    dst = block_indices.unsqueeze(0).repeat(len(block_indices), 1).flatten()
                    mask = src != dst
                    edges.append(torch.stack([src[mask], dst[mask]], dim=0))
                start_idx = end_idx
            
            if edges:
                edge_index = torch.cat(edges, dim=1)
            else:
                edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
            
            edges = (edge_index[0], edge_index[1])
            
            print(f"  edges: {len(edges[0])} edges")
            
            # Forward pass
            H_out, block_repr, graph_repr, pred_Z = encoder(H, Z, block_id, batch_id, edges)
            
            print(f"Output shapes:")
            print(f"  H_out: {H_out.shape}")
            print(f"  block_repr: {block_repr.shape}")
            print(f"  graph_repr: {graph_repr.shape}")
            
            print("✓ Forward pass successful!")
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    success = test_gcn_esa_model()
    if success:
        print("\n🎉 GCN+ESA model test passed!")
    else:
        print("\n❌ GCN+ESA model test failed!")
