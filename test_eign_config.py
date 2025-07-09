#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Test script to verify EIGN model parameters and configuration
"""
import json
import torch
from utils.logger import print_log
from utils.random_seed import setup_seed
from data.pdb_utils import VOCAB
import models


def test_eign_config():
    """Test EIGN configuration and model creation"""
    
    print_log("Testing EIGN configuration...")
    
    # Load configuration
    config_path = "./eign_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print_log("Configuration loaded:")
    for key, value in config.items():
        print_log(f"  {key}: {value}")
    
    # Convert config dict to args-like object
    class Args:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
            # Add missing attributes with defaults
            self.valid_batch_size = 32
            self.valid_max_n_vertex_per_gpu = 2000
            self.patience = 30
            self.num_workers = 4
            self.grad_clip = 1.0
            self.warmup = 5
            self.noisy_sigma = 0.0
            self.gpus = [0]
            self.local_rank = -1
            self.pretrain_ckpt = None
            self.pdb_dir = None
            self.train_set2 = None
            self.valid_set2 = None
            self.train_set3 = None
            self.fragment = None
            self.pretrain = False
    
    args = Args(config)
    
    # Setup
    setup_seed(args.seed)
    VOCAB.load_tokenizer(args.fragment)
    
    print_log("\nCreating EIGN model...")
    
    # Create model
    model = models.create_model(args)
    print_log(f"Model type: {type(model)}")
    print_log(f"Model encoder type: {type(model.encoder)}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print_log(f"\nModel parameters:")
    print_log(f"  Total parameters: {total_params:,}")
    print_log(f"  Trainable parameters: {trainable_params:,}")
    print_log(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Test model with dummy input
    print_log("\nTesting model forward pass...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create dummy batch data
    batch_size = 2
    n_atoms = 100
    
    Z = torch.randint(0, 50, (n_atoms,)).to(device)  # atom types
    B = torch.randint(0, 20, (n_atoms,)).to(device)  # block types
    A = torch.randint(0, 5, (n_atoms,)).to(device)   # atom positions
    atom_positions = torch.randn(n_atoms, 3).to(device)
    block_lengths = torch.tensor([50, 50]).to(device)
    lengths = torch.tensor([50, 50]).to(device)
    segment_ids = torch.cat([torch.zeros(50), torch.ones(50)]).long().to(device)
    label = torch.randn(batch_size).to(device)
    
    print_log(f"Dummy input shapes:")
    print_log(f"  Z: {Z.shape}")
    print_log(f"  B: {B.shape}")
    print_log(f"  A: {A.shape}")
    print_log(f"  atom_positions: {atom_positions.shape}")
    print_log(f"  block_lengths: {block_lengths.shape}")
    print_log(f"  lengths: {lengths.shape}")
    print_log(f"  segment_ids: {segment_ids.shape}")
    print_log(f"  label: {label.shape}")
    
    try:
        with torch.no_grad():
            output = model(Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label)
        print_log(f"Model output shape: {output.shape}")
        print_log("Forward pass successful!")
        
        # Print some model details
        print_log(f"\nModel architecture details:")
        print_log(f"  Hidden size: {args.hidden_size}")
        print_log(f"  Number of layers: {args.n_layers}")
        print_log(f"  Dropout rate: {args.dropout}")
        print_log(f"  K-neighbors: {args.k_neighbors}")
        print_log(f"  Radial size: {args.radial_size}")
        print_log(f"  Cutoff: {args.cutoff}")
        
    except Exception as e:
        print_log(f"Error in forward pass: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print_log("\nConfiguration test completed!")


if __name__ == "__main__":
    test_eign_config()
