#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Training script for EIGN model on PDBbind identity30 dataset
"""
import os
import json
import torch
import argparse
from torch.utils.data import DataLoader

from utils.logger import print_log
from utils.random_seed import setup_seed, SEED

# Import necessary modules
from data.dataset import PDBBindBenchmark, DynamicBatchWrapper
import models
import trainers
from utils.nn_utils import count_parameters
from data.pdb_utils import VOCAB


def train_eign_model():
    """Train EIGN model with predefined configuration"""
    
    # Load configuration
    config_path = "./eign_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Convert config dict to args-like object
    class Args:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
            # Add missing attributes with defaults
            self.valid_batch_size = None
            self.valid_max_n_vertex_per_gpu = None
            self.patience = -1
            self.num_workers = 4
            self.grad_clip = 1.0
            self.warmup = 0
            self.noisy_sigma = 0.0
            self.gpus = [0]  # Use GPU 0
            self.local_rank = -1
            self.pretrain_ckpt = None
            self.pdb_dir = None
            self.train_set2 = None
            self.valid_set2 = None
            self.train_set3 = None
            self.fragment = None
            self.pretrain = False
            self.batch_size = 16
    
    args = Args(config)
    
    # Setup
    setup_seed(args.seed)
    VOCAB.load_tokenizer(args.fragment)
    
    print_log(f"Training EIGN model on PDBBind identity30 dataset")
    print_log(f"Configuration: {config}")
    
    # Create model
    model = models.create_model(args)
    print_log(f"Model created: {type(model)}")
    print_log(f"Model parameters: {count_parameters(model)}")
    
    # Load datasets
    train_set = PDBBindBenchmark(args.train_set)
    valid_set = PDBBindBenchmark(args.valid_set)
    print_log(f'Train: {len(train_set)}, validation: {len(valid_set)}')
    
    # Setup dynamic batching if specified
    if args.max_n_vertex_per_gpu is not None:
        if args.valid_max_n_vertex_per_gpu is None:
            args.valid_max_n_vertex_per_gpu = args.max_n_vertex_per_gpu
        train_set = DynamicBatchWrapper(train_set, args.max_n_vertex_per_gpu)
        valid_set = DynamicBatchWrapper(valid_set, args.valid_max_n_vertex_per_gpu)
        args.batch_size, args.valid_batch_size = 1, 1
        args.num_workers = 1
    
    # Create data loaders
    collate_fn = train_set.collate_fn
    
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    if args.valid_batch_size is None:
        args.valid_batch_size = args.batch_size
        
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.valid_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Setup training
    step_per_epoch = len(train_loader)
    train_config = trainers.TrainConfig(
        args.save_dir, 
        args.lr, 
        args.max_epoch,
        warmup=args.warmup,
        patience=args.patience,
        grad_clip=args.grad_clip,
        save_topk=args.save_topk
    )
    train_config.add_parameter(
        step_per_epoch=step_per_epoch,
        final_lr=args.final_lr
    )
    
    # Create trainer
    trainer = trainers.AffinityTrainer(model, train_loader, valid_loader, train_config)
    
    # Move to GPU if available
    if torch.cuda.is_available() and args.gpus[0] >= 0:
        device = torch.device(f'cuda:{args.gpus[0]}')
        model = model.to(device)
        print_log(f"Model moved to {device}")
    else:
        print_log("Using CPU")
    
    # Start training
    print_log("Starting training...")
    trainer.train(args.gpus, args.local_rank)
    print_log("Training completed!")


if __name__ == "__main__":
    train_eign_model()
