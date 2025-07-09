#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import argparse
import torch
from torch.utils.data import DataLoader

from utils.logger import print_log
from utils.random_seed import setup_seed, SEED

# Import necessary modules
from data.dataset import BlockGeoAffDataset, PDBBindBenchmark, DynamicBatchWrapper
import models
import trainers
from utils.nn_utils import count_parameters
from data.pdb_utils import VOCAB


def train_eign_model():
    print_log("Training EIGN model on PDBBind identity30 dataset")
    
    # Configuration for EIGN model
    config = {
        "train_set": "./datasets/PDBBind/processed/identity30/train.pkl",
        "valid_set": "./datasets/PDBBind/processed/identity30/valid.pkl", 
        "save_dir": "./datasets/PDBBind/processed/identity30/models/EIGN",
        "task": "PDBBind",
        "lr": 0.0001,
        "final_lr": 0.0001,
        "max_epoch": 5,  # Reduced for quick validation
        "save_topk": 1,
        "max_n_vertex_per_gpu": 1500,
        "shuffle": True,
        "model_type": "EIGN",
        "hidden_size": 64,
        "n_layers": 3,
        "n_channel": 1,
        "n_rbf": 32,
        "cutoff": 7.0,
        "radial_size": 64,
        "k_neighbors": 9,
        "n_head": 4,
        "atom_level": True,
        "hierarchical": False,
        "no_block_embedding": False,
        "seed": 2023
    }
    
    print_log(f"Configuration: {config}")
    
    # Create args object from config
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    args = Args(**config)
    args.gpus = [0]
    args.local_rank = -1
    args.pretrain_ckpt = None
    args.noisy_sigma = 0.0
    args.valid_batch_size = None
    args.patience = -1
    args.grad_clip = 1.0
    args.warmup = 0
    args.num_workers = 1
    args.fragment = None
    
    # Set up
    setup_seed(args.seed)
    VOCAB.load_tokenizer(args.fragment)
    
    # Create model
    model = models.create_model(args)
    print_log(f"Model created: {type(model)}")
    print_log(f"Model parameters: {count_parameters(model)}")
    
    # Create datasets
    train_set = PDBBindBenchmark(args.train_set)
    valid_set = PDBBindBenchmark(args.valid_set)
    print_log(f'Train: {len(train_set)}, validation: {len(valid_set)}')
    
    # Wrap datasets for dynamic batching
    train_set = DynamicBatchWrapper(train_set, args.max_n_vertex_per_gpu)
    valid_set = DynamicBatchWrapper(valid_set, args.max_n_vertex_per_gpu)
    args.batch_size, args.valid_batch_size = 1, 1
    args.num_workers = 1
    
    # Create data loaders  
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=args.shuffle, collate_fn=train_set.collate_fn,
        num_workers=args.num_workers, pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_set, batch_size=args.valid_batch_size,
        shuffle=False, collate_fn=valid_set.collate_fn,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Create trainer config
    step_per_epoch = len(train_loader)
    trainer_config = trainers.TrainConfig(
        args.save_dir, args.lr, args.max_epoch,
        warmup=args.warmup,
        patience=args.patience,
        grad_clip=args.grad_clip,
        save_topk=args.save_topk
    )
    trainer_config.add_parameter(step_per_epoch=step_per_epoch, final_lr=args.final_lr)
    
    # Create trainer
    trainer = trainers.AffinityTrainer(model, train_loader, valid_loader, trainer_config)
    
    # Move model to GPU
    device = torch.device(f'cuda:{args.gpus[0]}' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print_log(f"Model moved to {device}")
    
    print_log("Starting training...")
    
    # Train model
    try:
        trainer.train(args.gpus, args.local_rank)
        print_log("Training completed successfully!")
    except KeyboardInterrupt:
        print_log("Training interrupted by user")
    except Exception as e:
        print_log(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    train_eign_model()
