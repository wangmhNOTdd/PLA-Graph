#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train MACE-En model on v2020-other-PL dataset
Enhanced version with improved numerical stability
"""

import os
import sys
import warnings
import argparse
import pickle
from typing import Dict, Any, Optional
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

# MACE imports
from models.MACE.modules.tools.torch_tools import init_device, init_wandb
from models.MACE.modules.tools.utils import AtomicNumberTable
from models.MACE.modules.blocks import InteractionBlock, RealAgnosticInteractionBlock
from utils.convert import dtype_mapping, init_dataloader, load_data
from utils.logger import Logger

# Enhanced MACE-En model
from models.MACE.model import MACE_MP  # Base MP model as reference
from models.MACE_En.model import MACE_En, EnhancedRadialEmbeddingBlock

# Suppress warnings
warnings.filterwarnings("ignore")


def build_mace_en_model(config: Dict[str, Any], z_table: AtomicNumberTable, 
                       r_max: float, device: torch.device) -> nn.Module:
    """Build MACE-En model with enhanced numerical stability."""
    
    print(f"Building MACE-En model with config:")
    print(f"  - hidden_irreps: {config['hidden_irreps']}")
    print(f"  - r_max: {r_max}")
    print(f"  - num_bessel: {config['num_bessel']}")
    print(f"  - radial_type: {config.get('radial_type', 'enhanced_bessel')}")
    print(f"  - num_polynomial_cutoff: {config['num_polynomial_cutoff']}")
    print(f"  - max_ell: {config['max_ell']}")
    print(f"  - correlation: {config['correlation']}")
    print(f"  - num_interactions: {config['num_interactions']}")
    print(f"  - numerical_eps: {config.get('numerical_eps', 1e-8)}")
    
    # Atomic energies (can be zeros for protein-ligand binding)
    atomic_energies = torch.zeros(len(z_table.zs), dtype=torch.get_default_dtype())
    
    # Select interaction class
    if config.get('interaction_cls') == 'RealAgnosticInteractionBlock':
        interaction_cls = RealAgnosticInteractionBlock
    else:
        interaction_cls = InteractionBlock
    
    model = MACE_En(
        r_max=r_max,
        num_bessel=config['num_bessel'],
        num_polynomial_cutoff=config['num_polynomial_cutoff'],
        max_ell=config['max_ell'],
        interaction_cls=interaction_cls,
        interaction_cls_first=interaction_cls,
        num_interactions=config['num_interactions'],
        num_elements=len(z_table.zs),
        hidden_irreps=config['hidden_irreps'],
        MLP_irreps=config['MLP_irreps'],
        atomic_energies=atomic_energies.numpy(),
        avg_num_neighbors=config.get('avg_num_neighbors', 50.0),
        atomic_numbers=z_table.zs,
        correlation=config['correlation'],
        gate=F.silu,
        radial_MLP=config.get('radial_MLP', [64, 64, 64]),
        radial_type=config.get('radial_type', 'enhanced_bessel'),
        numerical_eps=config.get('numerical_eps', 1e-8),
    )
    
    return model.to(device)


def compute_loss(prediction: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    """Compute MSE loss between prediction and target energies."""
    predicted_energy = prediction['energy']
    
    # Numerical stability checks
    if torch.isnan(predicted_energy).any():
        print("Warning: NaN detected in predicted energy")
        predicted_energy = torch.nan_to_num(predicted_energy, nan=0.0)
    
    if torch.isinf(predicted_energy).any():
        print("Warning: Inf detected in predicted energy")
        predicted_energy = torch.clamp(predicted_energy, min=-1000, max=1000)
    
    loss = F.mse_loss(predicted_energy, target)
    
    # Additional stability check for loss
    if torch.isnan(loss) or torch.isinf(loss):
        print("Warning: Invalid loss detected, returning large penalty")
        return torch.tensor(1e6, device=loss.device, requires_grad=True)
    
    return loss


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
                device: torch.device, gradient_clip: float = 1.0) -> float:
    """Train for one epoch with enhanced stability."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        try:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            prediction = model(batch.to_dict(), training=True)
            
            # Compute loss
            loss = compute_loss(prediction, batch.y)
            
            # Backward pass with gradient clipping
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            # Check gradients for stability
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            if total_norm > 100:  # Large gradient warning
                print(f"Warning: Large gradient norm {total_norm:.2f} at batch {batch_idx}")
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}, Grad Norm: {total_norm:.3f}")
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    return total_loss / max(num_batches, 1)


def validate_epoch(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    """Validate with enhanced stability checks."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            try:
                batch = batch.to(device)
                prediction = model(batch.to_dict(), training=False)
                loss = compute_loss(prediction, batch.y)
                total_loss += loss.item()
                num_batches += 1
            except Exception as e:
                print(f"Validation error: {e}")
                continue
    
    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description='Train MACE-En on v2020-other-PL dataset')
    parser.add_argument('--data_dir', type=str, 
                       default='./datasets/v2020-other-PL/processed_get_format/',
                       help='Directory containing processed data')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--radial_type', type=str, default='enhanced_bessel',
                       choices=['enhanced_bessel', 'hybrid', 'adaptive'],
                       help='Type of radial basis function')
    parser.add_argument('--num_bessel', type=int, default=8, help='Number of Bessel functions')
    parser.add_argument('--numerical_eps', type=float, default=1e-8, help='Numerical stability epsilon')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/mace_en_v2020',
                       help='Directory to save checkpoints')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Setup device
    device = init_device('cuda')
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup logger
    logger = Logger(str(args.save_dir), 'train.log')
    logger.info(f"Starting MACE-En training with args: {args}")
    
    # Setup wandb
    if args.use_wandb:
        wandb.init(
            project="MACE-En-v2020",
            config=vars(args),
            name=f"MACE_En_{args.radial_type}_{args.num_bessel}bessel"
        )
    
    # Load data
    print("Loading data...")
    train_data = load_data(os.path.join(args.data_dir, 'train.pkl'))
    val_data = load_data(os.path.join(args.data_dir, 'valid.pkl'))
    
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    
    # Setup atomic number table
    z_table = AtomicNumberTable([1, 6, 7, 8, 9, 15, 16, 17, 35, 53])  # Common elements
    
    # Model configuration
    config = {
        'hidden_irreps': '128x0e + 128x1o',
        'MLP_irreps': '16x0e',
        'num_bessel': args.num_bessel,
        'num_polynomial_cutoff': 5,
        'max_ell': 3,
        'correlation': 3,
        'num_interactions': 2,
        'avg_num_neighbors': 50.0,
        'radial_MLP': [64, 64],
        'radial_type': args.radial_type,
        'numerical_eps': args.numerical_eps,
    }
    
    # Determine r_max from data
    r_max = 5.0  # Default for protein-ligand interactions
    
    # Build model
    print("Building MACE-En model...")
    model = build_mace_en_model(config, z_table, r_max, device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Setup training
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, args.gradient_clip)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, device)
        
        # Update scheduler
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
                'args': vars(args)
            }
            
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"New best model saved with val_loss: {val_loss:.6f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print("Training completed!")
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
