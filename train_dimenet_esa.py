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
from torch_geometric.nn import radius_graph
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pickle
import math

from utils.logger import print_log
from utils.random_seed import setup_seed, SEED

from data.dataset import PDBBindBenchmark
import trainers
from utils.nn_utils import count_parameters
from data.pdb_utils import VOCAB

# Import DimeNet components
from models.DimeNet.dimenet import (
    BesselBasisLayer, SphericalBasisLayer, EmbeddingBlock, 
    InteractionBlock, triplets, stable_norm
)
from torch_geometric.nn.inits import glorot_orthogonal


class TrainingMonitor:
    """训练监控类，记录和可视化训练过程中的各项指标"""
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.history = defaultdict(list)
        
    def log_epoch(self, epoch, train_loss, valid_metrics=None, test_metrics=None):
        """记录每个epoch的指标"""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        
        if valid_metrics:
            for key, value in valid_metrics.items():
                self.history[f'valid_{key}'].append(value)
        
        # 为了保持列表长度一致，在没有测试指标时添加None
        if test_metrics:
            for key, value in test_metrics.items():
                if f'test_{key}' not in self.history:
                    # 如果是第一次添加测试指标，需要填充之前的空值
                    self.history[f'test_{key}'] = [None] * (len(self.history['epoch']) - 1)
                self.history[f'test_{key}'].append(value)
        else:
            # 如果这个epoch没有测试指标，为已存在的测试指标添加None
            for key in list(self.history.keys()):
                if key.startswith('test_'):
                    if len(self.history[key]) < len(self.history['epoch']):
                        self.history[key].append(None)
    
    def save_history(self):
        """保存训练历史"""
        with open(os.path.join(self.save_dir, 'training_history.pkl'), 'wb') as f:
            pickle.dump(dict(self.history), f)
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('DimeNet+ESA 训练监控曲线', fontsize=16, fontweight='bold')
        
        epochs = self.history['epoch']
        
        # 1. 损失曲线
        ax = axes[0, 0]
        ax.plot(epochs, self.history['train_loss'], 'b-', label='训练损失', linewidth=2)
        if 'valid_loss' in self.history:
            ax.plot(epochs, self.history['valid_loss'], 'r-', label='验证损失', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('损失函数曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Pearson相关系数
        ax = axes[0, 1]
        if 'valid_pearson' in self.history and len(self.history['valid_pearson']) > 0:
            valid_epochs = epochs[:len(self.history['valid_pearson'])]
            ax.plot(valid_epochs, self.history['valid_pearson'], 'g-', label='验证集', linewidth=2)
        if 'test_pearson' in self.history and len(self.history['test_pearson']) > 0:
            # 只绘制在训练过程中记录的测试结果
            test_data = [x for x in self.history['test_pearson'] if x is not None]
            if test_data:
                test_epochs = [epochs[i] for i in range(len(epochs)) if i < len(self.history['test_pearson']) and self.history['test_pearson'][i] is not None]
                ax.plot(test_epochs, test_data, 'orange', label='测试集', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Pearson Correlation')
        ax.set_title('Pearson 相关系数')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Spearman相关系数
        ax = axes[0, 2]
        if 'valid_spearman' in self.history and len(self.history['valid_spearman']) > 0:
            valid_epochs = epochs[:len(self.history['valid_spearman'])]
            ax.plot(valid_epochs, self.history['valid_spearman'], 'g-', label='验证集', linewidth=2)
        if 'test_spearman' in self.history and len(self.history['test_spearman']) > 0:
            test_data = [x for x in self.history['test_spearman'] if x is not None]
            if test_data:
                test_epochs = [epochs[i] for i in range(len(epochs)) if i < len(self.history['test_spearman']) and self.history['test_spearman'][i] is not None]
                ax.plot(test_epochs, test_data, 'orange', label='测试集', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Spearman Correlation')
        ax.set_title('Spearman 相关系数')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. RMSE
        ax = axes[1, 0]
        if 'valid_rmse' in self.history and len(self.history['valid_rmse']) > 0:
            valid_epochs = epochs[:len(self.history['valid_rmse'])]
            ax.plot(valid_epochs, self.history['valid_rmse'], 'g-', label='验证集', linewidth=2)
        if 'test_rmse' in self.history and len(self.history['test_rmse']) > 0:
            test_data = [x for x in self.history['test_rmse'] if x is not None]
            if test_data:
                test_epochs = [epochs[i] for i in range(len(epochs)) if i < len(self.history['test_rmse']) and self.history['test_rmse'][i] is not None]
                ax.plot(test_epochs, test_data, 'orange', label='测试集', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('均方根误差')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. MAE
        ax = axes[1, 1]
        if 'valid_mae' in self.history and len(self.history['valid_mae']) > 0:
            valid_epochs = epochs[:len(self.history['valid_mae'])]
            ax.plot(valid_epochs, self.history['valid_mae'], 'g-', label='验证集', linewidth=2)
        if 'test_mae' in self.history and len(self.history['test_mae']) > 0:
            test_data = [x for x in self.history['test_mae'] if x is not None]
            if test_data:
                test_epochs = [epochs[i] for i in range(len(epochs)) if i < len(self.history['test_mae']) and self.history['test_mae'][i] is not None]
                ax.plot(test_epochs, test_data, 'orange', label='测试集', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE')
        ax.set_title('平均绝对误差')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. 性能总结图
        ax = axes[1, 2]
        if all(key in self.history for key in ['test_pearson', 'test_spearman', 'test_rmse', 'test_mae']):
            # 获取最后的有效测试结果
            last_pearson = next((x for x in reversed(self.history['test_pearson']) if x is not None), None)
            last_spearman = next((x for x in reversed(self.history['test_spearman']) if x is not None), None)
            last_rmse = next((x for x in reversed(self.history['test_rmse']) if x is not None), None)
            last_mae = next((x for x in reversed(self.history['test_mae']) if x is not None), None)
            
            if all(x is not None for x in [last_pearson, last_spearman, last_rmse, last_mae]):
                metrics = ['Pearson', 'Spearman', 'RMSE', 'MAE']
                values = [last_pearson, last_spearman, last_rmse, last_mae]
                colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightyellow']
                bars = ax.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
                ax.set_title('最终测试集性能')
                ax.set_ylabel('Metric Value')
                
                # 在柱状图上添加数值标签
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
                
                ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Waiting for test results...', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def final_test_result(self, test_metrics):
        """记录最终测试结果"""
        print("\n" + "="*60)
        print("FINAL TEST RESULTS")
        print("="*60)
        for key, value in test_metrics.items():
            print(f"{key.upper()}: {value:.4f}")
        print("="*60)
        
        # 更新history中的最后一个测试结果
        for key, value in test_metrics.items():
            if f'test_{key}' not in self.history:
                self.history[f'test_{key}'] = [None] * len(self.history['epoch'])
            if len(self.history[f'test_{key}']) == len(self.history['epoch']):
                self.history[f'test_{key}'][-1] = value
            else:
                self.history[f'test_{key}'].append(value)


class GeometricAtomProcessor(nn.Module):
    """Geometric-aware atom processor inspired by DimeNet concepts"""
    
    def __init__(self, hidden_channels, num_blocks=2, num_radial=6, num_spherical=7, 
                 cutoff=8.0, envelope_exponent=5):
        super().__init__()
        
        self.cutoff = cutoff
        self.num_blocks = num_blocks
        self.hidden_channels = hidden_channels
        
        # Radial basis functions (simplified)
        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        
        # Spherical basis functions (simplified)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff, envelope_exponent)
        
        # Message passing layers
        self.interaction_layers = nn.ModuleList()
        for _ in range(num_blocks):
            self.interaction_layers.append(nn.Sequential(
                nn.Linear(hidden_channels * 2 + num_radial, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels)
            ))
        
        # Update functions
        self.update_layers = nn.ModuleList()
        for _ in range(num_blocks):
            self.update_layers.append(nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels)
            ))
            
    def forward(self, atom_features, coordinates, batch_id):
        """
        Args:
            atom_features: [N_atoms, hidden_size] - initial atom embeddings
            coordinates: [N_atoms, 3] - atom coordinates
            batch_id: [N_atoms] - batch indices for each atom
        Returns:
            processed_features: [N_atoms, hidden_size] - processed atom features
        """
        device = atom_features.device
        num_atoms = atom_features.size(0)
        
        # Build edges using radius graph
        edge_index = radius_graph(coordinates, r=self.cutoff, batch=batch_id, max_num_neighbors=32)
        
        if edge_index.size(1) == 0:
            # No edges, return original features
            return atom_features
            
        i, j = edge_index[0], edge_index[1]
        
        # Calculate distances
        dist = torch.norm(coordinates[i] - coordinates[j], dim=-1)
        
        # Radial basis functions
        rbf = self.rbf(dist)
        
        x = atom_features
        
        # Apply interaction layers
        for layer_idx in range(self.num_blocks):
            # Create messages
            x_i = x[i]  # [num_edges, hidden_channels]
            x_j = x[j]  # [num_edges, hidden_channels]
            
            # Combine features with distance information
            edge_features = torch.cat([x_i, x_j, rbf], dim=-1)  # [num_edges, 2*hidden + num_radial]
            
            # Process messages
            messages = self.interaction_layers[layer_idx](edge_features)
            
            # Aggregate messages
            aggregated = torch.zeros_like(x)
            aggregated.index_add_(0, i, messages)
            
            # Update node features
            update_input = torch.cat([x, aggregated], dim=-1)
            x = self.update_layers[layer_idx](update_input) + x  # Residual connection
        
        return x


class GeometricESAModel(nn.Module):
    """Geometric-aware ESA Model for Protein-Ligand Affinity Prediction"""
    
    def __init__(self, hidden_size=128, geometric_blocks=2, num_heads=8, dropout=0.1, 
                 cutoff=8.0, num_radial=6, num_spherical=7):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.cutoff = cutoff
        
        # Atom type embedding
        self.atom_embedding = nn.Embedding(VOCAB.get_num_atom_type(), hidden_size)
        self.position_embedding = nn.Embedding(VOCAB.get_num_atom_pos(), hidden_size)
        self.block_embedding = nn.Embedding(len(VOCAB), hidden_size)
        
        # Geometric processor instead of GCN
        self.geometric_processor = GeometricAtomProcessor(
            hidden_channels=hidden_size,
            num_blocks=geometric_blocks,
            num_radial=num_radial,
            num_spherical=num_spherical,
            cutoff=cutoff
        )
        
        # Attention for block-level interactions
        self.block_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Edge MLP for ESA
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
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
            n_atoms_in_batch = sum(block_lengths[atom_start:atom_start + block_length.item()])
            atom_batch_id.extend([i] * n_atoms_in_batch)
            atom_start += block_length.item()
        atom_batch_id = torch.tensor(atom_batch_id, dtype=torch.long, device=device)
        
        # Initial embeddings
        atom_features = self.atom_embedding(A) + self.position_embedding(atom_positions)
        block_features = self.block_embedding(B)
        
        # Add block embedding to atoms
        atom_features = atom_features + block_features[block_id]
        
        # Get atom coordinates
        coords = Z.squeeze(-2) if Z.dim() == 3 else Z  # [N_atoms, 3]
        
        # Apply Geometric processor on atoms (replacing GCN)
        processed_atom_features = self.geometric_processor(atom_features, coords, atom_batch_id)
        
        # Pool atoms to blocks
        block_features = scatter_mean(processed_atom_features, block_id, dim=0)  # [N_blocks, hidden_size]
        
        # Create batch_id for blocks
        block_batch_id = []
        start = 0
        for i, length in enumerate(lengths):
            block_batch_id.extend([i] * length.item())
            start += length.item()
        block_batch_id = torch.tensor(block_batch_id, dtype=torch.long, device=device)
        
        # Build block-level edges for ESA
        # Get block center coordinates
        block_coords = scatter_mean(coords, block_id, dim=0)  # [N_blocks, 3]
        
        # Apply Edge Set Attention (ESA) at block level
        graph_features = []
        for batch_idx in range(batch_size):
            batch_mask = block_batch_id == batch_idx
            batch_blocks = block_features[batch_mask]  # [n_blocks_in_batch, hidden_size]
            batch_coords = block_coords[batch_mask]     # [n_blocks_in_batch, 3]
            
            if batch_blocks.shape[0] > 1:
                # Build block-level edges using spatial proximity
                block_edges = radius_graph(batch_coords, r=self.cutoff * 2, 
                                         max_num_neighbors=min(32, batch_blocks.shape[0]-1))
                
                if block_edges.shape[1] > 0:
                    # Create edge features for ESA
                    src_features = batch_blocks[block_edges[0]]
                    dst_features = batch_blocks[block_edges[1]]
                    edge_features = torch.cat([src_features, dst_features], dim=-1)
                    edge_features = self.edge_mlp(edge_features)  # [n_edges, hidden_size]
                    
                    # Apply attention on edges (simplified ESA)
                    edge_attn, _ = self.block_attention(
                        edge_features.unsqueeze(0),
                        edge_features.unsqueeze(0),
                        edge_features.unsqueeze(0)
                    )
                    edge_attn = edge_attn.squeeze(0)
                    
                    # Aggregate edge information back to nodes
                    aggregated_features = scatter_mean(edge_attn, block_edges[0], dim=0, dim_size=batch_blocks.shape[0])
                    batch_blocks = batch_blocks + aggregated_features
                
                # Global attention on updated blocks
                attended_blocks, _ = self.block_attention(
                    batch_blocks.unsqueeze(0),
                    batch_blocks.unsqueeze(0), 
                    batch_blocks.unsqueeze(0)
                )
                attended_blocks = attended_blocks.squeeze(0)
                
                # Pool to graph representation
                graph_repr = attended_blocks.mean(dim=0)
            elif batch_blocks.shape[0] == 1:
                graph_repr = batch_blocks.squeeze(0)
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


def main():
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets/PDBBind/processed/identity30')
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--save_dir', type=str, default='checkpoints/dimenet_esa')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--dimenet_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--cutoff', type=float, default=8.0)
    parser.add_argument('--num_radial', type=int, default=6)
    parser.add_argument('--num_spherical', type=int, default=7)
    args = parser.parse_args()
    
    # 设置随机种子
    setup_seed(SEED)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # 初始化训练监控
    monitor = TrainingMonitor(args.save_dir)
    
    # 保存配置
    config = vars(args)
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    print("Loading datasets...")
    train_dataset = PDBBindBenchmark(os.path.join(args.data_dir, 'train.pkl'))
    valid_dataset = PDBBindBenchmark(os.path.join(args.data_dir, 'valid.pkl'))
    test_dataset = PDBBindBenchmark(os.path.join(args.data_dir, 'test.pkl'))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=valid_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # 创建模型
    model = GeometricESAModel(
        hidden_size=args.hidden_size,
        geometric_blocks=args.dimenet_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
        cutoff=args.cutoff,
        num_radial=args.num_radial,
        num_spherical=args.num_spherical
    ).to(device)
    
    # 打印模型参数
    total_params = count_parameters(model)
    print(f"Model parameters: {total_params:,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # 训练循环
    best_valid_loss = float('inf')
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            # 移动到设备
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # 前向传播
            optimizer.zero_grad()
            loss = model(
                batch['X'], batch['B'], batch['A'],
                batch['atom_positions'], batch['block_lengths'],
                batch['lengths'], batch['segment_ids'], batch['label']
            )
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = np.mean(train_losses)
        
        # 验证阶段
        model.eval()
        valid_losses = []
        valid_preds = []
        valid_labels = []
        
        with torch.no_grad():
            for batch in valid_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # 计算损失
                loss = model(
                    batch['X'], batch['B'], batch['A'],
                    batch['atom_positions'], batch['block_lengths'],
                    batch['lengths'], batch['segment_ids'], batch['label']
                )
                valid_losses.append(loss.item())
                
                # 获取预测
                pred = model(
                    batch['X'], batch['B'], batch['A'],
                    batch['atom_positions'], batch['block_lengths'],
                    batch['lengths'], batch['segment_ids']
                )
                valid_preds.extend(pred.cpu().numpy())
                valid_labels.extend(batch['label'].cpu().numpy())
        
        avg_valid_loss = np.mean(valid_losses)
        
        # 计算验证指标
        valid_preds = np.array(valid_preds)
        valid_labels = np.array(valid_labels)
        
        from scipy.stats import pearsonr, spearmanr
        
        valid_pearson = pearsonr(valid_preds, valid_labels)[0]
        valid_spearman = spearmanr(valid_preds, valid_labels)[0]
        valid_rmse = np.sqrt(np.mean((valid_preds - valid_labels) ** 2))
        valid_mae = np.mean(np.abs(valid_preds - valid_labels))
        
        valid_metrics = {
            'loss': avg_valid_loss,
            'pearson': valid_pearson,
            'spearman': valid_spearman,
            'rmse': valid_rmse,
            'mae': valid_mae
        }
        
        # 记录训练历史
        monitor.log_epoch(epoch + 1, avg_train_loss, valid_metrics)
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Valid Loss: {avg_valid_loss:.4f}, Pearson: {valid_pearson:.4f}, Spearman: {valid_spearman:.4f}")
        print(f"Valid RMSE: {valid_rmse:.4f}, MAE: {valid_mae:.4f}")
        
        # 学习率调度
        scheduler.step(avg_valid_loss)
        
        # 早停检查
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_valid_loss': best_valid_loss,
                'config': config
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print("Saved best model!")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
            
            if patience_counter >= args.patience:
                print(f"Early stopping after {patience_counter} epochs without improvement")
                break
    
    # 测试阶段
    print("\nEvaluating on test set...")
    
    # 加载最佳模型
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            pred = model(
                batch['X'], batch['B'], batch['A'],
                batch['atom_positions'], batch['block_lengths'],
                batch['lengths'], batch['segment_ids']
            )
            test_preds.extend(pred.cpu().numpy())
            test_labels.extend(batch['label'].cpu().numpy())
    
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    
    test_pearson = pearsonr(test_preds, test_labels)[0]
    test_spearman = spearmanr(test_preds, test_labels)[0]
    test_rmse = np.sqrt(np.mean((test_preds - test_labels) ** 2))
    test_mae = np.mean(np.abs(test_preds - test_labels))
    
    test_metrics = {
        'pearson': test_pearson,
        'spearman': test_spearman,
        'rmse': test_rmse,
        'mae': test_mae
    }
    
    # 记录最终测试结果
    monitor.final_test_result(test_metrics)
    
    # 保存训练历史和绘制曲线
    monitor.save_history()
    monitor.plot_training_curves()
    
    print(f"\nTraining completed! Results saved to {args.save_dir}")
    

if __name__ == "__main__":
    main()
