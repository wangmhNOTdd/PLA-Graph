#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
GCN+ESA模型训练脚本 - 带监控和可视化功能
支持实时监控训练过程，保存训练指标，生成曲线图
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.nn import GCNConv, radius_graph
from scipy.stats import pearsonr, spearmanr

from utils.logger import print_log
from utils.random_seed import setup_seed, SEED
from data.dataset import PDBBindBenchmark
from utils.nn_utils import count_parameters
from data.pdb_utils import VOCAB

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TrainingMonitor:
    """训练监控器 - 记录和可视化训练过程"""
    
    def __init__(self, save_dir, model_name="GCN+ESA"):
        self.save_dir = save_dir
        self.model_name = model_name
        self.metrics_dir = os.path.join(save_dir, 'metrics')
        self.plots_dir = os.path.join(save_dir, 'plots')
        
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # 训练指标记录
        self.train_metrics = {
            'epoch': [],
            'loss': [],
            'lr': []
        }
        
        # 验证指标记录
        self.valid_metrics = {
            'epoch': [],
            'loss': [],
            'pearson': [],
            'spearman': [],
            'rmse': [],
            'mae': []
        }
        
        # 测试指标记录
        self.test_metrics = {}
        
        # 训练开始时间
        self.start_time = time.time()
        
    def log_train_metrics(self, epoch, loss, lr):
        """记录训练指标"""
        self.train_metrics['epoch'].append(epoch)
        self.train_metrics['loss'].append(loss)
        self.train_metrics['lr'].append(lr)
        
    def log_valid_metrics(self, epoch, metrics):
        """记录验证指标"""
        self.valid_metrics['epoch'].append(epoch)
        self.valid_metrics['loss'].append(metrics['loss'])
        self.valid_metrics['pearson'].append(metrics['pearson'])
        self.valid_metrics['spearman'].append(metrics['spearman'])
        self.valid_metrics['rmse'].append(metrics['rmse'])
        self.valid_metrics['mae'].append(metrics['mae'])
        
    def log_test_metrics(self, metrics):
        """记录测试指标"""
        self.test_metrics = metrics.copy()
        
    def save_metrics(self):
        """保存所有指标到文件"""
        # 保存训练指标
        train_df = pd.DataFrame(self.train_metrics)
        train_df.to_csv(os.path.join(self.metrics_dir, 'train_metrics.csv'), index=False)
        
        # 保存验证指标
        if self.valid_metrics['epoch']:
            valid_df = pd.DataFrame(self.valid_metrics)
            valid_df.to_csv(os.path.join(self.metrics_dir, 'valid_metrics.csv'), index=False)
        
        # 保存测试指标
        if self.test_metrics:
            with open(os.path.join(self.metrics_dir, 'test_metrics.json'), 'w') as f:
                json.dump(self.test_metrics, f, indent=2)
                
        # 保存训练摘要
        training_time = time.time() - self.start_time
        summary = {
            'model': self.model_name,
            'training_time_minutes': training_time / 60,
            'total_epochs': len(self.train_metrics['epoch']),
            'best_valid_loss': min(self.valid_metrics['loss']) if self.valid_metrics['loss'] else None,
            'best_valid_pearson': max(self.valid_metrics['pearson']) if self.valid_metrics['pearson'] else None,
            'test_metrics': self.test_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.metrics_dir, 'training_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        if not self.train_metrics['epoch']:
            return
            
        # 设置绘图样式
        sns.set_style("whitegrid")
        plt.style.use('seaborn-v0_8')
        
        # 1. 训练和验证损失曲线
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{self.model_name} 训练监控报告', fontsize=16, fontweight='bold')
        
        # 损失曲线
        ax1 = axes[0, 0]
        ax1.plot(self.train_metrics['epoch'], self.train_metrics['loss'], 
                label='训练损失', color='blue', linewidth=2, marker='o', markersize=4)
        if self.valid_metrics['epoch']:
            ax1.plot(self.valid_metrics['epoch'], self.valid_metrics['loss'], 
                    label='验证损失', color='red', linewidth=2, marker='s', markersize=4)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('训练和验证损失')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 学习率曲线
        ax2 = axes[0, 1]
        ax2.plot(self.train_metrics['epoch'], self.train_metrics['lr'], 
                color='green', linewidth=2, marker='d', markersize=4)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('学习率变化')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Pearson相关系数
        if self.valid_metrics['epoch']:
            ax3 = axes[0, 2]
            ax3.plot(self.valid_metrics['epoch'], self.valid_metrics['pearson'], 
                    color='purple', linewidth=2, marker='^', markersize=4)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Pearson Correlation')
            ax3.set_title('Pearson相关系数')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim([0, 1])
        
        # Spearman相关系数
        if self.valid_metrics['epoch']:
            ax4 = axes[1, 0]
            ax4.plot(self.valid_metrics['epoch'], self.valid_metrics['spearman'], 
                    color='orange', linewidth=2, marker='v', markersize=4)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Spearman Correlation')
            ax4.set_title('Spearman相关系数')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim([0, 1])
        
        # RMSE
        if self.valid_metrics['epoch']:
            ax5 = axes[1, 1]
            ax5.plot(self.valid_metrics['epoch'], self.valid_metrics['rmse'], 
                    color='brown', linewidth=2, marker='>', markersize=4)
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('RMSE')
            ax5.set_title('均方根误差')
            ax5.grid(True, alpha=0.3)
        
        # MAE
        if self.valid_metrics['epoch']:
            ax6 = axes[1, 2]
            ax6.plot(self.valid_metrics['epoch'], self.valid_metrics['mae'], 
                    color='pink', linewidth=2, marker='<', markersize=4)
            ax6.set_xlabel('Epoch')
            ax6.set_ylabel('MAE')
            ax6.set_title('平均绝对误差')
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'training_curves.png'), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.plots_dir, 'training_curves.pdf'), 
                   bbox_inches='tight')
        plt.close()
        
    def plot_metrics_summary(self):
        """绘制指标汇总图"""
        if not self.valid_metrics['epoch']:
            return
            
        # 创建指标汇总图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.model_name} 验证指标汇总', fontsize=14, fontweight='bold')
        
        epochs = self.valid_metrics['epoch']
        
        # 相关系数对比
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.valid_metrics['pearson'], label='Pearson', 
                linewidth=2, marker='o', markersize=3)
        ax1.plot(epochs, self.valid_metrics['spearman'], label='Spearman', 
                linewidth=2, marker='s', markersize=3)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Correlation')
        ax1.set_title('相关系数对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # 误差指标对比 (标准化)
        ax2 = axes[0, 1]
        rmse_norm = np.array(self.valid_metrics['rmse']) / np.max(self.valid_metrics['rmse'])
        mae_norm = np.array(self.valid_metrics['mae']) / np.max(self.valid_metrics['mae'])
        ax2.plot(epochs, rmse_norm, label='RMSE (normalized)', 
                linewidth=2, marker='^', markersize=3)
        ax2.plot(epochs, mae_norm, label='MAE (normalized)', 
                linewidth=2, marker='v', markersize=3)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Normalized Error')
        ax2.set_title('误差指标对比 (标准化)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 最佳指标显示
        ax3 = axes[1, 0]
        best_metrics = {
            'Best Pearson': f"{np.max(self.valid_metrics['pearson']):.4f}",
            'Best Spearman': f"{np.max(self.valid_metrics['spearman']):.4f}",
            'Best RMSE': f"{np.min(self.valid_metrics['rmse']):.4f}",
            'Best MAE': f"{np.min(self.valid_metrics['mae']):.4f}",
            'Best Loss': f"{np.min(self.valid_metrics['loss']):.4f}"
        }
        
        y_pos = np.arange(len(best_metrics))
        colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'lavender']
        
        for i, (metric, value) in enumerate(best_metrics.items()):
            ax3.barh(i, 1, color=colors[i], alpha=0.7)
            ax3.text(0.5, i, f"{metric}: {value}", ha='center', va='center', 
                    fontweight='bold', fontsize=10)
        
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(best_metrics.keys())
        ax3.set_xlim(0, 1)
        ax3.set_title('最佳验证指标')
        ax3.set_xticks([])
        
        # 训练进度总结
        ax4 = axes[1, 1]
        training_time = (time.time() - self.start_time) / 60
        total_epochs = len(self.train_metrics['epoch'])
        avg_time_per_epoch = training_time / max(total_epochs, 1)
        
        summary_text = f"""
训练总结:
• 总训练时间: {training_time:.1f} 分钟
• 训练轮数: {total_epochs}
• 平均每轮时间: {avg_time_per_epoch:.2f} 分钟
• 最佳验证损失: {np.min(self.valid_metrics['loss']):.4f}
• 最佳Pearson: {np.max(self.valid_metrics['pearson']):.4f}
• 最佳Spearman: {np.max(self.valid_metrics['spearman']):.4f}
"""
        
        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'metrics_summary.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_report(self):
        """生成完整的训练报告"""
        self.save_metrics()
        self.plot_training_curves()
        self.plot_metrics_summary()
        
        print_log(f"训练报告已生成:")
        print_log(f"  - 指标数据: {self.metrics_dir}")
        print_log(f"  - 可视化图表: {self.plots_dir}")
        
        return self.test_metrics if self.test_metrics else None


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


def parse():
    parser = argparse.ArgumentParser(description='GCN+ESA Training with Monitoring')
    
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
    
    # Monitoring
    parser.add_argument('--model_name', type=str, default='GCN+ESA')
    parser.add_argument('--plot_interval', type=int, default=5, 
                       help='Plot training curves every N epochs')
    
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
    
    # 创建监控器
    monitor = TrainingMonitor(args.save_dir, args.model_name)
    
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
        current_lr = optimizer.param_groups[0]['lr']
        
        print_log(f'Train Loss: {train_loss:.4f}, LR: {current_lr:.2e}')
        
        # 记录训练指标
        monitor.log_train_metrics(epoch + 1, train_loss, current_lr)
        
        # Validate
        if valid_loader:
            valid_metrics = evaluate(model, valid_loader, device)
            print_log(f'Valid Loss: {valid_metrics["loss"]:.4f}, '
                     f'Pearson: {valid_metrics["pearson"]:.4f}, '
                     f'Spearman: {valid_metrics["spearman"]:.4f}, '
                     f'RMSE: {valid_metrics["rmse"]:.4f}')
            
            # 记录验证指标
            monitor.log_valid_metrics(epoch + 1, valid_metrics)
            
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
                
            # 定期生成中间报告
            if (epoch + 1) % args.plot_interval == 0:
                monitor.plot_training_curves()
                print_log(f'Updated training curves at epoch {epoch + 1}')
                
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
        
        # 记录测试指标
        monitor.log_test_metrics(test_metrics)
    
    # 生成完整报告
    final_metrics = monitor.generate_report()
    print_log("训练完成！完整的监控报告已生成。")


if __name__ == '__main__':
    main()
