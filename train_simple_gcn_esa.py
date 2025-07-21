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
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pickle

from utils.logger import print_log
from utils.random_seed import setup_seed, SEED

from data.dataset import PDBBindBenchmark
import trainers
from utils.nn_utils import count_parameters
from data.pdb_utils import VOCAB


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
        fig.suptitle('GCN+ESA 训练监控曲线', fontsize=16, fontweight='bold')
        
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
            ax.plot(valid_epochs, self.history['valid_spearman'], 'purple', label='验证集', linewidth=2)
        if 'test_spearman' in self.history and len(self.history['test_spearman']) > 0:
            test_data = [x for x in self.history['test_spearman'] if x is not None]
            if test_data:
                test_epochs = [epochs[i] for i in range(len(epochs)) if i < len(self.history['test_spearman']) and self.history['test_spearman'][i] is not None]
                ax.plot(test_epochs, test_data, 'brown', label='测试集', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Spearman Correlation')
        ax.set_title('Spearman 相关系数')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. RMSE
        ax = axes[1, 0]
        if 'valid_rmse' in self.history and len(self.history['valid_rmse']) > 0:
            valid_epochs = epochs[:len(self.history['valid_rmse'])]
            ax.plot(valid_epochs, self.history['valid_rmse'], 'red', label='验证集', linewidth=2)
        if 'test_rmse' in self.history and len(self.history['test_rmse']) > 0:
            test_data = [x for x in self.history['test_rmse'] if x is not None]
            if test_data:
                test_epochs = [epochs[i] for i in range(len(epochs)) if i < len(self.history['test_rmse']) and self.history['test_rmse'][i] is not None]
                ax.plot(test_epochs, test_data, 'darkred', label='测试集', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('均方根误差 (RMSE)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. MAE
        ax = axes[1, 1]
        if 'valid_mae' in self.history and len(self.history['valid_mae']) > 0:
            valid_epochs = epochs[:len(self.history['valid_mae'])]
            ax.plot(valid_epochs, self.history['valid_mae'], 'cyan', label='验证集', linewidth=2)
        if 'test_mae' in self.history and len(self.history['test_mae']) > 0:
            test_data = [x for x in self.history['test_mae'] if x is not None]
            if test_data:
                test_epochs = [epochs[i] for i in range(len(epochs)) if i < len(self.history['test_mae']) and self.history['test_mae'][i] is not None]
                ax.plot(test_epochs, test_data, 'darkcyan', label='测试集', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE')
        ax.set_title('平均绝对误差 (MAE)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. 学习率曲线（如果有记录的话）
        ax = axes[1, 2]
        if 'learning_rate' in self.history:
            ax.plot(epochs, self.history['learning_rate'], 'black', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('学习率变化')
            ax.set_yscale('log')
        else:
            # 显示最佳性能总结
            if 'valid_pearson' in self.history:
                best_epoch = np.argmax(self.history['valid_pearson'])
                best_pearson = self.history['valid_pearson'][best_epoch]
                best_spearman = self.history['valid_spearman'][best_epoch] if 'valid_spearman' in self.history else 0
                best_rmse = self.history['valid_rmse'][best_epoch] if 'valid_rmse' in self.history else 0
                
                ax.text(0.1, 0.8, f'最佳验证性能:', fontsize=14, fontweight='bold', transform=ax.transAxes)
                ax.text(0.1, 0.7, f'Epoch: {epochs[best_epoch]}', fontsize=12, transform=ax.transAxes)
                ax.text(0.1, 0.6, f'Pearson: {best_pearson:.4f}', fontsize=12, transform=ax.transAxes)
                ax.text(0.1, 0.5, f'Spearman: {best_spearman:.4f}', fontsize=12, transform=ax.transAxes)
                ax.text(0.1, 0.4, f'RMSE: {best_rmse:.4f}', fontsize=12, transform=ax.transAxes)
                
                # 最终测试性能
                if 'final_test_pearson' in self.history:
                    final_test_pearson = self.history['final_test_pearson']
                    final_test_spearman = self.history.get('final_test_spearman', 0)
                    final_test_rmse = self.history.get('final_test_rmse', 0)
                    
                    ax.text(0.1, 0.2, f'最终测试性能:', fontsize=14, fontweight='bold', transform=ax.transAxes)
                    ax.text(0.1, 0.1, f'Pearson: {final_test_pearson:.4f}', fontsize=12, transform=ax.transAxes)
                    ax.text(0.1, 0.05, f'Spearman: {final_test_spearman:.4f}', fontsize=12, transform=ax.transAxes)
                    ax.text(0.1, 0.0, f'RMSE: {final_test_rmse:.4f}', fontsize=12, transform=ax.transAxes)
            
            ax.set_title('性能总结')
            ax.set_xticks([])
            ax.set_yticks([])
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.save_dir, 'training_curves.pdf'), bbox_inches='tight')
        print_log(f"训练曲线已保存到: {os.path.join(self.save_dir, 'training_curves.png')}")
        
        # 显示图表
        try:
            plt.show()
        except:
            print_log("无法显示图表，但已保存到文件")
        
        plt.close()
    
    def print_summary(self):
        """打印训练总结"""
        if not self.history['epoch']:
            return
            
        print_log("\n" + "="*50)
        print_log("训练总结")
        print_log("="*50)
        
        total_epochs = len(self.history['epoch'])
        print_log(f"总训练轮数: {total_epochs}")
        
        if 'valid_pearson' in self.history:
            best_epoch = np.argmax(self.history['valid_pearson'])
            print_log(f"\n最佳验证性能 (Epoch {self.history['epoch'][best_epoch]}):")
            print_log(f"  Pearson: {self.history['valid_pearson'][best_epoch]:.4f}")
            if 'valid_spearman' in self.history:
                print_log(f"  Spearman: {self.history['valid_spearman'][best_epoch]:.4f}")
            if 'valid_rmse' in self.history:
                print_log(f"  RMSE: {self.history['valid_rmse'][best_epoch]:.4f}")
            if 'valid_mae' in self.history:
                print_log(f"  MAE: {self.history['valid_mae'][best_epoch]:.4f}")
        
        if 'final_test_pearson' in self.history:
            print_log(f"\n最终测试性能:")
            print_log(f"  Pearson: {self.history['final_test_pearson']:.4f}")
            if 'final_test_spearman' in self.history:
                print_log(f"  Spearman: {self.history['final_test_spearman']:.4f}")
            if 'final_test_rmse' in self.history:
                print_log(f"  RMSE: {self.history['final_test_rmse']:.4f}")
            if 'final_test_mae' in self.history:
                print_log(f"  MAE: {self.history['final_test_mae']:.4f}")
        
        print_log("="*50)


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
            n_atoms_in_batch = sum(block_lengths[atom_start:atom_start + block_length.item()])
            atom_batch_id.extend([i] * n_atoms_in_batch)
            atom_start += block_length.item()
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
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize training monitor
    monitor = TrainingMonitor(args.save_dir)
    
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
    
    # Save training configuration
    config = vars(args)
    config['model_parameters'] = count_parameters(model)
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Training loop
    best_valid_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.max_epoch):
        print_log(f'Epoch {epoch + 1}/{args.max_epoch}')
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print_log(f'Train Loss: {train_loss:.4f}')
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Validate
        valid_metrics = None
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
        
        # Test (optional during training for monitoring)
        test_metrics = None
        if test_loader and (epoch + 1) % 5 == 0:  # Test every 5 epochs
            test_metrics = evaluate(model, test_loader, device)
            print_log(f'Test (Epoch {epoch+1}) - Pearson: {test_metrics["pearson"]:.4f}, '
                     f'Spearman: {test_metrics["spearman"]:.4f}, RMSE: {test_metrics["rmse"]:.4f}')
        
        # Log metrics to monitor
        monitor.log_epoch(epoch + 1, train_loss, valid_metrics, test_metrics)
        
        # Record learning rate
        monitor.history['learning_rate'].append(current_lr)
    
    # Final test evaluation
    if test_loader:
        model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth')))
        final_test_metrics = evaluate(model, test_loader, device)
        print_log(f'Final Test Results - Loss: {final_test_metrics["loss"]:.4f}, '
                 f'Pearson: {final_test_metrics["pearson"]:.4f}, '
                 f'Spearman: {final_test_metrics["spearman"]:.4f}, '
                 f'RMSE: {final_test_metrics["rmse"]:.4f}')
        
        # 单独记录最终测试结果，不添加到epoch历史中
        monitor.history['final_test_pearson'] = final_test_metrics["pearson"]
        monitor.history['final_test_spearman'] = final_test_metrics["spearman"]
        monitor.history['final_test_rmse'] = final_test_metrics["rmse"]
        monitor.history['final_test_mae'] = final_test_metrics["mae"]
    
    # Save training history and generate plots
    monitor.save_history()
    monitor.plot_training_curves()
    monitor.print_summary()
    
    print_log(f"训练完成! 所有结果保存在: {args.save_dir}")


if __name__ == '__main__':
    main()
