#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
DSAN优化版编码器 - 向量化块处理
保持核心创新点不变，大幅提升计算效率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum, scatter_max
import numpy as np


class VectorizedPMA(nn.Module):
    """向量化的池化多头注意力（PMA）实现"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # 可学习的种子向量S
        self.seed = nn.Parameter(torch.randn(1, 1, dim))
        
        # 注意力权重
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim) 
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x, mask=None):
        """
        标准的PMA前向传播
        Args:
            x: [batch_size, seq_len, dim] - 块内原子特征
            mask: [batch_size, seq_len] - 可选的掩码
        Returns:
            [batch_size, 1, dim] - 块级特征
        """
        batch_size, seq_len, _ = x.size()
        
        # 扩展种子向量
        query = self.seed.expand(batch_size, -1, -1)  # [batch_size, 1, dim]
        
        # 计算Q, K, V
        Q = self.q_proj(query).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 应用掩码（如果提供）
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(~mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, 1, self.dim)
        
        # 输出投影
        out = self.out_proj(out)
        
        return out
    
    def batch_forward(self, batched_features, block_sizes):
        """
        批量处理多个块的PMA
        Args:
            batched_features: [n_blocks, max_block_size, dim] - 预填充的特征
            block_sizes: [n_blocks] - 每个块的实际大小
        Returns:
            [n_blocks, dim] - 块级特征
        """
        n_blocks, max_block_size, dim = batched_features.shape
        
        # 创建掩码矩阵
        mask = torch.arange(max_block_size, device=batched_features.device).unsqueeze(0) < block_sizes.unsqueeze(1)
        
        # 批量计算PMA
        block_features = self.forward(batched_features, mask)  # [n_blocks, 1, dim]
        
        return block_features.squeeze(1)  # [n_blocks, dim]


class BatchedGeometryComputation(nn.Module):
    """批量化的几何特征计算"""
    def __init__(self, rbf_dim=16, cutoff=10.0):
        super().__init__()
        self.rbf_dim = rbf_dim
        self.cutoff = cutoff
        
        # 创建高斯中心
        self.centers = nn.Parameter(torch.linspace(0, cutoff, rbf_dim))
        self.widths = nn.Parameter(torch.ones(rbf_dim) * (cutoff / rbf_dim))
        
    def compute_rbf(self, distances):
        """批量计算RBF特征"""
        distances = distances.unsqueeze(-1)  # [..., 1]
        centers = self.centers.unsqueeze(0)  # [1, rbf_dim]
        widths = self.widths.unsqueeze(0)  # [1, rbf_dim]
        
        # 计算高斯RBF
        rbf = torch.exp(-((distances - centers) ** 2) / (2 * widths ** 2))
        return rbf
    
    def batch_compute_geometry(self, atom_positions, block_id):
        """
        批量计算所有块的几何特征
        Args:
            atom_positions: [n_atoms, 3] 原子坐标
            block_id: [n_atoms] 块ID
        Returns:
            dict: 包含质心、相对位置、距离和RBF特征的字典
        """
        device = atom_positions.device
        unique_blocks = torch.unique(block_id)
        n_blocks = len(unique_blocks)
        
        # 预分配结果张量
        centroids = torch.zeros(n_blocks, 3, device=device)
        block_masks = []
        all_rel_positions = []
        all_distances = []
        all_rbf_features = []
        
        # 批量计算质心
        for i, block_idx in enumerate(unique_blocks):
            mask = (block_id == block_idx)
            block_masks.append(mask)
            
            # 计算质心
            block_positions = atom_positions[mask]
            centroids[i] = torch.mean(block_positions, dim=0)
            
            # 计算相对位置和距离
            rel_pos = block_positions - centroids[i]
            distances = torch.norm(rel_pos, dim=1)
            
            # 计算RBF特征
            rbf_feat = self.compute_rbf(distances)
            
            all_rel_positions.append(rel_pos)
            all_distances.append(distances)
            all_rbf_features.append(rbf_feat)
        
        return {
            'centroids': centroids,
            'block_masks': block_masks,
            'rel_positions': all_rel_positions,
            'distances': all_distances,
            'rbf_features': all_rbf_features,
            'unique_blocks': unique_blocks
        }


class OptimizedGeometryAwareCrossAttention(nn.Module):
    """优化的几何感知交叉注意力模块"""
    def __init__(self, hidden_size, rbf_dim=16, cutoff=10.0, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.rbf_dim = rbf_dim
        
        # 几何计算模块
        self.geometry_computer = BatchedGeometryComputation(rbf_dim, cutoff)
        self.geom_proj = nn.Linear(rbf_dim, hidden_size // 4)
        
        # 交叉注意力组件
        self.cross_attn_q = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_k = nn.Linear(hidden_size + hidden_size // 4, hidden_size)
        self.cross_attn_v = nn.Linear(hidden_size + hidden_size // 4, hidden_size)
        
        # 融合MLP
        self.context_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 原子级前馈网络
        self.atomic_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # 层归一化
        self.ln_atom1 = nn.LayerNorm(hidden_size)
        self.ln_atom2 = nn.LayerNorm(hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, atom_features, atom_positions, block_features, block_id, geometry_cache=None):
        """
        优化的几何感知块内原子更新
        """
        device = atom_features.device
        
        # 使用缓存的几何特征或重新计算
        if geometry_cache is None:
            geometry_cache = self.geometry_computer.batch_compute_geometry(atom_positions, block_id)
        
        updated_atom_features = torch.zeros_like(atom_features)
        
        # 批量处理所有块
        for i, (block_idx, mask) in enumerate(zip(geometry_cache['unique_blocks'], geometry_cache['block_masks'])):
            atom_indices = mask.nonzero(as_tuple=True)[0]
            
            if len(atom_indices) == 0:
                continue
            
            # 获取预计算的几何特征
            block_atom_feat = atom_features[atom_indices]
            rbf_feat = geometry_cache['rbf_features'][i]
            geom_feat = self.geom_proj(rbf_feat)
            
            # 增强原子特征
            augmented_atom_feat = torch.cat([block_atom_feat, geom_feat], dim=1)
            
            # 几何感知的交叉注意力
            Q = self.cross_attn_q(block_features[i].unsqueeze(0))  # [1, hidden_size]
            K = self.cross_attn_k(augmented_atom_feat)  # [n_atoms_in_block, hidden_size]
            V = self.cross_attn_v(augmented_atom_feat)  # [n_atoms_in_block, hidden_size]
            
            # 计算注意力权重
            attention_scores = torch.matmul(Q, K.transpose(0, 1)) / np.sqrt(self.hidden_size)
            attention_weights = F.softmax(attention_scores, dim=1)  # [1, n_atoms_in_block]
            
            # 计算上下文向量
            context_vector = torch.matmul(attention_weights, V)  # [1, hidden_size]
            
            # 生成原子更新
            atomic_update = self.context_mlp(context_vector)  # [1, hidden_size]
            atomic_update = atomic_update.expand(len(atom_indices), -1)  # [n_atoms_in_block, hidden_size]
            
            # 更新原子特征
            updated_atom_feat = self.ln_atom1(block_atom_feat + self.dropout_layer(atomic_update))
            updated_atom_feat = self.ln_atom2(updated_atom_feat + self.dropout_layer(self.atomic_ffn(updated_atom_feat)))
            
            # 将更新后的特征放回正确位置
            updated_atom_features[atom_indices] = updated_atom_feat
        
        return updated_atom_features


class OptimizedDSANLayer(nn.Module):
    """优化的DSAN-v2层实现"""
    def __init__(self, hidden_size, num_heads=8, k_neighbors=9, dropout=0.1, 
                 use_geometry=True, rbf_dim=16, cutoff=10.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.k_neighbors = k_neighbors
        self.dropout = dropout
        self.use_geometry = use_geometry
        self.rbf_dim = rbf_dim
        self.cutoff = cutoff
        
        # 1. 优化的PMA - 块级消息提取
        self.pma = VectorizedPMA(
            dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 2. ESA - 块间信息交换（保持不变）
        from .encoder import ESAModule
        self.esa = ESAModule(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 3. 优化的几何感知交叉注意力
        self.geo_cross_attn = OptimizedGeometryAwareCrossAttention(
            hidden_size=hidden_size,
            rbf_dim=rbf_dim,
            cutoff=cutoff,
            dropout=dropout
        )
        
        # 几何计算模块（共享）
        self.geometry_computer = BatchedGeometryComputation(rbf_dim, cutoff)
        
        # 块级层归一化
        self.ln_block = nn.LayerNorm(hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        
    def vectorized_extract_block_features(self, atom_features, block_id):
        """向量化的块特征提取"""
        device = atom_features.device
        unique_blocks = torch.unique(block_id)
        n_blocks = len(unique_blocks)
        
        # 创建块掩码矩阵 [n_blocks, n_atoms]
        block_masks = block_id.unsqueeze(0) == unique_blocks.unsqueeze(1)
        
        # 计算块大小
        block_sizes = block_masks.sum(dim=1)  # [n_blocks]
        max_block_size = block_sizes.max().item()
        
        if max_block_size == 0:
            return torch.zeros(n_blocks, self.hidden_size, device=device), unique_blocks
        
        # 预分配批量特征张量
        batched_features = torch.zeros(n_blocks, max_block_size, self.hidden_size, 
                                     device=device, dtype=atom_features.dtype)
        
        # 向量化填充特征
        for i, mask in enumerate(block_masks):
            indices = mask.nonzero(as_tuple=True)[0]
            if len(indices) > 0:
                batched_features[i, :len(indices)] = atom_features[indices]
        
        # 批量PMA处理
        block_features = self.pma.batch_forward(batched_features, block_sizes)
        
        return block_features, unique_blocks
        
    def forward(self, atom_features, atom_positions, block_id, inter_edges):
        """
        优化的DSAN-v2层前向传播
        """
        device = atom_features.device
        
        # === 模块1: 向量化的PMA - 块级消息提取 ===
        block_features, unique_blocks = self.vectorized_extract_block_features(atom_features, block_id)
        
        # === 模块2: ESA - 3D感知的块间信息交换 ===
        # 映射全局块ID到局部索引
        block_id_to_local = {unique_blocks[i].item(): i for i in range(len(unique_blocks))}
        
        valid_edges = []
        if inter_edges.size(1) > 0:
            for i in range(inter_edges.size(1)):
                src_global, dst_global = inter_edges[0, i].item(), inter_edges[1, i].item()
                if src_global in block_id_to_local and dst_global in block_id_to_local:
                    src_local = block_id_to_local[src_global]
                    dst_local = block_id_to_local[dst_global]
                    valid_edges.append([src_local, dst_local])
        
        if valid_edges:
            valid_edges = torch.tensor(valid_edges, device=device).T  # [2, n_valid_edges]
            updated_block_features = self.esa(block_features, valid_edges)
            updated_block_features = self.ln_block(block_features + self.dropout_layer(updated_block_features - block_features))
        else:
            updated_block_features = block_features
        
        # === 预计算几何特征（一次性计算） ===
        geometry_cache = self.geometry_computer.batch_compute_geometry(atom_positions, block_id)
        
        # === 模块3: 优化的几何感知块内原子更新 ===
        updated_atom_features = self.geo_cross_attn(
            atom_features, atom_positions, updated_block_features, block_id, geometry_cache
        )
        
        return updated_atom_features


class OptimizedDSANEncoder(nn.Module):
    """优化的DSAN编码器"""
    def __init__(self, hidden_size, n_layers=3, num_heads=8, k_neighbors=9, 
                 dropout=0.1, use_geometry=True, rbf_dim=16, cutoff=10.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.k_neighbors = k_neighbors
        self.dropout = dropout
        self.use_geometry = use_geometry
        self.rbf_dim = rbf_dim
        self.cutoff = cutoff
        
        # 堆叠多个优化的DSAN层
        self.dsan_layers = nn.ModuleList([
            OptimizedDSANLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                k_neighbors=k_neighbors,
                dropout=dropout,
                use_geometry=use_geometry,
                rbf_dim=rbf_dim,
                cutoff=cutoff
            ) for _ in range(n_layers)
        ])
        
        # 全局读出层
        self.global_readout = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None):
        """
        优化的DSAN编码器前向传播
        """
        device = H.device
        
        # 处理坐标：取第一个通道的坐标
        if Z.dim() == 3:
            atom_positions = Z[:, 0, :]  # [N_atoms, 3]
        else:
            atom_positions = Z  # [N_atoms, 3]
        
        # 初始化原子特征
        current_atom_features = H
        
        # 通过多个优化的DSAN层
        for layer in self.dsan_layers:
            current_atom_features = layer(
                current_atom_features, atom_positions, block_id, edges
            )
        
        # 计算块表示
        max_block_id = int(block_id.max().item()) + 1 if len(block_id) > 0 else 1
        block_repr = scatter_sum(current_atom_features, block_id, dim=0, dim_size=max_block_id)
        block_repr = F.normalize(block_repr, dim=-1)
        
        # 计算图表示
        unique_blocks = torch.unique(block_id)
        atom_batch_id = torch.zeros_like(block_id)
        
        for block_idx in unique_blocks:
            atom_mask = (block_id == block_idx)
            if block_idx < len(batch_id):
                atom_batch_id[atom_mask] = batch_id[block_idx]
        
        max_batch_id = int(atom_batch_id.max().item()) + 1 if len(atom_batch_id) > 0 else 1
        graph_repr = scatter_sum(current_atom_features, atom_batch_id, dim=0, dim_size=max_batch_id)
        graph_repr = F.normalize(graph_repr, dim=-1)
        
        # 返回预测坐标（这里简单返回原坐标）
        pred_Z = Z
        
        return current_atom_features, block_repr, graph_repr, pred_Z
