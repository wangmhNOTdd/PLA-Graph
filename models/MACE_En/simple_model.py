#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
简化的MACE-En模型实现
"""

import torch
import torch.nn as nn
from e3nn.util.jit import compile_mode

from .modules.enhanced_radial import EnhancedBesselBasis
from ..MACE.modules.blocks import ScaleShiftBlock
from ..MACE.modules.tools.scatter import scatter_sum


@compile_mode("script")
class SimpleMACEEn(torch.nn.Module):
    """
    简化版MACE-En模型，专注于数值稳定性
    """
    
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        hidden_size: int,
        num_elements: int,
        atomic_inter_scale: float = 1.0,
        atomic_inter_shift: float = 0.0,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.r_max = r_max
        
        # Enhanced radial embedding
        self.radial_embedding = EnhancedBesselBasis(
            r_max=r_max, 
            num_basis=num_bessel,
            eps=1e-8
        )
        
        # Simple node embedding
        self.node_embedding = nn.Linear(num_elements, hidden_size)
        
        # Message passing layers
        self.message_mlp = nn.Sequential(
            nn.Linear(num_bessel + hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(), 
            nn.Linear(hidden_size, hidden_size),
        )
        
        # Energy prediction
        self.energy_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        
        # Scale and shift
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, 
            shift=atomic_inter_shift
        )
        
    def forward(self, data):
        """
        前向传播
        
        Args:
            data: 包含以下键的字典
                - node_attrs: [n_nodes, num_elements] 
                - positions: [n_nodes, 3]
                - edge_index: [2, n_edges]
                - batch: [n_nodes]
                - num_graphs: int
        """
        
        node_attrs = data["node_attrs"]
        positions = data["positions"] 
        edge_index = data["edge_index"]
        batch = data["batch"]
        num_graphs = data["num_graphs"]
        
        # 节点嵌入
        h = self.node_embedding(node_attrs)  # [n_nodes, hidden_size]
        
        # 计算边长度
        edge_vectors = positions[edge_index[1]] - positions[edge_index[0]]
        edge_lengths = torch.norm(edge_vectors, dim=-1, keepdim=True)  # [n_edges, 1]
        
        # 径向特征 (使用增强的数值稳定性)
        edge_attr = self.radial_embedding(edge_lengths)  # [n_edges, num_bessel]
        
        # 确保edge_attr是2D的 - 更完整的维度处理
        while edge_attr.dim() > 2:
            edge_attr = edge_attr.squeeze(-1)
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        
        # 消息传递
        src, dst = edge_index
        h_src = h[src]  # [n_edges, hidden_size]
        
        # 组合节点和边特征
        message_input = torch.cat([h_src, edge_attr], dim=-1)  # [n_edges, hidden_size + num_bessel]
        messages = self.message_mlp(message_input)  # [n_edges, hidden_size]
        
        # 聚合消息
        h_agg = scatter_sum(messages, dst, dim=0, dim_size=h.size(0))  # [n_nodes, hidden_size]
        
        # 更新节点特征
        update_input = torch.cat([h, h_agg], dim=-1)  # [n_nodes, hidden_size * 2]
        h = self.update_mlp(update_input)  # [n_nodes, hidden_size]
        
        # 预测节点能量
        node_energies = self.energy_mlp(h).squeeze(-1)  # [n_nodes]
        
        # 聚合为图能量
        graph_energies = scatter_sum(
            node_energies, batch, dim=0, dim_size=num_graphs
        )  # [num_graphs]
        
        # 应用scale和shift
        graph_energies = self.scale_shift(graph_energies)
        
        return {"energy": graph_energies}


# 为了兼容性，创建别名
ScaleShiftMACEEn = SimpleMACEEn
