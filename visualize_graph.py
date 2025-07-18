#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
图可视化工具 - 用于可视化GET模型的输入图结构
支持PDBbind identity30数据集的分子复合物图结构可视化
"""
import os
import sys
import json
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from typing import List, Dict, Tuple, Optional
import seaborn as sns

# 添加项目路径
PROJ_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(PROJ_DIR)

from data.dataset import BlockGeoAffDataset, PDBBindBenchmark
from data.pdb_utils import VOCAB
from utils.logger import print_log

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class GraphVisualizer:
    """分子复合物图可视化器"""
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        self.colors = sns.color_palette("husl", 20)  # 为不同类型的节点设置颜色
        
    def load_data(self, dataset_path: str, index: int = 0):
        """加载数据集中的一个样本"""
        print_log(f"Loading dataset from {dataset_path}...")
        
        if dataset_path.endswith('.pkl'):
            with open(dataset_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, list):
                if index >= len(data):
                    raise IndexError(f"Index {index} out of range. Dataset has {len(data)} samples.")
                sample = data[index]
            else:
                # 如果是dataset对象
                if hasattr(data, '__getitem__'):
                    sample = data[index]
                else:
                    raise ValueError("Unknown data format")
        else:
            # 尝试作为dataset处理
            dataset = BlockGeoAffDataset(dataset_path)
            sample = dataset[index]
            
        return sample
    
    def parse_sample_data(self, sample: Dict):
        """解析样本数据结构"""
        data = sample.get('data', {})
        info = {
            'id': sample.get('id', 'Unknown'),
            'affinity': sample.get('affinity', 'Unknown'),
            'n_blocks': len(data.get('B', [])),
            'n_atoms': len(data.get('A', [])),
            'coordinates': data.get('X', np.array([])),
            'block_types': data.get('B', []),
            'atom_types': data.get('A', []),
            'block_lengths': data.get('block_lengths', []),
            'segment_ids': data.get('segment_ids', [])
        }
        return info
    
    def create_graph_structure(self, sample: Dict, k_neighbors: int = 9):
        """创建图结构用于可视化"""
        X = sample['data']['X']  # 原子坐标 [n_atoms, 3]
        B = sample['data']['B']  # 块类型 [n_blocks]
        A = sample['data']['A']  # 原子类型 [n_atoms]
        block_lengths = sample['data']['block_lengths']  # 每个块的原子数量
        segment_ids = sample['data']['segment_ids']  # 每个块属于哪个段(蛋白质/配体)
        
        # 创建NetworkX图
        G = nx.Graph()
        
        # 添加原子节点
        atom_idx = 0
        block_start_idx = 0
        
        for block_idx, (block_type, block_len, segment_id) in enumerate(zip(B, block_lengths, segment_ids)):
            # 添加块节点
            block_center = np.mean(X[atom_idx:atom_idx + block_len], axis=0)
            G.add_node(f"block_{block_idx}", 
                      type='block',
                      block_type=block_type,
                      segment_id=segment_id,
                      pos=block_center,
                      size=block_len * 100)  # 节点大小根据原子数量调整
            
            # 添加原子节点并连接到块节点
            for i in range(block_len):
                atom_id = f"atom_{atom_idx + i}"
                G.add_node(atom_id,
                          type='atom',
                          atom_type=A[atom_idx + i],
                          block_idx=block_idx,
                          segment_id=segment_id,
                          pos=X[atom_idx + i],
                          size=50)
                
                # 原子与块的连接
                G.add_edge(f"block_{block_idx}", atom_id, edge_type='block_atom')
            
            atom_idx += block_len
        
        # 添加k近邻边（块之间的连接）
        block_positions = np.array([G.nodes[f"block_{i}"]['pos'] for i in range(len(B))])
        for i in range(len(B)):
            distances = np.linalg.norm(block_positions - block_positions[i], axis=1)
            # 找到k个最近的邻居（除了自己）
            neighbor_indices = np.argsort(distances)[1:k_neighbors+1]
            for j in neighbor_indices:
                if distances[j] < 10.0:  # 距离阈值
                    G.add_edge(f"block_{i}", f"block_{j}", 
                              edge_type='block_block',
                              distance=distances[j])
        
        return G
    
    def plot_3d_structure(self, sample: Dict, save_path: Optional[str] = None):
        """绘制3D分子结构"""
        info = self.parse_sample_data(sample)
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        X = info['coordinates']
        A = info['atom_types']
        B = info['block_types']
        block_lengths = info['block_lengths']
        segment_ids = info['segment_ids']
        
        # 绘制原子
        atom_idx = 0
        legend_elements = []
        segment_colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple'}
        
        for block_idx, (block_type, block_len, segment_id) in enumerate(zip(B, block_lengths, segment_ids)):
            color = segment_colors.get(segment_id, 'gray')
            
            # 绘制块中的原子
            for i in range(block_len):
                if atom_idx < len(X):
                    pos = X[atom_idx]
                    ax.scatter(pos[0], pos[1], pos[2], 
                             c=color, s=60, alpha=0.7)
                    atom_idx += 1
            
            # 添加块中心点
            if block_len > 0:
                block_center = np.mean(X[atom_idx-block_len:atom_idx], axis=0)
                ax.scatter(block_center[0], block_center[1], block_center[2], 
                          c=color, s=200, alpha=0.5, marker='s')
        
        # 设置图形属性
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(f'3D Molecular Structure\nID: {info["id"]}, Affinity: {info["affinity"]}\n'
                    f'Blocks: {info["n_blocks"]}, Atoms: {info["n_atoms"]}')
        
        # 添加图例
        legend_labels = ['Segment 0', 'Segment 1', 'Segment 2', 'Segment 3']
        for i, label in enumerate(legend_labels):
            if i in segment_ids:
                ax.scatter([], [], [], c=segment_colors[i], s=100, label=label)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print_log(f"3D structure saved to {save_path}")
        
        plt.show()
    
    def plot_2d_graph(self, sample: Dict, k_neighbors: int = 9, save_path: Optional[str] = None):
        """绘制2D图结构"""
        info = self.parse_sample_data(sample)
        G = self.create_graph_structure(sample, k_neighbors)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 左图：显示块级别的连接
        block_nodes = [node for node in G.nodes() if node.startswith('block_')]
        block_subgraph = G.subgraph(block_nodes)
        
        # 使用3D坐标投影到2D
        pos_3d = {node: G.nodes[node]['pos'] for node in block_nodes}
        pos_2d = {node: (pos_3d[node][0], pos_3d[node][1]) for node in block_nodes}
        
        # 根据segment_id着色
        segment_colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple'}
        node_colors = [segment_colors.get(G.nodes[node]['segment_id'], 'gray') for node in block_nodes]
        node_sizes = [G.nodes[node]['size'] for node in block_nodes]
        
        # 绘制块连接图
        nx.draw_networkx_nodes(block_subgraph, pos_2d, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.7, ax=ax1)
        nx.draw_networkx_edges(block_subgraph, pos_2d, alpha=0.5, ax=ax1)
        
        # 添加节点标签
        labels = {node: f"{node.split('_')[1]}" for node in block_nodes}
        nx.draw_networkx_labels(block_subgraph, pos_2d, labels, font_size=8, ax=ax1)
        
        ax1.set_title(f'Block-level Graph\nID: {info["id"]}, Blocks: {info["n_blocks"]}')
        ax1.set_xlabel('X (Å)')
        ax1.set_ylabel('Y (Å)')
        ax1.axis('equal')
        
        # 右图：显示统计信息
        ax2.axis('off')
        
        # 统计信息
        stats_text = f"""
        分子复合物信息:
        ─────────────────────────
        ID: {info["id"]}
        结合亲和力: {info["affinity"]}
        
        图结构统计:
        ─────────────────────────
        块数量: {info["n_blocks"]}
        原子数量: {info["n_atoms"]}
        边数量: {G.number_of_edges()}
        
        段分布:
        ─────────────────────────
        """
        
        # 统计每个segment的块数量
        segment_count = {}
        for segment_id in info['segment_ids']:
            segment_count[segment_id] = segment_count.get(segment_id, 0) + 1
        
        for segment_id, count in segment_count.items():
            stats_text += f"Segment {segment_id}: {count} blocks\n"
        
        # 原子类型统计
        atom_type_count = {}
        for atom_type in info['atom_types']:
            atom_type_count[atom_type] = atom_type_count.get(atom_type, 0) + 1
        
        stats_text += f"\n原子类型分布:\n"
        stats_text += "─────────────────────────\n"
        for atom_type, count in sorted(atom_type_count.items()):
            try:
                atom_symbol = VOCAB.atom_idx_to_symbol(atom_type)
                stats_text += f"{atom_symbol}: {count}\n"
            except:
                stats_text += f"Type {atom_type}: {count}\n"
        
        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print_log(f"2D graph saved to {save_path}")
        
        plt.show()
    
    def plot_connectivity_matrix(self, sample: Dict, k_neighbors: int = 9, save_path: Optional[str] = None):
        """绘制连接矩阵热图"""
        info = self.parse_sample_data(sample)
        G = self.create_graph_structure(sample, k_neighbors)
        
        # 创建邻接矩阵
        block_nodes = [node for node in G.nodes() if node.startswith('block_')]
        block_nodes.sort(key=lambda x: int(x.split('_')[1]))
        
        n_blocks = len(block_nodes)
        adj_matrix = np.zeros((n_blocks, n_blocks))
        
        for i, node1 in enumerate(block_nodes):
            for j, node2 in enumerate(block_nodes):
                if G.has_edge(node1, node2):
                    adj_matrix[i, j] = 1
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 创建热图
        sns.heatmap(adj_matrix, annot=True, fmt='.0f', cmap='Blues', 
                   xticklabels=range(n_blocks), yticklabels=range(n_blocks),
                   ax=ax)
        
        ax.set_title(f'Block Connectivity Matrix\nID: {info["id"]}, {n_blocks} blocks')
        ax.set_xlabel('Block Index')
        ax.set_ylabel('Block Index')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print_log(f"Connectivity matrix saved to {save_path}")
        
        plt.show()
    
    def visualize_complete(self, dataset_path: str, index: int = 0, k_neighbors: int = 9, 
                          output_dir: Optional[str] = None):
        """完整的可视化流程"""
        # 加载数据
        sample = self.load_data(dataset_path, index)
        info = self.parse_sample_data(sample)
        
        print_log(f"Visualizing sample {index}:")
        print_log(f"  ID: {info['id']}")
        print_log(f"  Affinity: {info['affinity']}")
        print_log(f"  Blocks: {info['n_blocks']}, Atoms: {info['n_atoms']}")
        
        # 创建输出目录
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            sample_id = str(info['id']).replace('/', '_')
            prefix = os.path.join(output_dir, f"sample_{index}_{sample_id}")
        else:
            prefix = None
        
        # 3D结构可视化
        save_path_3d = f"{prefix}_3d_structure.png" if prefix else None
        self.plot_3d_structure(sample, save_path_3d)
        
        # 2D图结构可视化
        save_path_2d = f"{prefix}_2d_graph.png" if prefix else None
        self.plot_2d_graph(sample, k_neighbors, save_path_2d)
        
        # 连接矩阵可视化
        save_path_matrix = f"{prefix}_connectivity_matrix.png" if prefix else None
        self.plot_connectivity_matrix(sample, k_neighbors, save_path_matrix)


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize molecular graphs in GET project')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to the dataset file (e.g., train.pkl, valid.pkl, test.pkl)')
    parser.add_argument('--index', type=int, default=0,
                       help='Index of the sample to visualize')
    parser.add_argument('--k_neighbors', type=int, default=9,
                       help='Number of neighbors for k-NN graph construction')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save visualization plots')
    parser.add_argument('--figsize', type=int, nargs=2, default=[15, 10],
                       help='Figure size (width, height)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建可视化器
    visualizer = GraphVisualizer(figsize=tuple(args.figsize))
    
    # 执行可视化
    visualizer.visualize_complete(
        dataset_path=args.dataset,
        index=args.index,
        k_neighbors=args.k_neighbors,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
