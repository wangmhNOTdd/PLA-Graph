#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
PDBbind Identity30 训练集前三个复合物可视化
使用方法：
python visualize_pdbbind_example.py        # 可视化训练集前三个复合物并保存
python visualize_pdbbind_example.py --demo # 快速演示训练集第一个复合物（不保存）
"""
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 添加项目路径
PROJ_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJ_DIR)

from visualize_graph import GraphVisualizer
from utils.logger import print_log

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def main():
    """仅可视化训练集的前三个复合物"""
    
    # 只关注训练集
    train_dataset_path = './datasets/PDBBind/processed/identity30/train.pkl'
    
    # 检查训练集文件是否存在
    if not os.path.exists(train_dataset_path):
        print_log(f"Training dataset not found: {train_dataset_path}")
        print_log("Please check the path and ensure the dataset is processed.")
        return
    
    print_log(f"Found training dataset: {train_dataset_path}")
    
    # 创建可视化器
    visualizer = GraphVisualizer(figsize=(15, 10))
    
    # 创建输出目录
    output_dir = './visualization_output/train_first_3'
    os.makedirs(output_dir, exist_ok=True)
    
    print_log(f"\n{'='*60}")
    print_log(f"可视化训练集前三个复合物")
    print_log(f"{'='*60}")
    
    try:
        # 获取训练集大小
        dataset_size = get_dataset_size(train_dataset_path)
        print_log(f"Training dataset size: {dataset_size}")
        
        # 仅可视化前3个样本
        max_samples = min(3, dataset_size)
        print_log(f"Will visualize {max_samples} samples from training set")
        
        for sample_idx in range(max_samples):
            print_log(f"\n{'='*40}")
            print_log(f"正在可视化训练集样本 {sample_idx + 1}/{max_samples}")
            print_log(f"{'='*40}")
            
            # 执行可视化
            visualizer.visualize_complete(
                dataset_path=train_dataset_path,
                index=sample_idx,
                k_neighbors=9,  # 使用默认的k值
                output_dir=output_dir
            )
            
            print_log(f"✓ 样本 {sample_idx + 1} 可视化完成")
            
    except Exception as e:
        print_log(f"Error visualizing training dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print_log(f"\n{'='*60}")
    print_log(f"所有可视化完成！结果保存在: {output_dir}")
    print_log(f"{'='*60}")
    print_log("\n生成的文件:")
    print_log("  - *_3d_structure.png: 3D分子结构图")
    print_log("  - *_2d_graph.png: 2D图结构和统计信息")
    print_log("  - *_connectivity_matrix.png: 块连接矩阵热图")


def get_dataset_size(dataset_path):
    """获取数据集大小"""
    try:
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, list):
            return len(data)
        elif hasattr(data, '__len__'):
            return len(data)
        else:
            return 1
    except Exception as e:
        print_log(f"Error getting dataset size: {e}")
        return 1


def quick_demo():
    """快速演示 - 可视化训练集第一个样本"""
    print_log("Running quick demo for training set...")
    
    # 只关注训练集
    train_dataset_path = './datasets/PDBBind/processed/identity30/train.pkl'
    
    if not os.path.exists(train_dataset_path):
        print_log(f"Training dataset not found: {train_dataset_path}")
        print_log("Please check the path and ensure the dataset is processed.")
        return
    
    visualizer = GraphVisualizer()
    print_log(f"Visualizing first sample from training set: {train_dataset_path}")
    
    # 仅可视化第一个样本，不保存
    visualizer.visualize_complete(
        dataset_path=train_dataset_path,
        index=0,
        k_neighbors=9,
        output_dir=None
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PDBbind Identity30 训练集前三个复合物可视化')
    parser.add_argument('--demo', action='store_true', 
                       help='快速演示模式 (仅可视化训练集第一个样本，不保存图片)')
    
    args = parser.parse_args()
    
    if args.demo:
        quick_demo()
    else:
        main()
