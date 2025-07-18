#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简单的可视化测试脚本
"""

import os
import sys
import pickle

def test_visualization():
    """测试可视化功能"""
    print("测试训练集前三个复合物的可视化...")
    print("=" * 50)
    
    # 检查训练集文件
    train_dataset_path = './datasets/PDBBind/processed/identity30/train.pkl'
    if not os.path.exists(train_dataset_path):
        print(f"ERROR: 训练集文件不存在: {train_dataset_path}")
        print("请确保已经处理过 PDBbind identity30 数据集")
        return False
    
    print(f"OK: 找到训练集文件: {train_dataset_path}")
    
    # 检查可视化脚本是否存在
    viz_script = './visualize_pdbbind_example.py'
    if not os.path.exists(viz_script):
        print(f"ERROR: 可视化脚本不存在: {viz_script}")
        return False
    
    print(f"OK: 找到可视化脚本: {viz_script}")
    
    # 尝试加载前三个样本
    try:
        with open(train_dataset_path, 'rb') as f:
            train_data = pickle.load(f)
        
        print(f"OK: 成功加载训练集，共有 {len(train_data)} 个样本")
        
        # 检查前三个样本
        for i in range(min(3, len(train_data))):
            sample = train_data[i]
            print(f"  样本 {i+1}: type={type(sample)}, keys={list(sample.keys()) if isinstance(sample, dict) else 'N/A'}")
            
        print("OK: 训练集前三个样本检查完成")
        return True
        
    except Exception as e:
        print(f"ERROR: 无法加载训练集: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_visualization()
    if success:
        print("\n所有检查通过！可以开始可视化。")
    else:
        print("\n检查失败，请修复问题后重试。")
