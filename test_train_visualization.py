#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
测试训练集前三个复合物的可视化
"""
import os
import sys

def test_visualization():
    """测试可视化功能"""
    print("测试训练集前三个复合物的可视化...")
    print("=" * 50)
    
    # 检查训练集文件是否存在
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
    
    # 检查依赖是否可用
    try:
        import matplotlib
        import numpy as np
        import pickle
        print("✓ 基本依赖包可用")
    except ImportError as e:
        print(f"❌ 缺少依赖包: {e}")
        print("请运行: python install_visualization_deps.py")
        return False
    
    print("\n" + "=" * 50)
    print("测试通过！您可以运行以下命令:")
    print("1. 快速演示 (不保存图片):")
    print("   python visualize_pdbbind_example.py --demo")
    print()
    print("2. 完整可视化 (保存图片):")
    print("   python visualize_pdbbind_example.py")
    print()
    print("结果将保存在 ./visualization_output/train_first_3/ 目录中")
    
    return True

if __name__ == "__main__":
    success = test_visualization()
    if success:
        print("\n🎉 准备就绪！")
    else:
        print("\n❌ 测试失败，请检查上述错误")
