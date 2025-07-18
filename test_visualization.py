#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
快速测试可视化功能
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 添加项目路径
PROJ_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJ_DIR)

from utils.logger import print_log

def test_visualization_imports():
    """测试可视化相关导入"""
    print_log("Testing visualization imports...")
    
    try:
        import matplotlib.pyplot as plt
        print_log("✓ matplotlib imported successfully")
        
        import seaborn as sns
        print_log("✓ seaborn imported successfully")
        
        import networkx as nx
        print_log("✓ networkx imported successfully")
        
        import numpy as np
        print_log("✓ numpy imported successfully")
        
        import pandas as pd
        print_log("✓ pandas imported successfully")
        
        from visualize_graph import GraphVisualizer
        print_log("✓ GraphVisualizer imported successfully")
        
        return True
        
    except ImportError as e:
        print_log(f"✗ Import error: {e}")
        return False

def create_test_data():
    """创建测试数据"""
    print_log("Creating test data...")
    
    # 创建一个简单的测试数据结构
    test_data = {
        'id': 'test_molecule',
        'affinity': 7.5,
        'X': np.random.randn(20, 3) * 5,  # 20个原子的3D坐标
        'B': [0, 1, 2, 3, 4, 5, 6, 7],  # 8个块
        'A': list(range(20)),  # 20个原子
        'block_lengths': [3, 2, 3, 2, 3, 2, 3, 2],  # 每个块的原子数量
        'segment_ids': [0, 0, 1, 1, 0, 0, 1, 1],  # 段ID
        'atom_positions': list(range(20))
    }
    
    return test_data

def test_visualization_functionality():
    """测试可视化功能"""
    print_log("Testing visualization functionality...")
    
    try:
        from visualize_graph import GraphVisualizer
        
        # 创建可视化器
        visualizer = GraphVisualizer(figsize=(10, 6))
        print_log("✓ GraphVisualizer created successfully")
        
        # 创建测试数据
        test_data = create_test_data()
        print_log("✓ Test data created successfully")
        
        # 解析数据
        info = visualizer.parse_sample_data(test_data)
        print_log(f"✓ Sample data parsed: {info['n_blocks']} blocks, {info['n_atoms']} atoms")
        
        # 创建图结构
        graph = visualizer.create_graph_structure(test_data, k_neighbors=5)
        print_log(f"✓ Graph structure created: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # 测试3D可视化
        print_log("Testing 3D visualization...")
        plt.figure(figsize=(8, 6))
        visualizer.plot_3d_structure(test_data)
        plt.close()
        print_log("✓ 3D visualization test passed")
        
        # 测试2D可视化
        print_log("Testing 2D visualization...")
        plt.figure(figsize=(12, 6))
        visualizer.plot_2d_graph(test_data, k_neighbors=5)
        plt.close()
        print_log("✓ 2D visualization test passed")
        
        # 测试连接矩阵可视化
        print_log("Testing connectivity matrix visualization...")
        plt.figure(figsize=(8, 6))
        visualizer.plot_connectivity_matrix(test_data, k_neighbors=5)
        plt.close()
        print_log("✓ Connectivity matrix visualization test passed")
        
        return True
        
    except Exception as e:
        print_log(f"✗ Visualization test failed: {e}")
        return False

def test_file_existence():
    """测试文件是否存在"""
    print_log("Testing file existence...")
    
    files_to_check = [
        'visualize_graph.py',
        'visualize_pdbbind_example.py',
        'molecular_graph_visualization.ipynb',
        'VISUALIZATION_README.md'
    ]
    
    all_exist = True
    for file in files_to_check:
        if os.path.exists(file):
            print_log(f"✓ {file} exists")
        else:
            print_log(f"✗ {file} not found")
            all_exist = False
    
    return all_exist

def show_usage_examples():
    """显示使用示例"""
    print_log("\n" + "="*60)
    print_log("可视化工具使用示例")
    print_log("="*60)
    
    print_log("\n1. 命令行基本使用:")
    print_log("   python visualize_graph.py --dataset path/to/dataset.pkl --index 0")
    
    print_log("\n2. 快速演示:")
    print_log("   python visualize_pdbbind_example.py --demo")
    
    print_log("\n3. 完整可视化:")
    print_log("   python visualize_pdbbind_example.py")
    
    print_log("\n4. Jupyter Notebook:")
    print_log("   jupyter notebook molecular_graph_visualization.ipynb")
    
    print_log("\n5. 获取帮助:")
    print_log("   python visualize_graph.py --help")
    
    print_log("\n6. 指定输出目录:")
    print_log("   python visualize_graph.py --dataset path/to/dataset.pkl --output_dir ./output")
    
    print_log("\n" + "="*60)

def main():
    """主函数"""
    print_log("GET项目可视化工具测试")
    print_log("="*50)
    
    # 测试导入
    if not test_visualization_imports():
        print_log("❌ 导入测试失败")
        return
    
    # 测试文件存在性
    if not test_file_existence():
        print_log("❌ 文件检查失败")
        return
    
    # 测试可视化功能
    if not test_visualization_functionality():
        print_log("❌ 可视化功能测试失败")
        return
    
    print_log("\n" + "="*50)
    print_log("✅ 所有测试通过!")
    print_log("可视化工具已成功安装和配置")
    print_log("="*50)
    
    # 显示使用示例
    show_usage_examples()

if __name__ == "__main__":
    main()
