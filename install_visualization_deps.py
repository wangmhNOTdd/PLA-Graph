#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
安装可视化所需的依赖包
"""
import subprocess
import sys
import os

def install_package(package):
    """安装Python包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        return False

def main():
    """安装可视化所需的所有依赖"""
    print("Installing visualization dependencies...")
    print("=" * 50)
    
    # 可视化所需的包
    required_packages = [
        "matplotlib",
        "seaborn", 
        "networkx",
        "numpy",
        "pandas",
        "scipy",
        "tqdm",
        "ipywidgets",  # for Jupyter notebook widgets
        "jupyter",     # for Jupyter notebook
    ]
    
    # 检查并安装包
    success_count = 0
    for package in required_packages:
        if install_package(package):
            success_count += 1
    
    print("\n" + "=" * 50)
    print(f"Installation complete: {success_count}/{len(required_packages)} packages installed successfully")
    
    if success_count == len(required_packages):
        print("\n✓ All dependencies installed successfully!")
        print("\nYou can now run the visualization tools:")
        print("  1. python test_train_visualization.py  # 测试环境")
        print("  2. python visualize_pdbbind_example.py --demo  # 快速演示")
        print("  3. python visualize_pdbbind_example.py  # 完整可视化训练集前3个复合物")
        print("  4. jupyter notebook molecular_graph_visualization.ipynb  # 交互式可视化")
    else:
        print(f"\n⚠️ {len(required_packages) - success_count} packages failed to install")
        print("Please check the error messages above and install them manually")

if __name__ == "__main__":
    main()
