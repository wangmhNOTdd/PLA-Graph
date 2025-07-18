#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
一键运行训练集前三个复合物可视化
"""
import os
import sys
import subprocess

def run_visualization():
    """运行可视化"""
    print("=" * 60)
    print("GET项目 - 训练集前三个复合物可视化")
    print("=" * 60)
    
    # 检查环境
    print("\n1. 检查环境...")
    result = subprocess.run([sys.executable, "test_train_visualization.py"], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ 环境检查失败:")
        print(result.stdout)
        print(result.stderr)
        return False
    
    print("✓ 环境检查通过")
    
    # 询问用户选择
    print("\n2. 选择运行模式:")
    print("   1 - 快速演示 (仅显示第一个复合物，不保存图片)")
    print("   2 - 完整可视化 (保存前三个复合物的图片)")
    
    while True:
        choice = input("\n请选择 (1 或 2): ").strip()
        if choice in ['1', '2']:
            break
        print("❌ 无效选择，请输入 1 或 2")
    
    # 运行可视化
    print(f"\n3. 开始可视化...")
    
    if choice == '1':
        print("运行快速演示...")
        result = subprocess.run([sys.executable, "visualize_pdbbind_example.py", "--demo"])
    else:
        print("运行完整可视化...")
        result = subprocess.run([sys.executable, "visualize_pdbbind_example.py"])
    
    if result.returncode == 0:
        print("\n🎉 可视化完成！")
        if choice == '2':
            print("图片已保存到: ./visualization_output/train_first_3/")
    else:
        print("\n❌ 可视化失败")
        return False
    
    return True

if __name__ == "__main__":
    success = run_visualization()
    if not success:
        print("\n请检查错误信息并重试")
        sys.exit(1)
