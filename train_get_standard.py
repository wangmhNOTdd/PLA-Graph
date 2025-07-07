#!/usr/bin/env python3
"""
启动GET标准配置训练 - v2020-other-PL + CASF-2016数据集
"""

import subprocess
import sys
import os

def main():
    print("🚀 启动GET标准配置训练...")
    print("=" * 50)
    
    # 配置文件路径
    config_file = "get_v2020_standard_config.json"
    
    # 构建训练命令
    cmd = [
        sys.executable, "train.py",
        "--train_set", "./datasets/v2020-other-PL/processed_get_format/train.pkl",
        "--valid_set", "./datasets/v2020-other-PL/processed_get_format/valid.pkl", 
        "--save_dir", "./datasets/v2020-other-PL/processed_get_format/models/GET_standard",
        "--task", "PDBBind",
        "--lr", "0.0001",
        "--final_lr", "0.0001",
        "--max_epoch", "100",
        "--save_topk", "3",
        "--max_n_vertex_per_gpu", "1500",
        "--shuffle",
        "--model_type", "GET",
        "--hidden_size", "128",
        "--n_layers", "3",
        "--n_channel", "1", 
        "--n_rbf", "32",
        "--cutoff", "7.0",
        "--radial_size", "64",
        "--k_neighbors", "9",
        "--n_head", "4",
        "--seed", "2023",
        "--gpus", "0"
    ]
    
    print("📋 训练配置:")
    print(f"   模型: GET (标准配置)")
    print(f"   层数: 3 (轻量化)")
    print(f"   学习率: 0.0001")
    print(f"   训练集: 12245 样本")
    print(f"   验证集: 1360 样本") 
    print(f"   最大epoch: 100")
    print(f"   截断距离: 7.0")
    print(f"   邻居数: 9")
    print("=" * 50)
    
    # 执行训练
    print("🏃‍♂️ 开始训练...")
    try:
        subprocess.run(cmd, check=True)
        print("✅ 训练完成!")
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")
        return 1
    except KeyboardInterrupt:
        print("⛔ 训练被用户中断")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
