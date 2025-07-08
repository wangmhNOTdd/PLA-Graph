#!/usr/bin/env python3
"""
启动SchNet模型训练 - 标准PDBBind数据集 (更稳定的替代方案)
"""

import subprocess
import sys
import os

def main():
    print("🚀 启动SchNet模型训练...")
    print("=" * 50)
    
    # 构建训练命令
    cmd = [
        sys.executable, "train.py",
        "--train_set", "./datasets/PDBBind/processed/identity30/train.pkl",
        "--valid_set", "./datasets/PDBBind/processed/identity30/valid.pkl", 
        "--save_dir", "./datasets/PDBBind/processed/identity30/models/SchNet_standard",
        "--task", "PDBBind",
        "--lr", "0.0001",
        "--final_lr", "0.00001", 
        "--max_epoch", "100",
        "--save_topk", "3",
        "--max_n_vertex_per_gpu", "1500",
        "--grad_clip", "1.0",
        "--warmup", "1000",
        "--shuffle",
        "--model_type", "SchNet",
        "--hidden_size", "128",
        "--n_layers", "3",
        "--n_channel", "1", 
        "--n_rbf", "32",
        "--cutoff", "7.0",
        "--radial_size", "64",
        "--seed", "2023",
        "--gpus", "0"
    ]
    
    print("📋 SchNet训练配置:")
    print(f"   模型: SchNet (稳定的经典模型)")
    print(f"   数据集: 标准PDBBind identity30")
    print(f"   训练集: 3507 样本")
    print(f"   学习率: 0.0001 → 0.00001")
    print(f"   层数: 3")
    print(f"   隐藏维度: 128")
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
