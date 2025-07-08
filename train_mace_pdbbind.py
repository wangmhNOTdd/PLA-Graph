#!/usr/bin/env python3
"""
启动MACE模型训练 - 标准PDBBind数据集 (identity30)
"""

import subprocess
import sys
import os

def main():
    print("🚀 启动MACE模型训练...")
    print("=" * 50)
    
    # 构建训练命令
    cmd = [
        sys.executable, "train.py",
        "--train_set", "./datasets/PDBBind/processed/identity30/train.pkl",
        "--valid_set", "./datasets/PDBBind/processed/identity30/valid.pkl", 
        "--save_dir", "./datasets/PDBBind/processed/identity30/models/MACE_standard",
        "--task", "PDBBind",
        "--lr", "0.0001",
        "--final_lr", "0.00001", 
        "--max_epoch", "50",
        "--save_topk", "3",
        "--max_n_vertex_per_gpu", "500",
        "--grad_clip", "1.0",
        "--warmup", "1000",
        "--shuffle",
        "--model_type", "MACE",
        "--hidden_size", "64",
        "--n_layers", "3",
        "--n_channel", "1", 
        "--n_rbf", "32",
        "--cutoff", "7.0",
        "--radial_size", "32",
        "--seed", "2023",
        "--gpus", "0"
    ]
    
    print("=" * 50)
    
    # 执行训练
    print("开始训练...")
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
