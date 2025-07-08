#!/usr/bin/env python3

import subprocess
import sys
import os

def main():
    
    # 构建训练命令 - 使用更保守的参数
    cmd = [
        sys.executable, "train.py",
        "--train_set", "./datasets/PDBBind/processed/identity30/train.pkl",
        "--valid_set", "./datasets/PDBBind/processed/identity30/valid.pkl", 
        "--save_dir", "./datasets/PDBBind/processed/identity30/models/MACE_Neo",
        "--task", "PDBBind",
        "--lr", "0.0001",
        "--final_lr", "0.0000001", 
        "--max_epoch", "50",
        "--save_topk", "5",
        "--max_n_vertex_per_gpu", "1000",
        "--grad_clip", "0.5",
        "--warmup", "2000",
        "--shuffle",
        "--model_type", "MACE",
        "--hidden_size", "64",
        "--n_layers", "3",
        "--n_channel", "1",
        "--n_rbf", "16",
        "--cutoff", "6.0",
        "--radial_size", "32",
        "--seed", "2023",
        "--gpus", "0"
    ]
    
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
