#!/usr/bin/env python3

import subprocess
import sys
import os

def main():
    cmd = [
        sys.executable, "train.py",
        "--train_set", "./datasets/PDBBind/processed/identity30/train.pkl",
        "--valid_set", "./datasets/PDBBind/processed/identity30/valid.pkl", 
        "--save_dir", "./datasets/PDBBind/processed/identity30/models/MACE_gaussian",
        "--task", "PDBBind",
        "--lr", "0.0001",  # 降低学习率
        "--final_lr", "0.0001", 
        "--max_epoch", "10",
        "--save_topk", "5",
        "--max_n_vertex_per_gpu", "300",  # 减小batch size
        "--grad_clip", "0.5",  # 强梯度裁剪
        "--warmup", "2000",  # 增加warmup
        "--shuffle",
        "--model_type", "MACE",
        "--hidden_size", "64",  # 减小模型复杂度
        "--n_layers", "2",  # 减少层数
        "--n_channel", "1", 
        "--n_rbf", "16",  # 减少基函数数量
        "--cutoff", "6.0",  # 减小cutoff
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
