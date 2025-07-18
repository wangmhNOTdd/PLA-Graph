#!/usr/bin/env python3
"""
训练DSAN模型在PDBBind identity30数据集上
"""

import subprocess
import sys
import os

def main():
    print("🚀 启动DSAN模型训练...")
    print("=" * 50)
    
    # 检查GPU状态
    print("GPU状态检查:")
    import torch
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU数量: {torch.cuda.device_count()}")
        print(f"   GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"   显存大小: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 50)
    
    # 构建训练命令
    cmd = [
        sys.executable, "train.py",
        "--train_set", "./datasets/PDBBind/processed/identity30/train.pkl",
        "--valid_set", "./datasets/PDBBind/processed/identity30/valid.pkl", 
        "--save_dir", "./datasets/PDBBind/processed/identity30/models/DSAN_optimized",
        "--task", "PDBBind",
        "--lr", "0.001",
        "--final_lr", "0.0001", 
        "--max_epoch", "100",
        "--save_topk", "5",
        "--batch_size", "1",  # 从8增加到16以提高GPU利用率
        "--valid_batch_size", "1",
        "--grad_clip", "1.0",
        "--warmup", "1000",
        "--shuffle",
        "--model_type", "DSAN",  # 使用优化后的DSAN
        "--hidden_size", "64",
        "--n_layers", "3",  # 减少层数以加快训练
        "--n_channel", "1", 
        "--n_rbf", "32",
        "--cutoff", "10.0",
        "--n_head", "8",
        "--radial_size", "16",
        "--k_neighbors", "9",
        "--seed", "2024",
        "--gpus", "0"
    ]
    
    # 执行训练
    print("开始训练...")
    try:
        subprocess.run(cmd, check=True)
        print("✅ 训练完成!")
        print(f"模型已保存到: ./datasets/PDBBind/processed/identity30/models/DSAN_optimized")
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")
        return 1
    except KeyboardInterrupt:
        print("⛔ 训练被用户中断")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
