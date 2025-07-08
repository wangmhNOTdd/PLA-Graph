#!/usr/bin/env python3
"""
启动MACE模型训练 - v2020-other-PL + CASF-2016数据集
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
        "--train_set", "./datasets/v2020-other-PL/processed_get_format/train.pkl",
        "--valid_set", "./datasets/v2020-other-PL/processed_get_format/valid.pkl", 
        "--save_dir", "./datasets/v2020-other-PL/processed_get_format/models/MACE_standard",
        "--task", "PDBBind",
        "--lr", "0.0001",
        "--final_lr", "0.0001",
        "--max_epoch", "100",
        "--save_topk", "3",
        "--max_n_vertex_per_gpu", "1000",
        "--shuffle",
        "--model_type", "MACE",
        "--hidden_size", "128",
        "--n_layers", "3",
        "--n_channel", "1", 
        "--n_rbf", "32",
        "--cutoff", "7.0",
        "--radial_size", "64",
        "--seed", "2023",
        "--gpus", "0"
    ]
    
    print("📋 MACE训练配置:")
    print(f"   模型: MACE (等变消息传递)")
    print(f"   层数: 3")
    print(f"   学习率: 0.0001")
    print(f"   训练集: 12245 样本")
    print(f"   验证集: 1360 样本") 
    print(f"   最大epoch: 100")
    print(f"   截断距离: 7.0")
    print(f"   RBF数量: 32")
    print(f"   隐藏维度: 128")
    print("=" * 50)
    
    # 创建保存目录
    save_dir = "./datasets/v2020-other-PL/processed_get_format/models/MACE_standard"
    os.makedirs(save_dir, exist_ok=True)
    print(f"📁 模型保存目录: {save_dir}")
    
    # 执行训练
    print("🏃‍♂️ 开始MACE训练...")
    try:
        subprocess.run(cmd, check=True)
        print("✅ MACE训练完成!")
    except subprocess.CalledProcessError as e:
        print(f"❌ MACE训练失败: {e}")
        return 1
    except KeyboardInterrupt:
        print("⛔ MACE训练被用户中断")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
