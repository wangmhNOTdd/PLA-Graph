#!/usr/bin/env python3
"""
在PDBBind identity30数据集上训练MACE-En模型
使用改进的数值稳定性BesselBasis
"""

import subprocess
import sys
import os

def main():
    print("🚀 启动MACE-En模型训练 (PDBBind identity30)...")
    print("=" * 60)
    
    # 构建训练命令
    cmd = [
        sys.executable, "train.py",
        "--train_set", "./datasets/PDBBind/processed/identity30/train.pkl",
        "--valid_set", "./datasets/PDBBind/processed/identity30/valid.pkl", 
        "--save_dir", "./datasets/PDBBind/processed/identity30/models/MACE_En_standard",
        "--task", "PDBBind",
        "--lr", "0.0001",
        "--final_lr", "0.00005", 
        "--max_epoch", "100",
        "--save_topk", "5",
        "--max_n_vertex_per_gpu", "1000",
        "--grad_clip", "0.5",
        "--warmup", "1000",
        "--shuffle",
        "--model_type", "MACE-En",  # 使用MACE-En模型
        "--hidden_size", "128",
        "--n_layers", "3",
        "--n_channel", "1", 
        "--n_rbf", "32",
        "--cutoff", "7.0",
        "--radial_size", "64",
        "--seed", "2023",
        "--gpus", "0"
    ]
    
    print("📋 MACE-En训练配置:")
    print(f"   模型: MACE-En (增强数值稳定性)")
    print(f"   数据集: PDBBind identity30")
    print(f"   训练集: 3507 样本")
    print(f"   验证集: 466 样本") 
    print(f"   测试集: 490 样本")
    print(f"   学习率: 0.0001 → 0.00005")
    print(f"   最大epoch: 100")
    print(f"   层数: 3")
    print(f"   隐藏维度: 128")
    print(f"   RBF数量: 32")
    print(f"   梯度裁剪: 0.5 (数值稳定)")
    print(f"   Warmup: 1000步")
    print("\n🔧 MACE-En特性:")
    print("   ✅ 增强的BesselBasis (避免NaN)")
    print("   ✅ 数值稳定的Taylor展开")
    print("   ✅ 自适应epsilon处理")
    print("   ✅ 混合基函数支持")
    print("=" * 60)
    
    # 执行训练
    print("🏃‍♂️ 开始训练...")
    try:
        subprocess.run(cmd, check=True)
        print("✅ MACE-En训练完成!")
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")
        return 1
    except KeyboardInterrupt:
        print("⛔ 训练被用户中断")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
