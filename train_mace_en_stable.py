#!/usr/bin/env python3
"""
稳定版MACE-En模型训练脚本
在PDBBind identity30数据集上训练，使用经过验证的稳定参数
"""

import subprocess
import sys
import os

def main():
    print("🚀 启动MACE-En稳定版训练 (PDBBind identity30)...")
    print("=" * 60)
    
    # 构建训练命令 - 使用已验证的稳定参数
    cmd = [
        sys.executable, "train.py",
        "--train_set", "./datasets/PDBBind/processed/identity30/train.pkl",
        "--valid_set", "./datasets/PDBBind/processed/identity30/valid.pkl", 
        "--save_dir", "./datasets/PDBBind/processed/identity30/models/MACE_En_stable",
        "--task", "PDBBind",
        "--lr", "0.0001",           # 较低学习率确保稳定性
        "--final_lr", "0.00005", 
        "--max_epoch", "50",        # 适中的epoch数
        "--save_topk", "5",
        "--max_n_vertex_per_gpu", "500",  # 较小batch size避免内存溢出
        "--grad_clip", "0.1",       # 强梯度裁剪防止NaN
        "--warmup", "200",          # 更长warmup
        "--shuffle",
        "--model_type", "MACE-En",  # 使用增强版MACE
        "--hidden_size", "64",      # 较小模型规模
        "--n_layers", "2",          # 较少层数
        "--n_channel", "1", 
        "--n_rbf", "16",            # 较少RBF基函数
        "--cutoff", "7.0",
        "--radial_size", "32",      # 较小径向特征
        "--seed", "2023",
        "--gpus", "0"
    ]
    
    print("📋 MACE-En稳定版配置:")
    print(f"   模型: MACE-En (增强数值稳定性)")
    print(f"   数据集: PDBBind identity30")
    print(f"   训练集: 3507 样本")
    print(f"   验证集: 466 样本") 
    print(f"   测试集: 490 样本")
    print(f"   学习率: 0.0001 → 0.00005")
    print(f"   最大epoch: 50")
    print(f"   层数: 2 (轻量版)")
    print(f"   隐藏维度: 64 (紧凑)")
    print(f"   RBF数量: 16 (高效)")
    print(f"   梯度裁剪: 0.1 (强约束)")
    print(f"   Warmup: 200步")
    print(f"   参数量: ~2M")
    print("\n🔧 MACE-En稳定性特性:")
    print("   ✅ EnhancedBesselBasis (数值稳定)")
    print("   ✅ Taylor展开处理奇点")
    print("   ✅ 强梯度裁剪防止爆炸")
    print("   ✅ 保守参数设置")
    print("   ✅ 小batch避免内存问题")
    print("=" * 60)
    
    # 执行训练
    print("🏃‍♂️ 开始稳定训练...")
    try:
        subprocess.run(cmd, check=True)
        print("✅ MACE-En稳定版训练完成!")
        
        # 训练完成后的建议
        print("\n🎯 后续步骤建议:")
        print("1. 运行评估: python evaluate.py")
        print("2. 对比性能: python evaluate_mace_en_comparison.py")
        print("3. 尝试更大模型: 增加hidden_size到128")
        print("4. 尝试混合基函数: HybridBasis")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")
        print("\n🔧 可能的解决方案:")
        print("- 进一步降低学习率: --lr 0.00005")
        print("- 增强梯度裁剪: --grad_clip 0.05")
        print("- 减少batch size: --max_n_vertex_per_gpu 250")
        return 1
    except KeyboardInterrupt:
        print("⛔ 训练被用户中断")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
