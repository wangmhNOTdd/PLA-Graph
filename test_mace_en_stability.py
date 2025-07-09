#!/usr/bin/env python3
"""
MACE-En快速数值稳定性测试
短训练验证改进效果
"""

import subprocess
import sys
import os

def main():
    print("🚀 MACE-En 数值稳定性快速测试...")
    print("=" * 60)
    
    # 构建快速测试训练命令
    cmd = [
        sys.executable, "train.py",
        "--train_set", "./datasets/PDBBind/processed/identity30/train.pkl",
        "--valid_set", "./datasets/PDBBind/processed/identity30/valid.pkl", 
        "--save_dir", "./datasets/PDBBind/processed/identity30/models/MACE_En_quicktest",
        "--task", "PDBBind",
        "--lr", "0.0002",  # 稍高学习率测试稳定性
        "--final_lr", "0.0001", 
        "--max_epoch", "10",  # 快速测试
        "--save_topk", "3",
        "--max_n_vertex_per_gpu", "500",  # 小批次
        "--grad_clip", "1.0",  # 放宽梯度裁剪测试
        "--warmup", "200",  # 短warmup
        "--shuffle",
        "--model_type", "MACE-En",
        "--hidden_size", "128",
        "--n_layers", "3",
        "--n_channel", "1", 
        "--n_rbf", "32",
        "--cutoff", "7.0",
        "--radial_size", "64",
        "--seed", "42",  # 不同随机种子
        "--gpus", "0"
    ]
    
    print("📋 快速测试配置:")
    print(f"   目标: 验证MACE-En数值稳定性")
    print(f"   学习率: 0.0002 (较高)")
    print(f"   梯度裁剪: 1.0 (宽松)")
    print(f"   训练轮数: 10 (快速)")
    print(f"   预期: 无NaN损失，稳定收敛")
    print("\n🔧 测试重点:")
    print("   ✅ EnhancedBesselBasis防NaN")
    print("   ✅ Taylor展开处理小距离")
    print("   ✅ Epsilon保护除零运算")
    print("   ✅ 数值稳定的前向传播")
    print("=" * 60)
    
    # 执行训练
    print("🏃‍♂️ 开始快速测试...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("✅ MACE-En快速测试成功!")
        print("\n📊 关键观察点:")
        print("   - 是否出现NaN损失")
        print("   - 训练过程是否稳定")
        print("   - 收敛速度如何")
        
        # 检查日志中是否有NaN
        if "nan" in result.stdout.lower() or "nan" in result.stderr.lower():
            print("⚠️ 检测到NaN，需要进一步调优")
        else:
            print("✅ 未检测到NaN，数值稳定性良好")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 快速测试失败: {e}")
        print("可能的问题:")
        print("   1. MACE-En模型未正确注册")
        print("   2. 数值稳定性仍有问题") 
        print("   3. 参数配置需要调整")
        return 1
    except KeyboardInterrupt:
        print("⛔ 测试被用户中断")
        return 1
    
    print("\n🎯 下一步:")
    print("   1. 检查训练日志分析稳定性")
    print("   2. 如果成功，进行完整训练")
    print("   3. 与原MACE对比性能")
    
    return 0

if __name__ == "__main__":
    exit(main())
