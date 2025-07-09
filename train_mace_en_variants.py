#!/usr/bin/env python3
"""
MACE-En不同基函数配置对比训练
测试EnhancedBesselBasis、HybridBasis、AdaptiveBesselBasis的效果
"""

import subprocess
import sys
import os

def train_mace_en_variant(variant_name, additional_params, description):
    """训练MACE-En的特定变体"""
    
    print(f"\n🔍 训练MACE-En {variant_name}...")
    print("=" * 50)
    
    # 基础训练参数
    base_cmd = [
        sys.executable, "train.py",
        "--train_set", "./datasets/PDBBind/processed/identity30/train.pkl",
        "--valid_set", "./datasets/PDBBind/processed/identity30/valid.pkl", 
        "--save_dir", f"./datasets/PDBBind/processed/identity30/models/MACE_En_{variant_name}",
        "--task", "PDBBind",
        "--lr", "0.0001",
        "--final_lr", "0.00005", 
        "--max_epoch", "50",  # 较短训练用于对比
        "--save_topk", "3",
        "--max_n_vertex_per_gpu", "800",
        "--grad_clip", "0.5",
        "--warmup", "500",
        "--shuffle",
        "--model_type", "MACE-En",
        "--hidden_size", "128",
        "--n_layers", "3",
        "--n_channel", "1", 
        "--cutoff", "7.0",
        "--radial_size", "64",
        "--seed", "2023",
        "--gpus", "0"
    ]
    
    # 添加特定参数
    cmd = base_cmd + additional_params
    
    print(f"📋 {variant_name} 配置:")
    print(f"   描述: {description}")
    print(f"   特殊参数: {additional_params}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {variant_name} 训练完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {variant_name} 训练失败: {e}")
        return False

def main():
    print("🚀 MACE-En 基函数对比实验...")
    print("=" * 60)
    
    # 测试配置
    variants = [
        {
            "name": "Enhanced", 
            "params": ["--n_rbf", "32"],
            "description": "增强BesselBasis，数值稳定"
        },
        {
            "name": "Hybrid", 
            "params": ["--n_rbf", "16", "--basis_type", "hybrid"],
            "description": "混合Bessel+Gaussian基函数"
        },
        {
            "name": "Adaptive",
            "params": ["--n_rbf", "24", "--basis_type", "adaptive"], 
            "description": "自适应可学习频率BesselBasis"
        }
    ]
    
    results = {}
    
    # 依次训练每个变体
    for variant in variants:
        success = train_mace_en_variant(
            variant["name"], 
            variant["params"], 
            variant["description"]
        )
        results[variant["name"]] = success
    
    # 总结结果
    print("\n" + "=" * 60)
    print("🎉 MACE-En对比实验完成!")
    print("\n📊 训练结果总结:")
    for name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"   {name}: {status}")
    
    print("\n📁 模型保存位置:")
    print("   Enhanced: ./datasets/PDBBind/processed/identity30/models/MACE_En_Enhanced/")
    print("   Hybrid: ./datasets/PDBBind/processed/identity30/models/MACE_En_Hybrid/")
    print("   Adaptive: ./datasets/PDBBind/processed/identity30/models/MACE_En_Adaptive/")
    
    print("\n🔍 下一步:")
    print("   1. 检查各变体的验证损失对比")
    print("   2. 运行推断和评估脚本")
    print("   3. 分析数值稳定性改进效果")
    print("=" * 60)

if __name__ == "__main__":
    exit(main())
