#!/usr/bin/env python3
"""
评估MACE_gaussian两个版本的性能 - 标准PDBBind数据集
"""

import subprocess
import sys
import os

def evaluate_mace_version(version_num, best_ckpt, version_name):
    """评估指定版本的MACE模型"""
    
    print(f"\n🔍 开始评估 MACE_gaussian {version_name}...")
    print("=" * 60)
    
    # 创建结果目录
    results_dir = f"./results/mace_gaussian_{version_name.lower()}"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. 测试集推断
    print("📊 测试集推断...")
    cmd_test_inference = [
        sys.executable, "inference.py",
        "--test_set", "./datasets/PDBBind/processed/identity30/test.pkl",
        "--ckpt", best_ckpt,
        "--task", "PDBBind",
        "--save_path", f"{results_dir}/test_predictions.jsonl",
        "--gpu", "0"
    ]
    subprocess.run(cmd_test_inference, check=True)
    
    # 2. 验证集推断
    print("📊 验证集推断...")
    cmd_valid_inference = [
        sys.executable, "inference.py",
        "--test_set", "./datasets/PDBBind/processed/identity30/valid.pkl",
        "--ckpt", best_ckpt,
        "--task", "PDBBind",
        "--save_path", f"{results_dir}/valid_predictions.jsonl",
        "--gpu", "0"
    ]
    subprocess.run(cmd_valid_inference, check=True)
    
    # 3. 测试集评估
    print("📈 测试集评估...")
    print(f"MACE_gaussian {version_name} - 测试集性能:")
    cmd_test_eval = [
        sys.executable, "evaluate.py",
        "--predictions", f"{results_dir}/test_predictions.jsonl"
    ]
    subprocess.run(cmd_test_eval, check=True)
    
    # 4. 验证集评估
    print("📈 验证集评估...")
    print(f"MACE_gaussian {version_name} - 验证集性能:")
    cmd_valid_eval = [
        sys.executable, "evaluate.py",
        "--predictions", f"{results_dir}/valid_predictions.jsonl"
    ]
    subprocess.run(cmd_valid_eval, check=True)
    
    print(f"✅ MACE_gaussian {version_name} 评估完成！")

def main():
    """主函数 - 评估两个版本"""
    
    print("🚀 开始评估MACE_gaussian两个版本...")
    
    # Version 0 (最佳性能)
    version_0_ckpt = "./datasets/PDBBind/processed/identity30/models/MACE_gaussian/version_0/checkpoint/epoch5_step4560.ckpt"
    
    # Version 1
    version_1_ckpt = "./datasets/PDBBind/processed/identity30/models/MACE_gaussian/version_1/checkpoint/epoch5_step4560.ckpt"
    
    try:
        # 评估 Version 0
        evaluate_mace_version(0, version_0_ckpt, "Version_0")
        
        # 评估 Version 1  
        evaluate_mace_version(1, version_1_ckpt, "Version_1")
        
        print("\n" + "=" * 60)
        print("🎉 两个版本评估完成！")
        print("\n📊 性能对比:")
        print("   Version 0: 验证损失 2.4280 (最佳)")
        print("   Version 1: 验证损失 2.4889")
        print("\n📁 结果保存在:")
        print("   ./results/mace_gaussian_version_0/")
        print("   ./results/mace_gaussian_version_1/")
        print("=" * 60)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 评估失败: {e}")
        return 1
    except KeyboardInterrupt:
        print("⛔ 评估被用户中断")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
