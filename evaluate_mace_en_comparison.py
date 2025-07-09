#!/usr/bin/env python3
"""
评估MACE-En模型并对比原MACE性能
"""

import subprocess
import sys
import os
import json

def evaluate_model(model_name, ckpt_path, save_prefix):
    """评估单个模型"""
    
    print(f"\n🔍 评估 {model_name}...")
    
    # 创建结果目录
    results_dir = f"./results/{save_prefix}"
    os.makedirs(results_dir, exist_ok=True)
    
    # 验证集推断
    print("📊 验证集推断...")
    cmd_valid = [
        sys.executable, "inference.py",
        "--test_set", "./datasets/PDBBind/processed/identity30/valid.pkl",
        "--ckpt", ckpt_path,
        "--task", "PDBBind",
        "--save_path", f"{results_dir}/valid_predictions.jsonl",
        "--gpu", "0"
    ]
    subprocess.run(cmd_valid, check=True)
    
    # 测试集推断
    print("📊 测试集推断...")
    cmd_test = [
        sys.executable, "inference.py",
        "--test_set", "./datasets/PDBBind/processed/identity30/test.pkl",
        "--ckpt", ckpt_path,
        "--task", "PDBBind",
        "--save_path", f"{results_dir}/test_predictions.jsonl",
        "--gpu", "0"
    ]
    subprocess.run(cmd_test, check=True)
    
    # 评估性能
    print(f"📈 {model_name} - 验证集性能:")
    cmd_eval_valid = [
        sys.executable, "evaluate.py",
        "--predictions", f"{results_dir}/valid_predictions.jsonl"
    ]
    subprocess.run(cmd_eval_valid, check=True)
    
    print(f"📈 {model_name} - 测试集性能:")
    cmd_eval_test = [
        sys.executable, "evaluate.py",
        "--predictions", f"{results_dir}/test_predictions.jsonl"
    ]
    subprocess.run(cmd_eval_test, check=True)

def main():
    print("🚀 MACE-En vs 原MACE 性能对比评估...")
    print("=" * 60)
    
    # 检查可用的模型检查点
    models_to_evaluate = []
    
    # MACE-En Enhanced
    mace_en_dir = "./datasets/PDBBind/processed/identity30/models/MACE_En_standard"
    if os.path.exists(mace_en_dir):
        # 查找最佳检查点
        ckpt_dir = os.path.join(mace_en_dir, "version_0/checkpoint")
        if os.path.exists(ckpt_dir):
            topk_file = os.path.join(ckpt_dir, "topk_map.txt")
            if os.path.exists(topk_file):
                with open(topk_file, 'r') as f:
                    best_line = f.readline().strip()
                    best_ckpt = best_line.split(': ')[1]
                    models_to_evaluate.append({
                        "name": "MACE-En Enhanced",
                        "ckpt": best_ckpt,
                        "prefix": "mace_en_enhanced"
                    })
    
    # 原MACE Gaussian V1 (之前最佳)
    mace_orig_dir = "./datasets/PDBBind/processed/identity30/models/MACE_gaussian/version_1/checkpoint"
    if os.path.exists(mace_orig_dir):
        best_ckpt = os.path.join(mace_orig_dir, "epoch5_step4560.ckpt")
        if os.path.exists(best_ckpt):
            models_to_evaluate.append({
                "name": "MACE Original V1",
                "ckpt": best_ckpt,
                "prefix": "mace_original_v1"
            })
    
    # 检查是否有模型可评估
    if not models_to_evaluate:
        print("❌ 未找到可评估的模型检查点！")
        print("请先运行训练脚本生成模型。")
        return 1
    
    # 评估所有模型
    print(f"📋 发现 {len(models_to_evaluate)} 个模型待评估:")
    for model in models_to_evaluate:
        print(f"   - {model['name']}")
    
    try:
        for model in models_to_evaluate:
            evaluate_model(model["name"], model["ckpt"], model["prefix"])
        
        print("\n" + "=" * 60)
        print("🎉 所有模型评估完成!")
        print("\n📊 结果对比:")
        print("   检查各模型在验证集和测试集上的性能")
        print("   重点关注数值稳定性和预测精度的改进")
        print("\n📁 结果保存在 ./results/ 目录下")
        print("=" * 60)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 评估过程中出错: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
