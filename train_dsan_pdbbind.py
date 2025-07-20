#!/usr/bin/env python3
"""
训练DSAN模型在PDBBind identity30数据集上（显存优化版）
"""

import subprocess
import sys
import os

def main():
    print("🚀 启动DSAN模型训练（显存优化版）...")
    print("=" * 60)
    
    # 检查GPU状态
    print("GPU状态检查:")
    import torch
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU数量: {torch.cuda.device_count()}")
        print(f"   GPU名称: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   显存大小: {gpu_memory:.1f} GB")
        print(f"   当前显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"   显存利用率: {torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100:.1f}%")
    print("=" * 60)
    
    # 构建优化的训练命令
    cmd = [
        sys.executable, "train.py",
        "--train_set", "./datasets/PDBBind/processed/identity30/train.pkl",
        "--valid_set", "./datasets/PDBBind/processed/identity30/valid.pkl", 
        "--save_dir", "./datasets/PDBBind/processed/identity30/models/DSAN_memory_optimized",
        "--task", "PDBBind",
        "--lr", "0.001",
        "--final_lr", "0.0001", 
        "--max_epoch", "100",
        "--save_topk", "5",
        "--batch_size", "4",  # 增加到4测试修复效果
        "--valid_batch_size", "4",
        "--grad_clip", "1.0",
        "--warmup", "1000",
        "--shuffle",
        "--model_type", "DSAN",  # 使用显存优化的DSAN
        "--hidden_size", "128",
        "--n_layers", "3",  # 保持3层
        "--n_channel", "1", 
        "--n_rbf", "16",      # 从32减少到16，减少边特征维度
        "--cutoff", "8.0",    # 从10.0减少到8.0，减少边数量
        "--n_head", "8",      # 保持8个注意力头
        "--radial_size", "8", # 从16减少到8，减少几何特征维度
        "--k_neighbors", "6", # 从9减少到6，显著减少边数量（关键优化）
        "--seed", "2024",
        "--gpus", "0"
    ]
    
    print("DSAN训练配置（显存优化）:")
    print(f"   批次大小: 4 (测试修复效果)")
    print(f"   RBF维度: 16 (减少边特征)")
    print(f"   Cutoff距离: 8.0 (减少边数量)")
    print(f"   几何特征维度: 8 (减少计算)")
    print(f"   K近邻数: 6 (关键：减少ESA复杂度)")
    print(f"   注意力头数: 8")
    print(f"   层数: 3")
    print(f"   隐藏层大小: 128")
    print("=" * 60)
    
    print("显存优化特性:")
    print("   ✅ 批量块处理（Batch Block Processing）")
    print("   ✅ 向量化PMA（Vectorized PMA）")
    print("   ✅ 显存清理（Memory Cache Clearing）")
    print("   ✅ 分块几何计算（Chunked Geometry Computing）")
    print("   ⚠️  梯度检查点已暂时禁用（避免动态形状冲突）")
    print("=" * 60)
    
    # 执行训练
    print("🎯 开始训练... (Ctrl+C 中断)")
    import time
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True)
        
        end_time = time.time()
        training_time = (end_time - start_time) / 3600  # hours
        
        print("✅ 训练完成!")
        print(f"模型已保存到: ./datasets/PDBBind/processed/identity30/models/DSAN_memory_optimized")
        print(f"训练用时: {training_time:.2f} 小时")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")
        print("可能的解决方案:")
        print("   1. 检查数据集路径是否正确")
        print("   2. 进一步减少batch_size")
        print("   3. 减少k_neighbors或cutoff")
        print("   4. 检查GPU显存是否足够")
        return 1
    except KeyboardInterrupt:
        print("⛔ 训练被用户中断")
        return 1
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return 1
    finally:
        # 清理GPU显存
        if 'torch' in sys.modules:
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"最终显存占用: {final_memory:.2f} GB")
        print("=" * 60)
        print("训练结束")

if __name__ == "__main__":
    exit(main())
