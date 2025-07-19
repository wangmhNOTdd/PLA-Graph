#!/usr/bin/env python3
"""
RTX 4060显存压力测试脚本
测试优化后的DSAN模型在4060上的显存占用
"""

import torch
import torch.nn as nn
import time
import psutil
import numpy as np
from models.DSAN.encoder import DSANEncoder


def check_gpu_status():
    """检查GPU状态"""
    print("🔧 GPU状态检查:")
    if torch.cuda.is_available():
        print(f"   ✅ CUDA可用")
        print(f"   GPU数量: {torch.cuda.device_count()}")
        print(f"   GPU名称: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   显存总量: {gpu_memory:.1f} GB")
        print(f"   当前显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        return True, gpu_memory
    else:
        print("   ❌ CUDA不可用")
        return False, 0


def memory_monitor():
    """实时显存监控"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        free = (torch.cuda.get_device_properties(0).total_memory / 1024**3) - reserved
        
        status = f"GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free"
        
        # 4060警告阈值
        if allocated > 6.0:
            status += " ⚠️  高显存使用!"
        if allocated > 7.0:
            status += " 🚨 临界状态!"
            
        return status, allocated
    return "CUDA不可用", 0


def create_test_data(n_atoms=100, n_blocks=10, device='cuda'):
    """创建测试数据"""
    print(f"📊 创建测试数据: {n_atoms}个原子, {n_blocks}个块")
    
    # 原子特征
    H = torch.randn(n_atoms, 64, device=device)  # 减少到64维
    
    # 原子坐标
    Z = torch.randn(n_atoms, 1, 3, device=device)
    
    # 块ID (每个块大约10个原子)
    block_id = torch.randint(0, n_blocks, (n_atoms,), device=device)
    
    # 批次ID
    batch_id = torch.zeros(n_blocks, device=device, dtype=torch.long)
    
    # 块间边（随机生成）
    n_edges = min(n_blocks * 2, 50)  # 减少边数量
    edge_src = torch.randint(0, n_blocks, (n_edges,), device=device)
    edge_dst = torch.randint(0, n_blocks, (n_edges,), device=device)
    edges = torch.stack([edge_src, edge_dst], dim=0)
    
    return H, Z, block_id, batch_id, edges


def test_dsan_memory_usage():
    """测试DSAN模型显存使用"""
    print("🧪 开始DSAN显存测试...")
    print("=" * 60)
    
    # 检查GPU
    gpu_available, total_memory = check_gpu_status()
    if not gpu_available:
        return False
    
    device = 'cuda'
    torch.cuda.empty_cache()
    
    print("\n💾 基准显存状态:")
    baseline_status, baseline_memory = memory_monitor()
    print(f"   {baseline_status}")
    
    try:
        # 创建优化的DSAN模型 (4060参数)
        print("\n🏗️  创建DSAN模型 (4060优化参数):")
        model = DSANEncoder(
            hidden_size=64,      # 减少隐藏层大小
            n_layers=2,          # 减少层数  
            num_heads=4,         # 减少注意力头数
            k_neighbors=4,       # 减少K近邻
            dropout=0.1,
            use_geometry=True,
            rbf_dim=8,           # 减少RBF维度
            cutoff=6.0,          # 减少cutoff
            memory_efficient=True  # 开启显存优化
        ).to(device)
        
        model_status, model_memory = memory_monitor()
        print(f"   模型加载后: {model_status}")
        print(f"   模型占用显存: {model_memory - baseline_memory:.2f} GB")
        
        # 测试不同规模的数据
        test_cases = [
            (50, 5, "小规模"),
            (100, 10, "中等规模"),  
            (200, 20, "大规模"),
            (300, 30, "极限规模")
        ]
        
        max_memory = model_memory
        successful_cases = []
        
        for n_atoms, n_blocks, desc in test_cases:
            print(f"\n🎯 测试 {desc} ({n_atoms}原子, {n_blocks}块):")
            
            try:
                # 创建测试数据
                H, Z, block_id, batch_id, edges = create_test_data(n_atoms, n_blocks, device)
                
                # 前向传播
                torch.cuda.empty_cache()
                start_time = time.time()
                
                with torch.no_grad():
                    atom_features, block_repr, graph_repr, pred_Z = model(H, Z, block_id, batch_id, edges)
                
                torch.cuda.synchronize()
                end_time = time.time()
                
                # 监控显存
                test_status, test_memory = memory_monitor()
                inference_time = (end_time - start_time) * 1000
                
                print(f"   推理时间: {inference_time:.1f} ms")
                print(f"   显存状态: {test_status}")
                print(f"   峰值显存: {test_memory:.2f} GB")
                print(f"   输出形状: atom_features{atom_features.shape}, block_repr{block_repr.shape}")
                
                if test_memory > max_memory:
                    max_memory = test_memory
                    
                successful_cases.append((desc, n_atoms, n_blocks, test_memory, inference_time))
                
                # 清理
                del H, Z, block_id, batch_id, edges, atom_features, block_repr, graph_repr, pred_Z
                torch.cuda.empty_cache()
                
                print("   ✅ 测试通过")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"   ❌ 显存不足: {e}")
                    torch.cuda.empty_cache()
                    break
                else:
                    raise e
        
        # 测试总结
        print("\n" + "=" * 60)
        print("📈 测试总结:")
        print(f"   GPU型号: {torch.cuda.get_device_name(0)} ({total_memory:.1f}GB)")
        print(f"   基准显存: {baseline_memory:.2f} GB")
        print(f"   模型显存: {model_memory - baseline_memory:.2f} GB") 
        print(f"   峰值显存: {max_memory:.2f} GB")
        print(f"   显存利用率: {(max_memory / total_memory) * 100:.1f}%")
        
        print(f"\n🎉 成功测试的规模:")
        for desc, n_atoms, n_blocks, memory, time_ms in successful_cases:
            print(f"   {desc}: {n_atoms}原子/{n_blocks}块 -> {memory:.2f}GB, {time_ms:.1f}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    finally:
        # 清理显存
        torch.cuda.empty_cache()
        final_status, _ = memory_monitor()
        print(f"\n🧹 清理后状态: {final_status}")


def main():
    print("🚀 RTX 4060 DSAN显存压力测试")
    print("=" * 60)
    
    success = test_dsan_memory_usage()
    
    if success:
        print("\n✅ 显存测试完成! DSAN模型已针对RTX 4060优化")
        print("💡 建议使用以下训练参数:")
        print("   - batch_size: 1")
        print("   - hidden_size: 64")
        print("   - n_layers: 2")
        print("   - k_neighbors: 4")
        print("   - cutoff: 6.0")
        print("   - rbf_dim: 8")
        return 0
    else:
        print("\n❌ 显存测试失败")
        return 1


if __name__ == "__main__":
    exit(main())
