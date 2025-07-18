#!/usr/bin/env python3
"""
测试优化版DSAN模型的正确性和性能
"""

import torch
import torch.nn as nn
import time
import numpy as np
from models.DSAN.encoder import DSANEncoder
from models.DSAN.optimized_encoder import OptimizedDSANEncoder

def generate_test_data(n_atoms=100, n_blocks=10, hidden_size=128, device='cuda'):
    """生成测试数据"""
    
    # 原子特征
    H = torch.randn(n_atoms, hidden_size, device=device)
    
    # 原子坐标
    Z = torch.randn(n_atoms, 1, 3, device=device)
    
    # 块ID (每个块平均包含n_atoms/n_blocks个原子)
    block_id = torch.repeat_interleave(torch.arange(n_blocks, device=device), 
                                      n_atoms // n_blocks)
    if len(block_id) < n_atoms:
        # 处理不能整除的情况
        remainder = n_atoms - len(block_id)
        block_id = torch.cat([block_id, torch.full((remainder,), n_blocks-1, device=device)])
    
    # 批次ID
    batch_id = torch.zeros(n_blocks, dtype=torch.long, device=device)
    
    # 块间边（随机生成一些边）
    n_edges = min(50, n_blocks * (n_blocks - 1) // 4)  # 限制边数
    src_blocks = torch.randint(0, n_blocks, (n_edges,), device=device)
    dst_blocks = torch.randint(0, n_blocks, (n_edges,), device=device)
    # 确保不是自环
    mask = src_blocks != dst_blocks
    edges = torch.stack([src_blocks[mask], dst_blocks[mask]])
    
    return H, Z, block_id, batch_id, edges

def benchmark_model(model, H, Z, block_id, batch_id, edges, n_runs=10):
    """测试模型性能"""
    model.eval()
    
    # 预热
    with torch.no_grad():
        _ = model(H, Z, block_id, batch_id, edges)
    
    # 计时
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(H, Z, block_id, batch_id, edges)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / n_runs
    return avg_time

def test_correctness(original_model, optimized_model, H, Z, block_id, batch_id, edges, tolerance=1e-4):
    """测试两个模型输出的一致性"""
    original_model.eval()
    optimized_model.eval()
    
    with torch.no_grad():
        # 原始模型输出
        orig_atom, orig_block, orig_graph, orig_pred_z = original_model(H, Z, block_id, batch_id, edges)
        
        # 优化模型输出
        opt_atom, opt_block, opt_graph, opt_pred_z = optimized_model(H, Z, block_id, batch_id, edges)
        
        # 比较原子特征
        atom_diff = torch.abs(orig_atom - opt_atom).max().item()
        
        # 比较块特征
        block_diff = torch.abs(orig_block - opt_block).max().item()
        
        # 比较图特征
        graph_diff = torch.abs(orig_graph - opt_graph).max().item()
        
        print(f"原子特征最大差异: {atom_diff:.6f}")
        print(f"块特征最大差异: {block_diff:.6f}")
        print(f"图特征最大差异: {graph_diff:.6f}")
        
        # 检查是否在容忍范围内
        is_correct = (atom_diff < tolerance and 
                     block_diff < tolerance and 
                     graph_diff < tolerance)
        
        return is_correct, atom_diff, block_diff, graph_diff

def main():
    print("🔧 测试优化版DSAN模型...")
    print("=" * 50)
    
    # 检查CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print("=" * 50)
    
    # 模型参数
    hidden_size = 128
    n_layers = 3
    num_heads = 8
    k_neighbors = 9
    dropout = 0.1
    rbf_dim = 16
    cutoff = 10.0
    
    print("创建模型...")
    
    # 创建原始DSAN模型
    original_model = DSANEncoder(
        hidden_size=hidden_size,
        n_layers=n_layers,
        num_heads=num_heads,
        k_neighbors=k_neighbors,
        dropout=dropout,
        use_geometry=True,
        rbf_dim=rbf_dim,
        cutoff=cutoff
    ).to(device)
    
    # 创建优化版DSAN模型
    optimized_model = OptimizedDSANEncoder(
        hidden_size=hidden_size,
        n_layers=n_layers,
        num_heads=num_heads,
        k_neighbors=k_neighbors,
        dropout=dropout,
        use_geometry=True,
        rbf_dim=rbf_dim,
        cutoff=cutoff
    ).to(device)
    
    print(f"原始模型参数量: {sum(p.numel() for p in original_model.parameters()):,}")
    print(f"优化模型参数量: {sum(p.numel() for p in optimized_model.parameters()):,}")
    
    # 生成测试数据
    print("\n生成测试数据...")
    test_sizes = [
        (50, 5),    # 小规模
        (100, 10),  # 中规模
        (200, 20),  # 大规模
    ]
    
    for n_atoms, n_blocks in test_sizes:
        print(f"\n测试规模: {n_atoms} 原子, {n_blocks} 块")
        print("-" * 30)
        
        H, Z, block_id, batch_id, edges = generate_test_data(
            n_atoms=n_atoms, 
            n_blocks=n_blocks, 
            hidden_size=hidden_size, 
            device=device
        )
        
        print(f"数据形状:")
        print(f"  原子特征: {H.shape}")
        print(f"  坐标: {Z.shape}")
        print(f"  块ID: {block_id.shape}")
        print(f"  边数: {edges.shape[1]}")
        
        # 测试正确性
        print("\n测试输出一致性...")
        try:
            is_correct, atom_diff, block_diff, graph_diff = test_correctness(
                original_model, optimized_model, H, Z, block_id, batch_id, edges
            )
            
            if is_correct:
                print("✅ 输出一致性测试通过!")
            else:
                print("❌ 输出一致性测试失败!")
                print(f"   最大差异: 原子={atom_diff:.6f}, 块={block_diff:.6f}, 图={graph_diff:.6f}")
                
        except Exception as e:
            print(f"❌ 正确性测试出错: {e}")
            continue
        
        # 性能测试
        print("\n性能测试...")
        try:
            orig_time = benchmark_model(original_model, H, Z, block_id, batch_id, edges, n_runs=5)
            opt_time = benchmark_model(optimized_model, H, Z, block_id, batch_id, edges, n_runs=5)
            
            speedup = orig_time / opt_time
            
            print(f"原始模型平均时间: {orig_time*1000:.2f} ms")
            print(f"优化模型平均时间: {opt_time*1000:.2f} ms") 
            print(f"加速比: {speedup:.2f}x")
            
            if speedup > 1.0:
                print("✅ 性能优化成功!")
            else:
                print("⚠️  性能可能没有提升")
                
        except Exception as e:
            print(f"❌ 性能测试出错: {e}")
    
    print("\n" + "=" * 50)
    print("测试完成!")

if __name__ == "__main__":
    main()
