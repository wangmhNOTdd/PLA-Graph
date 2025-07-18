#!/usr/bin/env python3
"""
测试优化后的DSAN模型
"""

import torch
import numpy as np
from models.DSAN.encoder import DSANEncoder

def test_dsan():
    print("🔧 测试优化后的DSAN模型...")
    print("=" * 50)
    
    # 检查CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 模型参数
    hidden_size = 128
    n_layers = 3
    num_heads = 8
    k_neighbors = 9
    dropout = 0.1
    rbf_dim = 16
    cutoff = 10.0
    
    print("创建DSAN模型...")
    model = DSANEncoder(
        hidden_size=hidden_size,
        n_layers=n_layers,
        num_heads=num_heads,
        k_neighbors=k_neighbors,
        dropout=dropout,
        use_geometry=True,
        rbf_dim=rbf_dim,
        cutoff=cutoff
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 生成测试数据
    n_atoms = 100
    n_blocks = 10
    
    H = torch.randn(n_atoms, hidden_size, device=device)
    Z = torch.randn(n_atoms, 1, 3, device=device)
    
    # 块ID (每个块平均包含n_atoms/n_blocks个原子)
    block_id = torch.repeat_interleave(torch.arange(n_blocks, device=device), 
                                      n_atoms // n_blocks)
    if len(block_id) < n_atoms:
        remainder = n_atoms - len(block_id)
        block_id = torch.cat([block_id, torch.full((remainder,), n_blocks-1, device=device)])
    
    batch_id = torch.zeros(n_blocks, dtype=torch.long, device=device)
    
    # 块间边
    n_edges = 20
    src_blocks = torch.randint(0, n_blocks, (n_edges,), device=device)
    dst_blocks = torch.randint(0, n_blocks, (n_edges,), device=device)
    mask = src_blocks != dst_blocks
    edges = torch.stack([src_blocks[mask], dst_blocks[mask]])
    
    print(f"测试数据:")
    print(f"  原子数: {n_atoms}")
    print(f"  块数: {n_blocks}")
    print(f"  边数: {edges.shape[1]}")
    
    # 测试前向传播
    print("\n测试前向传播...")
    model.eval()
    
    try:
        with torch.no_grad():
            atom_features, block_repr, graph_repr, pred_Z = model(
                H, Z, block_id, batch_id, edges
            )
        
        print("✅ 前向传播成功!")
        print(f"  原子特征形状: {atom_features.shape}")
        print(f"  块表示形状: {block_repr.shape}")
        print(f"  图表示形状: {graph_repr.shape}")
        
        # 简单性能测试
        print("\n简单性能测试...")
        import time
        n_runs = 10
        
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time.time()
        
        for _ in range(n_runs):
            with torch.no_grad():
                _ = model(H, Z, block_id, batch_id, edges)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / n_runs
        print(f"平均推理时间: {avg_time*1000:.2f} ms")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 50)
    print("✅ DSAN优化测试完成!")
    return True

if __name__ == "__main__":
    test_dsan()
