#!/usr/bin/env python3
"""
专门调试CUDA索引问题的简化测试
"""

import torch
import sys
sys.path.append('.')

def create_simple_batch_data():
    """创建最简化的批次数据来隔离问题"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 2个简单分子的批次
    batch_size = 2
    
    # 分子1: 5个原子，分子2: 7个原子
    n_atoms_mol1 = 5
    n_atoms_mol2 = 7
    total_atoms = n_atoms_mol1 + n_atoms_mol2
    
    print(f"创建简化批次: 分子1={n_atoms_mol1}原子, 分子2={n_atoms_mol2}原子")
    
    # 原子坐标 [total_atoms, 1, 3]
    Z = torch.randn(total_atoms, 1, 3, device=device)
    
    # 创建简单的边连接（每个分子内部连接）
    edge_list = []
    # 分子1的边 (原子0-4)
    for i in range(n_atoms_mol1-1):
        edge_list.extend([[i, i+1], [i+1, i]])
    
    # 分子2的边 (原子5-11)  
    offset = n_atoms_mol1
    for i in range(n_atoms_mol2-1):
        edge_list.extend([[i+offset, i+1+offset], [i+1+offset, i+offset]])
    
    E_idx = torch.tensor(edge_list, device=device).T  # [2, num_edges]
    n_edges = E_idx.shape[1]
    E = torch.randn(n_edges, 16, device=device)  # 边特征
    
    # 关键：批次ID应该是原子级别的
    B = torch.cat([
        torch.zeros(n_atoms_mol1, dtype=torch.long, device=device),  # 分子1
        torch.ones(n_atoms_mol2, dtype=torch.long, device=device)    # 分子2
    ])
    
    # 原子特征
    A = torch.randn(total_atoms, device=device)
    
    print(f"数据形状验证:")
    print(f"  Z: {Z.shape}")  
    print(f"  E_idx: {E_idx.shape}, 边数={n_edges}")
    print(f"  E: {E.shape}")
    print(f"  B: {B.shape}, 内容={B.tolist()}")
    print(f"  A: {A.shape}")
    
    # 验证批次ID的合理性
    print(f"批次统计:")
    unique_batches = torch.unique(B)
    for batch_id in unique_batches:
        count = (B == batch_id).sum().item()
        print(f"  batch_{batch_id}: {count}个原子")
    
    # 验证边索引的合理性
    max_atom_idx = E_idx.max().item()
    print(f"边索引范围: 0 到 {max_atom_idx} (总原子数: {total_atoms})")
    
    if max_atom_idx >= total_atoms:
        print("❌ 错误：边索引超出原子范围！")
        return None
        
    return Z, E_idx, E, B, A

def test_dsan_encoder():
    """测试DSAN编码器的批次处理"""
    try:
        from models.DSAN.encoder import DSANEncoder
        
        # 创建数据
        data = create_simple_batch_data()
        if data is None:
            return False
            
        Z, E_idx, E, B, A = data
        device = Z.device
        
        print(f"\n=== 测试DSAN编码器 ===")
        
        # 创建编码器
        encoder = DSANEncoder(
            hidden_size=64,  # 更小的尺寸
            n_layers=1,      # 单层
            n_head=4,        # 更少的头
            n_rbf=8,         # 更少的RBF
            k_neighbors=3,   # 更少的邻居
            cutoff=5.0,      # 更小的截断
            use_adaptive_cutoff=False,  # 关闭自适应截断
            dropout=0.0      # 关闭dropout
        ).to(device)
        
        print("开始前向传播...")
        
        # 前向传播
        with torch.no_grad():
            output = encoder(Z, E_idx, E, B, A)
            print(f"✅ 成功！输出形状: {output.shape}")
            return True
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        
        # 详细的错误信息
        import traceback
        traceback.print_exc()
        
        # 如果是CUDA错误，提供额外信息
        if "CUDA" in str(e):
            print(f"\n🔍 CUDA错误详情:")
            print(f"   可能的原因：tensor索引越界")
            print(f"   建议：检查scatter操作的索引范围")
        
        return False

def main():
    """主函数"""
    print("🔍 DSAN批次索引调试")
    print("=" * 50)
    
    # 启用CUDA调试
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print("已启用CUDA同步调试模式")
    
    success = test_dsan_encoder()
    
    print("=" * 50)
    if success:
        print("🎉 调试测试通过！")
    else:
        print("⚠️ 仍存在问题，需要进一步调试")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
