#!/usr/bin/env python3
"""
DSAN批量训练测试脚本 - 验证batch_size=4是否能正常工作
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append('.')

from models.DSAN.encoder import DSANEncoder, DSANLayer
from models import create_model

def test_batch_processing():
    """测试DSAN的批量处理能力"""
    print("🧪 测试DSAN批量处理（batch_size=4）")
    print("=" * 60)
    
    # 模拟训练参数
    class Args:
        task = 'PDBBind'
        noisy_sigma = 0.0
        model_type = 'DSAN'
        hidden_size = 128
        n_channel = 1
        n_rbf = 16
        cutoff = 8.0
        radial_size = 8
        k_neighbors = 6
        n_layers = 2  # 减少层数加快测试
        n_head = 8
        atom_level = False
        hierarchical = False
        no_block_embedding = False
        pretrain_ckpt = None
    
    args = Args()
    
    try:
        # 创建模型
        model = create_model(args)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"✅ 模型创建成功，设备: {device}")
        
        # 创建批量测试数据 (batch_size=4)
        batch_size = 4
        max_atoms_per_graph = 100
        max_blocks_per_graph = 20
        n_channel = 1
        
        # 模拟不同大小的分子
        molecule_sizes = [80, 95, 60, 75]  # 不同分子的原子数
        block_sizes = [16, 19, 12, 15]     # 不同分子的块数
        
        # 构建批量数据
        all_atom_features = []
        all_positions = []
        all_block_features = []
        all_atom_positions = []
        all_block_lengths = []
        all_lengths = []
        all_segment_ids = []
        
        atom_offset = 0
        block_offset = 0
        
        for batch_idx in range(batch_size):
            n_atoms = molecule_sizes[batch_idx]
            n_blocks = block_sizes[batch_idx]
            
            # 原子特征和坐标
            Z = torch.randn(n_atoms, n_channel, 3, device=device)
            B = torch.randint(0, 10, (n_blocks,), device=device)  # 块类型
            A = torch.randint(0, 10, (n_atoms,), device=device)  # 原子类型
            atom_positions = torch.randn(n_atoms, 3, device=device)
            
            # 块长度（每个块包含的原子数）
            atoms_per_block = n_atoms // n_blocks
            block_lengths_single = torch.full((n_blocks,), atoms_per_block, device=device)
            # 调整最后一个块的长度
            remaining_atoms = n_atoms - (atoms_per_block * n_blocks)
            if remaining_atoms > 0:
                block_lengths_single[-1] += remaining_atoms
            
            all_atom_features.append(Z)
            all_positions.append(Z)
            all_block_features.append(B)
            all_atom_positions.append(atom_positions)
            all_block_lengths.append(block_lengths_single)
            all_lengths.append(torch.tensor([n_blocks], device=device))
            all_segment_ids.append(torch.full((n_blocks,), batch_idx, device=device))
        
        # 合并批量数据
        Z_batch = torch.cat(all_atom_features, dim=0)  # [total_atoms, n_channel, 3]
        B_batch = torch.cat(all_block_features, dim=0)  # [total_blocks]
        A_batch = torch.cat([torch.randint(0, 10, (mol_size,), device=device) 
                            for mol_size in molecule_sizes], dim=0)  # [total_atoms]
        atom_positions_batch = torch.cat(all_atom_positions, dim=0)  # [total_atoms, 3]
        block_lengths_batch = torch.cat(all_block_lengths, dim=0)  # [total_blocks]
        lengths_batch = torch.tensor([sum(block_sizes[:i+1]) for i in range(batch_size)], device=device)
        segment_ids_batch = torch.cat(all_segment_ids, dim=0)  # [total_blocks]
        
        print(f"批量数据形状:")
        print(f"   Z: {Z_batch.shape}")
        print(f"   B: {B_batch.shape}")
        print(f"   A: {A_batch.shape}")
        print(f"   atom_positions: {atom_positions_batch.shape}")
        print(f"   block_lengths: {block_lengths_batch.shape}")
        print(f"   lengths: {lengths_batch.shape}")
        print(f"   segment_ids: {segment_ids_batch.shape}")
        
        # 创建标签（亲和力）
        label = torch.randn(batch_size, device=device)
        
        print("🔄 执行前向传播...")
        
        # 测试前向传播
        with torch.no_grad():  # 节省内存
            try:
                loss = model(
                    Z=Z_batch,
                    B=B_batch, 
                    A=A_batch,
                    atom_positions=atom_positions_batch,
                    block_lengths=block_lengths_batch,
                    lengths=lengths_batch,
                    segment_ids=segment_ids_batch,
                    label=label
                )
                
                print(f"✅ 前向传播成功!")
                print(f"   损失值: {loss.item():.6f}")
                print(f"   损失形状: {loss.shape}")
                
                return True
                
            except RuntimeError as e:
                if "illegal memory access" in str(e).lower():
                    print(f"❌ CUDA非法内存访问错误: {e}")
                    print("这表明仍存在索引越界问题")
                    return False
                else:
                    print(f"❌ 其他运行时错误: {e}")
                    return False
            except Exception as e:
                print(f"❌ 前向传播失败: {e}")
                import traceback
                traceback.print_exc()
                return False
        
    except Exception as e:
        print(f"❌ 测试设置失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_memory_status():
    """检查内存状态"""
    if torch.cuda.is_available():
        print(f"\n💾 内存状态:")
        print(f"   已分配显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"   已预留显存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   总显存: {total_memory:.1f} GB")
        usage_percent = (torch.cuda.memory_reserved() / torch.cuda.get_device_properties(0).total_memory) * 100
        print(f"   显存利用率: {usage_percent:.1f}%")

def main():
    """主测试函数"""
    print("🚀 DSAN批量处理测试")
    print(f"PyTorch版本: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA版本: {torch.version.cuda}")
    
    # 初始内存状态
    check_memory_status()
    
    # 执行测试
    success = test_batch_processing()
    
    # 清理内存
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 最终内存状态
    check_memory_status()
    
    print("=" * 60)
    if success:
        print("🎉 批量处理测试成功！可以使用batch_size=4")
        print("现在可以放心地进行批量训练")
        return 0
    else:
        print("⚠️ 批量处理测试失败，需要进一步调试")
        print("建议继续使用batch_size=1或尝试更小的批次")
        return 1

if __name__ == "__main__":
    exit(main())
