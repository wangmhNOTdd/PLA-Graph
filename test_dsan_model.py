#!/usr/bin/env python3
"""
DSAN模型测试脚本 - 验证模型初始化和前向传播
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目路径
sys.path.append('.')

from models.DSAN.encoder import DSANEncoder, DSANLayer
from models import create_model
from utils.nn_utils import count_parameters

def test_dsan_encoder():
    """测试DSAN编码器基本功能"""
    print("=" * 60)
    print("🧪 测试DSAN编码器基本功能")
    print("=" * 60)
    
    # 设置参数
    hidden_size = 128
    n_layers = 2
    num_heads = 8
    k_neighbors = 6
    dropout = 0.1
    rbf_dim = 8
    cutoff = 8.0
    
    # 创建DSAN编码器
    try:
        encoder = DSANEncoder(
            hidden_size=hidden_size,
            n_layers=n_layers,
            num_heads=num_heads,
            k_neighbors=k_neighbors,
            dropout=dropout,
            use_geometry=True,
            rbf_dim=rbf_dim,
            cutoff=cutoff,
            memory_efficient=True  # 启用显存优化
        )
        print("✅ DSAN编码器创建成功")
        print(f"   参数数量: {count_parameters(encoder) / 1e6:.2f}M")
    except Exception as e:
        print(f"❌ DSAN编码器创建失败: {e}")
        return False
    
    # 创建测试数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = encoder.to(device)
    
    batch_size = 2
    n_atoms_per_graph = 50
    n_blocks_per_graph = 10
    n_channel = 1
    
    # 生成测试数据
    H = torch.randn(batch_size * n_atoms_per_graph, hidden_size, device=device)
    Z = torch.randn(batch_size * n_atoms_per_graph, n_channel, 3, device=device)
    
    # 块ID和批次ID
    block_id = torch.repeat_interleave(torch.arange(batch_size * n_blocks_per_graph), 
                                       n_atoms_per_graph // n_blocks_per_graph).to(device)
    batch_id = torch.repeat_interleave(torch.arange(batch_size), n_atoms_per_graph).to(device)
    
    # 简单的边索引（块间边）
    n_edges = 20
    edges = torch.randint(0, batch_size * n_blocks_per_graph, (2, n_edges), device=device)
    
    print(f"测试数据形状:")
    print(f"   H: {H.shape}")
    print(f"   Z: {Z.shape}")
    print(f"   block_id: {block_id.shape}")
    print(f"   batch_id: {batch_id.shape}")
    print(f"   edges: {edges.shape}")
    
    # 测试前向传播
    try:
        with torch.no_grad():
            atom_features, block_repr, graph_repr, pred_Z = encoder(
                H, Z, block_id, batch_id, edges
            )
        
        print("✅ 前向传播成功")
        print(f"   atom_features: {atom_features.shape}")
        print(f"   block_repr: {block_repr.shape}")
        print(f"   graph_repr: {graph_repr.shape}")
        print(f"   pred_Z: {pred_Z.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dsan_layer():
    """测试单个DSAN层"""
    print("=" * 60)
    print("🧪 测试单个DSAN层")
    print("=" * 60)
    
    # 参数
    hidden_size = 128
    num_heads = 8
    k_neighbors = 6
    
    try:
        layer = DSANLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            k_neighbors=k_neighbors,
            memory_efficient=True
        )
        print("✅ DSAN层创建成功")
        print(f"   参数数量: {count_parameters(layer) / 1e6:.3f}M")
    except Exception as e:
        print(f"❌ DSAN层创建失败: {e}")
        return False
    
    # 测试数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    layer = layer.to(device)
    
    n_atoms = 100
    atom_features = torch.randn(n_atoms, hidden_size, device=device)
    atom_positions = torch.randn(n_atoms, 3, device=device)
    block_id = torch.randint(0, 20, (n_atoms,), device=device)
    inter_edges = torch.randint(0, 20, (2, 30), device=device)
    
    try:
        with torch.no_grad():
            updated_features = layer(atom_features, atom_positions, block_id, inter_edges)
        
        print("✅ DSAN层前向传播成功")
        print(f"   输入特征: {atom_features.shape}")
        print(f"   输出特征: {updated_features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ DSAN层前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dsan_integration():
    """测试DSAN与训练框架的集成"""
    print("=" * 60)
    print("🧪 测试DSAN与训练框架集成")
    print("=" * 60)
    
    # 模拟训练参数
    class Args:
        task = 'PDBBind'  # 添加必需的task参数
        noisy_sigma = 0.0  # 添加噪声参数
        model_type = 'DSAN'
        hidden_size = 128
        n_channel = 1
        n_rbf = 16
        cutoff = 8.0
        radial_size = 8
        k_neighbors = 6
        n_layers = 2
        n_head = 8
        atom_level = False
        hierarchical = False
        no_block_embedding = False
        pretrain_ckpt = None  # 添加预训练检查点参数
    
    args = Args()
    
    try:
        model = create_model(args)
        print("✅ DSAN模型集成创建成功")
        print(f"   模型类型: {type(model)}")
        print(f"   参数数量: {count_parameters(model) / 1e6:.2f}M")
        
        # 检查编码器类型
        if hasattr(model, 'encoder'):
            print(f"   编码器类型: {type(model.encoder)}")
            print(f"   编码器参数: {count_parameters(model.encoder) / 1e6:.2f}M")
        
        return True
        
    except Exception as e:
        print(f"❌ DSAN模型集成失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_memory_usage():
    """检查显存使用情况"""
    print("=" * 60)
    print("💾 显存使用情况")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name()}")
        print(f"总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"已分配显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"已预留显存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"可用显存: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3:.1f} GB")
    else:
        print("CUDA不可用，使用CPU")

def main():
    """主测试函数"""
    print("🚀 DSAN模型完整测试")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU名称: {torch.cuda.get_device_name()}")
    
    # 初始显存检查
    check_memory_usage()
    
    # 运行所有测试
    tests = [
        ("DSAN层测试", test_dsan_layer),
        ("DSAN编码器测试", test_dsan_encoder),
        ("DSAN集成测试", test_dsan_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n▶️ 运行 {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name} 通过")
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
            results.append((test_name, False))
        
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 最终显存检查
    print("\n")
    check_memory_usage()
    
    # 汇总结果
    print("=" * 60)
    print("📋 测试结果汇总")
    print("=" * 60)
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 个测试通过")
    
    if passed == len(results):
        print("🎉 所有测试通过！DSAN模型准备就绪")
        return 0
    else:
        print("⚠️ 部分测试失败，请检查问题")
        return 1

if __name__ == "__main__":
    exit(main())
