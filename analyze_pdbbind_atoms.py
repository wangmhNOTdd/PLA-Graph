#!/usr/bin/env python3
"""
统计PDBbind数据集中复合物的原子个数
包括极端值、平均值、中位数等统计信息
"""

import pickle
import numpy as np
from collections import defaultdict
import os
import sys

def load_dataset(pkl_path):
    """加载pickle数据集"""
    print(f"正在加载数据集: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    print(f"数据集大小: {len(data)} 个复合物")
    return data

def analyze_atom_counts(data, dataset_name):
    """分析原子个数统计"""
    print(f"\n=== {dataset_name} 数据集原子个数统计 ===")
    
    atom_counts = []
    block_counts = []
    segment_info = defaultdict(list)
    
    for i, item in enumerate(data):
        if isinstance(item, dict) and 'data' in item:
            # PDBBindBenchmark 格式
            sample_data = item['data']
        else:
            # 直接的数据格式
            sample_data = item
            
        # 统计原子个数
        if 'A' in sample_data:
            n_atoms = len(sample_data['A'])
            atom_counts.append(n_atoms)
        
        # 统计块个数  
        if 'B' in sample_data:
            n_blocks = len(sample_data['B'])
            block_counts.append(n_blocks)
            
        # 统计每个segment的信息
        if 'segment_ids' in sample_data:
            segments = sample_data['segment_ids']
            unique_segments = set(segments)
            for seg_id in unique_segments:
                seg_blocks = [i for i, s in enumerate(segments) if s == seg_id]
                segment_info[seg_id].append(len(seg_blocks))
    
    atom_counts = np.array(atom_counts)
    block_counts = np.array(block_counts)
    
    # 基本统计
    print(f"\n📊 原子个数统计:")
    print(f"   样本数量: {len(atom_counts)}")
    print(f"   最小值: {np.min(atom_counts)} 个原子")
    print(f"   最大值: {np.max(atom_counts)} 个原子") 
    print(f"   平均值: {np.mean(atom_counts):.2f} 个原子")
    print(f"   中位数: {np.median(atom_counts):.2f} 个原子")
    print(f"   标准差: {np.std(atom_counts):.2f} 个原子")
    
    # 百分位数
    percentiles = [5, 10, 25, 75, 90, 95]
    print(f"\n📈 原子个数百分位数:")
    for p in percentiles:
        val = np.percentile(atom_counts, p)
        print(f"   {p}%: {val:.0f} 个原子")
    
    # 块统计
    if len(block_counts) > 0:
        print(f"\n🧱 块个数统计:")
        print(f"   最小值: {np.min(block_counts)} 个块")
        print(f"   最大值: {np.max(block_counts)} 个块")
        print(f"   平均值: {np.mean(block_counts):.2f} 个块")
        print(f"   中位数: {np.median(block_counts):.2f} 个块")
    
    # Segment分析
    if segment_info:
        print(f"\n🏗️ Segment分析:")
        for seg_id, block_counts_per_seg in segment_info.items():
            if len(block_counts_per_seg) > 0:
                avg_blocks = np.mean(block_counts_per_seg)
                print(f"   Segment {seg_id}: 平均 {avg_blocks:.2f} 个块")
    
    # 分布区间统计
    print(f"\n📈 原子个数分布:")
    ranges = [(0, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, float('inf'))]
    for low, high in ranges:
        if high == float('inf'):
            count = np.sum(atom_counts >= low)
            print(f"   {low}+ 个原子: {count} 个复合物 ({count/len(atom_counts)*100:.1f}%)")
        else:
            count = np.sum((atom_counts >= low) & (atom_counts < high))
            print(f"   {low}-{high} 个原子: {count} 个复合物 ({count/len(atom_counts)*100:.1f}%)")
    
    return atom_counts, block_counts

def main():
    """主函数"""
    print("🧬 PDBbind数据集原子个数统计分析")
    print("=" * 60)
    
    # 数据集路径
    datasets = {
        'Train': './datasets/PDBBind/processed/identity30/train.pkl',
        'Valid': './datasets/PDBBind/processed/identity30/valid.pkl', 
        'Test': './datasets/PDBBind/processed/identity30/test.pkl'
    }
    
    atom_counts_dict = {}
    
    # 分析每个数据集
    for name, path in datasets.items():
        if os.path.exists(path):
            try:
                data = load_dataset(path)
                atom_counts, block_counts = analyze_atom_counts(data, name)
                atom_counts_dict[name] = atom_counts
            except Exception as e:
                print(f"❌ 处理{name}数据集时出错: {e}")
        else:
            print(f"⚠️  数据集文件不存在: {path}")
    
    # 合并统计
    if atom_counts_dict:
        print(f"\n=== 合并统计 ===")
        all_atom_counts = np.concatenate(list(atom_counts_dict.values()))
        print(f"📊 所有数据集合并统计:")
        print(f"   总样本数: {len(all_atom_counts)}")
        print(f"   最小值: {np.min(all_atom_counts)} 个原子")
        print(f"   最大值: {np.max(all_atom_counts)} 个原子")
        print(f"   平均值: {np.mean(all_atom_counts):.2f} 个原子")
        print(f"   中位数: {np.median(all_atom_counts):.2f} 个原子")
        print(f"   标准差: {np.std(all_atom_counts):.2f} 个原子")
        
        # 寻找极端值样本
        print(f"\n🔍 极端值样本:")
        min_idx = np.argmin(all_atom_counts)
        max_idx = np.argmax(all_atom_counts)
        print(f"   最小原子数样本: {np.min(all_atom_counts)} 个原子")
        print(f"   最大原子数样本: {np.max(all_atom_counts)} 个原子")
        
        # 绘制分布图（需要matplotlib）
        # try:
        #     plot_distribution(atom_counts_dict, './analysis_results')
        # except Exception as e:
        #     print(f"⚠️  绘图时出错: {e}")
        
        print(f"\n💾 统计结果已保存到终端输出")
    
    print("\n✅ 分析完成!")

if __name__ == "__main__":
    main()
