#!/usr/bin/env python3
"""
重新处理v2020-other-PL数据集
1. 从INDEX_general_PL.2020文件读取亲和力数据
2. 训练集：v2020-other-PL中除去CASF-2016的数据
3. 验证集：从训练集中随机抽取10%
4. 测试集：CASF-2016的所有285条数据
5. 处理后数据保存在./datasets/v2020-other-PL/processed/
"""

import os
import pickle
import random
import pandas as pd
import re
import math
from pathlib import Path

def get_casf_2016_entries():
    """从CASF-2016/power_scoring/CoreSet.dat获取所有条目"""
    casf_file = "./datasets/v2020-other-PL/CASF-2016/power_scoring/CoreSet.dat"
    casf_entries = set()
    
    print(f"Reading CASF-2016 entries from {casf_file}")
    with open(casf_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 1:
                    pdb_id = parts[0].lower()  # 转换为小写以匹配目录名
                    casf_entries.add(pdb_id)
    
    print(f"Found {len(casf_entries)} CASF-2016 entries")
    return casf_entries

def parse_affinity_value(affinity_str):
    """解析亲和力字符串并转换为-log(Ka)值，只保留确定的值"""
    try:
        # 去除注释部分
        if '//' in affinity_str:
            affinity_str = affinity_str.split('//')[0].strip()
        
        # 跳过所有不确定的值（包含不等号或约等号）
        if any(symbol in affinity_str for symbol in ['<', '>', '~', '<=', '>=']):
            return None
        
        # 更广泛的正则表达式来匹配Ki, Kd, IC50
        # 支持更多数值格式，包括科学计数法
        patterns = [
            r'Ki=([0-9.]+(?:[eE][+-]?[0-9]+)?)([upnfmM]?)M?',      # Ki值
            r'Kd=([0-9.]+(?:[eE][+-]?[0-9]+)?)([upnfmM]?)M?',      # Kd值  
            r'IC50=([0-9.]+(?:[eE][+-]?[0-9]+)?)([upnfmM]?)M?',    # IC50值
        ]
        
        for pattern in patterns:
            match = re.search(pattern, affinity_str)
            if match:
                value = float(match.group(1))
                unit = match.group(2).lower() if match.group(2) else ''
                
                # 根据单位转换为摩尔浓度
                if unit == 'p':  # pM
                    value *= 1e-12
                elif unit == 'n':  # nM
                    value *= 1e-9
                elif unit == 'u':  # uM
                    value *= 1e-6
                elif unit == 'f':  # fM
                    value *= 1e-15
                elif unit == 'm':  # mM
                    value *= 1e-3
                else:  # 无单位或M，假设为M
                    pass
                
                # 转换为-log(Ka)
                # 注意：对于IC50，这是一个近似转换
                if value > 0:
                    return -math.log10(value)
        
        return None
    except Exception as e:
        print(f"Warning: Cannot parse affinity '{affinity_str}': {e}")
        return None

def load_affinity_data():
    """从INDEX_general_PL.2020文件加载亲和力数据"""
    index_file = "./datasets/v2020-other-PL/v2020-other-PL/index/INDEX_general_PL.2020"
    affinity_data = {}
    
    print(f"Loading affinity data from {index_file}")
    
    with open(index_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(None, 4)  # 分割为最多5部分
                if len(parts) >= 4:
                    pdb_id = parts[0].lower()
                    resolution = parts[1]
                    year = parts[2]
                    affinity_str = parts[3]
                    
                    affinity_value = parse_affinity_value(affinity_str)
                    if affinity_value is not None:
                        affinity_data[pdb_id] = affinity_value
    
    print(f"Loaded affinity data for {len(affinity_data)} complexes")
    return affinity_data

def get_v2020_other_pl_entries():
    """获取v2020-other-PL数据集中所有条目"""
    v2020_dir = "./datasets/v2020-other-PL/v2020-other-PL"
    all_entries = []
    
    print(f"Scanning {v2020_dir} for entries...")
    for entry in os.listdir(v2020_dir):
        entry_path = os.path.join(v2020_dir, entry)
        if os.path.isdir(entry_path) and entry not in ['index', 'readme']:
            # 检查是否包含必要文件
            pdb_file = os.path.join(entry_path, f"{entry}_protein.pdb")
            sdf_file = os.path.join(entry_path, f"{entry}_ligand.sdf")
            
            if os.path.exists(pdb_file) and os.path.exists(sdf_file):
                all_entries.append(entry)
    
    print(f"Found {len(all_entries)} valid entries in v2020-other-PL")
    return all_entries

def parse_affinity_from_casf(pdb_id):
    """从CASF-2016/power_scoring/CoreSet.dat解析亲和力值"""
    casf_file = "./datasets/v2020-other-PL/CASF-2016/power_scoring/CoreSet.dat"
    
    with open(casf_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 4 and parts[0].lower() == pdb_id.lower():
                    try:
                        log_ka = float(parts[3])
                        return log_ka
                    except ValueError:
                        print(f"Warning: Cannot parse logKa for {pdb_id}: {parts[3]}")
                        return None
    return None

def create_data_entry(pdb_id, affinity_value, data_split, is_casf=False):
    """创建数据条目"""
    if is_casf:
        # CASF-2016数据路径
        return {
            'id': pdb_id,
            'affinity': {'neglog_aff': affinity_value},
            'protein_path': f"./datasets/v2020-other-PL/CASF-2016/coreset/{pdb_id}/{pdb_id}_protein.pdb",
            'ligand_path': f"./datasets/v2020-other-PL/CASF-2016/coreset/{pdb_id}/{pdb_id}_ligand.sdf",
            'split': data_split
        }
    else:
        # v2020-other-PL数据路径
        return {
            'id': pdb_id,
            'affinity': {'neglog_aff': affinity_value},
            'protein_path': f"./datasets/v2020-other-PL/v2020-other-PL/{pdb_id}/{pdb_id}_protein.pdb",
            'ligand_path': f"./datasets/v2020-other-PL/v2020-other-PL/{pdb_id}/{pdb_id}_ligand.sdf",
            'split': data_split
        }

def main():
    # 设置随机种子
    random.seed(42)
    
    # 创建输出目录
    output_dir = "./datasets/v2020-other-PL/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载亲和力数据
    affinity_data = load_affinity_data()
    
    # 获取CASF-2016条目
    casf_entries = get_casf_2016_entries()
    
    # 获取所有v2020-other-PL条目
    all_v2020_entries = get_v2020_other_pl_entries()
    
    # 找出用于训练的条目（v2020-other-PL中除去CASF-2016的数据）
    train_valid_entries = []
    for entry in all_v2020_entries:
        if entry not in casf_entries and entry in affinity_data:
            train_valid_entries.append(entry)
    
    print(f"Training+Validation entries (v2020-other-PL excluding CASF-2016): {len(train_valid_entries)}")
    
    # 随机打乱训练+验证条目
    random.shuffle(train_valid_entries)
    
    # 分割为训练集(90%)和验证集(10%)
    n_train_valid = len(train_valid_entries)
    n_valid = int(n_train_valid * 0.1)
    n_train = n_train_valid - n_valid
    
    train_entries = train_valid_entries[:n_train]
    valid_entries = train_valid_entries[n_train:]
    
    print(f"Training entries: {len(train_entries)}")
    print(f"Validation entries: {len(valid_entries)}")
    
    # 处理训练集数据
    train_data = []
    print("Processing training data...")
    for i, pdb_id in enumerate(train_entries):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(train_entries)} training entries")
        
        affinity = affinity_data[pdb_id]
        data_entry = create_data_entry(pdb_id, affinity, 'train', is_casf=False)
        train_data.append(data_entry)
    
    # 处理验证集数据
    valid_data = []
    print("Processing validation data...")
    for i, pdb_id in enumerate(valid_entries):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(valid_entries)} validation entries")
        
        affinity = affinity_data[pdb_id]
        data_entry = create_data_entry(pdb_id, affinity, 'valid', is_casf=False)
        valid_data.append(data_entry)
    
    # 处理测试集数据（CASF-2016所有285条数据）
    test_data = []
    print("Processing test data (CASF-2016 coreset)...")
    for i, pdb_id in enumerate(sorted(casf_entries)):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(casf_entries)} test entries")
        
        affinity = parse_affinity_from_casf(pdb_id)
        
        if affinity is not None:
            data_entry = create_data_entry(pdb_id, affinity, 'test', is_casf=True)
            test_data.append(data_entry)
        else:
            print(f"Warning: No affinity found for test entry {pdb_id}")
    
    # 保存数据
    train_file = os.path.join(output_dir, "train.pkl")
    valid_file = os.path.join(output_dir, "valid.pkl")
    test_file = os.path.join(output_dir, "test.pkl")
    
    print(f"\nSaving data...")
    print(f"Training set: {len(train_data)} entries -> {train_file}")
    print(f"Validation set: {len(valid_data)} entries -> {valid_file}")
    print(f"Test set: {len(test_data)} entries -> {test_file}")
    
    with open(train_file, 'wb') as f:
        pickle.dump(train_data, f)
    
    with open(valid_file, 'wb') as f:
        pickle.dump(valid_data, f)
    
    with open(test_file, 'wb') as f:
        pickle.dump(test_data, f)
    
    print(f"\n✅ Data processing completed successfully!")
    print(f"📊 Dataset statistics:")
    print(f"   Training: {len(train_data)} complexes (90% of v2020-other-PL)")
    print(f"   Validation: {len(valid_data)} complexes (10% of v2020-other-PL)")
    print(f"   Test: {len(test_data)} complexes (CASF-2016 coreset)")
    print(f"   Total: {len(train_data) + len(valid_data) + len(test_data)} complexes")
    
    # 保存数据集统计信息
    stats_file = os.path.join(output_dir, "dataset_stats.txt")
    with open(stats_file, 'w') as f:
        f.write("v2020-other-PL Dataset Statistics (Final)\n")
        f.write("==========================================\n")
        f.write(f"Training set: {len(train_data)} complexes (90% of v2020-other-PL excluding CASF-2016)\n")
        f.write(f"Validation set: {len(valid_data)} complexes (10% of v2020-other-PL excluding CASF-2016)\n")
        f.write(f"Test set: {len(test_data)} complexes (CASF-2016 coreset)\n")
        f.write(f"Total: {len(train_data) + len(valid_data) + len(test_data)} complexes\n")
        f.write(f"\nCASF-2016 total entries: {len(casf_entries)}\n")
        f.write(f"CASF-2016 entries with affinity data: {len(test_data)}\n")
        f.write(f"v2020-other-PL total entries: {len(all_v2020_entries)}\n")
        f.write(f"v2020-other-PL entries with affinity data: {len(affinity_data)}\n")

if __name__ == "__main__":
    main()
