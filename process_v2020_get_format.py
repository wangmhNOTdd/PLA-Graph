#!/usr/bin/env python3
"""
按照GET处理PDBBind的方式处理v2020-other-PL + CASF-2016数据集
"""

import os
import sys
import json
import pickle
import argparse
import random
import re
import math
from pathlib import Path

# 添加项目根目录到Python路径
PROJ_DIR = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..'
)
print(f'Project directory: {PROJ_DIR}')
sys.path.append(PROJ_DIR)

from utils.logger import print_log
from data.pdb_utils import Residue, VOCAB
from data.dataset import blocks_interface, blocks_to_data
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.mol2_to_blocks import mol2_to_blocks
# from data.converter.sdf_to_blocks import sdf_to_blocks  # 不存在，改用mol2_to_blocks

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
                    affinity_str = parts[3]
                    
                    affinity_value = parse_affinity_value(affinity_str)
                    if affinity_value is not None:
                        affinity_data[pdb_id] = affinity_value
    
    print(f"Loaded affinity data for {len(affinity_data)} complexes")
    return affinity_data

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

def process_one_v2020(pdb_id, label, interface_dist_th=8.0):
    """处理一个v2020-other-PL条目"""
    
    item = {}
    item['id'] = pdb_id
    item['affinity'] = {'neglog_aff': label}
    
    # v2020-other-PL文件路径
    prot_fname = f"./datasets/v2020-other-PL/v2020-other-PL/{pdb_id}/{pdb_id}_protein.pdb"
    ligand_fname = f"./datasets/v2020-other-PL/v2020-other-PL/{pdb_id}/{pdb_id}_ligand.mol2"
    
    try:
        list_blocks1 = pdb_to_list_blocks(prot_fname)
    except Exception as e:
        print_log(f'{pdb_id} protein parsing failed: {e}', level='ERROR')
        return None
    
    try:
        blocks2 = mol2_to_blocks(ligand_fname)
    except Exception as e:
        print_log(f'{pdb_id} ligand parsing failed: {e}', level='ERROR')
        return None
    
    blocks1 = []
    for b in list_blocks1:
        blocks1.extend(b)
    
    # 构建接口
    blocks1, _ = blocks_interface(blocks1, blocks2, interface_dist_th)
    if len(blocks1) == 0:
        print_log(f'{pdb_id} has no interface', level='ERROR')
        return None
    
    data = blocks_to_data(blocks1, blocks2)
    for key in data:
        if hasattr(data[key], 'tolist'):  # numpy array
            data[key] = data[key].tolist()
    
    item['data'] = data
    return item

def process_one_casf(pdb_id, label, interface_dist_th=8.0):
    """处理一个CASF-2016条目"""
    
    item = {}
    item['id'] = pdb_id
    item['affinity'] = {'neglog_aff': label}
    
    # CASF-2016文件路径
    prot_fname = f"./datasets/v2020-other-PL/CASF-2016/coreset/{pdb_id}/{pdb_id}_protein.pdb"
    ligand_fname = f"./datasets/v2020-other-PL/CASF-2016/coreset/{pdb_id}/{pdb_id}_ligand.mol2"
    
    try:
        list_blocks1 = pdb_to_list_blocks(prot_fname)
    except Exception as e:
        print_log(f'{pdb_id} protein parsing failed: {e}', level='ERROR')
        return None
    
    try:
        blocks2 = mol2_to_blocks(ligand_fname)
    except Exception as e:
        print_log(f'{pdb_id} ligand parsing failed: {e}', level='ERROR')
        return None
    
    blocks1 = []
    for b in list_blocks1:
        blocks1.extend(b)
    
    # 构建接口
    blocks1, _ = blocks_interface(blocks1, blocks2, interface_dist_th)
    if len(blocks1) == 0:
        print_log(f'{pdb_id} has no interface', level='ERROR')
        return None
    
    data = blocks_to_data(blocks1, blocks2)
    for key in data:
        if hasattr(data[key], 'tolist'):  # numpy array
            data[key] = data[key].tolist()
    
    item['data'] = data
    return item

def main():
    # 设置随机种子
    random.seed(42)
    
    # 创建输出目录
    output_dir = "./datasets/v2020-other-PL/processed_get_format"
    os.makedirs(output_dir, exist_ok=True)
    
    print_log("开始处理v2020-other-PL + CASF-2016数据集")
    
    # 加载亲和力数据
    affinity_data = load_affinity_data()
    
    # 获取CASF-2016条目
    casf_entries = get_casf_2016_entries()
    
    # 获取所有v2020-other-PL条目
    v2020_dir = "./datasets/v2020-other-PL/v2020-other-PL"
    all_v2020_entries = []
    
    print(f"Scanning {v2020_dir} for entries...")
    for entry in os.listdir(v2020_dir):
        entry_path = os.path.join(v2020_dir, entry)
        if os.path.isdir(entry_path) and entry not in ['index', 'readme']:
            pdb_file = os.path.join(entry_path, f"{entry}_protein.pdb")
            sdf_file = os.path.join(entry_path, f"{entry}_ligand.sdf")
            
            if os.path.exists(pdb_file) and os.path.exists(sdf_file):
                all_v2020_entries.append(entry)
    
    print(f"Found {len(all_v2020_entries)} valid entries in v2020-other-PL")
    
    # 找出用于训练+验证的条目（v2020-other-PL中除去CASF-2016且有亲和力数据的）
    train_valid_entries = []
    for entry in all_v2020_entries:
        if entry not in casf_entries and entry in affinity_data:
            train_valid_entries.append(entry)
    
    print(f"Training+Validation entries: {len(train_valid_entries)}")
    
    # 随机打乱并分割
    random.shuffle(train_valid_entries)
    n_train_valid = len(train_valid_entries)
    n_valid = int(n_train_valid * 0.1)
    n_train = n_train_valid - n_valid
    
    train_entries = train_valid_entries[:n_train]
    valid_entries = train_valid_entries[n_train:]
    
    print(f"Training entries: {len(train_entries)}")
    print(f"Validation entries: {len(valid_entries)}")
    
    # 处理训练集
    print_log("Processing training data...")
    train_data = []
    for i, pdb_id in enumerate(train_entries):
        if i % 1000 == 0:
            print_log(f"Processed {i}/{len(train_entries)} training entries")
        
        label = affinity_data[pdb_id]
        item = process_one_v2020(pdb_id, label)
        if item is not None:
            train_data.append(item)
        else:
            print_log(f"Failed to process training entry: {pdb_id}")
    
    # 处理验证集
    print_log("Processing validation data...")
    valid_data = []
    for i, pdb_id in enumerate(valid_entries):
        if i % 1000 == 0:
            print_log(f"Processed {i}/{len(valid_entries)} validation entries")
        
        label = affinity_data[pdb_id]
        item = process_one_v2020(pdb_id, label)
        if item is not None:
            valid_data.append(item)
        else:
            print_log(f"Failed to process validation entry: {pdb_id}")
    
    # 处理测试集（CASF-2016）
    print_log("Processing test data (CASF-2016 coreset)...")
    test_data = []
    for i, pdb_id in enumerate(sorted(casf_entries)):
        if i % 100 == 0:
            print_log(f"Processed {i}/{len(casf_entries)} test entries")
        
        label = parse_affinity_from_casf(pdb_id)
        if label is not None:
            item = process_one_casf(pdb_id, label)
            if item is not None:
                test_data.append(item)
            else:
                print_log(f"Failed to process test entry: {pdb_id}")
        else:
            print_log(f"No affinity found for test entry: {pdb_id}")
    
    # 保存数据
    train_file = os.path.join(output_dir, "train.pkl")
    valid_file = os.path.join(output_dir, "valid.pkl")
    test_file = os.path.join(output_dir, "test.pkl")
    
    print_log(f"Saving training data: {len(train_data)} entries -> {train_file}")
    with open(train_file, 'wb') as f:
        pickle.dump(train_data, f)
    
    print_log(f"Saving validation data: {len(valid_data)} entries -> {valid_file}")
    with open(valid_file, 'wb') as f:
        pickle.dump(valid_data, f)
    
    print_log(f"Saving test data: {len(test_data)} entries -> {test_file}")
    with open(test_file, 'wb') as f:
        pickle.dump(test_data, f)
    
    print_log("✅ Data processing completed successfully!")
    print_log(f"📊 Dataset statistics:")
    print_log(f"   Training: {len(train_data)} complexes")
    print_log(f"   Validation: {len(valid_data)} complexes")
    print_log(f"   Test: {len(test_data)} complexes")
    print_log(f"   Total: {len(train_data) + len(valid_data) + len(test_data)} complexes")

if __name__ == "__main__":
    main()
