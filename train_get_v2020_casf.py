#!/usr/bin/env python3
"""
训练GET模型在v2020-other-PL + CASF-2016数据集上
"""

import os
import json
import subprocess
import sys
from datetime import datetime

def run_training():
    """运行GET模型训练"""
    
    # 配置文件路径
    config_file = "./get_v2020_casf_config.json"
    
    # 检查配置文件是否存在
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return
    
    # 读取配置
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print("🚀 开始训练GET模型")
    print("=" * 60)
    print(f"📊 数据集配置:")
    print(f"   训练集: {config['train_set']}")
    print(f"   验证集: {config['valid_set']}")
    print(f"   保存目录: {config['save_dir']}")
    print(f"🔧 模型配置:")
    print(f"   模型类型: {config['model_type']}")
    print(f"   隐藏层大小: {config['hidden_size']}")
    print(f"   层数: {config['n_layers']}")
    print(f"   学习率: {config['lr']}")
    print(f"   最大轮数: {config['max_epoch']}")
    print("=" * 60)
    
    # 确保保存目录存在
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # 构建训练命令
    cmd = [
        "python", "train.py",
        "--train_set", config["train_set"],
        "--valid_set", config["valid_set"],
        "--save_dir", config["save_dir"],
        "--task", config["task"],
        "--lr", str(config["lr"]),
        "--final_lr", str(config["final_lr"]),
        "--max_epoch", str(config["max_epoch"]),
        "--save_topk", str(config["save_topk"]),
        "--max_n_vertex_per_gpu", str(config["max_n_vertex_per_gpu"]),
        "--valid_max_n_vertex_per_gpu", str(config["valid_max_n_vertex_per_gpu"]),
        "--model_type", config["model_type"],
        "--hidden_size", str(config["hidden_size"]),
        "--embed_dim", str(config["embed_dim"]),
        "--n_layers", str(config["n_layers"]),
        "--n_channel", str(config["n_channel"]),
        "--n_rbf", str(config["n_rbf"]),
        "--cutoff", str(config["cutoff"]),
        "--radial_size", str(config["radial_size"]),
        "--radial_dist_cutoff", str(config["radial_dist_cutoff"]),
        "--k_neighbors", str(config["k_neighbors"]),
        "--n_head", str(config["n_head"]),
        "--warmup", str(config["warmup"]),
        "--grad_clip", str(config["grad_clip"]),
        "--num_workers", str(config["num_workers"]),
        "--seed", str(config["seed"]),
        "--patience", str(config["patience"]),
        "--gpus", "0"  # 使用第一个GPU
    ]
    
    # 添加布尔参数
    if config.get("shuffle", False):
        cmd.append("--shuffle")
    if config.get("atom_level", False):
        cmd.append("--atom_level")
    if config.get("hierarchical", False):
        cmd.append("--hierarchical")
    if config.get("no_block_embedding", False):
        cmd.append("--no_block_embedding")
    
    print(f"🔥 执行命令: {' '.join(cmd)}")
    print("=" * 60)
    
    # 记录开始时间
    start_time = datetime.now()
    print(f"⏰ 训练开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 执行训练命令
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        # 记录结束时间
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("=" * 60)
        print(f"✅ 训练完成!")
        print(f"⏰ 结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  训练耗时: {duration}")
        print(f"📁 模型保存在: {config['save_dir']}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")
        return False
    except KeyboardInterrupt:
        print(f"⚠️  训练被用户中断")
        return False
    
    return True

if __name__ == "__main__":
    success = run_training()
    if success:
        print("\\n🎉 GET模型训练成功完成!")
    else:
        print("\\n💥 GET模型训练失败!")
        sys.exit(1)
