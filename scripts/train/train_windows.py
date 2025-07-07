#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Windows版本的训练脚本，替代原始的train.sh
"""
import os
import sys
import json
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Windows training script for GET')
    parser.add_argument('config_path', type=str, help='Path to config JSON file')
    parser.add_argument('--gpu', type=str, default='0', help='GPU devices to use (comma separated)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 获取项目根目录
    code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    print(f"Locate the project folder at {code_dir}")
    
    # 读取配置文件
    if not os.path.exists(args.config_path):
        print(f"Config file not found: {args.config_path}")
        sys.exit(1)
    
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    # 解析配置为命令行参数
    config_args = []
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                config_args.append(f'--{key}')
        else:
            config_args.extend([f'--{key}', str(value)])
    
    # 设置GPU环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(f"Using GPUs: {args.gpu}")
    
    # GPU列表
    gpu_list = args.gpu.split(',') if args.gpu != '-1' else []
    
    # 构建训练命令
    train_script = os.path.join(code_dir, 'train.py')
    cmd = [sys.executable, train_script]
    
    # 添加GPU参数
    if gpu_list:
        cmd.extend(['--gpus'] + [str(i) for i in range(len(gpu_list))])
    else:
        cmd.extend(['--gpus', '-1'])
    
    # 添加其他配置参数
    cmd.extend(config_args)
    
    print(f"Executing command: {' '.join(cmd)}")
    
    # 切换到项目目录并执行训练
    os.chdir(code_dir)
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Training failed with return code {e.returncode}")
        return e.returncode

if __name__ == '__main__':
    sys.exit(main())
