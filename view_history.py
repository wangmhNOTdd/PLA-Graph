#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
查看训练历史的简单脚本
"""

import pickle
import sys
import os

def view_training_history(history_path):
    """查看训练历史"""
    try:
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        
        print("="*60)
        print("训练历史数据概览")
        print("="*60)
        
        print(f"可用的指标: {list(history.keys())}")
        print(f"训练轮数: {len(history.get('epoch', []))}")
        
        if 'epoch' in history:
            print(f"Epoch 范围: {history['epoch'][0]} - {history['epoch'][-1]}")
        
        print("\n训练损失:")
        if 'train_loss' in history:
            losses = history['train_loss']
            print(f"  初始损失: {losses[0]:.4f}")
            print(f"  最终损失: {losses[-1]:.4f}")
            print(f"  最低损失: {min(losses):.4f}")
        
        print("\n验证性能:")
        if 'valid_pearson' in history:
            pearson_values = history['valid_pearson']
            best_idx = max(range(len(pearson_values)), key=lambda i: pearson_values[i])
            print(f"  最佳 Pearson: {pearson_values[best_idx]:.4f} (Epoch {history['epoch'][best_idx]})")
            
        if 'valid_spearman' in history:
            spearman_values = history['valid_spearman']
            best_idx = max(range(len(spearman_values)), key=lambda i: spearman_values[i])
            print(f"  最佳 Spearman: {spearman_values[best_idx]:.4f} (Epoch {history['epoch'][best_idx]})")
            
        if 'valid_rmse' in history:
            rmse_values = history['valid_rmse']
            best_idx = min(range(len(rmse_values)), key=lambda i: rmse_values[i])
            print(f"  最佳 RMSE: {rmse_values[best_idx]:.4f} (Epoch {history['epoch'][best_idx]})")
        
        print("\n最终测试性能:")
        if 'final_test_pearson' in history:
            print(f"  Pearson: {history['final_test_pearson']:.4f}")
        if 'final_test_spearman' in history:
            print(f"  Spearman: {history['final_test_spearman']:.4f}")
        if 'final_test_rmse' in history:
            print(f"  RMSE: {history['final_test_rmse']:.4f}")
        if 'final_test_mae' in history:
            print(f"  MAE: {history['final_test_mae']:.4f}")
        
        print("\n详细数据:")
        for key, values in history.items():
            if isinstance(values, list):
                print(f"  {key}: {len(values)} 个值")
            else:
                print(f"  {key}: {values}")
                
    except Exception as e:
        print(f"读取训练历史失败: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python view_history.py <training_history.pkl路径>")
        sys.exit(1)
    
    history_path = sys.argv[1]
    if not os.path.exists(history_path):
        print(f"文件不存在: {history_path}")
        sys.exit(1)
    
    view_training_history(history_path)
