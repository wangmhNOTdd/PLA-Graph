#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
训练结果分析工具
用于分析和可视化已完成的训练结果
"""

import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def load_training_history(history_path):
    """加载训练历史"""
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    return history


def plot_comparison_curves(history_list, labels, save_dir):
    """绘制多个训练历史的对比曲线"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('GCN+ESA 训练对比分析', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, (history, label) in enumerate(zip(history_list, labels)):
        color = colors[i % len(colors)]
        epochs = history.get('epoch', [])
        
        if not epochs:
            continue
        
        # 1. 损失曲线
        ax = axes[0, 0]
        if 'train_loss' in history:
            ax.plot(epochs, history['train_loss'], color=color, linestyle='-', 
                   label=f'{label} (训练)', linewidth=2, alpha=0.7)
        if 'valid_loss' in history:
            ax.plot(epochs, history['valid_loss'], color=color, linestyle='--', 
                   label=f'{label} (验证)', linewidth=2, alpha=0.7)
        
        # 2. Pearson相关系数
        ax = axes[0, 1]
        if 'valid_pearson' in history:
            ax.plot(epochs, history['valid_pearson'], color=color, linestyle='-',
                   label=f'{label} (验证)', linewidth=2, alpha=0.7)
        
        # 3. Spearman相关系数
        ax = axes[0, 2]
        if 'valid_spearman' in history:
            ax.plot(epochs, history['valid_spearman'], color=color, linestyle='-',
                   label=f'{label} (验证)', linewidth=2, alpha=0.7)
        
        # 4. RMSE
        ax = axes[1, 0]
        if 'valid_rmse' in history:
            ax.plot(epochs, history['valid_rmse'], color=color, linestyle='-',
                   label=f'{label} (验证)', linewidth=2, alpha=0.7)
        
        # 5. MAE
        ax = axes[1, 1]
        if 'valid_mae' in history:
            ax.plot(epochs, history['valid_mae'], color=color, linestyle='-',
                   label=f'{label} (验证)', linewidth=2, alpha=0.7)
    
    # 设置子图标题和标签
    titles = ['损失函数', 'Pearson 相关系数', 'Spearman 相关系数', 'RMSE', 'MAE', '性能对比']
    y_labels = ['Loss', 'Pearson Correlation', 'Spearman Correlation', 'RMSE', 'MAE', '']
    
    for i, (ax, title, ylabel) in enumerate(zip(axes.flat[:5], titles[:5], y_labels[:5])):
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 6. 性能对比表格
    ax = axes[1, 2]
    ax.set_title('最佳性能对比')
    
    # 创建性能对比表格
    table_data = []
    for i, (history, label) in enumerate(zip(history_list, labels)):
        if 'valid_pearson' in history and history['valid_pearson']:
            best_epoch = np.argmax(history['valid_pearson'])
            best_pearson = history['valid_pearson'][best_epoch]
            best_spearman = history['valid_spearman'][best_epoch] if 'valid_spearman' in history else 0
            best_rmse = history['valid_rmse'][best_epoch] if 'valid_rmse' in history else 0
            
            table_data.append([
                label[:15] + '...' if len(label) > 15 else label,
                f'{best_pearson:.3f}',
                f'{best_spearman:.3f}', 
                f'{best_rmse:.3f}'
            ])
    
    if table_data:
        headers = ['模型', 'Pearson', 'Spearman', 'RMSE']
        table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # 设置表格样式
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f2f2f2')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_curves.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'comparison_curves.pdf'), bbox_inches='tight')
    print(f"对比曲线已保存到: {os.path.join(save_dir, 'comparison_curves.png')}")
    
    try:
        plt.show()
    except:
        print("无法显示图表，但已保存到文件")
    
    plt.close()


def print_detailed_analysis(history, label=""):
    """打印详细的训练分析"""
    print("\n" + "="*60)
    print(f"详细分析: {label}")
    print("="*60)
    
    if not history.get('epoch'):
        print("没有找到训练历史数据")
        return
    
    total_epochs = len(history['epoch'])
    print(f"总训练轮数: {total_epochs}")
    
    # 训练损失分析
    if 'train_loss' in history:
        train_losses = history['train_loss']
        print(f"\n训练损失:")
        print(f"  初始: {train_losses[0]:.4f}")
        print(f"  最终: {train_losses[-1]:.4f}")
        print(f"  最低: {min(train_losses):.4f}")
        print(f"  改善: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%")
    
    # 验证性能分析
    if 'valid_pearson' in history:
        pearson_values = history['valid_pearson']
        best_epoch = np.argmax(pearson_values)
        best_pearson = pearson_values[best_epoch]
        
        print(f"\n验证集性能:")
        print(f"  最佳 Epoch: {history['epoch'][best_epoch]}")
        print(f"  最佳 Pearson: {best_pearson:.4f}")
        
        if 'valid_spearman' in history:
            print(f"  最佳 Spearman: {history['valid_spearman'][best_epoch]:.4f}")
        if 'valid_rmse' in history:
            print(f"  最佳 RMSE: {history['valid_rmse'][best_epoch]:.4f}")
        if 'valid_mae' in history:
            print(f"  最佳 MAE: {history['valid_mae'][best_epoch]:.4f}")
    
    # 测试性能分析
    if 'test_pearson' in history and history['test_pearson']:
        final_test_pearson = history['test_pearson'][-1]
        print(f"\n最终测试性能:")
        print(f"  Pearson: {final_test_pearson:.4f}")
        if 'test_spearman' in history:
            print(f"  Spearman: {history['test_spearman'][-1]:.4f}")
        if 'test_rmse' in history:
            print(f"  RMSE: {history['test_rmse'][-1]:.4f}")
        if 'test_mae' in history:
            print(f"  MAE: {history['test_mae'][-1]:.4f}")
    
    # 训练稳定性分析
    if 'valid_pearson' in history and len(history['valid_pearson']) > 10:
        pearson_values = np.array(history['valid_pearson'])
        # 计算后半段的标准差作为稳定性指标
        second_half = pearson_values[len(pearson_values)//2:]
        stability = np.std(second_half)
        print(f"\n训练稳定性:")
        print(f"  后半段 Pearson 标准差: {stability:.4f}")
        print(f"  稳定性评级: {'高' if stability < 0.02 else '中' if stability < 0.05 else '低'}")


def main():
    parser = argparse.ArgumentParser(description='训练结果分析工具')
    parser.add_argument('--history_files', type=str, nargs='+', required=True,
                        help='训练历史文件路径 (training_history.pkl)')
    parser.add_argument('--labels', type=str, nargs='+', 
                        help='对应的标签名称')
    parser.add_argument('--save_dir', type=str, default='./analysis_results',
                        help='分析结果保存目录')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载训练历史
    history_list = []
    labels = args.labels if args.labels else []
    
    for i, history_file in enumerate(args.history_files):
        if os.path.exists(history_file):
            history = load_training_history(history_file)
            history_list.append(history)
            
            # 如果没有提供标签，使用文件名
            if i >= len(labels):
                labels.append(os.path.basename(os.path.dirname(history_file)))
        else:
            print(f"警告: 文件不存在 {history_file}")
    
    if not history_list:
        print("没有找到有效的训练历史文件")
        return
    
    # 打印详细分析
    for history, label in zip(history_list, labels):
        print_detailed_analysis(history, label)
    
    # 绘制对比曲线
    if len(history_list) > 1:
        plot_comparison_curves(history_list, labels, args.save_dir)
    else:
        # 单个历史，使用原有的绘制方法
        from train_simple_gcn_esa import TrainingMonitor
        monitor = TrainingMonitor(args.save_dir)
        monitor.history = history_list[0]
        monitor.plot_training_curves()
    
    print(f"\n分析完成！结果保存在: {args.save_dir}")


if __name__ == '__main__':
    main()
