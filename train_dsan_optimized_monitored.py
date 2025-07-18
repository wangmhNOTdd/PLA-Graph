#!/usr/bin/env python3
"""
DSAN性能监控脚本
监控训练时的GPU使用率、内存占用和训练速度
"""

import subprocess
import sys
import time
import threading
import torch
import psutil
import os

class PerformanceMonitor:
    def __init__(self):
        self.monitoring = False
        self.stats = {
            'gpu_memory': [],
            'gpu_utilization': [],
            'cpu_usage': [],
            'ram_usage': [],
            'timestamps': []
        }
        
    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # 获取当前时间
                current_time = time.time()
                
                # GPU内存使用
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                    # GPU使用率（近似）
                    gpu_util = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
                else:
                    gpu_memory = 0
                    gpu_util = 0
                
                # CPU和RAM使用率
                cpu_usage = psutil.cpu_percent()
                ram_usage = psutil.virtual_memory().percent
                
                # 记录数据
                self.stats['gpu_memory'].append(gpu_memory)
                self.stats['gpu_utilization'].append(gpu_util)
                self.stats['cpu_usage'].append(cpu_usage)
                self.stats['ram_usage'].append(ram_usage)
                self.stats['timestamps'].append(current_time)
                
                time.sleep(1)  # 每秒采样一次
                
            except Exception as e:
                print(f"监控出错: {e}")
                time.sleep(1)
    
    def get_summary(self):
        """获取监控摘要"""
        if not self.stats['timestamps']:
            return "没有监控数据"
        
        duration = self.stats['timestamps'][-1] - self.stats['timestamps'][0]
        
        summary = f"""
性能监控摘要:
={'='*40}
监控时长: {duration:.1f} 秒
GPU内存使用: 平均 {sum(self.stats['gpu_memory'])/len(self.stats['gpu_memory']):.2f} GB, 峰值 {max(self.stats['gpu_memory']):.2f} GB
CPU使用率: 平均 {sum(self.stats['cpu_usage'])/len(self.stats['cpu_usage']):.1f}%
RAM使用率: 平均 {sum(self.stats['ram_usage'])/len(self.stats['ram_usage']):.1f}%
"""
        return summary

def main():
    print("🚀 启动DSAN优化版模型训练 (带性能监控)...")
    print("=" * 50)
    
    # 检查GPU状态
    print("GPU状态检查:")
    import torch
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU数量: {torch.cuda.device_count()}")
        print(f"   GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"   显存大小: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 50)
    
    # 启动性能监控
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    print("性能监控已启动...")
    print("训练参数:")
    print(f"   模型类型: DSAN_Optimized (向量化优化版)")
    print(f"   数据集: PDBBind identity30")
    print(f"   隐藏层大小: 128")
    print(f"   层数: 3")
    print(f"   注意力头数: 8")
    print(f"   学习率: 0.001 -> 0.0001")
    print(f"   最大轮数: 100")
    print(f"   批次大小: 16 (从8优化)")
    print(f"   设备: GPU-0")
    print(f"   优化特性: 向量化块处理 + 内存高效注意力")
    print("=" * 50)
    
    # 构建训练命令
    cmd = [
        sys.executable, "train.py",
        "--train_set", "./datasets/PDBBind/processed/identity30/train.pkl",
        "--valid_set", "./datasets/PDBBind/processed/identity30/valid.pkl", 
        "--save_dir", "./datasets/PDBBind/processed/identity30/models/DSAN_optimized",
        "--task", "PDBBind",
        "--lr", "0.001",
        "--final_lr", "0.0001", 
        "--max_epoch", "100",
        "--save_topk", "5",
        "--batch_size", "16",  # 优化的批次大小
        "--valid_batch_size", "16",
        "--grad_clip", "1.0",
        "--warmup", "1000",
        "--shuffle",
        "--model_type", "DSAN_Optimized",  # 使用优化版DSAN
        "--hidden_size", "128",
        "--n_layers", "3",
        "--n_channel", "1", 
        "--n_rbf", "32",
        "--cutoff", "10.0",
        "--n_head", "8",
        "--radial_size", "16",  # 从16减少到16保持不变（也可以减少到8）
        "--k_neighbors", "9",
        "--seed", "2024",
        "--gpus", "0"
    ]
    
    # 记录开始时间
    start_time = time.time()
    
    # 执行训练
    print("开始训练...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        end_time = time.time()
        training_time = end_time - start_time
        
        print("✅ 训练完成!")
        print(f"模型已保存到: ./datasets/PDBBind/processed/identity30/models/DSAN_optimized")
        print(f"总训练时间: {training_time/3600:.2f} 小时")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")
        return_code = 1
    except KeyboardInterrupt:
        print("⛔ 训练被用户中断")
        return_code = 1
    else:
        return_code = 0
    finally:
        # 停止监控并显示摘要
        monitor.stop_monitoring()
        print("\n" + monitor.get_summary())
    
    return return_code

if __name__ == "__main__":
    exit(main())
