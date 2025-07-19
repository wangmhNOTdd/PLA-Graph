#!/usr/bin/env python3
"""
DSAN显存监控训练脚本
实时监控GPU显存使用，防止爆显存
"""

import subprocess
import sys
import os
import time
import threading
import torch
import psutil

class MemoryMonitor:
    def __init__(self, check_interval=5, warning_threshold=20.0):
        self.check_interval = check_interval
        self.warning_threshold = warning_threshold  # GB
        self.running = False
        self.max_gpu_memory = 0.0
        self.max_cpu_memory = 0.0
        self.warning_count = 0
        
    def start_monitoring(self):
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("🔍 显存监控已启动...")
        
    def stop_monitoring(self):
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2)
            
    def _monitor_loop(self):
        while self.running:
            try:
                if torch.cuda.is_available():
                    # GPU显存监控
                    gpu_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    gpu_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                    gpu_free = (torch.cuda.get_device_properties(0).total_memory / 1024**3) - gpu_reserved
                    
                    # CPU内存监控
                    cpu_memory = psutil.virtual_memory().used / 1024**3   # GB
                    cpu_percent = psutil.virtual_memory().percent
                    
                    # 更新最大使用量
                    if gpu_allocated > self.max_gpu_memory:
                        self.max_gpu_memory = gpu_allocated
                    if cpu_memory > self.max_cpu_memory:
                        self.max_cpu_memory = cpu_memory
                    
                    # 显示当前状态
                    status_msg = (f"[Memory] GPU: {gpu_allocated:.1f}GB ({gpu_reserved:.1f}GB reserved, "
                                f"{gpu_free:.1f}GB free) | CPU: {cpu_memory:.1f}GB ({cpu_percent:.1f}%) | "
                                f"Max GPU: {self.max_gpu_memory:.1f}GB")
                    print(status_msg)
                    
                    # 显存警告和清理
                    if gpu_allocated > self.warning_threshold:
                        self.warning_count += 1
                        print(f"⚠️  显存使用过高! ({gpu_allocated:.1f}GB > {self.warning_threshold}GB) "
                              f"警告次数: {self.warning_count}")
                        
                        # 强制清理显存
                        torch.cuda.empty_cache()
                        print("🧹 已执行显存清理")
                        
                        # 如果连续多次警告，降低监控频率避免日志刷屏
                        if self.warning_count > 5:
                            time.sleep(self.check_interval * 2)
                            continue
                    
                    # 临界值检查（7GB，接近4060的8GB极限）
                    if gpu_allocated > 7.0:
                        print("🚨 显存严重不足! 可能即将OOM!")
                        print("建议立即检查模型参数或减小batch_size")
                        
                else:
                    print("[Memory] CUDA不可用，无法监控GPU显存")
                    
            except Exception as e:
                print(f"⚠️  显存监控出错: {e}")
                
            time.sleep(self.check_interval)
    
    def get_summary(self):
        return (f"\n显存监控总结:\n"
                f"最大GPU显存使用: {self.max_gpu_memory:.2f} GB\n"
                f"最大CPU内存使用: {self.max_cpu_memory:.2f} GB\n"
                f"显存警告次数: {self.warning_count}")

def main():
    print("🚀 启动DSAN模型训练（带显存监控）...")
    print("=" * 60)
    
    # 检查GPU状态
    print("GPU状态检查:")
    if torch.cuda.is_available():
        print(f"   CUDA可用: ✅")
        print(f"   GPU数量: {torch.cuda.device_count()}")
        print(f"   GPU名称: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   显存大小: {gpu_memory:.1f} GB")
        print(f"   当前显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    else:
        print("   CUDA可用: ❌")
        return 1
    
    print("=" * 60)
    
    # 启动显存监控 (RTX 4060: 8GB显存)
    monitor = MemoryMonitor(check_interval=8, warning_threshold=6.0)  # 4060设置为6GB警告
    monitor.start_monitoring()
    
    try:
        # 构建优化的训练命令
        cmd = [
            sys.executable, "train.py",
            "--train_set", "./datasets/PDBBind/processed/identity30/train.pkl",
            "--valid_set", "./datasets/PDBBind/processed/identity30/valid.pkl", 
            "--save_dir", "./datasets/PDBBind/processed/identity30/models/DSAN_memory_optimized",
            "--task", "PDBBind",
            "--lr", "0.001",
            "--final_lr", "0.0001", 
            "--max_epoch", "100",
            "--save_topk", "5",
            "--batch_size", "1",  # 4060显存较小，使用最小批次
            "--valid_batch_size", "1",
            "--grad_clip", "1.0",
            "--warmup", "1000",
            "--shuffle",
            "--model_type", "DSAN",
            "--hidden_size", "64",   # 减少隐藏层大小
            "--n_layers", "2",       # 减少层数
            "--n_channel", "1", 
            "--n_rbf", "8",          # 大幅减少RBF维度
            "--cutoff", "6.0",       # 大幅减少cutoff
            "--n_head", "4",         # 减少注意力头数
            "--radial_size", "4",    # 大幅减少几何特征维度
            "--k_neighbors", "4",    # 大幅减少K近邻数
            "--seed", "2024",
            "--gpus", "0"
        ]
        
        print("训练配置（4060显存优化）:")
        print(f"   批次大小: 1 (4060保守设置)")
        print(f"   RBF维度: 8 (大幅减少)")
        print(f"   Cutoff: 6.0 (大幅减少边数量)")
        print(f"   几何特征维度: 4 (大幅减少)")
        print(f"   K近邻数: 4 (大幅减少)")
        print(f"   隐藏层大小: 64 (减少参数量)")
        print(f"   显存警告阈值: 6GB")
        print("=" * 60)
        
        # 执行训练
        print("🎯 开始训练... (Ctrl+C 中断)")
        start_time = time.time()
        
        result = subprocess.run(cmd, check=True)
        
        end_time = time.time()
        training_time = (end_time - start_time) / 3600  # hours
        
        print("✅ 训练完成!")
        print(f"模型已保存到: ./datasets/PDBBind/processed/identity30/models/DSAN_memory_optimized")
        print(f"训练用时: {training_time:.2f} 小时")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")
        return 1
    except KeyboardInterrupt:
        print("⛔ 训练被用户中断")
        return 1
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return 1
    finally:
        # 停止监控并显示摘要
        monitor.stop_monitoring()
        print(monitor.get_summary())
        print("=" * 60)
        print("训练结束")

if __name__ == "__main__":
    exit(main())
