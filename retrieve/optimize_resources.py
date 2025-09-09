#!/usr/bin/env python3
"""
資源使用率優化測試腳本
系統性地測試不同參數組合，找到最佳資源利用率配置
"""

import subprocess
import time
import psutil
import json
import os
from datetime import datetime
import argparse

class ResourceMonitor:
    def __init__(self):
        self.metrics = []
    
    def get_system_metrics(self):
        """獲取系統資源指標"""
        # CPU 使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 記憶體使用率
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        
        # GPU 使用率 (簡化版本)
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_percent = gpus[0].load * 100
                gpu_memory_percent = gpus[0].memoryUtil * 100
            else:
                gpu_percent = 0
                gpu_memory_percent = 0
        except:
            gpu_percent = 0
            gpu_memory_percent = 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_used_gb': memory_used_gb,
            'gpu_percent': gpu_percent,
            'gpu_memory_percent': gpu_memory_percent
        }
    
    def start_monitoring(self, duration=300):  # 5分鐘監控
        """開始監控系統資源"""
        print(f"開始監控系統資源 {duration} 秒...")
        start_time = time.time()
        
        while time.time() - start_time < duration:
            metrics = self.get_system_metrics()
            self.metrics.append(metrics)
            print(f"CPU: {metrics['cpu_percent']:.1f}% | "
                  f"Memory: {metrics['memory_percent']:.1f}% | "
                  f"GPU: {metrics['gpu_percent']:.1f}% | "
                  f"GPU Memory: {metrics['gpu_memory_percent']:.1f}%")
            time.sleep(10)  # 每10秒記錄一次
    
    def get_average_metrics(self):
        """計算平均資源使用率"""
        if not self.metrics:
            return None
        
        avg_cpu = sum(m['cpu_percent'] for m in self.metrics) / len(self.metrics)
        avg_memory = sum(m['memory_percent'] for m in self.metrics) / len(self.metrics)
        avg_gpu = sum(m['gpu_percent'] for m in self.metrics) / len(self.metrics)
        avg_gpu_memory = sum(m['gpu_memory_percent'] for m in self.metrics) / len(self.metrics)
        
        return {
            'avg_cpu_percent': avg_cpu,
            'avg_memory_percent': avg_memory,
            'avg_gpu_percent': avg_gpu,
            'avg_gpu_memory_percent': avg_gpu_memory,
            'total_samples': len(self.metrics)
        }

class ResourceOptimizer:
    def __init__(self, dataset='webqsp'):
        self.dataset = dataset
        self.results = []
    
    def run_experiment(self, config, test_duration=300):
        """運行單個實驗"""
        print(f"\n{'='*60}")
        print(f"測試配置: {config}")
        print(f"{'='*60}")
        
        # 構建命令
        cmd = [
            'python', 'train.py',
            '-d', self.dataset,
            '--batch_size', str(config['batch_size']),
            '--num_workers', str(config['num_workers']),
            '--samples_per_epoch', str(config['samples_per_epoch']),
            '--samples_per_batch_load', str(config['samples_per_batch_load']),
            '-id_sup', f"opt_{config['batch_size']}_{config['num_workers']}_{config['samples_per_epoch']}_{config['samples_per_batch_load']}"
        ]
        
        # 啟動監控
        monitor = ResourceMonitor()
        
        try:
            # 啟動訓練進程
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 監控資源使用
            monitor.start_monitoring(test_duration)
            
            # 等待進程完成或超時
            try:
                stdout, stderr = process.communicate(timeout=test_duration + 60)
                return_code = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                return_code = -1
            
            # 獲取平均指標
            avg_metrics = monitor.get_average_metrics()
            
            result = {
                'config': config,
                'return_code': return_code,
                'avg_metrics': avg_metrics,
                'stdout': stdout.decode() if stdout else '',
                'stderr': stderr.decode() if stderr else '',
                'timestamp': datetime.now().isoformat()
            }
            
            self.results.append(result)
            
            print(f"實驗完成 - 返回碼: {return_code}")
            if avg_metrics:
                print(f"平均資源使用率:")
                print(f"  CPU: {avg_metrics['avg_cpu_percent']:.1f}%")
                print(f"  Memory: {avg_metrics['avg_memory_percent']:.1f}%")
                print(f"  GPU: {avg_metrics['avg_gpu_percent']:.1f}%")
                print(f"  GPU Memory: {avg_metrics['avg_gpu_memory_percent']:.1f}%")
            
            return result
            
        except Exception as e:
            print(f"實驗失敗: {e}")
            return None
    
    def run_quick_test(self):
        """快速測試模式"""
        print("開始快速資源使用率測試...")
        
        # 快速測試配置
        quick_configs = [
            {'batch_size': 1, 'num_workers': 0, 'samples_per_epoch': 500, 'samples_per_batch_load': 32},
            {'batch_size': 4, 'num_workers': 4, 'samples_per_epoch': 1000, 'samples_per_batch_load': 64},
            {'batch_size': 8, 'num_workers': 8, 'samples_per_epoch': 2000, 'samples_per_batch_load': 128},
        ]
        
        for i, config in enumerate(quick_configs):
            print(f"\n進度: {i+1}/{len(quick_configs)}")
            self.run_experiment(config, test_duration=60)  # 1分鐘測試
            
            # 保存中間結果
            self.save_results(f"quick_optimization_results_{self.dataset}_{i+1}.json")
        
        # 分析結果
        self.analyze_results()
    
    def analyze_results(self):
        """分析測試結果並推薦最佳配置"""
        print(f"\n{'='*60}")
        print("資源使用率分析結果")
        print(f"{'='*60}")
        
        # 按 GPU 使用率排序
        valid_results = [r for r in self.results if r['avg_metrics'] and r['return_code'] == 0]
        
        if not valid_results:
            print("沒有成功的實驗結果")
            return
        
        # 按 GPU 使用率排序
        gpu_sorted = sorted(valid_results, key=lambda x: x['avg_metrics']['avg_gpu_percent'], reverse=True)
        
        print("Top 3 GPU 使用率配置:")
        for i, result in enumerate(gpu_sorted[:3]):
            config = result['config']
            metrics = result['avg_metrics']
            print(f"{i+1}. GPU: {metrics['avg_gpu_percent']:.1f}% | "
                  f"CPU: {metrics['avg_cpu_percent']:.1f}% | "
                  f"Memory: {metrics['avg_memory_percent']:.1f}%")
            print(f"   配置: batch_size={config['batch_size']}, "
                  f"num_workers={config['num_workers']}, "
                  f"samples_per_epoch={config['samples_per_epoch']}, "
                  f"samples_per_batch_load={config['samples_per_batch_load']}")
        
        # 推薦最佳配置
        if gpu_sorted:
            best_config = gpu_sorted[0]['config']
            print(f"\n推薦最佳配置:")
            print(f"python train.py -d {self.dataset} \\")
            print(f"  --batch_size {best_config['batch_size']} \\")
            print(f"  --num_workers {best_config['num_workers']} \\")
            print(f"  --samples_per_epoch {best_config['samples_per_epoch']} \\")
            print(f"  --samples_per_batch_load {best_config['samples_per_batch_load']}")
    
    def save_results(self, filename):
        """保存測試結果"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"結果已保存到: {filename}")

def main():
    parser = argparse.ArgumentParser(description='資源使用率優化測試')
    parser.add_argument('-d', '--dataset', type=str, default='webqsp', 
                        choices=['webqsp', 'cwq'], help='數據集名稱')
    
    args = parser.parse_args()
    
    optimizer = ResourceOptimizer(args.dataset)
    optimizer.run_quick_test()

if __name__ == '__main__':
    main()
