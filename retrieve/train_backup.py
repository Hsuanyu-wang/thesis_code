import numpy as np
import os, psutil, time, torch, wandb
import pandas as pd
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from collections import defaultdict
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

from src.config.retriever import load_yaml
from src.dataset.retriever import OptimizedRetrieverDataset, optimized_collate_retriever, GroupedByFileBatchSampler
from src.model.retriever import Retriever
from src.setup import set_seed

# ---------------- System usage monitor (CPU/GPU) ----------------
import threading
try:
    import pynvml
    _HAS_PYNVML = True
except Exception:
    _HAS_PYNVML = False

from torch.utils.data import SubsetRandomSampler

# Helper: safely construct Adam without conflicting fused/foreach flags
def build_adam_optimizer(parameters, base_cfg: dict):
    cfg = dict(base_cfg) if base_cfg is not None else {}
    # Respect explicit user settings but avoid invalid combo
    requested_fused = cfg.pop('fused', None)
    requested_foreach = cfg.pop('foreach', None)

    use_fused = False
    if torch.cuda.is_available():
        if requested_fused is True or requested_fused is None:
            try:
                # Probe fused support on this build/device
                _ = Adam([torch.zeros(1, device='cuda', requires_grad=True)], lr=1e-3, fused=True)
                use_fused = True
            except Exception:
                use_fused = False
    # When fused is used, foreach must be False/omitted
    if use_fused:
        return Adam(parameters, fused=True, **cfg)
    # Otherwise, fall back to foreach (default to True if not specified)
    if requested_foreach is None:
        requested_foreach = True
    return Adam(parameters, foreach=requested_foreach, **cfg)

class SystemUsageMonitor:
    def __init__(self, interval_sec: float = 1.0):
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread = None
        self.cpu_samples = []
        self.gpu_util_samples = []
        self.gpu_mem_used_samples = []
        self.gpu_mem_total = None
        self._nvml_inited = False

    def _init_nvml(self):
        if torch.cuda.is_available() and _HAS_PYNVML and not self._nvml_inited:
            try:
                pynvml.nvmlInit()
                self._nvml_inited = True
            except Exception:
                self._nvml_inited = False

    def _sample_once(self):
        # CPU
        try:
            self.cpu_samples.append(psutil.cpu_percent(interval=None))
        except Exception:
            pass
        # GPU
        if torch.cuda.is_available():
            if self._nvml_inited:
                try:
                    device_index = 0
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    self.gpu_util_samples.append(util.gpu)
                    self.gpu_mem_used_samples.append(mem.used / (1024**2))
                    if self.gpu_mem_total is None:
                        self.gpu_mem_total = mem.total / (1024**2)
                except Exception:
                    pass
            else:
                # Fallback: best-effort via torch (util not available), only mem
                try:
                    used = torch.cuda.max_memory_allocated(0) / (1024**2)
                    self.gpu_mem_used_samples.append(used)
                    if self.gpu_mem_total is None:
                        self.gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                except Exception:
                    pass

    def _run(self):
        # Prime CPU percent to get accurate next reading
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass
        self._init_nvml()
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_sec)

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._thread = None
            if self._nvml_inited:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass

    def get_stats(self):
        def _fmt(samples):
            if not samples:
                return (None, None)
            return (max(samples), sum(samples) / max(len(samples), 1))
        cpu_max, cpu_avg = _fmt(self.cpu_samples)
        gpu_util_max, gpu_util_avg = _fmt(self.gpu_util_samples)
        gpu_mem_max, gpu_mem_avg = _fmt(self.gpu_mem_used_samples)
        return {
            'cpu_max': cpu_max,
            'cpu_avg': cpu_avg,
            'gpu_util_max': gpu_util_max,
            'gpu_util_avg': gpu_util_avg,
            'gpu_mem_max': gpu_mem_max,
            'gpu_mem_avg': gpu_mem_avg,
            'gpu_mem_total': self.gpu_mem_total
        }

    def report(self, prefix: str = "System usage"):
        stats = self.get_stats()
        print("\n===== " + prefix + " summary =====")
        if stats['cpu_max'] is not None:
            print(f"CPU utilization  max/avg: {stats['cpu_max']:.1f}% / {stats['cpu_avg']:.1f}%")
        else:
            print("CPU utilization  max/avg: N/A")
        if torch.cuda.is_available():
            if stats['gpu_util_max'] is not None:
                print(f"GPU utilization  max/avg: {stats['gpu_util_max']:.1f}% / {stats['gpu_util_avg']:.1f}%")
            else:
                print("GPU utilization  max/avg: N/A (utilization requires NVML)")
            if stats['gpu_mem_max'] is not None:
                if stats['gpu_mem_total']:
                    print(f"GPU memory usage max/avg: {stats['gpu_mem_max']:.0f}MB / {stats['gpu_mem_avg']:.0f}MB (of ~{stats['gpu_mem_total']:.0f}MB)")
                else:
                    print(f"GPU memory usage max/avg: {stats['gpu_mem_max']:.0f}MB / {stats['gpu_mem_avg']:.0f}MB")
            else:
                print("GPU memory usage max/avg: N/A")
        print("===============================\n")

    def recommend(self, args, context_label: str = ""):
        stats = self.get_stats()
        cpu_avg = stats['cpu_avg'] or 0.0
        gpu_util_avg = stats['gpu_util_avg'] or 0.0
        gpu_mem_max = stats['gpu_mem_max'] or 0.0
        gpu_mem_total = stats['gpu_mem_total'] or 0.0
        mem_headroom = (gpu_mem_total - gpu_mem_max) if (gpu_mem_total and gpu_mem_max) else None
        print("💡 Parameter recommendations" + (f" ({context_label})" if context_label else "") + ":")
        recs = []
        # Baselines
        nw = max(0, getattr(args, 'num_workers', 0))
        pf = max(2, getattr(args, 'prefetch_factor', 2))
        bs = max(1, getattr(args, 'batch_size', 1))
        gas = max(1, getattr(args, 'grad_accum_steps', 1))
        pin = bool(getattr(args, 'pin_memory', False))
        # If GPU under-utilized and memory headroom, increase batch size or grad accumulation
        if torch.cuda.is_available() and gpu_util_avg < 60.0:
            if mem_headroom is not None and mem_headroom > 0.3 * (gpu_mem_total or 1):
                recs.append(f"Increase batch_size (e.g., {bs} -> {min(bs*2, bs+32)}) or increase grad_accum_steps (e.g., {gas} -> {gas+1}).")
            else:
                recs.append("GPU util low but memory tight; consider small increase of prefetch_factor and num_workers.")
        # If CPU under-utilized, increase workers/prefetch
        if cpu_avg < 50.0:
            if nw == 0:
                recs.append("Increase num_workers from 0 to 2-4 to parallelize data loading.")
            else:
                recs.append(f"Increase num_workers (e.g., {nw} -> {min(nw*2, nw+8)}) and/or prefetch_factor (e.g., {pf} -> {min(pf*2, pf+2)}).")
        # If CPU very high, reduce loader pressure
        if cpu_avg > 85.0 and nw > 1:
            recs.append(f"CPU very busy; reduce num_workers (e.g., {nw} -> {max(1, nw//2)}) or lower prefetch_factor (e.g., {pf} -> {max(2, pf-1)}).")
        # If GPU memory near full, reduce batch
        if torch.cuda.is_available() and gpu_mem_total and gpu_mem_max / gpu_mem_total > 0.90:
            recs.append(f"GPU memory near capacity; reduce batch_size (e.g., {bs} -> {max(1, bs//2)}) or reduce grad_accum_steps ({gas} -> {max(1, gas-1)}).")
        # Pin memory suggestion
        if torch.cuda.is_available() and not pin:
            recs.append("Enable --pin_memory to speed up H2D transfer.")
        # Sensible defaults if nothing matched
        if not recs:
            recs.append("Current settings look balanced. Fine-tune gradually: tweak num_workers/prefetch by small steps and monitor.")
        for r in recs:
            print(" - " + r)
        print("")


def _move_listfield_concat_split(batch: dict, key: str, device: torch.device):
    """Helper: concat a list of tensors along dim=0 on CPU, one-shot H2D, then split back on GPU.
    Keeps original per-sample boundaries; replaces batch[key] with list of GPU tensors.
    """
    if key not in batch or not isinstance(batch[key], list) or len(batch[key]) == 0:
        return
    cpu_list = batch[key]
    # Ensure all tensors are on CPU first (they should be from DataLoader)
    # Determine concat dim and sizes
    first = cpu_list[0]
    if not torch.is_tensor(first):
        return
    dim0_sizes = [int(t.shape[0]) for t in cpu_list]
    try:
        flat = torch.cat(cpu_list, dim=0)
    except Exception:
        # If shapes inconsistent, skip optimization and fallback to per-item move
        batch[key] = [t.to(device, non_blocking=True) for t in cpu_list]
        return
    # Pin then move once
    if flat.device.type == 'cpu':
        try:
            flat = flat.pin_memory()
        except Exception:
            pass
    flat_gpu = flat.to(device, non_blocking=True)
    # Split back by sizes
    try:
        split_list = list(flat_gpu.split(dim0_sizes, dim=0))
        batch[key] = split_list
    except Exception:
        # Fallback
        batch[key] = [t.to(device, non_blocking=True) for t in cpu_list]



def move_batch_to_device_optimized(batch: dict, device: torch.device) -> dict:
    """Move a heterogeneous batch dict to device efficiently.
    - Direct-move dense stacked tensors
    - For list fields, concat on CPU -> one-shot H2D -> split back on GPU
    The function mutates and returns the same dict.
    """
    # Direct tensors (stacked/padded)
    for k in ['q_emb', 'entity_embs', 'relation_embs', 'topic_entity_one_hot', 'target_triple_probs']:
        if k in batch and torch.is_tensor(batch[k]):
            batch[k] = batch[k].to(device, non_blocking=True)

    # List fields to optimize via concat+split
    list_keys = [
        'entity_embs_list',
        'relation_embs_list',
        'topic_entity_one_hot_list',
        'target_triple_probs_list',
        'h_id_tensors',
        'r_id_tensors',
        't_id_tensors',
    ]
    for key in list_keys:
        _move_listfield_concat_split(batch, key, device)

    # Non-tensor python lists that must be kept on CPU (IDs for metrics)
    # a_entity_id_lists and num_non_text_entities stay as-is
    return batch



def create_dummy_batch(device, batch_size=2):
    """創建虛擬批次數據用於測試"""
    # 創建虛擬的嵌入向量
    emb_size = 768  # 假設嵌入維度為768
    
    # 動態生成每個樣本的實體和關係數量
    random.seed(42)  # 確保可重現性
    
    # 為每個樣本創建不同數量的實體和關係
    num_entities_per_sample = [random.randint(6, 12) for _ in range(batch_size)]
    num_relations_per_sample = [random.randint(4, 8) for _ in range(batch_size)]
    num_triples_per_sample = [random.randint(10, 20) for _ in range(batch_size)]
    
    dummy_batch = {
        # 查詢嵌入 - 批次格式
        'q_emb': torch.randn(batch_size, emb_size).to(device),
        
        # 實體嵌入 - 列表格式（每個樣本不同數量）
        'entity_embs_list': [
            torch.randn(num_entities_per_sample[i], emb_size).to(device) 
            for i in range(batch_size)
        ],
        
        # 關係嵌入 - 列表格式（每個樣本不同數量）
        'relation_embs_list': [
            torch.randn(num_relations_per_sample[i], emb_size).to(device) 
            for i in range(batch_size)
        ],
        
        # 主題實體one-hot - 列表格式
        'topic_entity_one_hot_list': [
            torch.randn(num_entities_per_sample[i], 2).to(device)  # 假設one-hot維度為2
            for i in range(batch_size)
        ],
        
        # ID張量 - 列表格式
        'h_id_tensors': [
            torch.randint(0, num_entities_per_sample[i], (num_triples_per_sample[i],)).to(device)
            for i in range(batch_size)
        ],
        'r_id_tensors': [
            torch.randint(0, num_relations_per_sample[i], (num_triples_per_sample[i],)).to(device)
            for i in range(batch_size)
        ],
        't_id_tensors': [
            torch.randint(0, num_entities_per_sample[i], (num_triples_per_sample[i],)).to(device)
            for i in range(batch_size)
        ],
        
        # 目標三元組概率 - 列表格式
        'target_triple_probs_list': [
            torch.randint(0, 2, (num_triples_per_sample[i],)).float().to(device)
            for i in range(batch_size)
        ],
        
        # 答案實體ID列表
        'a_entity_id_lists': [
            torch.randint(0, num_entities_per_sample[i], (3,)).to(device)  # 假設每個樣本有3個答案實體
            for i in range(batch_size)
        ],
        
        # 非文本實體數量
        'num_non_text_entities': [random.randint(2, 4) for _ in range(batch_size)]
    }
    return dummy_batch

def test_pipeline(args):
    """測試模式：快速驗證pipeline是否能正常運行"""
    print("🧪 進入測試模式 - 跳過完整數據加載，使用虛擬數據進行pipeline測試")
    usage = SystemUsageMonitor(interval_sec=1.0)
    usage.start()
    
    # 加載配置
    config_file = f'configs/retriever/{args.dataset}.yaml'
    config = load_yaml(config_file)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(config['env']['num_threads'])
    set_seed(config['env']['seed'])
    
    print(f"✅ 配置加載成功")
    print(f"✅ 設備: {device}")
    print(f"✅ 數據集: {args.dataset}")
    
    # 創建模型
    emb_size = 768  # 使用固定嵌入維度
    model = Retriever(emb_size, **config['retriever'], use_dropout=args.use_dropout, dropout_p=args.dropout_p).to(device)
    # optimizer = Adam(model.parameters(), **config['optimizer'])
    optimizer = build_adam_optimizer(model.parameters(), config['optimizer'])
    scaler = GradScaler()
    
    print(f"✅ 模型創建成功")
    print(f"✅ 優化器創建成功")
    
    # 創建虛擬數據
    dummy_batch = create_dummy_batch(device, batch_size=args.batch_size)
    print(f"✅ 虛擬批次數據創建成功 (batch_size={args.batch_size})")
    
    # 測試前向傳播
    model.train()
    with autocast('cuda' if device.type == 'cuda' else 'cpu'):
        try:
            pred_triple_logits_batch = model(dummy_batch)
            print(f"✅ 模型前向傳播成功，輸出形狀: {[pred.shape for pred in pred_triple_logits_batch]}")
        except Exception as e:
            print(f"❌ 模型前向傳播失敗: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # 測試損失計算
    try:
        loss = 0
        valid_samples_in_batch = 0
        for j in range(len(pred_triple_logits_batch)):
            pred_logits = pred_triple_logits_batch[j].reshape(-1)
            target = dummy_batch['target_triple_probs_list'][j]
            
            if pred_logits.numel() == 0:
                continue
                
            # 對齊長度
            if target.numel() != pred_logits.numel():
                min_len = min(target.numel(), pred_logits.numel())
                pred_logits = pred_logits[:min_len]
                target = target[:min_len]
            
            num_positive = target.sum().item()
            num_total = len(target)
            pos_weight = torch.tensor([(num_total - num_positive) / num_total if num_positive > 0 else 1.0], device=device)
            loss += F.binary_cross_entropy_with_logits(pred_logits, target, pos_weight=pos_weight)
            valid_samples_in_batch += 1

        if valid_samples_in_batch > 0:
            loss /= valid_samples_in_batch
            print(f"✅ 損失計算成功: {loss.item():.4f}")
        else:
            print("❌ 沒有有效的樣本用於損失計算")
            return False
    except Exception as e:
        print(f"❌ 損失計算失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 測試反向傳播
    try:
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print(f"✅ 反向傳播和優化器更新成功")
    except Exception as e:
        print(f"❌ 反向傳播失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 測試評估模式
    model.eval()
    with torch.inference_mode():
        try:
            pred_triple_logits_batch = model(dummy_batch)
            print(f"✅ 評估模式前向傳播成功")
        except Exception as e:
            print(f"❌ 評估模式前向傳播失敗: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("🎉 所有pipeline測試通過！模型可以正常運行。")
    usage.stop()
    stats = usage.report(prefix="Test pipeline system usage")
    usage.recommend(args, context_label="test")
    return True

def check_training_pipeline(args):
    """檢查正式訓練pipeline：使用真實數據集進行快速驗證"""
    print("🔍 進入檢查模式 - 使用真實數據集快速驗證訓練pipeline")
    usage = SystemUsageMonitor(interval_sec=1.0)
    usage.start()
    
    # 加載配置
    config_file = f'configs/retriever/{args.dataset}.yaml'
    config = load_yaml(config_file)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(config['env']['num_threads'])
    set_seed(config['env']['seed'])
    
    print(f"✅ 配置加載成功")
    print(f"✅ 設備: {device}")
    print(f"✅ 數據集: {args.dataset}")
    
    method = args.method if hasattr(args, 'method') else 'default'
    freq_weight = (method == 'freq_weight')
    inv_freq_weight = (method == 'inv_freq_weight')
    print(f"✅ 方法: {method}, 頻率權重: {freq_weight}, 逆頻率權重: {inv_freq_weight}")
    print(f"✅ 最短路徑後 reweight 方法: {getattr(args, 'method_sp', 'none')}")
    
    # 檢查數據集加載
    try:
        print(" 正在加載訓練數據集...")
        train_set = OptimizedRetrieverDataset(
            config=config,
            split='train',
            freq_weight=freq_weight,
            weight_mode=(
                'freq' if method == 'freq_weight' else (
                'inv' if method == 'inv_freq_weight' else (
                args.method_sp if args.method_sp in ['spcount', 'spcount_inv'] else 'none'))),
        )
        print(f"✅ 訓練數據集加載成功，樣本數: {len(train_set)}")
        
        print(" 正在加載驗證數據集...")
        val_set = OptimizedRetrieverDataset(
            config=config,
            split='val',
            freq_weight=freq_weight,
            weight_mode=(
                'freq' if method == 'freq_weight' else (
                'inv' if method == 'inv_freq_weight' else (
                args.method_sp if args.method_sp in ['spcount', 'spcount_inv'] else 'none'))),
        )
        print(f"✅ 驗證數據集加載成功，樣本數: {len(val_set)}")
    except Exception as e:
        print(f"❌ 數據集加載失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 檢查DataLoader創建
    try:
        print("🔄 正在創建DataLoader...")
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=optimized_collate_retriever,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=(args.num_workers > 0)
        )
        print(f"✅ 訓練DataLoader創建成功")
        
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=optimized_collate_retriever,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=(args.num_workers > 0),
            drop_last=True
        )
        print(f"✅ 驗證DataLoader創建成功")
    except Exception as e:
        print(f"❌ DataLoader創建失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 檢查數據批次加載
    try:
        print(" 正在測試數據批次加載...")
        train_batch = next(iter(train_loader))
        print(f"✅ 訓練批次加載成功，批次大小: {len(train_batch['q_emb'])}")
        print(f"   批次鍵: {list(train_batch.keys())}")
        
        val_batch = next(iter(val_loader))
        print(f"✅ 驗證批次加載成功，批次大小: {len(val_batch['q_emb'])}")
    except Exception as e:
        print(f"❌ 數據批次加載失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 檢查模型創建
    try:
        print("🧠 正在創建模型...")
        sample_data = train_set[0]
        emb_size = sample_data['q_emb'].shape[-1]
        print(f"   嵌入維度: {emb_size}")
        
        model = Retriever(emb_size, **config['retriever'], use_dropout=args.use_dropout, dropout_p=args.dropout_p).to(device)
        # optimizer = Adam(model.parameters(), **config['optimizer'])
        optimizer = build_adam_optimizer(model.parameters(), config['optimizer'])
        scaler = GradScaler()
        print(f"✅ 模型創建成功")
        print(f"✅ 優化器創建成功")
    except Exception as e:
        print(f"❌ 模型創建失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 檢查前向傳播
    try:
        print("⚡ 正在測試前向傳播...")
        model.train()
        # 將批次數據移動到device（使用合併傳輸優化）
        train_batch = move_batch_to_device_optimized(train_batch, device)
        with autocast('cuda' if device.type == 'cuda' else 'cpu'):
            pred_triple_logits_batch = model(train_batch)
            print(f"✅ 前向傳播成功，輸出形狀: {[pred.shape for pred in pred_triple_logits_batch]}")
    except Exception as e:
        print(f"❌ 前向傳播失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 檢查損失計算
    try:
        print("📉 正在測試損失計算...")
        loss = 0
        valid_samples_in_batch = 0
        for j in range(len(pred_triple_logits_batch)):
            pred_logits = pred_triple_logits_batch[j]
            if 'target_triple_probs_list' in train_batch:
                target = train_batch['target_triple_probs_list'][j]
            else:
                target = train_batch['target_triple_probs'][j]
            # 取正樣本權重因子（若存在）
            pos_factors = None
            if 'pos_weight_factors_list' in train_batch:
                pos_factors = train_batch['pos_weight_factors_list'][j]
            elif 'pos_weight_factors' in train_batch:
                pos_factors = train_batch['pos_weight_factors'][j]
            
            pred_logits = pred_logits.reshape(-1)
            target = target.reshape(-1)
            if pos_factors is not None:
                pos_factors = pos_factors.reshape(-1)
            
            if pred_logits.numel() == 0:
                continue
            
            # 對齊長度
            if target.numel() != pred_logits.numel():
                min_len = min(target.numel(), pred_logits.numel())
                pred_logits = pred_logits[:min_len]
                target = target[:min_len]
                if pos_factors is not None:
                    pos_factors = pos_factors[:min_len]
            
            num_positive = target.sum().item()
            num_total = len(target)

            if pos_factors is not None:
                # 將每個正樣本的權重合併到 pos_weight（平均縮放避免過大）
                mean_factor = float(pos_factors[target > 0].mean().item()) if (target > 0).any() else 1.0
                pos_weight = torch.tensor([(num_total - num_positive) / num_total if num_positive > 0 else 1.0], device=device) * mean_factor
            else:
                pos_weight = torch.tensor([(num_total - num_positive) / num_total if num_positive > 0 else 1.0], device=device)
            loss += F.binary_cross_entropy_with_logits(pred_logits, target, pos_weight=pos_weight)
            valid_samples_in_batch += 1

        if valid_samples_in_batch > 0:
            loss /= valid_samples_in_batch
            print(f"✅ 損失計算成功: {loss.item():.4f}")
        else:
            print("❌ 沒有有效的樣本用於損失計算")
            return False
    except Exception as e:
        print(f"❌ 損失計算失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 檢查反向傳播
    try:
        print(" 正在測試反向傳播...")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print(f"✅ 反向傳播和優化器更新成功")
    except Exception as e:
        print(f"❌ 反向傳播失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 檢查評估模式
    try:
        print("📊 正在測試評估模式...")
        model.eval()
        with torch.inference_mode():
            # 一次性將批次資料移動到 GPU
            val_batch = move_batch_to_device_optimized(val_batch, device)
            pred_triple_logits_batch = model(val_batch)
            print(f"✅ 評估模式前向傳播成功")
    except Exception as e:
        print(f"❌ 評估模式前向傳播失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 檢查wandb初始化（但不實際記錄）
    try:
        print(" 正在檢查wandb配置...")
        ts = time.strftime('%b%d-%H:%M:%S', time.localtime(time.time() + 8 * 3600))
        exp_prefix = config['train']['save_prefix']
        # Build base name, then append all active feature/method tags
        exp_name_base = f'{exp_prefix}_{ts}'
        feature_tags = []
        # Methods (mutually exclusive by CLI, but keep additive logic)
        if inv_freq_weight:
            feature_tags.append('inv_freq_weight')
        elif freq_weight:
            feature_tags.append('freq_weight')
        # Post-SP reweight method can coexist
        if method_sp in ['spcount', 'spcount_inv']:
            feature_tags.append(method_sp)
        # Feature flags
        if getattr(args, 'use_dropout', False):
            try:
                dp = float(getattr(args, 'dropout_p', 0.2))
                dp_str = str(dp).replace('.', '_')
                feature_tags.append(f'drop{dp_str}')
            except Exception:
                feature_tags.append('drop')
        if getattr(args, 'legacy_mode', False):
            feature_tags.append('legacy')
        # Compose
        exp_name = exp_name_base if not feature_tags else f"{exp_name_base}_{'_'.join(feature_tags)}"
        # Optional user suffix
        if args.id_sup is not None:
            exp_name = f'{exp_name}_{args.id_sup}'
        
        config_df = pd.json_normalize(config, sep='/')
        print(f"✅ wandb配置準備完成，實驗名稱: {exp_name}")
        print(f"   項目: {args.dataset}_retriever")
    except Exception as e:
        print(f"❌ wandb配置檢查失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("🎉 所有訓練pipeline檢查通過！正式訓練可以正常運行。")
    usage.stop()
    stats = usage.report(prefix="Check pipeline system usage")
    usage.recommend(args, context_label="check")
    return True

def check_and_warn_resources(args):
    """在訓練開始前檢查資源配置並給出警告"""
    print("\n" + "="*20 + " 資源配置檢查 " + "="*20)
    warnings = []
    
    # 1. 檢查 num_workers
    try:
        cpu_cores = os.cpu_count()
        if cpu_cores and args.num_workers > cpu_cores:
            warnings.append(f"⚠️ 警告: DataLoader 的 num_workers ({args.num_workers}) 大於系統 CPU 邏輯核心數 ({cpu_cores})。"
                            f"這可能導致過多的程序切換開銷，建議設為 {cpu_cores} 或更小。")
    except NotImplementedError:
        warnings.append("無法自動偵測 CPU 核心數。請手動確保 num_workers 設定合理。")

    # 2. 檢查記憶體
    try:
        mem_info = psutil.virtual_memory()
        total_gb = mem_info.total / (1024**3)
        available_gb = mem_info.available / (1024**3)
        print(f"📊 系統記憶體: 總共 {total_gb:.1f} GB, 目前可用 {available_gb:.1f} GB。")
        if args.num_workers > available_gb:
            warnings.append(f"⚠️ 警告: 可用記憶體 ({available_gb:.1f} GB) 不多，而 num_workers ({args.num_workers}) 較高。"
                            "如果遇到記憶體不足(ООМ)錯誤，請優先考慮降低 num_workers。")
    except (ImportError, NameError):
         warnings.append("提示: 未安裝 psutil，無法檢查記憶體。建議安裝以獲得更佳的資源提示。")

    # 3. 檢查 batch_size
    if args.batch_size > 32:
        warnings.append(f"💡 提示: batch_size ({args.batch_size}) 較大，請密切關注 GPU 記憶體使用情況。"
                        "如果發生 GPU OOM，請降低此數值。")

    if not warnings:
        print("✅ 資源配置看起來合理。")
    else:
        for w in warnings:
            print(w)
    print("="*55 + "\n")

@torch.no_grad()
def eval_epoch(config, device, data_loader, model, epoch=None, total_epochs=None):
    model.eval()
    metric_dict = defaultdict(list)
    total_val_loss = 0
    num_samples = 0

    # 創建驗證進度條
    eval_desc = f"Eval Epoch {epoch+1}/{total_epochs}" if epoch is not None and total_epochs is not None else "Validation"
    eval_pbar = tqdm(data_loader, desc=eval_desc, leave=False, 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for batch in eval_pbar:
        # 高效移動整個批次到 device
        batch = move_batch_to_device_optimized(batch, device)
        
        with autocast('cuda'):
            pred_triple_logits_batch = model(batch)
        
        # 現在 pred_triple_logits_batch 是一個 list of tensors，需要逐個處理
        for i in range(len(pred_triple_logits_batch)):
            pred_triple_logits = pred_triple_logits_batch[i].reshape(-1)
            # 選擇目標：優先使用 list 版，否則使用 padded 版的第 i 列
            if 'target_triple_probs_list' in batch:
                target = batch['target_triple_probs_list'][i].to(device, non_blocking=True)
            else:
                target = batch['target_triple_probs'][i].to(device, non_blocking=True)
            # 取正樣本權重因子（若存在）
            pos_factors = None
            if 'pos_weight_factors_list' in batch:
                pos_factors = batch['pos_weight_factors_list'][i].to(device, non_blocking=True)
            elif 'pos_weight_factors' in batch:
                pos_factors = batch['pos_weight_factors'][i].to(device, non_blocking=True)
            # 對齊長度
            if target.numel() != pred_triple_logits.numel():
                min_len = min(target.numel(), pred_triple_logits.numel())
                pred_triple_logits = pred_triple_logits[:min_len]
                target = target[:min_len]
                if pos_factors is not None:
                    pos_factors = pos_factors[:min_len]
            h_id_tensor = batch['h_id_tensors'][i]
            t_id_tensor = batch['t_id_tensors'][i]
            a_entity_id_list = batch['a_entity_id_lists'][i]
            
            # --- 以下是單一樣本的評估邏輯 ---
            if len(h_id_tensor) > 0:
                num_positive = target.sum().item()
                num_total = len(target)
                if pos_factors is not None and (target > 0).any():
                    mean_factor = float(pos_factors[target > 0].mean().item())
                    pos_weight = torch.tensor([(num_total - num_positive) / num_positive if num_positive > 0 else 1.0], device=device) * mean_factor
                else:
                    pos_weight = torch.tensor([(num_total - num_positive) / num_positive if num_positive > 0 else 1.0], device=device)
                val_loss = F.binary_cross_entropy_with_logits(pred_triple_logits, target, pos_weight=pos_weight)
                total_val_loss += val_loss.item()
                num_samples += 1
            
            sorted_triple_ids_pred = torch.argsort(pred_triple_logits, descending=True).cpu()
            triple_ranks_pred = torch.empty_like(sorted_triple_ids_pred)
            triple_ranks_pred[sorted_triple_ids_pred] = torch.arange(len(triple_ranks_pred))
            
            target_triple_ids = target.nonzero().squeeze(-1).cpu()
            num_target_triples = len(target_triple_ids)
            if num_target_triples == 0:
                continue

            # 推斷實體數量（支援 list 或 padded）
            if 'entity_embs_list' in batch:
                num_total_entities = len(batch['entity_embs_list'][i]) + batch['num_non_text_entities'][i]
            else:
                num_total_entities = int(batch['entity_embs'][i].shape[0]) + batch['num_non_text_entities'][i]
            for k in config['eval']['k_list']:
                recall_k_sample = (triple_ranks_pred[target_triple_ids] < k).sum().item()
                metric_dict[f'triple_recall@{k}'].append(recall_k_sample / num_target_triples)
                
                triple_mask_k = triple_ranks_pred < k
                entity_mask_k = torch.zeros(num_total_entities)
                entity_mask_k[h_id_tensor[triple_mask_k]] = 1.
                entity_mask_k[t_id_tensor[triple_mask_k]] = 1.
                
                recall_k_sample_ans = entity_mask_k[a_entity_id_list].sum().item()
                metric_dict[f'ans_recall@{k}'].append(recall_k_sample_ans / len(a_entity_id_list))

        if num_samples > 0:
            eval_pbar.set_postfix({'val_loss': f'{total_val_loss / num_samples:.4f}'})

    eval_pbar.close()
    for key, val in metric_dict.items():
        metric_dict[key] = np.mean(val) if val else 0.0
    
    metric_dict['val_loss'] = total_val_loss / num_samples if num_samples > 0 else 0.0
    return metric_dict

def train_epoch(device, train_loader, model, optimizer, scaler, grad_accum_steps=1, epoch=None, total_epochs=None):
    model.train()
    total_loss = 0
    num_samples_processed = 0
    
    train_desc = f"Train Epoch {epoch+1}/{total_epochs}" if epoch is not None and total_epochs is not None else "Training"
    train_pbar = tqdm(train_loader, desc=train_desc, leave=False)

    optimizer.zero_grad(set_to_none=True)
    
    for i, batch in enumerate(train_pbar):
        # 高效移動整個批次到 device
        batch = move_batch_to_device_optimized(batch, device)

        with autocast('cuda'):
            pred_triple_logits_batch = model(batch) # 直接傳遞整個 batch dict

            # 逐樣本計算 loss
            loss = 0
            valid_samples_in_batch = 0
            for j in range(len(pred_triple_logits_batch)):
                pred_logits = pred_triple_logits_batch[j]
                # 選擇目標：優先使用 list 版，否則使用 padded 版的第 j 列
                if 'target_triple_probs_list' in batch:
                    target = batch['target_triple_probs_list'][j]
                else:
                    target = batch['target_triple_probs'][j]
                # 取正樣本權重因子（若存在）
                pos_factors = None
                if 'pos_weight_factors_list' in batch:
                    pos_factors = batch['pos_weight_factors_list'][j]
                elif 'pos_weight_factors' in batch:
                    pos_factors = batch['pos_weight_factors'][j]
                
                pred_logits = pred_logits.reshape(-1)
                target = target.reshape(-1)
                if pos_factors is not None:
                    pos_factors = pos_factors.reshape(-1)
                
                if pred_logits.numel() == 0: continue
                
                # 對齊長度
                if target.numel() != pred_logits.numel():
                    min_len = min(target.numel(), pred_logits.numel())
                    pred_logits = pred_logits[:min_len]
                    target = target[:min_len]
                    if pos_factors is not None:
                        pos_factors = pos_factors[:min_len]
                
                num_positive = target.sum().item()
                num_total = len(target)
                if pos_factors is not None:
                    mean_factor = float(pos_factors[target > 0].mean().item()) if (target > 0).any() else 1.0
                    pos_weight = torch.tensor([(num_total - num_positive) / num_total if num_positive > 0 else 1.0], device=device) * mean_factor
                else:
                    pos_weight = torch.tensor([(num_total - num_positive) / num_total if num_positive > 0 else 1.0], device=device)
                loss += F.binary_cross_entropy_with_logits(pred_logits, target, pos_weight=pos_weight)
                valid_samples_in_batch += 1

        if valid_samples_in_batch > 0:
            # 梯度累加
            loss_to_scale = loss / grad_accum_steps
            scaler.scale(loss_to_scale).backward()
            
            total_loss += loss.item() * valid_samples_in_batch
            num_samples_processed += valid_samples_in_batch

        if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if num_samples_processed > 0:
            train_pbar.set_postfix({'epoch_loss': f'{total_loss / num_samples_processed:.4f}'})

    train_pbar.close()
    
    avg_loss = total_loss / num_samples_processed if num_samples_processed > 0 else 0.0
    return {'loss': avg_loss}

# Legacy 模式訓練函數（復現原版 SubgraphRAG）
def train_epoch_legacy(device, train_loader, model, optimizer):
    """Legacy 模式訓練函數 - 復現原版 SubgraphRAG 訓練流程"""
    model.train()
    epoch_loss = 0
    
    for sample in tqdm(train_loader):
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
        num_non_text_entities, relation_embs, topic_entity_one_hot,\
        target_triple_probs, a_entity_id_list = prepare_sample(device, sample)
            
        if len(h_id_tensor) == 0:
            continue

        pred_triple_logits = model(
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
            num_non_text_entities, relation_embs, topic_entity_one_hot)
        target_triple_probs = target_triple_probs.to(device).unsqueeze(-1)
        
        # Legacy 模式：使用標準 BCE 損失（無 pos_weight）
        loss = F.binary_cross_entropy_with_logits(
            pred_triple_logits, target_triple_probs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss = loss.item()
        epoch_loss += loss
    
    epoch_loss /= len(train_loader)
    
    log_dict = {'loss': epoch_loss}
    return log_dict

# Legacy 模式評估函數（復現原版 SubgraphRAG）
@torch.no_grad()
def eval_epoch_legacy(config, device, data_loader, model):
    """Legacy 模式評估函數 - 復現原版 SubgraphRAG 評估流程"""
    model.eval()
    
    metric_dict = defaultdict(list)
    
    for sample in tqdm(data_loader):
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
        num_non_text_entities, relation_embs, topic_entity_one_hot,\
        target_triple_probs, a_entity_id_list = prepare_sample(device, sample)

        pred_triple_logits = model(
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
            num_non_text_entities, relation_embs, topic_entity_one_hot).reshape(-1)
        
        # Triple ranking
        sorted_triple_ids_pred = torch.argsort(
            pred_triple_logits, descending=True).cpu()
        triple_ranks_pred = torch.empty_like(sorted_triple_ids_pred)
        triple_ranks_pred[sorted_triple_ids_pred] = torch.arange(
            len(triple_ranks_pred))
        
        target_triple_ids = target_triple_probs.nonzero().squeeze(-1)
        num_target_triples = len(target_triple_ids)
        
        if num_target_triples == 0:
            continue

        num_total_entities = len(entity_embs) + num_non_text_entities
        for k in config['eval']['k_list']:
            recall_k_sample = (
                triple_ranks_pred[target_triple_ids] < k).sum().item()
            metric_dict[f'triple_recall@{k}'].append(
                recall_k_sample / num_target_triples)
            
            triple_mask_k = triple_ranks_pred < k
            entity_mask_k = torch.zeros(num_total_entities)
            entity_mask_k[h_id_tensor[triple_mask_k]] = 1.
            entity_mask_k[t_id_tensor[triple_mask_k]] = 1.
            recall_k_sample_ans = entity_mask_k[a_entity_id_list].sum().item()
            metric_dict[f'ans_recall@{k}'].append(
                recall_k_sample_ans / len(a_entity_id_list))

    for key, val in metric_dict.items():
        metric_dict[key] = np.mean(val)
    
    return metric_dict

def main(args):
    # 檢查是否使用 legacy 模式（復現原版 SubgraphRAG）
    use_legacy_mode = getattr(args, "legacy_mode", False)
    
    if use_legacy_mode:
        print("🔄 使用 Legacy 模式 - 復現原版 SubgraphRAG 訓練流程")
        # Legacy 模式：關閉所有優化功能
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    else:
        print("🚀 使用 Advanced 模式 - 啟用優化功能")
        # Advanced 模式：啟用優化功能
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    
    # 如果是測試模式，直接運行測試並返回
    if hasattr(args, 'test') and args.test:
        success = test_pipeline(args)
        if success:
            print("✅ 測試完成，pipeline運行正常")
        else:
            print("❌ 測試失敗，請檢查錯誤信息")
        return
    
    # 如果是檢查模式，運行檢查並返回
    if hasattr(args, 'check') and args.check:
        success = check_training_pipeline(args)
        if success:
            print("✅ 檢查完成，訓練pipeline運行正常")
        else:
            print("❌ 檢查失敗，請檢查錯誤信息")
        return
    
    config_file = f'configs/retriever/{args.dataset}.yaml'
    config = load_yaml(config_file)
    device = torch.device('cuda:0')
    
    torch.set_num_threads(config['env']['num_threads'])
    set_seed(config['env']['seed'])

    usage = SystemUsageMonitor(interval_sec=1.0)
    usage.start()

    method = args.method if hasattr(args, 'method') else 'default'
    freq_weight = (method == 'freq_weight')
    inv_freq_weight = (method == 'inv_freq_weight')
    method_sp = getattr(args, 'method_sp', 'none')

    ts = time.strftime('%b%d-%H:%M:%S', time.localtime(time.time() + 8 * 3600))
    exp_prefix = config['train']['save_prefix']
    # Build base name, then append all active feature/method tags
    exp_name_base = f'{exp_prefix}_{ts}'
    feature_tags = []
    # Methods (mutually exclusive by CLI, but keep additive logic)
    if inv_freq_weight:
        feature_tags.append('inv_freq_weight')
    elif freq_weight:
        feature_tags.append('freq_weight')
    # Post-SP reweight method can coexist
    if method_sp in ['spcount', 'spcount_inv']:
        feature_tags.append(method_sp)
    # Feature flags
    if getattr(args, 'use_dropout', False):
        try:
            dp = float(getattr(args, 'dropout_p', 0.2))
            dp_str = str(dp).replace('.', '_')
            feature_tags.append(f'drop{dp_str}')
        except Exception:
            feature_tags.append('drop')
    if getattr(args, 'legacy_mode', False):
        feature_tags.append('legacy')
    # Compose
    exp_name = exp_name_base if not feature_tags else f"{exp_name_base}_{'_'.join(feature_tags)}"
    # Optional user suffix
    if args.id_sup is not None:
        exp_name = f'{exp_name}_{args.id_sup}'
        
    config_df = pd.json_normalize(config, sep='/')
    wandb.init(
        project=f'{args.dataset}_retriever',
        name=exp_name,
        config=config_df.to_dict(orient='records')[0],
        mode='online'
    )
    
    num_epochs = args.num_epochs if hasattr(args, 'num_epochs') and args.num_epochs is not None else config['train']['num_epochs']
    patience = args.patience if hasattr(args, 'patience') and args.patience is not None else config['train']['patience']
    

    if hasattr(args, 'k_list') and args.k_list is not None:
        k_list_str = args.k_list.replace(' ', '')
        k_list = [int(k) for k in k_list_str.split(',')]
        config['eval']['k_list'] = k_list
    else:
        if isinstance(config['eval']['k_list'], str):
            config['eval']['k_list'] = [int(k) for k in config['eval']['k_list'].split(',')]
    
    print(f"=== Training Configuration ===")
    print(f"Dataset: {args.dataset}")
    print(f"Method: {method}")
    print(f"Method SP: {method_sp}")
    print(f"Frequency Weight: {'Enabled' if freq_weight else 'Disabled'}")
    print(f"Text Encoder: {config['dataset']['text_encoder_name']}")
    print(f"Patience: {patience}")
    print(f"K List: {config['eval']['k_list']}")
    print(f"Legacy Mode: {use_legacy_mode}")
    if args.id_sup is not None:
        print(f"ID Suffix: {args.id_sup}")
    print(f"===============================")
    
    check_and_warn_resources(args)
    
    if use_legacy_mode:
        print("🔄 使用 Legacy 資料集和 Collate 函數...")
        # Legacy 模式：使用原版資料集和 collate 函數
        from src.dataset.retriever import RetrieverDataset, collate_retriever
        
        train_set = RetrieverDataset(config=config, split='train')
        val_set = RetrieverDataset(config=config, split='val')
        
        # Legacy DataLoader 設定
        train_loader = DataLoader(
            train_set, batch_size=1, shuffle=True, collate_fn=collate_retriever)
        val_loader = DataLoader(
            val_set, batch_size=1, collate_fn=collate_retriever)
    else:
        print("🚀 使用 OptimizedRetrieverDataset 和 optimized_collate_retriever 進行訓練...")
        # Advanced 模式：使用優化版資料集
        train_set = OptimizedRetrieverDataset(
            config=config,
            split='train',
            freq_weight=freq_weight,
            weight_mode=(
                'freq' if method == 'freq_weight' else (
                'inv' if method == 'inv_freq_weight' else (
                method_sp if method_sp in ['spcount', 'spcount_inv'] else 'none'))),
        )
        
        val_set = OptimizedRetrieverDataset(
            config=config,
            split='val',
            freq_weight=freq_weight,
            weight_mode=(
                'freq' if method == 'freq_weight' else (
                'inv' if method == 'inv_freq_weight' else (
                method_sp if method_sp in ['spcount', 'spcount_inv'] else 'none'))),
        )
        
        # 支援每個 epoch 只取部分樣本進行訓練
        train_sampler = None
        if hasattr(args, 'samples_per_epoch') and args.samples_per_epoch is not None:
            num_samples = min(args.samples_per_epoch, len(train_set))
            train_sampler = SubsetRandomSampler(torch.randperm(len(train_set))[:num_samples])

        # 構建 DataLoader：若可用，使用分組 batch sampler 以提升 I/O 局部性
        use_grouped_sampler_train = (train_sampler is None) and hasattr(train_set, 'emb_dict') and hasattr(train_set.emb_dict, '_sample_to_batch')
        if use_grouped_sampler_train:
            train_batch_sampler = GroupedByFileBatchSampler(
                dataset=train_set,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False
            )
            train_loader = DataLoader(
                train_set,
                batch_sampler=train_batch_sampler,
                collate_fn=optimized_collate_retriever,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=(args.num_workers > 0 and args.persistent_workers)
            )
        else:
            train_loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                collate_fn=optimized_collate_retriever,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=(args.num_workers > 0 and args.persistent_workers)
            )

        use_grouped_sampler_val = hasattr(val_set, 'emb_dict') and hasattr(val_set.emb_dict, '_sample_to_batch')
        if use_grouped_sampler_val:
            val_batch_sampler = GroupedByFileBatchSampler(
                dataset=val_set,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=True
            )
            val_loader = DataLoader(
                val_set,
                batch_sampler=val_batch_sampler,
                collate_fn=optimized_collate_retriever,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=(args.num_workers > 0 and args.persistent_workers)
            )
        else:
            val_loader = DataLoader(
                val_set,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=optimized_collate_retriever,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=(args.num_workers > 0 and args.persistent_workers)
            )
        
        # 記錄triplet統計信息
        triplet_info = {
            'train': {
                'skipped_samples': train_set.num_skipped,
                'relevant_triples_median': train_set.median_num_relevant,
                'relevant_triples_mean': train_set.mean_num_relevant,
                'relevant_triples_max': train_set.max_num_relevant
            },
            'val': {
                'skipped_samples': val_set.num_skipped,
                'relevant_triples_median': val_set.median_num_relevant,
                'relevant_triples_mean': val_set.mean_num_relevant,
                'relevant_triples_max': val_set.max_num_relevant
            }
        }
        
    # --- model&optimizer ---
    sample_data = train_set[0]
    emb_size = sample_data['q_emb'].shape[-1]
    
    # 根據模式選擇模型初始化
    if use_legacy_mode:
        # Legacy 模式：使用原版模型（無 dropout）
        model = Retriever(emb_size, **config['retriever']).to(device)
        optimizer = Adam(model.parameters(), **config['optimizer'])
        scaler = None  # Legacy 模式不使用 AMP
    else:
        # Advanced 模式：使用增強版模型
        model = Retriever(emb_size, **config['retriever'], use_dropout=args.use_dropout, dropout_p=args.dropout_p).to(device)
        optimizer = build_adam_optimizer(model.parameters(), config['optimizer'])
        scaler = GradScaler()
    
     # --- Training ---
    start_epoch = 0
    best_val_metric = 0.0
    num_patient_epochs = 0

    if args.resume_best is not None:
        ckpt = torch.load(args.resume_best, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scaler_state_dict' in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        best_val_metric = ckpt.get('best_val_metric', 0.0)
        num_patient_epochs = ckpt.get('num_patient_epochs', 0)
        start_epoch = ckpt.get('epoch', -1) + 1
        exp_name = ckpt.get('exp_name', exp_name)  # 用 ckpt 的 exp_name 覆蓋

    main_pbar = tqdm(range(start_epoch, num_epochs), desc="Overall Training Progress")
    
    for epoch in main_pbar:
        # 1. 更新主進度條描述
        main_pbar.set_description(f"Training Progress (Epoch {epoch+1}/{num_epochs})")

        # 2. 訓練階段 (Train)
        if use_legacy_mode:
            # Legacy 模式：使用原版訓練函數
            train_log_dict = train_epoch_legacy(device, train_loader, model, optimizer)
        else:
            # Advanced 模式：使用增強版訓練函數
            train_log_dict = train_epoch(
                device, train_loader, model, optimizer, scaler, 
                args.grad_accum_steps, epoch, num_epochs
            )

        # 3. 驗證階段 (Validation)
        if use_legacy_mode:
            # Legacy 模式：使用原版評估函數
            val_eval_dict = eval_epoch_legacy(config, device, val_loader, model)
        else:
            # Advanced 模式：使用增強版評估函數
            val_eval_dict = eval_epoch(
                config, device, val_loader, model, epoch, num_epochs
            )
        target_val_metric = val_eval_dict.get('triple_recall@100', 0.0) # 使用 .get() 避免 KeyError

        # 4. 記錄與日誌 (Logging)
        #    將訓練和驗證的結果上傳到 wandb
        log_payload = {
            'epoch': epoch,
            'train_loss': train_log_dict.get('loss', 0.0),
            'num_patient_epochs': num_patient_epochs
        }
        # 添加所有驗證指標到日誌
        for key, val in val_eval_dict.items():
            log_payload[f'val/{key}'] = val
        wandb.log(log_payload)

        # 5. 模型評估與儲存 (Checkpointing)
        #    根據驗證結果判斷是否為最佳模型
        if target_val_metric > best_val_metric:
            print(f"\n📈 New best model found! Recall@100: {best_val_metric:.4f} -> {target_val_metric:.4f}")
            num_patient_epochs = 0  # 重置耐心計數
            best_val_metric = target_val_metric
            best_val_loss = val_eval_dict.get('val_loss', float('inf'))

            # 準備儲存的狀態字典
            best_state_dict = {
                'config': config,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'num_patient_epochs': num_patient_epochs,
                'best_val_metric': best_val_metric,
                'exp_name': exp_name,
            }
            
            # 添加 scaler 狀態（如果存在）
            if scaler is not None:
                best_state_dict['scaler_state_dict'] = scaler.state_dict()
            
            # 儲存模型權重
            if use_legacy_mode:
                # Legacy 模式：使用原版儲存路徑
                save_dir = exp_name
                os.makedirs(save_dir, exist_ok=True)
                torch.save(best_state_dict, os.path.join(save_dir, 'cpt.pth'))
            else:
                # Advanced 模式：使用新儲存路徑
                save_dir = os.path.join('/home/YX_thesis/retrieve/results/training', args.dataset, exp_name)
                os.makedirs(save_dir, exist_ok=True)
                torch.save(best_state_dict, os.path.join(save_dir, 'cpt.pth'))
                
                # 儲存統計資訊（僅 Advanced 模式）
                if not use_legacy_mode:
                    triplet_info_path = os.path.join(save_dir, 'triplet_info.txt')
                    with open(triplet_info_path, 'w') as f:
                        f.write(f"Dataset: {args.dataset}\n")
                        f.write(f"Experiment: {exp_name}\n")
                        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 8 * 3600))}\n\n")
                        f.write("Best Model achieved at Epoch: {epoch}\n")
                        f.write(f"Best Validation Recall@100: {best_val_metric:.4f}\n\n")

                        f.write("Training Set Statistics:\n")
                        f.write(f"  Skipped samples: {train_set.num_skipped}\n")
                        f.write(f"  Relevant triples - Median: {triplet_info['train']['relevant_triples_median']}\n")
                        f.write(f"  Relevant triples - Mean: {triplet_info['train']['relevant_triples_mean']}\n")
                        f.write(f"  Relevant triples - Max: {triplet_info['train']['relevant_triples_max']}\n\n")

                        f.write("Validation Set Statistics:\n")
                        f.write(f"  Skipped samples: {triplet_info['val']['skipped_samples']}\n")
                        f.write(f"  Relevant triples - Median: {triplet_info['val']['relevant_triples_median']}\n")
                        f.write(f"  Relevant triples - Mean: {triplet_info['val']['relevant_triples_mean']}\n")
                        f.write(f"  Relevant triples - Max: {triplet_info['val']['relevant_triples_max']}\n")

        else:
            # 如果模型表現沒有提升，增加耐心計數
            num_patient_epochs += 1

        # 6. 更新進度條顯示
        main_pbar.set_postfix({
            'train_loss': f"{train_log_dict.get('loss', 0.0):.4f}",
            'val_recall@100': f'{target_val_metric:.4f}',
            'best_recall@100': f'{best_val_metric:.4f}',
            'patience': f'{num_patient_epochs}/{patience}'
        })

        # 7. 提前停止 (Early Stopping)
        if num_patient_epochs >= patience:
            print(f"\n⌛ Early stopping triggered at epoch {epoch+1} after {patience} epochs with no improvement.")
            break
    
    main_pbar.close()
    print(f"\n✅ Training completed! Best validation recall@100: {best_val_metric:.4f}")
    usage.stop()
    stats = usage.report(prefix="Training system usage")
    usage.recommend(args, context_label="train")

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Train SubgraphRAG retriever model')
    
    parser.add_argument('-d', '--dataset', type=str, required=True, 
                        choices=['webqsp', 'cwq'], help='Dataset name')
    parser.add_argument('-e', '--num_epochs', type=int, default=None,
                        help='Number of training epochs (overrides config file)')
    parser.add_argument('-m', '--method', type=str, choices=['default', 'freq_weight', 'inv_freq_weight'], default='default',
                        help='Method for shortest path calculation: default (no weights), freq_weight (frequency-based), inv_freq_weight (inverse-frequency)')
    parser.add_argument('--method_sp', type=str, choices=['none', 'spcount', 'spcount_inv'], default='none',
                        help='Post-shortest-path reweighting by path frequency: spcount (more paths -> higher positive weight), spcount_inv (more paths -> lower positive weight)')
    parser.add_argument('-id_sup', '--id_sup', type=str, default=None,
                        help='Additional identifier suffix for experiment name (e.g., -id_sup abc will create folder_name_abc)')
    parser.add_argument('-p', '--patience', type=int, default=None,
                        help='Patience for early stopping (overrides config file)')
    parser.add_argument('-k', '--k_list', type=str, default=None,
                        help='Custom k values for evaluation (e.g., "5,10,50,100")')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='DataLoader num_workers override')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='DataLoader batch size override')
    parser.add_argument('--prefetch_factor', type=int, default=4,
                        help='DataLoader prefetch_factor to keep workers busy')
    parser.add_argument('--pin_memory', dest='pin_memory', action='store_true', default=True,
                        help='Enable pin_memory to speed up host-to-device transfer (default: on)')
    parser.add_argument('--no_pin_memory', dest='pin_memory', action='store_false',
                        help='Disable pin_memory')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Number of gradient accumulation steps to simulate larger batch size')
    parser.add_argument('--samples_per_epoch', type=int, default=None,
                        help='Limit number of training samples per epoch (useful for quick tests, esp. cwq)')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode: skip full data loading and use dummy data to verify pipeline')
    parser.add_argument('--check', action='store_true',
                        help='Run in check mode: use real dataset to verify training pipeline without full training')
    parser.add_argument('--resume_best', type=str, default=None,
                        help='Path to a checkpoint file to resume training from the best model state.')
    parser.add_argument('--wandb_resume', action='store_true',
                        help='If set, resume the same wandb run if possible.')
    parser.add_argument('--persistent_workers', action='store_true', default=True,
                        help='Keep DataLoader workers alive between epochs (default: on)')
    parser.add_argument('--no_persistent_workers', dest='persistent_workers', action='store_false',
                        help='Disable persistent DataLoader workers to release memory between epochs')
    # New: dropout control
    parser.add_argument('--use_dropout', action='store_true', default=False,
                        help='Enable dropout layer in prediction head')
    parser.add_argument('--dropout_p', type=float, default=0.2,
                        help='Dropout probability if --use_dropout is set')
    
    # Legacy 模式參數
    parser.add_argument('--legacy_mode', action='store_true', default=False,
                        help='Use legacy mode to reproduce original SubgraphRAG results')
    parser.add_argument('--no_pos_weight', action='store_true', default=False,
                        help='Disable pos_weight in BCE loss (legacy compatibility)')
    parser.add_argument('--no_amp', action='store_true', default=False,
                        help='Disable AMP (Automatic Mixed Precision) for legacy compatibility')
    parser.add_argument('--legacy_batch_size', type=int, default=1,
                        help='Batch size for legacy mode (default: 1)')
    parser.add_argument('--legacy_optimizer', action='store_true', default=False,
                        help='Use legacy Adam optimizer without fused/foreach optimizations')
    
    args = parser.parse_args()
    
    # 調試：打印參數
    ts_start = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 8 * 3600))
    start_time = time.time()
    print(f"========== Start training retriever {ts_start} ==========")
    print(f"method: {args.method}")
    print(f"freq_weight: {args.method == 'freq_weight'}")
    print(f"id_sup: {args.id_sup}")
    print(f"patience: {args.patience}")
    print(f"k_list: {args.k_list}")
    print(f"batch_size: {args.batch_size}")
    print(f"num_workers: {args.num_workers}")
    print(f"grad_accum_steps: {args.grad_accum_steps}")
    print(f"prefetch_factor: {args.prefetch_factor}")
    print(f"pin_memory: {args.pin_memory}")
    print(f"persistent_workers: {args.persistent_workers}")
    print(f"test_mode: {args.test}")
    print(f"check_mode: {args.check}")
    print(f"resume_best: {args.resume_best}")
    print(f"wandb_resume: {args.wandb_resume}")
    print(f"use_dropout: {args.use_dropout}")
    print(f"dropout_p: {args.dropout_p}")
    print(f"legacy_mode: {args.legacy_mode}")
    
    main(args)
    
    ts_end = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 8 * 3600))
    end_time = time.time()
    total_seconds = int(end_time - start_time)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    print(f"Retriever training time: {hours:02d}:{minutes:02d}:{seconds:02d}")
