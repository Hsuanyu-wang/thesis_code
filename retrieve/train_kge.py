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
from src.dataset.retriever import OptimizedRetrieverDataset, optimized_collate_retriever
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


# ---------------- KGE scorer (weights-only, manual scoring) ----------------
class InlineKGEScorer:
    def __init__(self, checkpoint_path: str, device: torch.device, freeze: bool = True, score_norm: str = 'logistic',
                 entity_map: dict = None, relation_map: dict = None, enable_cache: bool = False,
                 model_type: str = 'transe', cache_max_size: int = 0,
                 build_full_map: bool = False, map_tensor_max_elems: int = 0):
        self.device = torch.device(device)
        self.score_norm = score_norm
        self.entity_map = entity_map
        self.relation_map = relation_map
        self.enable_cache = enable_cache
        self._cache = {} if enable_cache else None
        self._cache_max_size = int(cache_max_size) if enable_cache else 0
        self.model_type = (model_type or 'transe').lower()
        self._build_full_map = bool(build_full_map)
        self._map_tensor_max_elems = int(map_tensor_max_elems) if map_tensor_max_elems else 0
        # weights
        self.entity_emb = None  # [num_entities, dim]
        self.relation_emb = None  # [num_relations, dim]
        self.num_entities = None
        self.num_relations = None
        self.emb_dim = None
        # vector maps
        self._entity_map_tensor = None  # cpu long
        self._relation_map_tensor = None  # cpu long
        self._load_weights(checkpoint_path)
        if freeze:
            # Nothing to freeze explicitly; we keep tensors detached
            pass

    def _load_weights(self, checkpoint_path: str):
        # PyTorch 2.6 defaults weights_only=True; try safe path first, then fallback
        obj = None
        try:
            obj = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except Exception:
            obj = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        # Heuristically extract state dict
        if isinstance(obj, dict):
            state_dict = None
            for key in ['model_state_dict', 'state_dict']:
                if key in obj and isinstance(obj[key], dict):
                    state_dict = obj[key]
                    break
            if state_dict is None:
                # Could be already a plain state dict
                state_dict = obj if all(isinstance(v, torch.Tensor) for v in obj.values()) else None
        else:
            state_dict = None
        if state_dict is None:
            raise RuntimeError(f"Cannot extract state_dict from checkpoint: {checkpoint_path}")

        # Find candidate 2D tensors
        candidates = [(k, v) for k, v in state_dict.items() if isinstance(v, torch.Tensor) and v.ndim == 2]
        if not candidates:
            raise RuntimeError("No 2D embedding matrices found in checkpoint state_dict.")

        # Prefer names containing 'entity' and 'relation'
        ent_cands = [(k, v) for k, v in candidates if ('ent' in k.lower() or 'entity' in k.lower()) and 'rel' not in k.lower()]
        rel_cands = [(k, v) for k, v in candidates if 'rel' in k.lower() or 'relation' in k.lower()]
        # Fallbacks
        if not ent_cands:
            # choose the matrix with the largest first dim as entities
            ent_cands = [max(candidates, key=lambda kv: kv[1].shape[0])]
        if not rel_cands:
            # choose a matrix with smaller first dim than entities as relations
            ent_rows = ent_cands[0][1].shape[0]
            smaller = [kv for kv in candidates if kv[1].shape[0] < ent_rows]
            rel_cands = [max(smaller, key=lambda kv: kv[1].shape[0])] if smaller else [candidates[0]]

        self.entity_emb = ent_cands[0][1].detach().to(self.device)
        self.relation_emb = rel_cands[0][1].detach().to(self.device)
        self.num_entities = int(self.entity_emb.shape[0])
        self.num_relations = int(self.relation_emb.shape[0])
        self.emb_dim = int(self.entity_emb.shape[1])
        # Free loader objects asap
        try:
            del obj
            del state_dict
        except Exception:
            pass
        import gc as _gc
        try:
            _gc.collect()
        except Exception:
            pass
        # Build vectorized mapping tensors if dicts provided
        if self._build_full_map:
            if self.entity_map is not None and isinstance(self.entity_map, dict) and len(self.entity_map) > 0:
                max_eid = max(int(k) for k in self.entity_map.keys())
                total = max_eid + 1
                if self._map_tensor_max_elems and total > self._map_tensor_max_elems:
                    print(f"⚠️ entity_map tensor size {total} exceeds limit {self._map_tensor_max_elems}; using per-batch dict mapping.")
                else:
                    ent_map = torch.full((total,), -1, dtype=torch.long)
                    for k, v in self.entity_map.items():
                        kk = int(k)
                        if kk >= 0:
                            try:
                                ent_map[kk] = int(v)
                            except Exception:
                                pass
                    self._entity_map_tensor = ent_map  # keep on CPU
            if self.relation_map is not None and isinstance(self.relation_map, dict) and len(self.relation_map) > 0:
                max_rid = max(int(k) for k in self.relation_map.keys())
                total = max_rid + 1
                if self._map_tensor_max_elems and total > self._map_tensor_max_elems:
                    print(f"⚠️ relation_map tensor size {total} exceeds limit {self._map_tensor_max_elems}; using per-batch dict mapping.")
                else:
                    rel_map = torch.full((total,), -1, dtype=torch.long)
                    for k, v in self.relation_map.items():
                        kk = int(k)
                        if kk >= 0:
                            try:
                                rel_map[kk] = int(v)
                            except Exception:
                                pass
                    self._relation_map_tensor = rel_map

    def _map_ids_vectorized(self, ids: torch.Tensor, map_tensor: torch.Tensor | None) -> torch.Tensor:
        # ids on CPU long
        if map_tensor is None:
            return ids
        if ids.dtype != torch.long:
            ids = ids.to(torch.long)
        out = torch.full_like(ids, -1)
        mask = (ids >= 0) & (ids < map_tensor.shape[0])
        if mask.any():
            out[mask] = map_tensor[ids[mask]]
        return out

    def _build_valid_indices(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        valid = (h >= 0) & (r >= 0) & (t >= 0)
        valid = valid & (h < self.num_entities) & (t < self.num_entities) & (r < self.num_relations)
        valid_idx = valid.nonzero(as_tuple=False).reshape(-1)
        return valid_idx

    def _manual_score(self, eh: torch.Tensor, rr: torch.Tensor, et: torch.Tensor) -> torch.Tensor:
        if self.model_type in ['transe', 'trans-e', 'trans_e']:
            diff = eh + rr - et
            # L2 norm negative as score
            scores = -torch.norm(diff, p=2, dim=1)
        elif self.model_type in ['distmult', 'dist-mult', 'dist_mult']:
            scores = (eh * rr * et).sum(dim=1)
        else:
            # default to DistMult if unknown
            scores = (eh * rr * et).sum(dim=1)
        return scores

    @torch.no_grad()
    def score_triples(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor):
        # Inputs are 1D cpu tensors or gpu; we will work on cpu for mapping then to device
        n = int(h.numel())
        if n == 0:
            return {'scores': torch.zeros(0, device=self.device), 'valid_idx': torch.zeros(0, dtype=torch.long)}
        if self.enable_cache:
            key = (tuple(h.tolist()), tuple(r.tolist()), tuple(t.tolist()))
            if key in self._cache:
                return self._cache[key]

        # ensure CPU long for mapping
        if h.device.type != 'cpu':
            h = h.cpu()
        if r.device.type != 'cpu':
            r = r.cpu()
        if t.device.type != 'cpu':
            t = t.cpu()
        h = h.to(torch.long)
        r = r.to(torch.long)
        t = t.to(torch.long)

        # Map IDs (prefer tensor mapping if built; else fallback to per-batch dict)
        if self.entity_map is not None and self._entity_map_tensor is not None:
            h_m = self._map_ids_vectorized(h, self._entity_map_tensor)
            t_m = self._map_ids_vectorized(t, self._entity_map_tensor)
        elif self.entity_map is not None:
            h_m = torch.tensor([self.entity_map.get(int(x), -1) for x in h.tolist()], dtype=torch.long)
            t_m = torch.tensor([self.entity_map.get(int(x), -1) for x in t.tolist()], dtype=torch.long)
        else:
            h_m, t_m = h, t
        if self.relation_map is not None and self._relation_map_tensor is not None:
            r_m = self._map_ids_vectorized(r, self._relation_map_tensor)
        elif self.relation_map is not None:
            r_m = torch.tensor([self.relation_map.get(int(x), -1) for x in r.tolist()], dtype=torch.long)
        else:
            r_m = r

        valid_idx = self._build_valid_indices(h_m, r_m, t_m)
        num_valid = int(valid_idx.numel())
        if num_valid == 0:
            print(f"ℹ️ KGE scoring: skipped {n}/{n} triples (unmapped/out-of-range).")
            result = {'scores': torch.zeros(0, device=self.device), 'valid_idx': valid_idx}
            if self.enable_cache:
                self._cache[key] = result
            return result
        if num_valid < n:
            # throttle: only print when large skip ratio
            skipped = n - num_valid
            if skipped / max(1, n) >= 0.25:
                print(f"ℹ️ KGE scoring: skipped {skipped}/{n} triples (unmapped/out-of-range).")

        h_sel = h_m[valid_idx].to(self.device, non_blocking=True)
        r_sel = r_m[valid_idx].to(self.device, non_blocking=True)
        t_sel = t_m[valid_idx].to(self.device, non_blocking=True)

        eh = self.entity_emb.index_select(0, h_sel)
        rr = self.relation_emb.index_select(0, r_sel)
        et = self.entity_emb.index_select(0, t_sel)
        scores = self._manual_score(eh, rr, et)

        # Normalize if requested
        if self.score_norm == 'logistic':
            scores = torch.sigmoid(scores)
        elif self.score_norm == 'zscore':
            mu = scores.mean()
            std = scores.std(unbiased=False) + 1e-8
            scores = (scores - mu) / std
        elif self.score_norm == 'minmax':
            mn = scores.min()
            mx = scores.max()
            scores = (scores - mn) / (mx - mn + 1e-8)

        result = {'scores': scores.detach(), 'valid_idx': valid_idx.to(self.device)}
        if self.enable_cache:
            # LRU size bound
            if self._cache_max_size and len(self._cache) >= self._cache_max_size:
                try:
                    # pop arbitrary oldest key (simulate LRU by FIFO for simplicity)
                    first_key = next(iter(self._cache))
                    self._cache.pop(first_key, None)
                except Exception:
                    self._cache.clear()
            self._cache[key] = result
        return result


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
    model = Retriever(emb_size, **config['retriever']).to(device)
    optimizer = Adam(model.parameters(), **config['optimizer'])
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
    with torch.no_grad():
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
    print(f"✅ 方法: {method}, 頻率權重: {freq_weight}")
    
    # 檢查數據集加載
    try:
        print(" 正在加載訓練數據集...")
        train_set = OptimizedRetrieverDataset(
            config=config,
            split='train',
            freq_weight=freq_weight,
        )
        print(f"✅ 訓練數據集加載成功，樣本數: {len(train_set)}")
        
        print(" 正在加載驗證數據集...")
        val_set = OptimizedRetrieverDataset(
            config=config,
            split='val',
            freq_weight=freq_weight,
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
        
        model = Retriever(emb_size, **config['retriever']).to(device)
        optimizer = Adam(model.parameters(), **config['optimizer'])
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
        
        # 將批次數據移動到device
        train_batch['q_emb'] = train_batch['q_emb'].to(device, non_blocking=True)
        if 'entity_embs' in train_batch:
            train_batch['entity_embs'] = train_batch['entity_embs'].to(device, non_blocking=True)
        if 'entity_embs_list' in train_batch:
            train_batch['entity_embs_list'] = [t.to(device, non_blocking=True) for t in train_batch['entity_embs_list']]
        if 'relation_embs' in train_batch:
            train_batch['relation_embs'] = train_batch['relation_embs'].to(device, non_blocking=True)
        if 'relation_embs_list' in train_batch:
            train_batch['relation_embs_list'] = [t.to(device, non_blocking=True) for t in train_batch['relation_embs_list']]
        if 'topic_entity_one_hot' in train_batch:
            train_batch['topic_entity_one_hot'] = train_batch['topic_entity_one_hot'].to(device, non_blocking=True)
        if 'target_triple_probs' in train_batch:
            train_batch['target_triple_probs'] = train_batch['target_triple_probs'].to(device, non_blocking=True)
        if 'target_triple_probs_list' in train_batch:
            train_batch['target_triple_probs_list'] = [t.to(device, non_blocking=True) for t in train_batch['target_triple_probs_list']]
        if 'h_id_tensors' in train_batch:
            train_batch['h_id_tensors'] = [t.to(device, non_blocking=True) for t in train_batch['h_id_tensors']]
        if 'r_id_tensors' in train_batch:
            train_batch['r_id_tensors'] = [t.to(device, non_blocking=True) for t in train_batch['r_id_tensors']]
        if 't_id_tensors' in train_batch:
            train_batch['t_id_tensors'] = [t.to(device, non_blocking=True) for t in train_batch['t_id_tensors']]
        
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
            
            pred_logits = pred_logits.reshape(-1)
            target = target.reshape(-1)
            
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
        with torch.no_grad():
            # 將批次資料移動到 GPU（同時支援 padded 與 list 版本）
            if 'q_emb' in val_batch:
                val_batch['q_emb'] = val_batch['q_emb'].to(device, non_blocking=True)
            if 'entity_embs' in val_batch:
                val_batch['entity_embs'] = val_batch['entity_embs'].to(device, non_blocking=True)
            if 'relation_embs' in val_batch:
                val_batch['relation_embs'] = val_batch['relation_embs'].to(device, non_blocking=True)
            if 'topic_entity_one_hot' in val_batch:
                val_batch['topic_entity_one_hot'] = val_batch['topic_entity_one_hot'].to(device, non_blocking=True)
            if 'target_triple_probs' in val_batch:
                val_batch['target_triple_probs'] = val_batch['target_triple_probs'].to(device, non_blocking=True)

            # list 版本
            if 'entity_embs_list' in val_batch:
                val_batch['entity_embs_list'] = [t.to(device, non_blocking=True) for t in val_batch['entity_embs_list']]
            if 'relation_embs_list' in val_batch:
                val_batch['relation_embs_list'] = [t.to(device, non_blocking=True) for t in val_batch['relation_embs_list']]
            if 'topic_entity_one_hot_list' in val_batch:
                val_batch['topic_entity_one_hot_list'] = [t.to(device, non_blocking=True) for t in val_batch['topic_entity_one_hot_list']]
            if 'target_triple_probs_list' in val_batch:
                val_batch['target_triple_probs_list'] = [t.to(device, non_blocking=True) for t in val_batch['target_triple_probs_list']]

            if 'h_id_tensors' in val_batch:
                val_batch['h_id_tensors'] = [t.to(device, non_blocking=True) for t in val_batch['h_id_tensors']]
            if 'r_id_tensors' in val_batch:
                val_batch['r_id_tensors'] = [t.to(device, non_blocking=True) for t in val_batch['r_id_tensors']]
            if 't_id_tensors' in val_batch:
                val_batch['t_id_tensors'] = [t.to(device, non_blocking=True) for t in val_batch['t_id_tensors']]
            
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
        exp_name = f'{exp_prefix}_freq_weight_{ts}' if freq_weight else f'{exp_prefix}_{ts}'
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
                            "如果遇到記憶體不足(OOM)錯誤，請優先考慮降低 num_workers。")
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
        # 將批次資料移動到 device（同時支援 padded 與 list 版本）
        if 'q_emb' in batch:
            batch['q_emb'] = batch['q_emb'].to(device, non_blocking=True)
        if 'entity_embs' in batch:
            batch['entity_embs'] = batch['entity_embs'].to(device, non_blocking=True)
        if 'relation_embs' in batch:
            batch['relation_embs'] = batch['relation_embs'].to(device, non_blocking=True)
        if 'topic_entity_one_hot' in batch:
            batch['topic_entity_one_hot'] = batch['topic_entity_one_hot'].to(device, non_blocking=True)
        if 'target_triple_probs' in batch:
            batch['target_triple_probs'] = batch['target_triple_probs'].to(device, non_blocking=True)
        # list 版本
        if 'entity_embs_list' in batch:
            batch['entity_embs_list'] = [t.to(device, non_blocking=True) for t in batch['entity_embs_list']]
        if 'relation_embs_list' in batch:
            batch['relation_embs_list'] = [t.to(device, non_blocking=True) for t in batch['relation_embs_list']]
        if 'topic_entity_one_hot_list' in batch:
            batch['topic_entity_one_hot_list'] = [t.to(device, non_blocking=True) for t in batch['topic_entity_one_hot_list']]
        if 'target_triple_probs_list' in batch:
            batch['target_triple_probs_list'] = [t.to(device, non_blocking=True) for t in batch['target_triple_probs_list']]
        if 'h_id_tensors' in batch:
            batch['h_id_tensors'] = [t.to(device, non_blocking=True) for t in batch['h_id_tensors']]
        if 'r_id_tensors' in batch:
            batch['r_id_tensors'] = [t.to(device, non_blocking=True) for t in batch['r_id_tensors']]
        if 't_id_tensors' in batch:
            batch['t_id_tensors'] = [t.to(device, non_blocking=True) for t in batch['t_id_tensors']]

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
            # 對齊長度
            if target.numel() != pred_triple_logits.numel():
                min_len = min(target.numel(), pred_triple_logits.numel())
                pred_triple_logits = pred_triple_logits[:min_len]
                target = target[:min_len]
            h_id_tensor = batch['h_id_tensors'][i]
            t_id_tensor = batch['t_id_tensors'][i]
            a_entity_id_list = batch['a_entity_id_lists'][i]
            
            # --- 以下是單一樣本的評估邏輯 ---
            if len(h_id_tensor) > 0:
                num_positive = target.sum().item()
                num_total = len(target)
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

def train_epoch(device, train_loader, model, optimizer, scaler, grad_accum_steps=1, epoch=None, total_epochs=None,
                kge_scorer: InlineKGEScorer = None, kge_loss_coef: float = 0.0):
    model.train()
    total_loss = 0
    total_kge_loss = 0.0
    num_kge_samples = 0
    num_samples_processed = 0
    
    train_desc = f"Train Epoch {epoch+1}/{total_epochs}" if epoch is not None and total_epochs is not None else "Training"
    train_pbar = tqdm(train_loader, desc=train_desc, leave=False)

    optimizer.zero_grad(set_to_none=True)
    
    for i, batch in enumerate(train_pbar):
        # 將批次資料移動到 device（同時支援 padded 與 list 版本）
        if 'q_emb' in batch:
            batch['q_emb'] = batch['q_emb'].to(device, non_blocking=True)
        if 'entity_embs' in batch:
            batch['entity_embs'] = batch['entity_embs'].to(device, non_blocking=True)
        if 'relation_embs' in batch:
            batch['relation_embs'] = batch['relation_embs'].to(device, non_blocking=True)
        if 'topic_entity_one_hot' in batch:
            batch['topic_entity_one_hot'] = batch['topic_entity_one_hot'].to(device, non_blocking=True)
        if 'target_triple_probs' in batch:
            batch['target_triple_probs'] = batch['target_triple_probs'].to(device, non_blocking=True)

        # list 版本
        if 'entity_embs_list' in batch:
            batch['entity_embs_list'] = [t.to(device, non_blocking=True) for t in batch['entity_embs_list']]
        if 'relation_embs_list' in batch:
            batch['relation_embs_list'] = [t.to(device, non_blocking=True) for t in batch['relation_embs_list']]
        if 'topic_entity_one_hot_list' in batch:
            batch['topic_entity_one_hot_list'] = [t.to(device, non_blocking=True) for t in batch['topic_entity_one_hot_list']]
        if 'target_triple_probs_list' in batch:
            batch['target_triple_probs_list'] = [t.to(device, non_blocking=True) for t in batch['target_triple_probs_list']]
        if 'h_id_tensors' in batch:
            batch['h_id_tensors'] = [t.to(device, non_blocking=True) for t in batch['h_id_tensors']]
        if 'r_id_tensors' in batch:
            batch['r_id_tensors'] = [t.to(device, non_blocking=True) for t in batch['r_id_tensors']]
        if 't_id_tensors' in batch:
            batch['t_id_tensors'] = [t.to(device, non_blocking=True) for t in batch['t_id_tensors']]

        with autocast('cuda'):
            pred_triple_logits_batch = model(batch) # 直接傳遞整個 batch dict

            # 逐樣本計算 loss
            loss = 0
            kge_distill_loss_sum = 0.0
            valid_samples_in_batch = 0
            for j in range(len(pred_triple_logits_batch)):
                pred_logits = pred_triple_logits_batch[j]
                # 選擇目標：優先使用 list 版，否則使用 padded 版的第 j 列
                if 'target_triple_probs_list' in batch:
                    target = batch['target_triple_probs_list'][j]
                else:
                    target = batch['target_triple_probs'][j]
                
                pred_logits = pred_logits.reshape(-1)
                target = target.reshape(-1)
                
                if pred_logits.numel() == 0: 
                    continue
                
                # 對齊長度（與 target、KGE 分數都需對齊）
                min_len = target.numel()
                if pred_logits.numel() != min_len:
                    min_len = min(min_len, pred_logits.numel())
                    pred_logits = pred_logits[:min_len]
                    target = target[:min_len]
                
                num_positive = target.sum().item()
                num_total = len(target)
                pos_weight = torch.tensor([(num_total - num_positive) / num_total if num_positive > 0 else 1.0], device=device)
                loss += F.binary_cross_entropy_with_logits(pred_logits, target, pos_weight=pos_weight)

                # KGE 蒸餾 loss（僅當啟用且有 scorer）
                if kge_scorer is not None and kge_loss_coef > 0.0 and 'h_id_tensors' in batch and 'r_id_tensors' in batch and 't_id_tensors' in batch:
                    h_ids = batch['h_id_tensors'][j].reshape(-1)
                    r_ids = batch['r_id_tensors'][j].reshape(-1)
                    t_ids = batch['t_id_tensors'][j].reshape(-1)
                    if h_ids.numel() > 0 and r_ids.numel() == h_ids.numel() and t_ids.numel() == h_ids.numel():
                        try:
                            kge_out = kge_scorer.score_triples(h_ids.detach().cpu(), r_ids.detach().cpu(), t_ids.detach().cpu())
                            kge_scores = kge_out['scores']  # shape [num_valid]
                            valid_idx = kge_out['valid_idx']  # indices into original order
                            if kge_scores.numel() == 0:
                                # nothing to distill for this sample
                                raise RuntimeError('no valid triples for KGE')
                            # align pred logits to valid indices
                            pred_for_kge = pred_logits.index_select(0, valid_idx.to(pred_logits.device))
                            # 蒸餾目標到 [0,1] 後與 sigmoid(logits) 用 MSE 更穩定
                            pred_probs = torch.sigmoid(pred_for_kge)
                            # 若 score_norm 不是 logistic，kge_scorer 可能返回非 [0,1]，此處做保守壓縮
                            if kge_scores.dtype != pred_probs.dtype:
                                kge_scores = kge_scores.to(pred_probs.dtype)
                            if kge_scorer.score_norm not in ['logistic', 'minmax']:
                                kge_scores = torch.sigmoid(kge_scores)
                            L_kge = F.mse_loss(pred_probs, kge_scores.to(pred_probs.device))
                            kge_distill_loss_sum += L_kge
                        except Exception:
                            pass
                valid_samples_in_batch += 1

        if valid_samples_in_batch > 0:
            # 合併 KGE loss
            if kge_loss_coef > 0.0 and kge_distill_loss_sum != 0.0:
                loss = loss + kge_loss_coef * kge_distill_loss_sum
                total_kge_loss += float(kge_distill_loss_sum.detach().cpu())
                num_kge_samples += 1

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
            postfix = {'epoch_loss': f'{total_loss / num_samples_processed:.4f}'}
            if num_kge_samples > 0:
                postfix['kge_loss'] = f'{(total_kge_loss / max(1, num_kge_samples)):.4f}'
            train_pbar.set_postfix(postfix)

    train_pbar.close()
    
    avg_loss = total_loss / num_samples_processed if num_samples_processed > 0 else 0.0
    avg_kge_loss = (total_kge_loss / max(1, num_kge_samples)) if num_kge_samples > 0 else 0.0
    return {'loss': avg_loss, 'kge_loss': avg_kge_loss}

def main(args):
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
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(config['env']['num_threads'])
    set_seed(config['env']['seed'])

    usage = SystemUsageMonitor(interval_sec=1.0)
    usage.start()

    method = args.method if hasattr(args, 'method') else 'default'
    freq_weight = (method == 'freq_weight')

    ts = time.strftime('%b%d-%H:%M:%S', time.localtime(time.time() + 8 * 3600))
    exp_prefix = config['train']['save_prefix']
    exp_name = f'{exp_prefix}_freq_weight_{ts}' if freq_weight else f'{exp_prefix}_{ts}'
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
    print(f"Frequency Weight: {'Enabled' if freq_weight else 'Disabled'}")
    print(f"Text Encoder: {config['dataset']['text_encoder_name']}")
    print(f"Patience: {patience}")
    print(f"K List: {config['eval']['k_list']}")
    if args.id_sup is not None:
        print(f"ID Suffix: {args.id_sup}")
    if getattr(args, 'use_kge_loss', False):
        print(f"Use KGE Loss: True | coef={args.kge_loss_coef} | norm={args.kge_score_norm}")
        print(f"KGE ckpt: {args.kge_model_path} | device: {args.kge_device} | freeze: {args.freeze_kge}")
    print(f"===============================")

    check_and_warn_resources(args)
    print("🚀 使用 OptimizedRetrieverDataset 和 optimized_collate_retriever 進行訓練...")

    train_set = OptimizedRetrieverDataset(
        config=config,
        split='train',
        freq_weight=freq_weight,
    )
    
    val_set = OptimizedRetrieverDataset(
        config=config,
        split='val',
        freq_weight=freq_weight,
    )
    
    # 支援每個 epoch 只取部分樣本進行訓練
    train_sampler = None
    if hasattr(args, 'samples_per_epoch') and args.samples_per_epoch is not None:
        num_samples = min(args.samples_per_epoch, len(train_set))
        train_sampler = SubsetRandomSampler(torch.randperm(len(train_set))[:num_samples])

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size, # 從 CLI 參數讀取
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=optimized_collate_retriever, # 使用新的 collate_fn
        num_workers=args.num_workers, # 從 CLI 參數讀取
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor, # 可由CLI設定
        persistent_workers=(args.num_workers > 0)
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size, # 驗證集也使用批次處理
        shuffle=False,
        collate_fn=optimized_collate_retriever,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=(args.num_workers > 0)
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
    
    model = Retriever(emb_size, **config['retriever']).to(device)
    optimizer = Adam(model.parameters(), **config['optimizer'])
    scaler = GradScaler()

    # --- optional: KGE scorer ---
    kge_scorer = None
    if getattr(args, 'use_kge_loss', False):
        if not args.kge_model_path or not os.path.exists(args.kge_model_path):
            raise FileNotFoundError(f"KGE checkpoint not found: {args.kge_model_path}")
        # Optional mapping files
        entity_map = None
        relation_map = None
        if getattr(args, 'kge_entity_map_path', None) and os.path.exists(args.kge_entity_map_path):
            try:
                import json
                with open(args.kge_entity_map_path, 'r') as f:
                    entity_map = {int(k): int(v) for k, v in json.load(f).items()}
            except Exception:
                print("⚠️ 讀取 kge_entity_map 失敗，將不使用映射")
        if getattr(args, 'kge_relation_map_path', None) and os.path.exists(args.kge_relation_map_path):
            try:
                import json
                with open(args.kge_relation_map_path, 'r') as f:
                    relation_map = {int(k): int(v) for k, v in json.load(f).items()}
            except Exception:
                print("⚠️ 讀取 kge_relation_map 失敗，將不使用映射")
        if hasattr(args, 'kge_device') and args.kge_device is not None:
            kge_device = torch.device(args.kge_device)
        else:
            kge_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        kge_scorer = InlineKGEScorer(
            checkpoint_path=args.kge_model_path,
            device=kge_device,
            freeze=getattr(args, 'freeze_kge', True),
            score_norm=getattr(args, 'kge_score_norm', 'logistic'),
            entity_map=entity_map,
            relation_map=relation_map,
            enable_cache=getattr(args, 'kge_cache_scores', False),
            model_type=(getattr(args, 'kge_model_type', None) or 'transe'),
            cache_max_size=getattr(args, 'kge_cache_max_size', 0),
            build_full_map=getattr(args, 'kge_build_full_map', False),
            map_tensor_max_elems=getattr(args, 'kge_map_tensor_max_elems', 0)
        )
    
     # --- Training ---
    num_patient_epochs = 0
    best_val_metric = 0.0

    main_pbar = tqdm(range(num_epochs), desc="Overall Training Progress")
    
    for epoch in main_pbar:
        # 1. 更新主進度條描述
        main_pbar.set_description(f"Training Progress (Epoch {epoch+1}/{num_epochs})")

        # 2. 訓練階段 (Train)
        #    模型在此階段學習並更新權重
        train_log_dict = train_epoch(
            device, train_loader, model, optimizer, scaler, 
            args.grad_accum_steps, epoch, num_epochs,
            kge_scorer=kge_scorer, kge_loss_coef=getattr(args, 'kge_loss_coef', 0.0)
        )

        # 3. 驗證階段 (Validation)
        #    使用訓練後的新權重在驗證集上評估模型表現
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
        if train_log_dict.get('kge_loss', 0.0) != 0.0:
            log_payload['train/kge_loss'] = train_log_dict['kge_loss']
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
                'epoch': epoch,
                'best_val_metric': best_val_metric
            }
            
            # 儲存模型權重
            save_dir = os.path.join('/home/YX_thesis/retrieve/results/training', args.dataset, exp_name)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(best_state_dict, os.path.join(save_dir, 'cpt.pth'))
            
            # 儲存統計資訊
            triplet_info_path = os.path.join(save_dir, 'triplet_info.txt')
            with open(triplet_info_path, 'w') as f:
                f.write(f"Dataset: {args.dataset}\n")
                f.write(f"Experiment: {exp_name}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 8 * 3600))}\n\n")
                f.write("Best Model achieved at Epoch: {epoch}\n")
                f.write(f"Best Validation Recall@100: {best_val_metric:.4f}\n\n")

                f.write("Training Set Statistics:\n")
                f.write(f"  Skipped samples: {train_set.num_skipped}\n") # 假設 train_set 有這些屬性
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
    parser.add_argument('-m', '--method', type=str, choices=['freq_weight', 'default'], default='default',
                        help='Method for shortest path calculation: freq_weight (frequency-based weights) or default (no weights)')
    parser.add_argument('-id_sup', '--id_sup', type=str, default=None,
                        help='Additional identifier suffix for experiment name (e.g., -id_sup abc will create folder_name_abc)')
    parser.add_argument('-p', '--patience', type=int, default=None,
                        help='Patience for early stopping (overrides config file)')
    parser.add_argument('-k', '--k_list', type=str, default=None,
                        help='Custom k values for evaluation (e.g., "5,10,50,100")')
    parser.add_argument('--num_workers', type=int, default=8,
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

    # --- KGE-related args ---
    parser.add_argument('--use_kge_loss', action='store_true', default=False,
                        help='Enable adding KGE score distillation loss during training')
    parser.add_argument('--kge_model_path', type=str, default=None,
                        help='Absolute path to PyKEEN checkpoint (.pt/.tar) used for scoring triples')
    parser.add_argument('--kge_model_type', type=str, default=None,
                        help='Optional: KGE model type for logging (e.g., transe, distmult, complex, rotate)')
    parser.add_argument('--kge_device', type=str, default=None,
                        help='Device for KGE scoring (e.g., cuda:0 or cpu). Defaults to training device.')
    parser.add_argument('--freeze_kge', action='store_true', default=True,
                        help='Freeze KGE model parameters (recommended)')
    parser.add_argument('--kge_loss_coef', type=float, default=0.1,
                        help='Weight for the additional KGE distillation loss term')
    parser.add_argument('--kge_score_norm', type=str, default='logistic', choices=['none','logistic','zscore','minmax'],
                        help='Normalization to apply to raw KGE scores before computing distillation loss')
    parser.add_argument('--kge_cache_scores', action='store_true', default=False,
                        help='Cache KGE scores per-sample to reduce repeated computation')
    parser.add_argument('--kge_entity_map_path', type=str, default=None,
                        help='Optional JSON mapping local_entity_id -> kge_entity_id')
    parser.add_argument('--kge_relation_map_path', type=str, default=None,
                        help='Optional JSON mapping local_relation_id -> kge_relation_id')
    parser.add_argument('--kge_cache_max_size', type=int, default=0,
                        help='Maximum size for KGE cache (0 for no limit)')
    parser.add_argument('--kge_build_full_map', action='store_true', default=False,
                        help='Build full-size vectorized mapping tensors (may use a lot of CPU memory). Default: off')
    parser.add_argument('--kge_map_tensor_max_elems', type=int, default=5000000,
                        help='Max allowed elements for a vectorized mapping tensor. If exceeded, fallback to per-batch dict mapping.')
    
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
    print(f"test_mode: {args.test}")
    print(f"check_mode: {args.check}")
    if args.use_kge_loss:
        print(f"use_kge_loss: {args.use_kge_loss} | kge_model_path: {args.kge_model_path} | kge_loss_coef: {args.kge_loss_coef} | norm: {args.kge_score_norm}")
    
    main(args)
    
    ts_end = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 8 * 3600))
    end_time = time.time()
    print(f"========== End training retriever {ts_end} ==========")
    total_seconds = int(end_time - start_time)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    print(f"Retriever training time: {hours:02d}:{minutes:02d}:{seconds:02d}")
