###################################################################//
# Ultra Memory-optimized training script for large datasets like cwq
# Uses additional memory optimization techniques
###################################################################\\
import numpy as np
import os
import pandas as pd
import time
import torch
import torch.nn.functional as F
import wandb
import gc
import psutil
import threading

from collections import defaultdict
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.retriever import load_yaml
from src.dataset.retriever import OptimizedRetrieverDataset, collate_retriever
from src.model.retriever import Retriever
from src.setup import set_seed, prepare_sample
from torch.cuda.amp import autocast, GradScaler

# GPU monitoring
try:
    import pynvml
    _HAS_PYNVML = True
except Exception:
    _HAS_PYNVML = False

GPU_MEM_FRACTION = 0.8

class SystemUsageMonitor:
    """系統使用率監控器，提供動態參數調整建議"""
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
                try:
                    used = torch.cuda.max_memory_allocated(0) / (1024**2)
                    self.gpu_mem_used_samples.append(used)
                    if self.gpu_mem_total is None:
                        self.gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                except Exception:
                    pass

    def _run(self):
        psutil.cpu_percent(interval=None)  # Prime CPU percent
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

    def recommend(self, args, context_label: str = ""):
        """根據系統使用率提供參數調整建議"""
        stats = self.get_stats()
        cpu_avg = stats['cpu_avg'] or 0.0
        gpu_util_avg = stats['gpu_util_avg'] or 0.0
        gpu_mem_max = stats['gpu_mem_max'] or 0.0
        gpu_mem_total = stats['gpu_mem_total'] or 0.0
        mem_headroom = (gpu_mem_total - gpu_mem_max) if (gpu_mem_total and gpu_mem_max) else None
        
        print("💡 Parameter recommendations" + (f" ({context_label})" if context_label else "") + ":")
        recs = []
        
        # 獲取當前參數
        nw = max(0, getattr(args, 'num_workers', 0))
        pf = max(2, getattr(args, 'prefetch_factor', 2))
        bs = max(1, getattr(args, 'train_bs', 1))
        gas = max(1, getattr(args, 'grad_accum_steps', 1))
        
        # GPU 使用率低且記憶體充足時，增加批次大小
        if torch.cuda.is_available() and gpu_util_avg < 60.0:
            if mem_headroom is not None and mem_headroom > 0.3 * (gpu_mem_total or 1):
                recs.append(f"Increase batch_size (e.g., {bs} -> {min(bs*2, bs+32)}) or decrease grad_accum_steps (e.g., {gas} -> {max(1, gas-1)}).")
            else:
                recs.append("GPU util low but memory tight; consider increasing prefetch_factor and num_workers.")
        
        # CPU 使用率低時，增加 workers/prefetch
        if cpu_avg < 50.0:
            if nw == 0:
                recs.append("Increase num_workers from 0 to 2-4 to parallelize data loading.")
            else:
                recs.append(f"Increase num_workers (e.g., {nw} -> {min(nw*2, nw+8)}) and/or prefetch_factor (e.g., {pf} -> {min(pf*2, pf+2)}).")
        
        # CPU 使用率過高時，減少 loader 壓力
        if cpu_avg > 85.0 and nw > 1:
            recs.append(f"CPU very busy; reduce num_workers (e.g., {nw} -> {max(1, nw//2)}) or lower prefetch_factor (e.g., {pf} -> {max(2, pf-1)}).")
        
        # GPU 記憶體接近滿時，減少批次大小
        if torch.cuda.is_available() and gpu_mem_total and gpu_mem_max / gpu_mem_total > 0.90:
            recs.append(f"GPU memory near capacity; reduce batch_size (e.g., {bs} -> {max(1, bs//2)}) or increase grad_accum_steps ({gas} -> {gas+1}).")
        
        if not recs:
            recs.append("Current settings look balanced. Fine-tune gradually.")
        
        for r in recs:
            print(" - " + r)
        print("")

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3

def optimize_memory():
    """Apply memory optimization techniques"""
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Set memory fraction for CUDA if needed
    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(GPU_MEM_FRACTION)
        except Exception:
            pass

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

@torch.no_grad()
def eval_epoch(config, device, data_loader, model, args=None, epoch=None, num_epochs=None):
    """Enhanced evaluation function with memory optimization"""
    model.eval()
    
    metric_dict = defaultdict(list)
    # Compute validation loss when early stop mode is provided ('and' or 'or')
    mode = getattr(args, 'early_stop_val', None) if args is not None else None
    compute_val_loss = (mode in ('and', 'or'))
    if compute_val_loss:
        val_loss_list = []
    # Enhanced progress bar for evaluation
    desc = f"Validation"
    if epoch is not None and num_epochs is not None:
        desc = f"Validation (Epoch {epoch+1}/{num_epochs})"
    
    eval_pbar = tqdm(data_loader, desc=desc, leave=False)
    scaler_enabled = torch.cuda.is_available()
    for sample in eval_pbar:
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
        num_non_text_entities, relation_embs, topic_entity_one_hot,\
        target_triple_probs, a_entity_id_list = prepare_sample(device, sample)

        with autocast(enabled=scaler_enabled):
            pred_triple_logits = model.forward_legacy(
                h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                num_non_text_entities, relation_embs, topic_entity_one_hot)
        
        ####################################################################################//
        # Calculate validation loss (similar to training loss computation) only when enabled
        if compute_val_loss:
            target_triple_probs_device = target_triple_probs.to(device).unsqueeze(-1)
            if args.spcount:
                sp_weights = target_triple_probs_device.clone()
                sp_weights = sp_weights + 1.0
                positive_mask = (target_triple_probs_device > 0).float()
                bce_loss = F.binary_cross_entropy_with_logits(
                    pred_triple_logits, positive_mask, reduction='none')
                weighted_loss = bce_loss * sp_weights.squeeze(-1)
                val_loss = weighted_loss.mean()
            elif args.spcount_inv:
                sp_weights = target_triple_probs_device.clone()
                positive_mask = (target_triple_probs_device > 0).float()
                inv_weights = torch.where(
                    sp_weights > 0, 
                    1.0 / (sp_weights + 1.0),
                    torch.ones_like(sp_weights)
                )
                bce_loss = F.binary_cross_entropy_with_logits(
                    pred_triple_logits, positive_mask, reduction='none')
                weighted_loss = bce_loss * inv_weights.squeeze(-1)
                val_loss = weighted_loss.mean()
            else:
                positive_mask = (target_triple_probs_device > 0).float()
                val_loss = F.binary_cross_entropy_with_logits(
                    pred_triple_logits, positive_mask)
            
            val_loss_list.append(val_loss.item())
        ####################################################################################\\
        
        # Triple ranking
        # Keep ranking on GPU, move to CPU only if needed
        pred_triple_logits_flat = pred_triple_logits.reshape(-1)
        sorted_triple_ids_pred = torch.argsort(
            pred_triple_logits_flat, descending=True)
        triple_ranks_pred = torch.empty_like(sorted_triple_ids_pred)
        triple_ranks_pred[sorted_triple_ids_pred] = torch.arange(
            len(triple_ranks_pred), device=triple_ranks_pred.device)
        
        target_triple_ids = target_triple_probs.to(device).nonzero().squeeze(-1)
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
            entity_mask_k = torch.zeros(num_total_entities, device=device)
            entity_mask_k[h_id_tensor[triple_mask_k]] = 1.
            entity_mask_k[t_id_tensor[triple_mask_k]] = 1.
            recall_k_sample_ans = entity_mask_k[torch.as_tensor(a_entity_id_list, device=device)].sum().item()
            metric_dict[f'ans_recall@{k}'].append(
                recall_k_sample_ans / len(a_entity_id_list))

    for key, val in metric_dict.items():
        metric_dict[key] = np.mean(val)
    
    if compute_val_loss:
        metric_dict['val_loss'] = np.mean(val_loss_list) if len(val_loss_list) > 0 else float('inf')
    
    return metric_dict

def train_epoch(device, train_loader, model, optimizer, args, epoch=None, num_epochs=None):
    """Enhanced training function with memory optimization"""
    model.train()
    epoch_loss = 0
    
    desc = f"Training"
    if epoch is not None and num_epochs is not None:
        desc = f"Training (Epoch {epoch+1}/{num_epochs})"
    
    train_pbar = tqdm(train_loader, desc=desc, leave=False)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    grad_accum_steps = max(1, int(getattr(args, 'grad_accum_steps', 8)))
    optimizer.zero_grad(set_to_none=True)
    for batch_idx, sample in enumerate(train_pbar):
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
        num_non_text_entities, relation_embs, topic_entity_one_hot,\
        target_triple_probs, a_entity_id_list = prepare_sample(device, sample)
            
        if len(h_id_tensor) == 0:
            continue

        with autocast(enabled=torch.cuda.is_available()):
            pred_triple_logits = model.forward_legacy(
                h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                num_non_text_entities, relation_embs, topic_entity_one_hot)
        
        target_triple_probs = target_triple_probs.to(device).unsqueeze(-1)
        ###################################################################//
        # 根據不同的方法設定權重
        if args.spcount:
            # spcount: 使用 shortest path 計數作為權重
            sp_weights = target_triple_probs.clone()
            sp_weights = sp_weights + 1.0
            
            positive_mask = (target_triple_probs > 0).float()
            
            bce_loss = F.binary_cross_entropy_with_logits(
                pred_triple_logits, positive_mask, reduction='none')
            weighted_loss = bce_loss * sp_weights.squeeze(-1)
            loss = weighted_loss.mean()
            
        elif args.spcount_inv:
            # spcount_inv: 使用 shortest path 計數的倒數作為權重
            sp_weights = target_triple_probs.clone()
            positive_mask = (target_triple_probs > 0).float()
            
            inv_weights = torch.where(
                sp_weights > 0, 
                1.0 / (sp_weights + 1.0),
                torch.ones_like(sp_weights)
            )
            
            bce_loss = F.binary_cross_entropy_with_logits(
                pred_triple_logits, positive_mask, reduction='none')
            weighted_loss = bce_loss * inv_weights.squeeze(-1)
            loss = weighted_loss.mean()
            
        else:
            # 標準的 BCE loss
            positive_mask = (target_triple_probs > 0).float()
            loss = F.binary_cross_entropy_with_logits(
                pred_triple_logits, positive_mask)
        ###################################################################\\
        # Gradient accumulation
        loss = loss / grad_accum_steps
        scaler.scale(loss).backward()
        if ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(train_loader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        loss = loss.item()
        epoch_loss += loss

        # Update progress bar with current loss and memory usage
        memory_usage = get_memory_usage()
        train_pbar.set_postfix({
            'loss': f'{loss:.4f}',
            'mem_gb': f'{memory_usage:.2f}',
            'accum': grad_accum_steps
        })
        
        # Periodic memory cleanup
        if batch_idx % 100 == 0:
            optimize_memory()
    
    train_pbar.close()
    epoch_loss /= len(train_loader)
    
    log_dict = {'loss': epoch_loss}
    return log_dict

###################################################################//
def check_and_warn_resources(args):
    """Display system resource information"""
    try:
        # Check CPU memory
        cpu_memory = psutil.virtual_memory()
        print(f"💾 System Memory: {cpu_memory.total / (1024**3):.1f} GB total, {cpu_memory.available / (1024**3):.1f} GB available")
        
        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(0)
                total_mem_gb = props.total_memory / (1024**3)
                print(f"🖥️ CUDA device: {props.name}, compute capability: {props.major}.{props.minor}, VRAM: {total_mem_gb:.1f} GB")
            except Exception:
                print("🖥️ CUDA is available (device 0 properties unavailable)")
        else:
            print("⚠️ CUDA not available; training will run on CPU and may be slow.")
    except Exception as e:
        print(f"(check_and_warn_resources) info unavailable: {e}")
###################################################################\\

def main(args):
    
    ###################################################################//
    # Print training start information
    ts_start = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 8 * 3600))
    start_time = time.time()
    print(f"========== Start training retriever (Ultra Memory Optimized) {ts_start} ==========")
    
    # Print configuration
    print(f"🔧 Configuration:")
    print(f"   Dataset: {args.dataset}")
    print(f"   Use dropout: {args.use_dropout}")
    if args.use_dropout:
        print(f"   Dropout rate: {args.dropout_rate}")
    print(f"   Frequency weight: {args.freq_weight}")
    print(f"   Inverse frequency weight: {args.freq_weight_inv}")
    print(f"   SP count: {args.spcount}")
    print(f"   SP count inverse: {args.spcount_inv}")
    print(f"   Memory optimization: ULTRA ENABLED")
    print(f"   GPU mem fraction: {args.gpu_mem_fraction}")
    print(f"===============================")
    
    # Check system resources
    check_and_warn_resources(args)
    
    # Initialize system usage monitor
    usage = SystemUsageMonitor(interval_sec=1.0)
    usage.start()
    
    # Set desired GPU memory fraction then apply initial memory optimization
    global GPU_MEM_FRACTION
    try:
        GPU_MEM_FRACTION = float(args.gpu_mem_fraction)
    except Exception:
        GPU_MEM_FRACTION = 0.95
    optimize_memory()
        
    # Modify the config file for advanced settings and extensions.
    config_file = f'configs/retriever/{args.dataset}.yaml'
    config = load_yaml(config_file)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(config['env']['num_threads'])
    set_seed(config['env']['seed'])
    ts = time.strftime('%b%d-%H:%M:%S', time.localtime(time.time() + 8 * 3600))
    config_df = pd.json_normalize(config, sep='/')
    exp_prefix = config['train']['save_prefix']
    exp_name_base = f'{exp_prefix}_{ts}'
    
    # Add feature tags to experiment name
    feature_tags = []
    if args.freq_weight:
        feature_tags.append('freq_weight')
    if args.freq_weight_inv:
        feature_tags.append('freq_weight_inv')
    if args.spcount:
        feature_tags.append('spcount')
    if args.spcount_inv:
        feature_tags.append('spcount_inv')
    if args.use_dropout:
        feature_tags.append(f'drop{str(args.dropout_rate).replace(".", "_")}')
    # Append early stop mode tag when enabled
    if args.early_stop_val in ('and', 'or'):
        feature_tags.append(f'esv_{args.early_stop_val}')
    # Add memory optimization tag
    feature_tags.append('ultra_mem_opt')
        
    exp_name = exp_name_base if not feature_tags else f"{exp_name_base}_{'_'.join(feature_tags)}"
    
    wandb.init(
        project=f'{args.dataset}_retriever',
        name=exp_name,
        config=config_df.to_dict(orient='records')[0]
    )

    print(f"🚀 Experiment: {exp_name}")
    print(f"📊 W&B project: {args.dataset}_retriever")

    # Load datasets with ultra memory optimization
    print("📂 Loading datasets (Ultra Memory Optimized)...")
    
    # Determine weight mode based on arguments
    weight_mode = 'none'
    if args.freq_weight:
        weight_mode = 'freq'
    elif args.freq_weight_inv:
        weight_mode = 'inv'
    elif args.spcount:
        weight_mode = 'spcount'
    elif args.spcount_inv:
        weight_mode = 'spcount_inv'
    
    # Use OptimizedRetrieverDataset with lazy loading
    train_set = OptimizedRetrieverDataset(
        config=config, 
        split='train', 
        skip_no_path=True, 
        freq_weight=args.freq_weight,
        weight_mode=weight_mode
    )
    val_set = OptimizedRetrieverDataset(
        config=config, 
        split='val', 
        skip_no_path=True, 
        freq_weight=args.freq_weight,
        weight_mode=weight_mode
    )
    
    print(f"   Training samples: {len(train_set)}")
    print(f"   Validation samples: {len(val_set)}")

    # Configure DataLoader with ultra memory optimization
    # Auto-tuned defaults when values are negative or unset (based on train.py)
    cpu_cores = os.cpu_count() or 4
    if args.num_workers < 0:
        # Use train.py proven settings: default 2 workers, conservative for CWQ
        if args.dataset == 'cwq':
            num_workers = min(4, cpu_cores)  # Conservative for CWQ
        else:
            num_workers = min(4, cpu_cores)   # More aggressive for webqsp
    else:
        num_workers = min(args.num_workers, cpu_cores)

    cuda_available = torch.cuda.is_available()
    pin_memory = True if cuda_available else False
    if args.pin_memory:
        pin_memory = True

    persistent_workers = (num_workers > 0)
    if args.persistent_workers:
        persistent_workers = True

    if args.prefetch_factor < 0:
        # Use train.py proven settings: default 4, conservative for CWQ
        if args.dataset == 'cwq':
            prefetch_factor = 2 if num_workers > 0 else None  # Conservative for CWQ
        else:
            prefetch_factor = 4 if num_workers > 0 else None   # Standard for webqsp
    else:
        prefetch_factor = args.prefetch_factor if num_workers > 0 else None

    if args.train_bs < 0:
        # Use train.py proven settings: batch size 64 for both datasets
        train_bs_resolved = 64  # train.py uses 64 for both datasets
    else:
        train_bs_resolved = args.train_bs

    if args.eval_bs < 0:
        # Use train.py proven settings: batch size 64 for both datasets
        eval_bs_resolved = 64  # train.py uses 64 for both datasets
    else:
        eval_bs_resolved = args.eval_bs

    train_loader = DataLoader(
        train_set, 
        batch_size=train_bs_resolved, 
        shuffle=True, 
        collate_fn=collate_retriever,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor  # tune prefetch for memory and throughput
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=eval_bs_resolved, 
        collate_fn=collate_retriever,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )
    
    emb_size = train_set[0]['q_emb'].shape[-1]
    print(f"🧠 Creating model with embedding size: {emb_size}")
    model = Retriever(emb_size, **config['retriever']).to(device)
    
    # Enable gradient checkpointing for memory optimization
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled for memory optimization")
    
    # Model compilation for faster execution (PyTorch 2.0+)
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("✅ Model compilation enabled for faster execution")
    except Exception as e:
        print(f"⚠️ Model compilation failed: {e}")
        print("   Continuing without compilation...")
    
    optimizer = Adam(model.parameters(), **config['optimizer'])

    # Create results directory
    save_dir = os.path.join('/home/YX_thesis/retrieve/results/training', args.dataset, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"💾 Results will be saved to: {save_dir}")
    
    # Training setup
    num_epochs = config['train']['num_epochs']
    patience = config['train']['patience']
    
    print(f"🏋️ Training configuration:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Patience: {patience}")
    print(f"   K values for evaluation: {config['eval']['k_list']}")
    print(f"   Train batch size: {train_bs_resolved}")
    print(f"   Eval batch size: {eval_bs_resolved}")
    print(f"   DataLoader workers: {num_workers}")
    print(f"   Prefetch factor: {prefetch_factor if prefetch_factor is not None else 'None'}")
    print(f"   Pin memory: {pin_memory}")
    print(f"   Persistent workers: {persistent_workers}")
    print(f"   Grad accumulation steps: {getattr(args, 'grad_accum_steps', 1)}")
    print(f"   Memory optimization: ULTRA ENABLED")
    print("===============================")
    ###################################################################\\
        
    # Training loop with main progress bar
    num_patient_epochs = 0
    num_patient_epochs_loss = 0  # 追蹤validation loss的patience
    best_val_metric = 0
    best_val_loss = float('inf')  # 追蹤最佳validation loss
    # Determine early stop mode: 'and'/'or'. Any other value disables loss-based early stopping
    early_stop_mode = args.early_stop_val
    enable_loss_early_stop = (early_stop_mode in ('and', 'or'))
    
    main_pbar = tqdm(range(num_epochs), desc="Overall Training Progress")
    
    for epoch in main_pbar:
        # Force garbage collection to free memory
        optimize_memory()
        
        # Update main progress bar description
        main_pbar.set_description(f"Training Progress (Epoch {epoch+1}/{num_epochs})")
        
        # Training phase
        train_log_dict = train_epoch(device, train_loader, model, optimizer, args, epoch, num_epochs)
        
        # Validation phase
        val_eval_dict = eval_epoch(config, device, val_loader, model, args, epoch, num_epochs)
        target_val_metric = val_eval_dict.get('triple_recall@100', 0.0)
        current_val_loss = val_eval_dict.get('val_loss', float('inf')) if enable_loss_early_stop else None
        
        # Logging to wandb
        log_payload = {
            'epoch': epoch,
            'train_loss': train_log_dict.get('loss', 0.0),
            'num_patient_epochs': num_patient_epochs,
            'num_patient_epochs_loss': num_patient_epochs_loss if enable_loss_early_stop else None,
            'memory_usage_gb': get_memory_usage()
        }
        
        # Add validation metrics
        for key, val in val_eval_dict.items():
            log_payload[f'val/{key}'] = val
        
        # Add system usage stats
        stats = usage.get_stats()
        log_payload.update({
            'system/cpu_avg': stats.get('cpu_avg', 0),
            'system/gpu_util_avg': stats.get('gpu_util_avg', 0),
            'system/gpu_mem_avg': stats.get('gpu_mem_avg', 0)
        })
        
        wandb.log(log_payload)
        
        # Provide system recommendations every 5 epochs
        if epoch % 5 == 0:
            usage.recommend(args, context_label=f"epoch_{epoch+1}")
        
        metric_improved = False
        # Model checkpointing and early stopping
        if target_val_metric > best_val_metric:
            print(f"\n📈 New best model found! Recall@100: {best_val_metric:.4f} -> {target_val_metric:.4f}")
            num_patient_epochs = 0
            best_val_metric = target_val_metric
            metric_improved = True
        else:
            metric_improved = False
            num_patient_epochs += 1
            
        # Track improvement in validation loss
        loss_improved = False
        if enable_loss_early_stop:
            # Consider any decrease as improvement when in loss-enabled modes
            if current_val_loss is not None and current_val_loss < best_val_loss:
                print(f"\n📉 Validation loss improved! Loss: {best_val_loss:.4f} -> {current_val_loss:.4f}")
                num_patient_epochs_loss = 0
                best_val_loss = current_val_loss
                loss_improved = True
            else:
                num_patient_epochs_loss += 1
            
        # Save model if either metric or loss improved
        if metric_improved or loss_improved:
            # Save best model
            best_state_dict = {
                'config': config,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_metric': best_val_metric,
                'exp_name': exp_name,
            }
            if enable_loss_early_stop:
                best_state_dict['best_val_loss'] = best_val_loss
            torch.save(best_state_dict, os.path.join(save_dir, 'cpt.pth'))
            
            # Save detailed information
            info_path = os.path.join(save_dir, 'training_info.txt')
            with open(info_path, 'w') as f:
                f.write(f"Dataset: {args.dataset}\n")
                f.write(f"Experiment: {exp_name}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 8 * 3600))}\n")
                f.write(f"Best Model achieved at Epoch: {epoch+1}\n")
                f.write(f"Best Validation Recall@100: {best_val_metric:.4f}\n")
                if enable_loss_early_stop:
                    f.write(f"Best Validation Loss: {best_val_loss:.4f}\n")
                f.write("\n")
                
                f.write("Configuration:\n")
                f.write(f"  Use dropout: {args.use_dropout}\n")
                if args.use_dropout:
                    f.write(f"  Dropout rate: {args.dropout_rate}\n")
                f.write(f"  Frequency weight: {args.freq_weight}\n")
                f.write(f"  Inverse frequency weight: {args.freq_weight_inv}\n")
                f.write(f"  SP count: {args.spcount}\n")
                f.write(f"  SP count inverse: {args.spcount_inv}\n")
                f.write(f"  Early stop mode: {early_stop_mode}\n")
                f.write(f"  Memory optimization: ULTRA ENABLED\n")
        
        # Update progress bar display
        memory_usage = get_memory_usage()
        if enable_loss_early_stop:
            main_pbar.set_postfix({
                'train_loss': f"{train_log_dict.get('loss', 0.0):.4f}",
                'val_loss': f'{current_val_loss:.4f}' if current_val_loss is not None else 'nan',
                'val_recall@100': f'{target_val_metric:.4f}',
                'best_recall@100': f'{best_val_metric:.4f}',
                'patience': f'{num_patient_epochs}/{patience}',
                'loss_patience': f'{num_patient_epochs_loss}/{patience}',
                'early_stop': early_stop_mode,
                'mem_gb': f'{memory_usage:.2f}'
            })
        else:
            main_pbar.set_postfix({
                'train_loss': f"{train_log_dict.get('loss', 0.0):.4f}",
                'val_recall@100': f'{target_val_metric:.4f}',
                'best_recall@100': f'{best_val_metric:.4f}',
                'patience': f'{num_patient_epochs}/{patience}',
                'mem_gb': f'{memory_usage:.2f}'
            })
        
        # Early stopping
        if enable_loss_early_stop:
            if early_stop_mode == 'and':
                should_stop = (num_patient_epochs >= patience and num_patient_epochs_loss >= patience)
            else:  # 'or'
                should_stop = (num_patient_epochs >= patience or num_patient_epochs_loss >= patience)
            if should_stop:
                print(f"\n⌛ Early stopping triggered at epoch {epoch+1}")
                print(f"   Validation metric no-improve epochs: {num_patient_epochs}/{patience}")
                print(f"   Validation loss no-improve epochs: {num_patient_epochs_loss}/{patience}")
                print(f"   Mode: {early_stop_mode.upper()} condition")
                break
        else:
            if num_patient_epochs >= patience:
                print(f"\n⌛ Early stopping triggered at epoch {epoch+1} after {patience} epochs with no improvement.")
                break
    
    main_pbar.close()
    
    # Stop system monitoring
    usage.stop()
    
    # Training completion
    ts_end = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 8 * 3600))
    end_time = time.time()
    total_seconds = int(end_time - start_time)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    print(f"\n✅ Training completed! Best validation recall@100: {best_val_metric:.4f}")
    print(f"⏱️ Training time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"🎯 Results saved to: {save_dir}")
    
    # Final system usage report
    print("\n" + "="*50)
    print("📊 Final System Usage Report")
    print("="*50)
    stats = usage.get_stats()
    if stats['cpu_avg'] is not None:
        print(f"CPU utilization  avg: {stats['cpu_avg']:.1f}%")
    if torch.cuda.is_available() and stats['gpu_util_avg'] is not None:
        print(f"GPU utilization  avg: {stats['gpu_util_avg']:.1f}%")
    if stats['gpu_mem_avg'] is not None and stats['gpu_mem_total']:
        print(f"GPU memory usage avg: {stats['gpu_mem_avg']:.0f}MB (of ~{stats['gpu_mem_total']:.0f}MB)")
    print("="*50)

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True, choices=['webqsp', 'cwq'], help='Dataset name')
    parser.add_argument('-dp', '--use_dropout', action='store_true', help='Enable dropout layer in the model.')
    parser.add_argument('-dpr', '--dropout_rate', type=float, default=0.2, help='Dropout rate if dropout is enabled.')
    parser.add_argument('-fw', '--freq_weight', action='store_true', help='Enable frequency-based weighting.')
    parser.add_argument('-fwi', '--freq_weight_inv', action='store_true', help='Enable inverse frequency-based weighting.')
    parser.add_argument('-sc', '--spcount', action='store_true', help='Enable SP count-based weighting.')
    parser.add_argument('-sci', '--spcount_inv', action='store_true', help='Enable inverse SP count-based weighting.')
    parser.add_argument('-esv', '--early_stop_val', type=str, default=None, choices=['none', 'and', 'or'], help='Early stop + validation metric.')
    
    # Throughput/memory tuning - Based on train.py proven settings
    parser.add_argument('-gas', '--grad_accum_steps', type=int, default=2, help='Number of steps to accumulate gradients before optimizer step')
    parser.add_argument('--train_bs', type=int, default=128, help='Training batch size (per step), -1 to auto')
    parser.add_argument('--eval_bs', type=int, default=128, help='Evaluation batch size, -1 to auto')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers, -1 to auto')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='DataLoader prefetch factor when workers>0, -1 to auto')
    parser.add_argument('--pin_memory', action='store_true', default=True, help='Enable DataLoader pin_memory when CUDA available')
    parser.add_argument('--persistent_workers', action='store_true', default=True, help='Keep workers alive across epochs')
    parser.add_argument('--gpu_mem_fraction', type=float, default=0.9, help='Target per-process GPU memory fraction [0-1]')
    args = parser.parse_args()
    
    main(args)
