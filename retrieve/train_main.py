###################################################################//
# cpt1
# + freq_weight/_inv
###################################################################\\
import numpy as np
import os
import pandas as pd
import time
import torch
import torch.nn.functional as F
import wandb

from collections import defaultdict
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.retriever import load_yaml
from src.dataset.retriever import RetrieverDataset, collate_retriever
from src.model.retriever import Retriever
from src.model.kge_weight_scorer import create_kge_weight_scorer
from src.setup import set_seed, prepare_sample

@torch.no_grad()
def eval_epoch(config, device, data_loader, model, args=None, epoch=None, num_epochs=None):
    """Enhanced evaluation function with better progress tracking"""
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
    for sample in eval_pbar:
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
        num_non_text_entities, relation_embs, topic_entity_one_hot,\
        target_triple_probs, a_entity_id_list = prepare_sample(device, sample)

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
            elif args.kge_weight:
                positive_mask = (target_triple_probs_device > 0).float()
                kge_scorer = getattr(args, 'kge_scorer', None)
                if kge_scorer is not None:
                    kge_weights = kge_scorer.compute_triple_weights(
                        h_id_tensor, r_id_tensor, t_id_tensor)
                    kge_weights = kge_weights.unsqueeze(-1)
                else:
                    kge_weights = torch.ones_like(target_triple_probs_device)
                bce_loss = F.binary_cross_entropy_with_logits(
                    pred_triple_logits, positive_mask, reduction='none')
                weighted_loss = bce_loss * kge_weights.squeeze(-1)
                val_loss = weighted_loss.mean()
            else:
                positive_mask = (target_triple_probs_device > 0).float()
                val_loss = F.binary_cross_entropy_with_logits(
                    pred_triple_logits, positive_mask)
            
            val_loss_list.append(val_loss.item())
        ####################################################################################\
        
        # Triple ranking
        pred_triple_logits_flat = pred_triple_logits.reshape(-1)
        sorted_triple_ids_pred = torch.argsort(
            pred_triple_logits_flat, descending=True).cpu()
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
    
    if compute_val_loss:
        metric_dict['val_loss'] = np.mean(val_loss_list) if len(val_loss_list) > 0 else float('inf')
    
    return metric_dict

def train_epoch(device, train_loader, model, optimizer, args, epoch=None, num_epochs=None):
    """Enhanced training function with better progress tracking"""
    model.train()
    epoch_loss = 0
    
    desc = f"Training"
    if epoch is not None and num_epochs is not None:
        desc = f"Training (Epoch {epoch+1}/{num_epochs})"
    
    train_pbar = tqdm(train_loader, desc=desc, leave=False)
    for batch_idx, sample in enumerate(train_pbar):
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
        num_non_text_entities, relation_embs, topic_entity_one_hot,\
        target_triple_probs, a_entity_id_list = prepare_sample(device, sample)
            
        if len(h_id_tensor) == 0:
            continue

        pred_triple_logits = model.forward_legacy(
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
            num_non_text_entities, relation_embs, topic_entity_one_hot)
        
        target_triple_probs = target_triple_probs.to(device).unsqueeze(-1)
        ###################################################################//
        # 根據不同的方法設定權重
        if args.spcount:
            # spcount: 使用 shortest path 計數作為權重
            # target_triple_probs 已經包含了每個 triple 在 shortest path 中出現的次數
            # 我們將這個計數+1作為權重（避免0權重）
            sp_weights = target_triple_probs.clone()
            sp_weights = sp_weights + 1.0  # 基礎權重為1，然後加上 shortest path 計數
            
            # 對於正樣本和負樣本分別處理
            positive_mask = (target_triple_probs > 0).float()
            
            # 計算加權的 BCE loss
            bce_loss = F.binary_cross_entropy_with_logits(
                pred_triple_logits, positive_mask, reduction='none')
            weighted_loss = bce_loss * sp_weights.squeeze(-1)
            loss = weighted_loss.mean()
            
        elif args.spcount_inv:
            # spcount_inv: 使用 shortest path 計數的倒數作為權重
            sp_weights = target_triple_probs.clone()
            positive_mask = (target_triple_probs > 0).float()
            
            # 計算倒數權重，避免除零
            inv_weights = torch.where(
                sp_weights > 0, 
                1.0 / (sp_weights + 1.0),  # 倒數權重
                torch.ones_like(sp_weights)  # 負樣本保持權重1
            )
            
            # 計算加權的 BCE loss
            bce_loss = F.binary_cross_entropy_with_logits(
                pred_triple_logits, positive_mask, reduction='none')
            weighted_loss = bce_loss * inv_weights.squeeze(-1)
            loss = weighted_loss.mean()
            
        elif args.kge_weight:
            # kge_weight: 使用 KGE 模型分數作為權重
            positive_mask = (target_triple_probs > 0).float()
            
            # 從 args 獲取 KGE 權重計算器
            kge_scorer = getattr(args, 'kge_scorer', None)
            if kge_scorer is not None:
                # 計算 KGE 權重
                kge_weights = kge_scorer.compute_triple_weights(
                    h_id_tensor, r_id_tensor, t_id_tensor)
                kge_weights = kge_weights.unsqueeze(-1)  # [E, 1]
            else:
                # 如果沒有 KGE scorer，使用標準權重
                kge_weights = torch.ones_like(target_triple_probs)
            
            # 計算加權的 BCE loss
            bce_loss = F.binary_cross_entropy_with_logits(
                pred_triple_logits, positive_mask, reduction='none')
            weighted_loss = bce_loss * kge_weights.squeeze(-1)
            loss = weighted_loss.mean()
            
        else:
            # 標準的 BCE loss
            positive_mask = (target_triple_probs > 0).float()
            loss = F.binary_cross_entropy_with_logits(
                pred_triple_logits, positive_mask)
        ###################################################################\\
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss = loss.item()
        epoch_loss += loss

        # Update progress bar with current loss
        train_pbar.set_postfix({'loss': f'{loss:.4f}'})
    
    train_pbar.close()
    epoch_loss /= len(train_loader)
    
    log_dict = {'loss': epoch_loss}
    return log_dict

def _get_kge_model_info(model_type, dataset_name):
    """
    根據模型類型和數據集名稱返回對應的路徑和類型
    
    Args:
        model_type: 模型類型 ('transe', 'distmult', 'complex', 'auto')
        dataset_name: 數據集名稱 ('webqsp', 'cwq')
    
    Returns:
        tuple: (model_path, model_type)
    """
    kge_models_dir = '/home/YX_thesis/pykeen/kge_models'
    
    if model_type == 'auto':
        # 自動選擇：根據數據集名稱和可用模型自動選擇
        if dataset_name == 'webqsp':
            preferred_models = [
                ('webqsp_transe.pt', 'TransE'),
                ('webqsp_DistMult.pt', 'DistMult'),
                ('webqsp_complex.pt', 'ComplEx'),
                ('webqsp_rotate.pt', 'Rotate'),
            ]
        else:  # cwq 或其他數據集
            # 嘗試使用 webqsp 訓練的模型作為通用模型
            preferred_models = [
                ('webqsp_transe.pt', 'TransE'),
                ('webqsp_DistMult.pt', 'DistMult'),
                ('webqsp_complex.pt', 'ComplEx'),
                ('webqsp_rotate.pt', 'Rotate'),
            ]
        
        for model_file, model_class in preferred_models:
            model_path = os.path.join(kge_models_dir, model_file)
            if os.path.exists(model_path):
                return model_path, model_class
        
        # 如果沒有找到預設模型，使用第一個可用的
        if os.path.exists(kge_models_dir):
            available_files = [f for f in os.listdir(kge_models_dir) if f.endswith('.pt')]
            if available_files:
                model_path = os.path.join(kge_models_dir, available_files[0])
                # 根據文件名推斷模型類型
                model_class = 'TransE'  # 預設
                if 'distmult' in available_files[0].lower():
                    model_class = 'DistMult'
                elif 'complex' in available_files[0].lower():
                    model_class = 'ComplEx'
                return model_path, model_class
    
    elif model_type == 'transe':
        # 優先使用數據集特定的模型，然後回退到通用模型
        model_files = [f'{dataset_name}_transe.pt', 'webqsp_transe.pt']
        for model_file in model_files:
            model_path = os.path.join(kge_models_dir, model_file)
            if os.path.exists(model_path):
                return model_path, 'TransE'
    
    elif model_type == 'distmult':
        # 優先使用數據集特定的模型，然後回退到通用模型
        model_files = [f'{dataset_name}_DistMult.pt', 'webqsp_DistMult.pt']
        for model_file in model_files:
            model_path = os.path.join(kge_models_dir, model_file)
            if os.path.exists(model_path):
                return model_path, 'DistMult'
    
    elif model_type == 'complex':
        # 優先使用數據集特定的模型，然後回退到通用模型
        model_files = [f'{dataset_name}_complex.pt', 'webqsp_complex.pt']
        for model_file in model_files:
            model_path = os.path.join(kge_models_dir, model_file)
            if os.path.exists(model_path):
                return model_path, 'ComplEx'
    
    elif model_type == 'rotate':
        # 優先使用數據集特定的模型，然後回退到通用模型
        model_files = [f'{dataset_name}_rotate.pt', 'webqsp_rotate.pt']
        for model_file in model_files:
            model_path = os.path.join(kge_models_dir, model_file)
            if os.path.exists(model_path):
                return model_path, 'Rotate'
    
    # 預設回退
    return os.path.join(kge_models_dir, 'webqsp_transe.pt'), 'TransE'

###################################################################//
def check_and_warn_resources(args):
    """Display system resource information"""
    try:
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
    print(f"========== Start training retriever {ts_start} ==========")
    
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
    print(f"   KGE weight: {args.kge_weight}")
    print(f"   KGE freq weight: {args.kge_shortest_path}")
    if args.kge_weight or args.kge_shortest_path:
        print(f"   KGE model: {args.kge_model}")
        print(f"   KGE weight mode: {args.kge_weight_mode}")
    print(f"===============================")
    
    # Check system resources
    check_and_warn_resources(args)
    
    # Initialize KGE weight scorer if needed
    kge_scorer = None
    if args.kge_weight or args.kge_shortest_path:
        print(f"🔧 Initializing KGE weight scorer...")
        
        # 根據選擇的模型確定路徑和類型
        kge_model_path, kge_model_type = _get_kge_model_info(args.kge_model, args.dataset)
        
        if os.path.exists(kge_model_path):
            print(f"   Model path: {kge_model_path}")
            print(f"   Model type: {kge_model_type}")
            print(f"   Weight mode: {args.kge_weight_mode}")
            
            kge_scorer = create_kge_weight_scorer(
                kge_model_path=kge_model_path,
                kge_model_type=kge_model_type,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                weight_mode=args.kge_weight_mode
            )
            if kge_scorer:
                print(f"   ✅ KGE weight scorer initialized successfully")
            else:
                print(f"   ❌ Failed to initialize KGE weight scorer")
        else:
            print(f"   ❌ KGE model not found at {kge_model_path}")
            print(f"   Available models in kge_models/:")
            kge_models_dir = '/home/YX_thesis/pykeen/kge_models'
            if os.path.exists(kge_models_dir):
                for f in os.listdir(kge_models_dir):
                    if f.endswith('.pt'):
                        print(f"     - {f}")
        
        # 將 KGE scorer 添加到 args 中
        args.kge_scorer = kge_scorer
        
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
    if args.kge_weight:
        feature_tags.append(f'kge_{args.kge_model}_{args.kge_weight_mode}')
    if args.kge_shortest_path:
        feature_tags.append(f'kge_shortest_path_{args.kge_model}_{args.kge_weight_mode}')
    if args.use_dropout:
        feature_tags.append(f'drop{str(args.dropout_rate).replace(".", "_")}')
    # Append early stop mode tag when enabled
    if args.early_stop_val in ('and', 'or'):
        feature_tags.append(f'esv_{args.early_stop_val}')
        
    exp_name = exp_name_base if not feature_tags else f"{exp_name_base}_{'_'.join(feature_tags)}"
    
    wandb.init(
        project=f'{args.dataset}_retriever',
        name=exp_name,
        config=config_df.to_dict(orient='records')[0]
    )
    # os.makedirs(exp_name, exist_ok=True)

    print(f"🚀 Experiment: {exp_name}")
    print(f"📊 W&B project: {args.dataset}_retriever")

    # Load datasets
    print("📂 Loading datasets...")
    train_set = RetrieverDataset(
        config=config, split='train', 
        freq_weight=args.freq_weight, freq_weight_inv=args.freq_weight_inv,
        kge_shortest_path=args.kge_shortest_path, kge_scorer=kge_scorer
    )
    val_set = RetrieverDataset(
        config=config, split='val', 
        freq_weight=args.freq_weight, freq_weight_inv=args.freq_weight_inv,
        kge_shortest_path=args.kge_shortest_path, kge_scorer=kge_scorer
    )
    print(f"   Training samples: {len(train_set)}")
    print(f"   Validation samples: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=collate_retriever)
    val_loader = DataLoader(val_set, batch_size=1, collate_fn=collate_retriever)
    
    emb_size = train_set[0]['q_emb'].shape[-1]
    print(f"🧠 Creating model with embedding size: {emb_size}")
    model = Retriever(emb_size, **config['retriever']).to(device)
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
        # num_patient_epochs += 1
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
            'num_patient_epochs_loss': num_patient_epochs_loss if enable_loss_early_stop else None
        }
        
        # Add validation metrics
        for key, val in val_eval_dict.items():
            log_payload[f'val/{key}'] = val
        
        wandb.log(log_payload)
        
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
                f.write(f"  SPO count: {args.spcount}\n")
                f.write(f"  SPO count inverse: {args.spcount_inv}\n")
                f.write(f"  KGE weight: {args.kge_weight}\n")
                f.write(f"  KGE freq weight: {args.kge_shortest_path}\n")
                if args.kge_weight or args.kge_shortest_path:
                    f.write(f"  KGE model: {args.kge_model}\n")
                    f.write(f"  KGE weight mode: {args.kge_weight_mode}\n")
                    # 記錄實際使用的 KGE 模型路徑和類型
                    if kge_scorer is not None:
                        f.write(f"  KGE model path: {kge_scorer.kge_model_path}\n")
                        f.write(f"  KGE model type: {kge_scorer.kge_model_type}\n")
                else:
                    f.write(f"  KGE model: none\n")
                f.write(f"  Early stop mode: {early_stop_mode}\n")
        
        # Update progress bar display
        if enable_loss_early_stop:
            main_pbar.set_postfix({
                'train_loss': f"{train_log_dict.get('loss', 0.0):.4f}",
                'val_loss': f'{current_val_loss:.4f}' if current_val_loss is not None else 'nan',
                'val_recall@100': f'{target_val_metric:.4f}',
                'best_recall@100': f'{best_val_metric:.4f}',
                'patience': f'{num_patient_epochs}/{patience}',
                'loss_patience': f'{num_patient_epochs_loss}/{patience}',
                'early_stop': early_stop_mode
            })
        else:
            main_pbar.set_postfix({
                'train_loss': f"{train_log_dict.get('loss', 0.0):.4f}",
                'val_recall@100': f'{target_val_metric:.4f}',
                'best_recall@100': f'{best_val_metric:.4f}',
                'patience': f'{num_patient_epochs}/{patience}'
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
    
    parser.add_argument('-kge_bce', '--kge_bce_weight', action='store_true', help='Enable KGE model-based weighting.')
    parser.add_argument('-kgsp', '--kge_shortest_path', action='store_true', help='SP by KGE score(affects shortest path triplet sets).')
    parser.add_argument('-km', '--kge_model', type=str, default='transe', 
                       choices=['transe', 'distmult', 'complex', 'rotate', 'auto'], 
                       help='KGE model type to use for weighting.')
    parser.add_argument('-kwm', '--kge_weight_mode', type=str, default='prob_inv', 
                       choices=['score', 'score_inv', 'prob', 'prob_inv'], 
                       help='KGE weight computation mode for BCE loss.')
    
    parser.add_argument('-esv', '--early_stop_val', type=str, default=None, choices=['none', 'and', 'or'], help='Early stop + validation metric.')
    args = parser.parse_args()
    
    main(args)
