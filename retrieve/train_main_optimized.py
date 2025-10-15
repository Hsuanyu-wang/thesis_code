###################################################################//
# Memory-optimized training script for large datasets like cwq
# Uses OptimizedRetrieverDataset with lazy loading to reduce memory usage
###################################################################\\
import numpy as np
import os
import pandas as pd
import time
import torch
import torch.nn.functional as F
import wandb
import gc

from collections import defaultdict
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.retriever import load_yaml
from src.dataset.retriever import OptimizedRetrieverDataset, collate_retriever
from src.model.retriever import Retriever
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
            else:
                positive_mask = (target_triple_probs_device > 0).float()
                val_loss = F.binary_cross_entropy_with_logits(
                    pred_triple_logits, positive_mask)
            
            val_loss_list.append(val_loss.item())
        ####################################################################################\\
        
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
    print(f"========== Start training retriever (Memory Optimized) {ts_start} ==========")
    
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
    print(f"   Memory optimization: ENABLED")
    print(f"===============================")
    
    # Check system resources
    check_and_warn_resources(args)
        
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
    feature_tags.append('mem_opt')
        
    exp_name = exp_name_base if not feature_tags else f"{exp_name_base}_{'_'.join(feature_tags)}"
    
    wandb.init(
        project=f'{args.dataset}_retriever',
        name=exp_name,
        config=config_df.to_dict(orient='records')[0]
    )

    print(f"🚀 Experiment: {exp_name}")
    print(f"📊 W&B project: {args.dataset}_retriever")

    # Load datasets with memory optimization
    print("📂 Loading datasets (Memory Optimized)...")
    
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

    # Configure DataLoader with memory optimization
    num_workers = min(4, os.cpu_count())  # Limit workers to prevent memory issues
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_set, 
        batch_size=1, 
        shuffle=True, 
        collate_fn=collate_retriever,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=1, 
        collate_fn=collate_retriever,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
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
    print(f"   DataLoader workers: {num_workers}")
    print(f"   Memory optimization: ENABLED")
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
        gc.collect()
        
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
                f.write(f"  Early stop mode: {early_stop_mode}\n")
                f.write(f"  Memory optimization: ENABLED\n")
        
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
    parser.add_argument('-esv', '--early_stop_val', type=str, default=None, choices=['none', 'and', 'or'], help='Early stop + validation metric.')
    args = parser.parse_args()
    
    main(args)
