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
from src.dataset.retriever import RetrieverDataset, collate_retriever, OptimizedRetrieverDataset, ImprovedRandomBatchRetrieverDataset, collate_retriever_batch
from src.model.retriever import Retriever
from src.setup import set_seed, prepare_sample

@torch.no_grad()
def eval_epoch(config, device, data_loader, model, epoch=None, total_epochs=None):
    model.eval()
    
    metric_dict = defaultdict(list)
    total_val_loss = 0
    num_batches = 0

    def iter_single_samples(sample):
        # Normalize loader outputs to single-sample tuples
        if isinstance(sample, dict):
            batch_size = len(sample['h_id_tensors'])
            for i in range(batch_size):
                h_id_tensor = sample['h_id_tensors'][i].to(device)
                r_id_tensor = sample['r_id_tensors'][i].to(device)
                t_id_tensor = sample['t_id_tensors'][i].to(device)
                q_emb = sample['q_embs'][i].to(device)
                entity_embs = sample['entity_embs_list'][i].to(device)
                relation_embs = sample['relation_embs_list'][i].to(device)
                topic_entity_one_hot = sample['topic_entity_one_hots'][i].to(device)
                num_non_text_entities = sample['num_non_text_entities_list'][i]
                target_triple_probs = sample['target_triple_probs_list'][i]
                a_entity_id_list = sample['a_entity_id_lists'][i]
                yield (
                    h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                    num_non_text_entities, relation_embs, topic_entity_one_hot,
                    target_triple_probs, a_entity_id_list
                )
        else:
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs, \
            num_non_text_entities, relation_embs, topic_entity_one_hot, \
            target_triple_probs, a_entity_id_list = prepare_sample(device, sample)
            yield (
                h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                num_non_text_entities, relation_embs, topic_entity_one_hot,
                target_triple_probs, a_entity_id_list
            )
    
    # 創建驗證進度條
    eval_desc = f"Eval Epoch {epoch+1}/{total_epochs}" if epoch is not None and total_epochs is not None else "Validation"
    eval_pbar = tqdm(data_loader, desc=eval_desc, leave=False, 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for sample in eval_pbar:
        for (
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
            num_non_text_entities, relation_embs, topic_entity_one_hot,
            target_triple_probs, a_entity_id_list
        ) in iter_single_samples(sample):

            pred_triple_logits = model((
                h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                num_non_text_entities, relation_embs, topic_entity_one_hot)).reshape(-1)
            
            # 計算validation loss
            if len(h_id_tensor) > 0:
                target_triple_probs = target_triple_probs.to(device)
                
                # 使用加權二元交叉熵來處理極度不平衡
                num_positive = target_triple_probs.sum().item()
                num_total = len(target_triple_probs)
                pos_weight = torch.tensor([(num_total - num_positive) / num_positive if num_positive > 0 else 1.0], device=device)
                val_loss = F.binary_cross_entropy_with_logits(
                    pred_triple_logits, target_triple_probs, pos_weight=pos_weight)
                total_val_loss += val_loss.item()
                num_batches += 1
            
            # Triple ranking
            sorted_triple_ids_pred = torch.argsort(
                pred_triple_logits, descending=True).cpu()
            triple_ranks_pred = torch.empty_like(sorted_triple_ids_pred)
            triple_ranks_pred[sorted_triple_ids_pred] = torch.arange(
                len(triple_ranks_pred))
            # 將 triple_ranks_pred 移到與 target_triple_ids 相同的設備
            triple_ranks_pred = triple_ranks_pred.to(target_triple_probs.device)
            
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
        
        # 更新進度條描述
        if num_batches > 0:
            current_loss = total_val_loss / num_batches
            eval_pbar.set_postfix({'val_loss': f'{current_loss:.4f}'})

    eval_pbar.close()

    for key, val in metric_dict.items():
        metric_dict[key] = np.mean(val)
    
    # 計算平均validation loss
    if num_batches > 0:
        metric_dict['val_loss'] = total_val_loss / num_batches
    else:
        metric_dict['val_loss'] = 0.0
    
    return metric_dict

def train_epoch(device, train_loader, model, optimizer, epoch=None, total_epochs=None):
    model.train()
    epoch_loss = 0
    num_batches = 0

    def iter_single_samples(sample):
        if isinstance(sample, dict):
            batch_size = len(sample['h_id_tensors'])
            for i in range(batch_size):
                h_id_tensor = sample['h_id_tensors'][i].to(device)
                r_id_tensor = sample['r_id_tensors'][i].to(device)
                t_id_tensor = sample['t_id_tensors'][i].to(device)
                q_emb = sample['q_embs'][i].to(device)
                entity_embs = sample['entity_embs_list'][i].to(device)
                relation_embs = sample['relation_embs_list'][i].to(device)
                topic_entity_one_hot = sample['topic_entity_one_hots'][i].to(device)
                num_non_text_entities = sample['num_non_text_entities_list'][i]
                target_triple_probs = sample['target_triple_probs_list'][i]
                a_entity_id_list = sample['a_entity_id_lists'][i]
                yield (
                    h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                    num_non_text_entities, relation_embs, topic_entity_one_hot,
                    target_triple_probs, a_entity_id_list
                )
        else:
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs, \
            num_non_text_entities, relation_embs, topic_entity_one_hot, \
            target_triple_probs, a_entity_id_list = prepare_sample(device, sample)
            yield (
                h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                num_non_text_entities, relation_embs, topic_entity_one_hot,
                target_triple_probs, a_entity_id_list
            )

    # 創建訓練進度條
    train_desc = f"Train Epoch {epoch+1}/{total_epochs}" if epoch is not None and total_epochs is not None else "Training"
    train_pbar = tqdm(train_loader, desc=train_desc, leave=False,
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    for sample in train_pbar:
        batch_loss = 0
        batch_samples = 0
        
        for (
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
            num_non_text_entities, relation_embs, topic_entity_one_hot,
            target_triple_probs, a_entity_id_list
        ) in iter_single_samples(sample):
            
            if len(h_id_tensor) == 0:
                continue

            pred_triple_logits = model((
                h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                num_non_text_entities, relation_embs, topic_entity_one_hot))
            target_triple_probs = target_triple_probs.to(device).unsqueeze(-1)
            
            # 使用加權二元交叉熵來處理極度不平衡
            num_positive = target_triple_probs.sum().item()
            num_total = len(target_triple_probs)
            pos_weight = torch.tensor([(num_total - num_positive) / num_positive if num_positive > 0 else 1.0], device=device)
            loss = F.binary_cross_entropy_with_logits(
                pred_triple_logits, target_triple_probs, pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss = loss.item()
            epoch_loss += loss
            batch_loss += loss
            batch_samples += 1
        
        num_batches += 1
        
        # 更新進度條描述
        if batch_samples > 0:
            avg_batch_loss = batch_loss / batch_samples
            avg_epoch_loss = epoch_loss / num_batches
            train_pbar.set_postfix({
                'batch_loss': f'{avg_batch_loss:.4f}',
                'epoch_loss': f'{avg_epoch_loss:.4f}'
            })
    
    train_pbar.close()
    
    epoch_loss /= max(num_batches, 1)
    
    log_dict = {'loss': epoch_loss}
    return log_dict

def main(args):
    # Modify the config file for advanced settings and extensions.
    config_file = f'configs/retriever/{args.dataset}.yaml'
    config = load_yaml(config_file)
    
    device = torch.device('cuda:0')
    torch.set_num_threads(config['env']['num_threads'])
    set_seed(config['env']['seed'])

    # 根據參數決定使用的方法
    method = args.method if hasattr(args, 'method') else 'default'
    freq_weight = (method == 'freq_weight')
    
    # 使用台灣時間 (UTC+8)
    ts = time.strftime('%b%d-%H:%M:%S', time.localtime(time.time() + 8 * 3600))
    config_df = pd.json_normalize(config, sep='/')
    exp_prefix = config['train']['save_prefix']
    
    # 根據使用的方法調整實驗名稱
    if freq_weight:
        exp_name = f'{exp_prefix}_freq_weight_{ts}'
    else:
        exp_name = f'{exp_prefix}_{ts}'
    
    # 如果有指定 id_sup，則在實驗名稱後加上後綴
    if args.id_sup is not None:
        exp_name = f'{exp_name}_{args.id_sup}'
    
    wandb.init(
        project=f'{args.dataset}_retriever',
        name=exp_name,
        config=config_df.to_dict(orient='records')[0],
        mode='online'
    )
    
    # 使用命令行參數的epoch數量，如果沒有指定則使用config中的值
    num_epochs = args.num_epochs if hasattr(args, 'num_epochs') and args.num_epochs is not None else config['train']['num_epochs']
    
    # 使用命令行參數的patience，如果沒有指定則使用config中的值
    patience = args.patience if hasattr(args, 'patience') and args.patience is not None else config['train']['patience']
    
    # 使用命令行參數的k_list，如果沒有指定則使用config中的值
    if hasattr(args, 'k_list') and args.k_list is not None:
        # 解析字符串格式的k_list
        k_list_str = args.k_list.replace(' ', '')  # 移除空格
        k_list = [int(k) for k in k_list_str.split(',')]
        config['eval']['k_list'] = k_list
    else:
        # 確保 config 中的 k_list 是列表格式
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
    print(f"===============================")
    
    # 設定資料與DataLoader的保守預設值（可由CLI覆蓋）
    if args.dataset == 'cwq':
        default_samples_per_epoch = 8000
        default_samples_per_batch_load = 64
        default_batch_size = 1
        default_num_workers = 0
    else:  # webqsp
        default_samples_per_epoch = 8000
        default_samples_per_batch_load = 64
        default_batch_size = 1
        default_num_workers = 0

    samples_per_epoch = args.samples_per_epoch if hasattr(args, 'samples_per_epoch') and args.samples_per_epoch is not None else default_samples_per_epoch
    samples_per_batch_load = args.samples_per_batch_load if hasattr(args, 'samples_per_batch_load') and args.samples_per_batch_load is not None else default_samples_per_batch_load
    batch_size = args.batch_size if hasattr(args, 'batch_size') and args.batch_size is not None else default_batch_size
    num_workers = args.num_workers if hasattr(args, 'num_workers') and args.num_workers is not None else default_num_workers
    pin_memory = False
    prefetch_factor = 2

    train_set = OptimizedRetrieverDataset(
        config=config,
        split='train',
        freq_weight=freq_weight,
        samples_per_epoch=samples_per_epoch,
        samples_per_batch_load=samples_per_batch_load
    )
    
    val_set = OptimizedRetrieverDataset(
        config=config, split='val',
        freq_weight=freq_weight,
        samples_per_epoch=samples_per_epoch,
        samples_per_batch_load=samples_per_batch_load
    )
    
    # train_set = ImprovedRandomBatchRetrieverDataset(
    #     config=config,
    #     split='train',
    #     freq_weight=freq_weight,
    #     samples_per_epoch=samples_per_epoch,
    #     batch_loading_size=samples_per_batch_load  # 修正：samples_per_batch_load -> batch_loading_size
    # )

    # val_set = ImprovedRandomBatchRetrieverDataset(
    #     config=config, split='val',
    #     freq_weight=freq_weight,
    #     samples_per_epoch=samples_per_epoch,
    #     batch_loading_size=samples_per_batch_load  # 修正：samples_per_batch_load -> batch_loading_size
    # )

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

    # 自動選擇 collate（batch大小與workers已上方決定，可由CLI覆蓋）
    if args.dataset == 'webqsp':
        selected_collate = collate_retriever
    elif args.dataset == 'cwq':
        selected_collate = collate_retriever_batch
    
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=selected_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=(num_workers > 0)
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        collate_fn=selected_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=(num_workers > 0)
    )
        
    emb_size = train_set[0]['q_emb'].shape[-1]  
    model = Retriever(emb_size, **config['retriever']).to(device)
    optimizer = Adam(model.parameters(), **config['optimizer'])

    num_patient_epochs = 0
    best_val_metric = 0
    best_val_loss = float('inf')  # 初始化最佳驗證損失
    
    # 創建整體訓練進度條
    main_pbar = tqdm(range(num_epochs), desc="Overall Training Progress", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for epoch in main_pbar:
        num_patient_epochs += 1
        
        # 更新主進度條描述
        main_pbar.set_description(f"Training Progress (Epoch {epoch+1}/{num_epochs})")
        
        val_eval_dict = eval_epoch(config, device, val_loader, model, epoch, num_epochs)
        target_val_metric = val_eval_dict['triple_recall@100']
        
        # 記錄validation metrics（包括validation loss）
        val_log = {'val/epoch': epoch}
        for key, val in val_eval_dict.items():
            val_log[f'val/{key}'] = val
        wandb.log(val_log)
        
        if target_val_metric > best_val_metric:
        # 同時考慮 recall 和 loss 的 early stopping 標準
        # if (target_val_metric > best_val_metric and val_eval_dict['val_loss'] < best_val_loss):
            num_patient_epochs = 0
            best_val_metric = target_val_metric
            best_val_loss = val_eval_dict['val_loss']
            best_state_dict = {
                'config': config,
                'model_state_dict': model.state_dict()
            }
            # 確保目錄存在
            save_dir = os.path.join('/home/SubgraphRAG/retrieve/results/training', args.dataset, exp_name)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(best_state_dict, os.path.join(save_dir, f'cpt.pth'))
            
            # 保存triplet統計信息
            triplet_info_path = os.path.join(save_dir, 'triplet_info.txt')
            with open(triplet_info_path, 'w') as f:
                f.write(f"Dataset: {args.dataset}\n")
                f.write(f"Experiment: {exp_name}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 8 * 3600))}\n\n")
                
                f.write("Training Set Statistics:\n")
                f.write(f"  Skipped samples: {triplet_info['train']['skipped_samples']}\n")
                f.write(f"  Relevant triples - Median: {triplet_info['train']['relevant_triples_median']}\n")
                f.write(f"  Relevant triples - Mean: {triplet_info['train']['relevant_triples_mean']}\n")
                f.write(f"  Relevant triples - Max: {triplet_info['train']['relevant_triples_max']}\n\n")
                
                f.write("Validation Set Statistics:\n")
                f.write(f"  Skipped samples: {triplet_info['val']['skipped_samples']}\n")
                f.write(f"  Relevant triples - Median: {triplet_info['val']['relevant_triples_median']}\n")
                f.write(f"  Relevant triples - Mean: {triplet_info['val']['relevant_triples_mean']}\n")
                f.write(f"  Relevant triples - Max: {triplet_info['val']['relevant_triples_max']}\n")

        train_log_dict = train_epoch(device, train_loader, model, optimizer, epoch, num_epochs)
        
        train_log_dict.update({
            'num_patient_epochs': num_patient_epochs,
            'epoch': epoch
        })
        wandb.log(train_log_dict)
        
        # 更新主進度條的後綴信息
        main_pbar.set_postfix({
            'best_recall@100': f'{best_val_metric:.4f}',
            'current_recall@100': f'{target_val_metric:.4f}',
            'patience': f'{num_patient_epochs}/{patience}',
            'train_loss': f'{train_log_dict["loss"]:.4f}'
        })
        
        if num_patient_epochs == patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    main_pbar.close()
    print(f"\nTraining completed! Best validation recall@100: {best_val_metric:.4f}")

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
    parser.add_argument('--samples_per_epoch', type=int, default=10000,
                        help='Number of samples per epoch for OptimizedRetrieverDataset')
    parser.add_argument('--samples_per_batch_load', type=int, default=64,
                        help='Number of samples to load per batch from disk in OptimizedRetrieverDataset')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='DataLoader batch size override')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader num_workers override')
    
    args = parser.parse_args()
    
    # 調試：打印參數
    ts_start = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 8 * 3600))
    start_time = time.time()
    print(f"========== Start training retriever {ts_start} ==========")
    print(f"Parsed arguments: {args}")
    print(f"method: {args.method}")
    print(f"freq_weight: {args.method == 'freq_weight'}")
    print(f"id_sup: {args.id_sup}")
    print(f"patience: {args.patience}")
    print(f"k_list: {args.k_list}")
    print(f"samples_per_epoch: {args.samples_per_epoch}")
    print(f"samples_per_batch_load: {args.samples_per_batch_load}")
    print(f"batch_size: {args.batch_size}")
    print(f"num_workers: {args.num_workers}")
    
    main(args)
    
    ts_end = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 8 * 3600))
    end_time = time.time()
    print(f"========== End training retriever {ts_end} ==========")
    print(f"Retriever training time: {end_time - start_time:.2f} seconds")
