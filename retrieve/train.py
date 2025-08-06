# 導入所需的套件
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
from src.model.kge_utils import create_kge_config_from_model
from src.setup import set_seed, prepare_sample

# 不進行梯度計算的評估函數
# 用於驗證集，計算各種 recall 指標
# 回傳 metric_dict: {triple_recall@k, ans_recall@k}
def eval_epoch(config, device, data_loader, model):
    model.eval()  # 設定模型為評估模式
    
    metric_dict = defaultdict(list)  # 用於儲存各種評估指標
    
    # 逐批次處理驗證資料
    for sample in tqdm(data_loader):
        # 準備樣本資料，包含各種張量與嵌入
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
        num_non_text_entities, relation_embs, topic_entity_one_hot,\
        target_triple_probs, a_entity_id_list = prepare_sample(device, sample)

        # 前向傳播，取得三元組預測分數
        pred_triple_logits, kge_score = model(
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
            num_non_text_entities, relation_embs, topic_entity_one_hot)
        
        # 確保 pred_triple_logits 是一維的
        pred_triple_logits = pred_triple_logits.squeeze(-1)
        
        # 三元組排序
        sorted_triple_ids_pred = torch.argsort(
            pred_triple_logits, descending=True).cpu()
        triple_ranks_pred = torch.empty_like(sorted_triple_ids_pred)
        triple_ranks_pred[sorted_triple_ids_pred] = torch.arange(
            len(triple_ranks_pred))
        
        target_triple_ids = target_triple_probs.nonzero().squeeze(-1)
        num_target_triples = len(target_triple_ids)
        
        if num_target_triples == 0:
            continue  # 若無標註三元組則跳過

        num_total_entities = len(entity_embs) + num_non_text_entities
        for k in config['eval']['k_list']:
            # 計算三元組 recall@k
            recall_k_sample = (
                triple_ranks_pred[target_triple_ids] < k).sum().item()
            metric_dict[f'triple_recall@{k}'].append(
                recall_k_sample / num_target_triples)
            
            # 計算答案實體 recall@k
            triple_mask_k = triple_ranks_pred < k
            entity_mask_k = torch.zeros(num_total_entities)
            entity_mask_k[h_id_tensor[triple_mask_k]] = 1.
            entity_mask_k[t_id_tensor[triple_mask_k]] = 1.
            recall_k_sample_ans = entity_mask_k[a_entity_id_list].sum().item()
            metric_dict[f'ans_recall@{k}'].append(
                recall_k_sample_ans / len(a_entity_id_list))

    # 對每個指標取平均
    for key, val in metric_dict.items():
        if isinstance(val, list):
            metric_dict[key] = float(np.mean(val)) if len(val) > 0 else 0.0
        else:
            metric_dict[key] = val
    
    return metric_dict

# 訓練一個 epoch 的函數
# 執行一個 epoch 的訓練，回傳 log 字典（平均 loss）
def train_epoch(device, train_loader, model, optimizer, config, kge_enabled=False):
    model.train()  # 設定模型為訓練模式
    epoch_loss = 0  # 累計損失
    epoch_kge_loss = 0  # 累計 KGE 損失
    
    # 取得 KGE loss weight
    kge_loss_weight = config.get('kge', {}).get('loss_weight', 1.0)
    
    for sample in tqdm(train_loader):
        # 準備樣本資料
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
        num_non_text_entities, relation_embs, topic_entity_one_hot,\
        target_triple_probs, a_entity_id_list = prepare_sample(device, sample)
            
        if len(h_id_tensor) == 0:
            continue  # 若無資料則跳過

        # 前向傳播
        pred_triple_logits, kge_score = model(
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
            num_non_text_entities, relation_embs, topic_entity_one_hot)
        target_triple_probs = target_triple_probs.to(device).unsqueeze(-1)
        
        # 計算 BCE 損失
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_triple_logits, target_triple_probs)
        
        # 計算 KGE margin ranking loss
        kge_loss = 0.0
        if kge_score is not None and kge_enabled:
            # 取得正樣本和負樣本的 KGE score
            positive_mask = target_triple_probs.squeeze(-1) > 0.5
            negative_mask = target_triple_probs.squeeze(-1) <= 0.5
            
            if positive_mask.sum() > 0 and negative_mask.sum() > 0:
                # 取得正樣本和負樣本的 KGE score
                pos_kge_scores = kge_score[positive_mask]
                neg_kge_scores = kge_score[negative_mask]
                
                # 計算 margin ranking loss
                margin = config.get('kge', {}).get('margin', 1.0)
                kge_loss = torch.clamp(
                    pos_kge_scores.unsqueeze(1) - neg_kge_scores.unsqueeze(0) + margin, 
                    min=0
                ).mean()
        
        # 總損失 = BCE loss + λ * KGE loss
        if kge_enabled:
            print(f"bce_loss: {bce_loss:.4f}, kge_loss: {kge_loss:.4f}")
        else:
            print(f"bce_loss: {bce_loss:.4f}, kge_loss: disabled")
        total_loss = bce_loss + kge_loss_weight * kge_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        epoch_loss += bce_loss.item()
        epoch_kge_loss += kge_loss.item() if isinstance(kge_loss, torch.Tensor) else kge_loss
    
    epoch_loss /= len(train_loader)  # 取平均
    epoch_kge_loss /= len(train_loader)  # 取平均
    
    log_dict = {
        'loss': epoch_loss,
        'kge_loss': epoch_kge_loss,
        'total_loss': epoch_loss + kge_loss_weight * epoch_kge_loss
    }
    return log_dict

# 主訓練腳本 (Main training script)
# 包含資料載入、模型初始化、訓練迴圈、驗證、早停、wandb log 等
def main(args):
    # 1. 載入資料集 (Load dataset)
    # 根據 dataset 讀取對應的 config 設定檔
    config_file = f'configs/retriever/{args.dataset}.yaml'
    config = load_yaml(config_file)
    
    device = torch.device('cuda:0')  # 使用第一張 GPU
    torch.set_num_threads(config['env']['num_threads'])  # 設定 CPU 執行緒數
    set_seed(config['env']['seed'])  # 設定隨機種子

    # 設定實驗名稱與 wandb 追蹤
    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    config_df = pd.json_normalize(config, sep='/')
    exp_prefix = config['train']['save_prefix']
    
    # Use experiment ID if provided, otherwise use timestamp
    if hasattr(args, 'exp_id') and args.exp_id is not None:
        exp_name = f'{exp_prefix}_{args.exp_id}'
    else:
        exp_name = f'{exp_prefix}_{ts}'
    
    # 新增：將訓練結果存放於 'training result' 資料夾下
    result_root = os.path.join('training result', exp_name)
    wandb.init(
        project=f'{args.dataset}',
        name=exp_name,
        config=config_df.to_dict(orient='records')[0]
    )
    os.makedirs(result_root, exist_ok=True)  # 建立訓練結果資料夾

    # 構建訓練與驗證資料集
    print("Loading training dataset...")
    train_set = RetrieverDataset(config=config, split='train')
    print("Loading validation dataset...")
    val_set = RetrieverDataset(config=config, split='val')

    # 建立 DataLoader
    train_loader = DataLoader(
        train_set, batch_size=1, shuffle=True, collate_fn=collate_retriever)
    val_loader = DataLoader(
        val_set, batch_size=1, collate_fn=collate_retriever)
    
    # 取得嵌入維度，初始化模型與優化器
    emb_size = train_set[0]['q_emb'].shape[-1]
    print(f"Embedding size: {emb_size}")
    
    # 若啟用 KGE，則載入 KGE 設定
    kge_config = None
    kge_enabled = False
    
    # 检查命令行是否指定了KGE模型，如果有则启用KGE，如果没有则禁用KGE
    if hasattr(args, 'kge_model') and args.kge_model is not None:
        # 命令行指定了KGE模型，启用KGE
        print("KGE is enabled via command line argument, loading KGE configuration...")
        dataset_name = config['dataset']['name']
        model_type = args.kge_model
        config['kge']['model_type'] = model_type
        config['kge']['enabled'] = True  # 確保配置文件中啟用 KGE
        print(f"Using KGE model type: {model_type}")
        
        kge_config = create_kge_config_from_model(dataset_name, model_type, 'train')
        if kge_config is None:
            print("Warning: KGE is enabled but no trained model found. Training without KGE.")
            kge_enabled = False
        else:
            print(f"KGE configuration loaded: {kge_config['model_type']} with {kge_config['embedding_dim']} dimensions")
            kge_enabled = True
    elif config.get('kge', {}).get('enabled', False):
        # 配置文件启用了KGE，且命令行没有指定
        print("KGE is enabled via config file, loading KGE configuration...")
        dataset_name = config['dataset']['name']
        model_type = config['kge']['model_type']
        
        kge_config = create_kge_config_from_model(dataset_name, model_type, 'train')
        if kge_config is None:
            print("Warning: KGE is enabled but no trained model found. Training without KGE.")
            kge_enabled = False
        else:
            print(f"KGE configuration loaded: {kge_config['model_type']} with {kge_config['embedding_dim']} dimensions")
            kge_enabled = True
    else:
        # 没有启用KGE
        print("KGE is disabled, training without KGE loss")
        config['kge']['enabled'] = False  # 確保配置文件中禁用 KGE
        kge_enabled = False
    
    print("Initializing model...")
    model = Retriever(emb_size, **config['retriever'], kge_config=kge_config).to(device)
    print("Initializing optimizer...")
    optimizer = Adam(model.parameters(), **config['optimizer'])
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    num_patient_epochs = 0  # 早停計數器
    best_val_metric = 0  # 最佳驗證指標
    
    # 3. 訓練主迴圈 (Main training loop)
    print(f"Starting training for {config['train']['num_epochs']} epochs...")
    epoch_pbar = tqdm(range(config['train']['num_epochs']), desc='Training Progress', position=0, leave=False)
    
    for epoch in epoch_pbar:
        num_patient_epochs += 1
        
        # 驗證集評估
        val_eval_dict = eval_epoch(config, device, val_loader, model)
        target_val_metric = val_eval_dict['triple_recall@100']
        if isinstance(target_val_metric, list):
            target_val_metric = float(np.mean(target_val_metric)) if len(target_val_metric) > 0 else 0.0
        else:
            target_val_metric = float(target_val_metric)
        
        # 若驗證指標提升則儲存模型
        if target_val_metric > best_val_metric:
            num_patient_epochs = 0
            best_val_metric = target_val_metric
            best_state_dict = {
                'config': config,
                'model_state_dict': model.state_dict()
            }
            torch.save(best_state_dict, os.path.join(result_root, f'cpt.pth'))

            val_log = {'val/epoch': epoch}
            for key, val in val_eval_dict.items():
                if isinstance(val, (float, int)):
                    val_log[f'val/{key}'] = float(val)
                elif isinstance(val, list):
                    val_log[f'val/{key}'] = str(val)
                else:
                    val_log[f'val/{key}'] = str(val)
            wandb.log(val_log)

        # 執行訓練一個 epoch
        train_log_dict = train_epoch(device, train_loader, model, optimizer, config, kge_enabled)
        
        # 更新進度條顯示目前指標
        postfix_dict = {
            'Epoch': f'{epoch+1}/{config["train"]["num_epochs"]}',
            'Val Recall@100': f'{target_val_metric:.4f}',
            'Best Recall@100': f'{best_val_metric:.4f}',
            'Patience': num_patient_epochs,
            'BCE Loss': f'{train_log_dict["loss"]:.4f}',
            'KGE Loss': f'{train_log_dict["kge_loss"]:.4f}',
            'Total Loss': f'{train_log_dict["total_loss"]:.4f}'
        }
        epoch_pbar.set_postfix(postfix_dict)
        
        # 處理 train_log_dict 型別
        safe_train_log_dict = {}
        for k, v in train_log_dict.items():
            if isinstance(v, (float, int)):
                safe_train_log_dict[k] = float(v)
            elif isinstance(v, list):
                safe_train_log_dict[k] = str(v)
            else:
                safe_train_log_dict[k] = str(v)
        safe_train_log_dict.update({
            'num_patient_epochs': num_patient_epochs,
            'epoch': epoch
        })
        wandb.log(safe_train_log_dict)
        if num_patient_epochs == config['train']['patience']:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break  # 早停
    
    epoch_pbar.close()
    print(f"\nTraining completed! Best validation recall@100: {best_val_metric:.4f}")

# 命令列執行進入點
if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True, 
                        choices=['webqsp', 'cwq', 'kgqagen'], help='Dataset name')
    parser.add_argument('--kge_model', type=str, default=None,
                        choices=['transe', 'distmult', 'ptranse', 'rotate', 'complex', 'simple', 'interht'],
                        help='KGE model type (可選: transe, distmult, ptranse, rotate, complex, simple, interht)')
    parser.add_argument('--exp_id', type=str, default=None,
                        help='Experiment ID for parallel runs')
    args = parser.parse_args()
    
    main(args)
