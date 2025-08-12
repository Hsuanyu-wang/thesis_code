# 嵌入計算腳本 (Embedding computation script)
# 匯入必要的套件
import os
import torch
import re
import gc

from datasets import load_dataset
from tqdm import tqdm

from src.config.emb import load_yaml
from src.dataset.emb import EmbInferDataset

def extract_entity_identifiers(dataset_name, dataset):
    """
    根據資料集來源提取實體識別符
    """
    entity_identifiers = set()
    
    # if dataset_name == 'kgqagen':
    # KGQAGen-10k (Wikidata 格式)
    print("Extracting Wikidata entity IDs from KGQAGen-10k...")
    for i, sample in enumerate(dataset):
        if i % 1000 == 0:
            print(f"Processing sample {i}/{len(dataset)}")
        
        # Extract entities from proof triples
        for triple in sample['proof']:
            if len(triple) >= 3:
                # Extract entity IDs from "Entity Name (Q123456)" format
                for entity in [triple[0], triple[2]]:
                    # Ensure entity is a string
                    if isinstance(entity, str):
                        # Use regex to extract Q-numbers
                        match = re.search(r'\(Q\d+\)', entity)
                        if match:
                            q_id = match.group(0)[1:-1]  # Remove parentheses
                            entity_identifiers.add(q_id)
    
    print(f"Found {len(entity_identifiers)} unique Wikidata entity IDs")
        
    # elif dataset_name in ['webqsp', 'cwq']:
    #     # WebQSP/CWQ (Freebase 格式) - 使用預設的實體識別符檔案
    #     print(f"Using existing entity identifiers for {dataset_name}")
    #     return None  # 表示使用預設檔案
    
    # else:
    #     raise NotImplementedError(f"Entity extraction not implemented for dataset: {dataset_name}")
    
    return sorted(list(entity_identifiers))

# 計算並儲存嵌入向量的函數 - 支援分批處理和記憶體管理
# subset: 資料集（已包裝成 EmbInferDataset）
# text_encoder: 文本編碼器（模型）
# save_file: 儲存嵌入向量的檔案路徑
# batch_size: 每批處理的樣本數量
def get_emb(subset, text_encoder, save_file, batch_size=500):
    total_samples = len(subset)
    
    print(f"Processing {total_samples} samples in batches of {batch_size}")
    
    # 創建基本的儲存目錄
    save_dir = os.path.dirname(save_file)
    temp_dir = os.path.join(save_dir, 'temp_batches')
    os.makedirs(temp_dir, exist_ok=True)
    
    # 分批處理資料集，每個批次單獨儲存
    batch_files = []
    
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        batch_num = batch_start // batch_size + 1
        print(f"Processing batch {batch_num}/{(total_samples + batch_size - 1)//batch_size}: samples {batch_start}-{batch_end-1}")
        
        # 處理當前批次
        batch_emb_dict = dict()
        
        for i in tqdm(range(batch_start, batch_end), desc=f"Batch {batch_num}"):
            id, q_text, text_entity_list, relation_list = subset[i]
            
            # 取得問題、實體、關係的嵌入向量
            q_emb, entity_embs, relation_embs = text_encoder(
                q_text, text_entity_list, relation_list)
            
            emb_dict_i = {
                'q_emb': q_emb,
                'entity_embs': entity_embs,
                'relation_embs': relation_embs
            }
            batch_emb_dict[id] = emb_dict_i
        
        # 儲存當前批次到單獨的檔案
        batch_file = os.path.join(temp_dir, f'batch_{batch_num}.pth')
        torch.save(batch_emb_dict, batch_file)
        batch_files.append(batch_file)
        
        print(f"Batch {batch_num} completed and saved to {batch_file}")
        
        # 強制清理記憶體
        del batch_emb_dict
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
    
    # 合併所有批次檔案到最終檔案
    print("Merging all batch files into final embedding file...")
    final_emb_dict = dict()
    
    for i, batch_file in enumerate(tqdm(batch_files, desc="Merging batches")):
        batch_dict = torch.load(batch_file, map_location='cpu')
        final_emb_dict.update(batch_dict)
        
        # 每處理幾個批次就清理一次記憶體
        if (i + 1) % 5 == 0:
            gc.collect()
        
        print(f"Merged batch {i+1}/{len(batch_files)}, total samples: {len(final_emb_dict)}")
    
    # 儲存最終結果
    torch.save(final_emb_dict, save_file)
    print(f"All embeddings saved to {save_file}")
    
    # 清理臨時檔案
    print("Cleaning up temporary files...")
    for batch_file in batch_files:
        if os.path.exists(batch_file):
            os.remove(batch_file)
    
    # 刪除臨時目錄
    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
        os.rmdir(temp_dir)
        
    print(f"Cleanup completed. Total processed samples: {len(final_emb_dict)}")

# 主程式入口
# args: 命令列參數
# 主要流程：讀取設定、資料集、初始化模型、計算嵌入並儲存

def main(args):
    # 1. 載入資料集 (Load dataset)
    # 根據指定資料集載入對應的 config 設定檔
    config_file = f'configs/emb/gte-large-en-v1.5/{args.dataset}.yaml'
    config = load_yaml(config_file)
    
    # 設定 PyTorch 執行緒數量
    torch.set_num_threads(config['env']['num_threads'])

    # 2. 初始化嵌入模型 (Initialize embedding model)
    # 根據資料集名稱選擇對應的資料來源
    if args.dataset == 'cwq':
        input_file = os.path.join('rmanluo', 'RoG-cwq')
    elif args.dataset == 'webqsp':
        input_file = os.path.join('ml1996', 'webqsp')
    elif args.dataset == 'kgqagen':
        input_file = os.path.join('lianglz', 'KGQAGen-10k')
    else:
        raise NotImplementedError(args.dataset)

    # 載入 train/val/test 資料集
    train_set = load_dataset(input_file, split='train')
    # if args.dataset == 'kgqagen':
    #     val_set = load_dataset(input_file, split='dev')
    # else:
    #     val_set = load_dataset(input_file, split='validation')
    # test_set = load_dataset(input_file, split='test')
    
    # 3. 計算所有實體/關係的嵌入 (Compute embeddings for all entities/relations)
    # 根據資料集來源提取或讀取實體識別符

    if args.dataset == 'kgqagen':
        val_set = load_dataset(input_file, split='dev')
        extracted_entities = extract_entity_identifiers(args.dataset, val_set)
        if extracted_entities is not None:
            # 使用提取的實體識別符
            entity_identifiers = set(extracted_entities)
            
            # 儲存提取的實體識別符到檔案
            entity_file = config['entity_identifier_file']
            os.makedirs(os.path.dirname(entity_file), exist_ok=True)
            with open(entity_file, 'w') as f:
                f.write(f"# {args.dataset.upper()} entity identifiers\n")
                f.write("# This file contains entity IDs that should not be used for text embedding\n")
                f.write("# Format: one entity ID per line\n")
                f.write(f"# Source: {args.dataset} dataset\n\n")
                for entity_id in extracted_entities:
                    f.write(f"{entity_id}\n")
            print(f"Saved {len(extracted_entities)} entity identifiers to {entity_file}")
    else:
        # 使用預設的實體識別符檔案
        entity_identifiers = []
        with open(config['entity_identifier_file'], 'r') as f:
            for line in f:
                entity_identifiers.append(line.strip())
        entity_identifiers = set(entity_identifiers)
    
    # 建立儲存處理後資料的資料夾
    save_dir = f'data_files/{args.dataset}/processed'
    os.makedirs(save_dir, exist_ok=True)

    # 將原始資料集包裝成 EmbInferDataset，並儲存處理後的 pkl 檔
    train_set = EmbInferDataset(
        train_set,
        entity_identifiers,
        os.path.join(save_dir, 'train.pkl'))

    # val_set = EmbInferDataset(
    #     val_set,
    #     entity_identifiers,
    #     os.path.join(save_dir, 'val.pkl'))

    # #可跳過沒有topic entity, answer entity的資料，預設為False
    # test_set = EmbInferDataset(
    #     test_set,
    #     entity_identifiers,
    #     os.path.join(save_dir, 'test.pkl'),
    #     skip_no_topic=False,
    #     skip_no_ans=False)
    
    # 4. 儲存嵌入結果 (Save embedding results)
    # 設定運算裝置為 GPU
    device = torch.device('cuda:0')
    
    # 根據 config 選擇對應的文本編碼器
    text_encoder_name = config['text_encoder']['name']
    if text_encoder_name == 'gte-large-en-v1.5':
        from src.model.text_encoders import GTELargeEN
        text_encoder = GTELargeEN(device)
    else:
        raise NotImplementedError(text_encoder_name)
    
    # 建立儲存嵌入向量的資料夾
    emb_save_dir = f'data_files/{args.dataset}/emb/{text_encoder_name}'
    os.makedirs(emb_save_dir, exist_ok=True)
    
    # 計算並儲存 train/val/test 的嵌入向量，使用指定的批次大小
    get_emb(train_set, text_encoder, os.path.join(emb_save_dir, 'train.pth'), batch_size=args.batch_size)
    # get_emb(val_set, text_encoder, os.path.join(emb_save_dir, 'val.pth'), batch_size=args.batch_size)
    # get_emb(test_set, text_encoder, os.path.join(emb_save_dir, 'test.pth'), batch_size=args.batch_size)

# 命令列執行進入點
if __name__ == '__main__':
    from argparse import ArgumentParser
    
    # 建立命令列參數解析器
    parser = ArgumentParser('Text Embedding Pre-Computation for Retrieval')
    parser.add_argument('-d', '--dataset', type=str, required=True, 
                        choices=['webqsp', 'cwq', 'kgqagen'], help='Dataset name')
    parser.add_argument('-b', '--batch_size', type=int, default=500, 
                        help='Batch size for processing samples (default: 500)')
    args = parser.parse_args()
    
    # 執行主程式
    main(args)
