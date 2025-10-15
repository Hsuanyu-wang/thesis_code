import os
import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
from tqdm import tqdm

from src.config.emb import load_yaml
from src.dataset.emb import EmbInferDataset
from src.dataset.retriever import RandomBatchRetrieverDataset, collate_retriever_batch

def get_dynamic_batch_size() -> int:
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory_gb < 8:
            return 128
        elif gpu_memory_gb < 16:
            return 256
        else:
            return 64
    return 64


def collate_fn_basic(batch):
    # EmbInferDataset 的 __getitem__ 回傳: id, q_text, text_entity_list, relation_list
    ids, q_texts, entity_lists, relation_lists = [], [], [], []
    for (id, q_text, text_entity_list, relation_list) in batch:
        ids.append(id)
        q_texts.append(q_text)
        entity_lists.append(text_entity_list)
        relation_lists.append(relation_list)
    return ids, q_texts, entity_lists, relation_lists


@torch.no_grad()
def get_emb_with_dataloader(subset, text_encoder, save_dir, split_name,
                            batch_size=None, num_workers=8, pin_memory=True, prefetch_factor=2):
    os.makedirs(save_dir, exist_ok=True)

    if batch_size is None:
        batch_size = get_dynamic_batch_size()

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory if torch.cuda.is_available() else False,
        collate_fn=collate_fn_basic,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=(num_workers > 0)
    )

    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {gpu_memory_gb:.1f}GB, using batch_size: {batch_size}, num_workers: {num_workers}, pin_memory: {pin_memory}")
    else:
        print(f"CPU mode, using batch_size: {batch_size}")

    batch_idx = 0
    for ids, q_texts, entity_lists, relation_lists in tqdm(loader, desc=f"{split_name}"):
        batch_emb_dict = {}
        try:
            # 逐筆呼叫 text_encoder（也可在 text_encoder 內部做子批處理）
            for id, q_text, text_entity_list, relation_list in zip(ids, q_texts, entity_lists, relation_lists):
                q_emb, entity_embs, relation_embs = text_encoder(q_text, text_entity_list, relation_list)
                batch_emb_dict[id] = {
                    'q_emb': q_emb,
                    'entity_embs': entity_embs,
                    'relation_embs': relation_embs
                }
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("OOM in current loader batch, attempting to continue...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                raise e

        # 直接落盤
        batch_file = os.path.join(save_dir, f'{split_name}_batch_{batch_idx:04d}.pth')
        torch.save(batch_emb_dict, batch_file)
        batch_idx += 1

        del batch_emb_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(args):
    # Modify the config file for advanced settings and extensions.
    config_file = f'configs/emb/gte-large-en-v1.5/{args.dataset}.yaml'
    config = load_yaml(config_file)

    torch.set_num_threads(config['env']['num_threads'])

    if args.dataset == 'cwq':
        input_file = os.path.join('rmanluo', 'RoG-cwq')
    else:
        input_file = os.path.join('ml1996', 'webqsp')

    # 原始資料集（Hugging Face 會快取到本地）
    train_set = load_dataset(input_file, split='train')
    val_set = load_dataset(input_file, split='validation')
    test_set = load_dataset(input_file, split='test')

    # 載入不可用文字之實體清單
    entity_identifiers = []
    with open(config['entity_identifier_file'], 'r') as f:
        for line in f:
            entity_identifiers.append(line.strip())
    entity_identifiers = set(entity_identifiers)

    # 前處理快取（pickle）
    save_dir = f'data_files/{args.dataset}/processed'
    os.makedirs(save_dir, exist_ok=True)

    train_set = EmbInferDataset(
        train_set,
        entity_identifiers,
        os.path.join(save_dir, 'train.pkl'))

    val_set = EmbInferDataset(
        val_set,
        entity_identifiers,
        os.path.join(save_dir, 'val.pkl'))

    test_set = EmbInferDataset(
        test_set,
        entity_identifiers,
        os.path.join(save_dir, 'test.pkl'),
        skip_no_topic=False,
        skip_no_ans=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    text_encoder_name = config['text_encoder']['name']
    if text_encoder_name == 'gte-large-en-v1.5':
        from src.model.text_encoders import GTELargeEN
        text_encoder = GTELargeEN(device)
    else:
        raise NotImplementedError(text_encoder_name)

    # emb_save_dir = f'data_files/{args.dataset}/emb/{text_encoder_name}'
    emb_save_dir = f'data_files/{args.dataset}/emb/batch_64'
    os.makedirs(emb_save_dir, exist_ok=True)

    # 逐 split 分批落盤（DataLoader 版本）
    get_emb_with_dataloader(train_set, text_encoder, emb_save_dir, 'train')
    get_emb_with_dataloader(val_set, text_encoder, emb_save_dir, 'val')
    get_emb_with_dataloader(test_set, text_encoder, emb_save_dir, 'test')


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser('Text Embedding Pre-Computation for Retrieval (DataLoader version)')
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        choices=['webqsp', 'cwq'], help='Dataset name')
    args = parser.parse_args()

    print(f"========== Start embedding with dataloader ==========")
    print(f"Parsed arguments: {args}")

    main(args) 
    
    print(f"========== End embedding with dataloader ==========")