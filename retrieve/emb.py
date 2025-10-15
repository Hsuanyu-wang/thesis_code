import os
import torch

from datasets import load_dataset
from tqdm import tqdm

from src.config.emb import load_yaml
from src.dataset.emb import EmbInferDataset


def get_dynamic_batch_size() -> int:
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory_gb < 8:
            return 16
        elif gpu_memory_gb < 16:
            return 32
        else:
            return 64
    return 16


@torch.no_grad()
def get_emb_in_batches(subset, text_encoder, save_dir, split_name, batch_size=None):
    os.makedirs(save_dir, exist_ok=True)
    if batch_size is None:
        batch_size = get_dynamic_batch_size()

    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {gpu_memory_gb:.1f}GB, using batch_size: {batch_size}")
    else:
        print(f"CPU mode, using batch_size: {batch_size}")

    total = len(subset)
    num_batches = (total + batch_size - 1) // batch_size

    for b in tqdm(range(num_batches), desc=f"{split_name}"):
        start = b * batch_size
        end = min(start + batch_size, total)

        batch_emb_dict = dict()

        for i in range(start, end):
            try:
                id, q_text, text_entity_list, relation_list = subset[i]
                q_emb, entity_embs, relation_embs = text_encoder(q_text, text_entity_list, relation_list)

                batch_emb_dict[id] = {
                    'q_emb': q_emb,
                    'entity_embs': entity_embs,
                    'relation_embs': relation_embs
                }
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM at sample {i}, skipping...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        # 直接落盤，避免常駐記憶體
        batch_file = os.path.join(save_dir, f'{split_name}_batch_{b:04d}.pth')
        torch.save(batch_emb_dict, batch_file)

        # 清理批次暫存
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

    emb_save_dir = f'data_files/{args.dataset}/emb/{text_encoder_name}'
    os.makedirs(emb_save_dir, exist_ok=True)

    # 逐 split 分批落盤
    get_emb_in_batches(train_set, text_encoder, emb_save_dir, 'train')
    get_emb_in_batches(val_set, text_encoder, emb_save_dir, 'val')
    get_emb_in_batches(test_set, text_encoder, emb_save_dir, 'test')


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser('Text Embedding Pre-Computation for Retrieval')
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        choices=['webqsp', 'cwq'], help='Dataset name')
    args = parser.parse_args()

    main(args)