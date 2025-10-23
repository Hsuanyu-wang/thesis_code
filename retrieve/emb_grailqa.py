#!/usr/bin/env python3
"""
Embedding computation for GrailQA dataset using the new adapter system.
This script demonstrates how to use the adapter system for new datasets.
"""

import os
import torch
from argparse import ArgumentParser

from datasets import load_dataset
from tqdm import tqdm

from src.config.emb import load_yaml
from src.dataset.emb import EmbInferDataset
from src.dataset.adapters import ensure_processed_data


@torch.no_grad()
def get_emb_in_batches(subset, text_encoder, save_dir, split_name, batch_size=None):
    """Compute embeddings in batches and save them."""
    os.makedirs(save_dir, exist_ok=True)
    
    if batch_size is None:
        batch_size = 64  # Default batch size for GrailQA
    
    from torch.utils.data import DataLoader
    
    def collate_fn_basic(batch):
        ids, q_texts, entity_lists, relation_lists = [], [], [], []
        for (id, q_text, text_entity_list, relation_list) in batch:
            ids.append(id)
            q_texts.append(q_text)
            entity_lists.append(text_entity_list)
            relation_lists.append(relation_list)
        return ids, q_texts, entity_lists, relation_lists
    
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_basic
    )
    
    batch_idx = 0
    for ids, q_texts, entity_lists, relation_lists in tqdm(loader, desc=f"Computing embeddings for {split_name}"):
        # Compute embeddings for this batch
        batch_embeddings = {}
        
        for i, (sample_id, q_text, text_entity_list, relation_list) in enumerate(
            zip(ids, q_texts, entity_lists, relation_lists)
        ):
            # Compute embeddings using the correct method
            q_emb, entity_embs, relation_embs = text_encoder(q_text, text_entity_list, relation_list)
            
            batch_embeddings[sample_id] = {
                'q_emb': q_emb,
                'entity_embs': entity_embs,
                'relation_embs': relation_embs
            }
        
        # Save batch
        batch_file = os.path.join(save_dir, f'{split_name}_batch_{batch_idx:04d}.pth')
        torch.save(batch_embeddings, batch_file)
        batch_idx += 1
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='grailQA', help='Dataset name')
    args = parser.parse_args()
    
    # Load configuration
    config_file = f'configs/emb/gte-large-en-v1.5/{args.dataset}.yaml'
    config = load_yaml(config_file)
    
    # Add dataset-specific configuration
    dataset_config = {
        'dataset': {
            'name': args.dataset,
            'source': args.dataset,
            'raw_data_path': f'data_files/{args.dataset}',
            'splits': ['train', 'test']
        }
    }
    config.update(dataset_config)
    
    torch.set_num_threads(config['env']['num_threads'])
    
    # Ensure processed data exists
    print("🔄 Ensuring processed data exists...")
    for split in config['dataset']['splits']:
        ensure_processed_data(args.dataset, config, split)
    
    # Load processed data
    print("📁 Loading processed data...")
    processed_dir = f'data_files/{args.dataset}/processed'
    
    # Load entity identifiers (create if not exists)
    entity_identifier_file = config['entity_identifier_file']
    if not os.path.exists(entity_identifier_file):
        print(f"⚠️  Entity identifier file not found: {entity_identifier_file}")
        print("Creating empty entity identifiers file...")
        os.makedirs(os.path.dirname(entity_identifier_file), exist_ok=True)
        with open(entity_identifier_file, 'w') as f:
            f.write("")  # Empty file for now
    
    entity_identifiers = set()
    if os.path.exists(entity_identifier_file):
        with open(entity_identifier_file, 'r') as f:
            for line in f:
                entity_identifiers.add(line.strip())
    
    # Initialize text encoder
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    text_encoder_name = config['text_encoder']['name']
    
    if text_encoder_name == 'gte-large-en-v1.5':
        from src.model.text_encoders import GTELargeEN
        text_encoder = GTELargeEN(device)
    else:
        raise NotImplementedError(f"Text encoder {text_encoder_name} not supported")
    
    # Process each split
    emb_save_dir = f'data_files/{args.dataset}/emb/gte-large-en-v1.5'
    # batch_size = 64
    os.makedirs(emb_save_dir, exist_ok=True)
    
    for split in config['dataset']['splits']:
        print(f"🔄 Processing {split} split...")
        
        # Load processed data
        processed_file = os.path.join(processed_dir, f'{split}.pkl')
        import pickle
        with open(processed_file, 'rb') as f:
            processed_data = pickle.load(f)
        
        # Create EmbInferDataset
        subset = EmbInferDataset(
            processed_data,
            entity_identifiers,
            processed_file,  # Use the same file as save_path
            skip_no_topic=False,  # Don't skip samples without topic entities for GrailQA
            skip_no_ans=False     # Don't skip samples without answer entities for GrailQA
        )
        
        # Compute embeddings
        get_emb_in_batches(subset, text_encoder, emb_save_dir, split)
    
    print("✅ Embedding computation completed!")


if __name__ == '__main__':
    main()
