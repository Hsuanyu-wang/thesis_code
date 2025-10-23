#!/usr/bin/env python3
"""
Training script for GrailQA dataset using the new adapter system.
This demonstrates how to integrate new datasets with the existing pipeline.
"""

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
from src.dataset.adapters import ensure_processed_data

# Try to import optimized dataset
try:
    from src.dataset.retriever import OptimizedRetrieverDataset, optimized_collate_retriever, GroupedByFileBatchSampler
    _HAS_OPTIMIZED_DS = True
except Exception:
    _HAS_OPTIMIZED_DS = False

from src.model.retriever import Retriever
from src.setup import set_seed, prepare_sample


def build_dataset(config, split, use_optimized=True):
    """Build dataset using the adapter system."""
    dataset_name = config['dataset']['name']
    
    # Ensure processed data exists
    print(f"🔄 Ensuring processed data exists for {dataset_name}/{split}...")
    ensure_processed_data(dataset_name, config, split)
    
    # Build dataset using existing pipeline
    if use_optimized and _HAS_OPTIMIZED_DS:
        print(f"📊 Using OptimizedRetrieverDataset for {split}")
        return OptimizedRetrieverDataset(
            config, 
            split, 
            skip_no_path=True, 
            freq_weight=config.get('freq_weight', False)
        )
    else:
        print(f"📊 Using RetrieverDataset for {split}")
        return RetrieverDataset(
            config, 
            split, 
            skip_no_path=True, 
            freq_weight=config.get('freq_weight', False)
        )


def train_model(config, device, train_set, val_set, args):
    """Train the retriever model."""
    print("🚀 Starting training...")
    
    # Initialize model
    model = Retriever(config).to(device)
    
    # Setup optimizer
    optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Setup data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_retriever,
        num_workers=config['training'].get('num_workers', 4)
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_retriever,
        num_workers=config['training'].get('num_workers', 4)
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience = config['training'].get('patience', 10)
    patience_counter = 0
    
    for epoch in range(config['training']['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}"):
            optimizer.zero_grad()
            
            # Prepare batch
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs, \
                num_non_text_entities, relation_embs, topic_entity_one_hot, \
                target_triple_probs, a_entity_id_list = prepare_sample(device, batch)
            
            # Forward pass
            if hasattr(model, 'forward_legacy'):
                logits = model.forward_legacy(
                    h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                    num_non_text_entities, relation_embs, topic_entity_one_hot
                )
            else:
                logits = model(
                    h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                    num_non_text_entities, relation_embs, topic_entity_one_hot
                )
            
            # Compute loss
            target_probs = target_triple_probs.reshape(-1)
            logits = logits.reshape(-1)
            
            loss = F.binary_cross_entropy_with_logits(logits, target_probs)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs, \
                    num_non_text_entities, relation_embs, topic_entity_one_hot, \
                    target_triple_probs, a_entity_id_list = prepare_sample(device, batch)
                
                if hasattr(model, 'forward_legacy'):
                    logits = model.forward_legacy(
                        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                        num_non_text_entities, relation_embs, topic_entity_one_hot
                    )
                else:
                    logits = model(
                        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                        num_non_text_entities, relation_embs, topic_entity_one_hot
                    )
                
                target_probs = target_triple_probs.reshape(-1)
                logits = logits.reshape(-1)
                
                loss = F.binary_cross_entropy_with_logits(logits, target_probs)
                val_loss += loss.item()
                val_batches += 1
        
        # Calculate average losses
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            save_dir = f"{config['dataset']['name']}_{time.strftime('%b%d-%H:%M:%S')}"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, 'cpt.pth'))
            print(f"💾 Saved best model to {save_dir}/cpt.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping after {epoch+1} epochs")
                break
    
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='grailQA', help='Dataset name')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--use-optimized', action='store_true', default=True, help='Use optimized dataset')
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config_file = args.config
    else:
        config_file = f'configs/retriever/{args.dataset}.yaml'
    
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        print("Creating default config for GrailQA...")
        
        # Create default config
        default_config = {
            'dataset': {
                'name': args.dataset,
                'source': args.dataset,
                'raw_data_path': f'data_files/{args.dataset}',
                'splits': ['train', 'val', 'test']
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 1e-4,
                'num_epochs': 100,
                'patience': 10,
                'num_workers': 4
            },
            'freq_weight': False
        }
        
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(default_config, f)
        print(f"✅ Created default config: {config_file}")
    
    config = load_yaml(config_file)
    
    # Set random seed
    set_seed(config.get('seed', 42))
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Build datasets
    print("📊 Building datasets...")
    train_set = build_dataset(config, 'train', use_optimized=args.use_optimized)
    val_set = build_dataset(config, 'val', use_optimized=args.use_optimized)
    
    print(f"📈 Train samples: {len(train_set)}")
    print(f"📈 Val samples: {len(val_set)}")
    
    # Train model
    model = train_model(config, device, train_set, val_set, args)
    
    print("✅ Training completed!")


if __name__ == '__main__':
    main()
