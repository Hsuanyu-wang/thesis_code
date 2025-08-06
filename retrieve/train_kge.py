import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
# from tqdm import tqdm  # 移除多餘的 tqdm
import random
from collections import defaultdict

from src.model.kge_models import create_kge_model, KGELoss
from src.config.retriever import load_yaml

class KGEDataset:
    """
    Dataset class for KGE training
    """
    def __init__(self, processed_data, num_negatives=1):
        self.triples = []
        self.num_negatives = num_negatives
        
        # Extract all triples from processed data
        print("Extracting triples from processed data...")
        for sample in processed_data:
            h_ids = sample['h_id_list']
            r_ids = sample['r_id_list']
            t_ids = sample['t_id_list']
            
            for h_id, r_id, t_id in zip(h_ids, r_ids, t_ids):
                self.triples.append((h_id, r_id, t_id))
        
        # Create entity and relation sets
        print("Building entity and relation sets...")
        self.entities = set()
        self.relations = set()
        for h_id, r_id, t_id in self.triples:
            self.entities.add(h_id)
            self.entities.add(t_id)
            self.relations.add(r_id)
        
        self.entities = list(self.entities)
        self.relations = list(self.relations)
        self.num_entities = len(self.entities)
        self.num_relations = len(self.relations)
        
        # Create entity and relation to id mappings
        self.entity_to_id = {entity: i for i, entity in enumerate(self.entities)}
        self.relation_to_id = {relation: i for i, relation in enumerate(self.relations)}
        
        # Convert triples to internal ids
        print("Converting triples to internal IDs...")
        self.triples = [(self.entity_to_id[h], self.relation_to_id[r], self.entity_to_id[t]) 
                       for h, r, t in self.triples]
        
        # Create negative sampling dictionary
        print("Creating negative sampling dictionary...")
        self.true_triples = set(self.triples)
        self.true_triples_with_rel = defaultdict(set)
        for h, r, t in self.triples:
            self.true_triples_with_rel[(h, r)].add(t)
            self.true_triples_with_rel[(t, r)].add(h)
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        
        # Generate negative samples
        neg_samples = []
        for _ in range(self.num_negatives):
            # Randomly corrupt head or tail
            if random.random() < 0.5:
                # Corrupt head
                neg_h = h
                while True:
                    neg_h = random.randint(0, self.num_entities - 1)
                    if (neg_h, r, t) not in self.true_triples:
                        break
                neg_samples.append((neg_h, r, t))
            else:
                # Corrupt tail
                neg_t = t
                while True:
                    neg_t = random.randint(0, self.num_entities - 1)
                    if (h, r, neg_t) not in self.true_triples:
                        break
                neg_samples.append((h, r, neg_t))
        
        return (h, r, t), neg_samples

def train_kge_model(config, dataset_name, split='train', force_retrain=False):
    """
    Train KGE model on the given dataset
    """
    # Load configuration
    kge_config = config['kge']
    model_type = kge_config['model_type']
    embedding_dim = kge_config['embedding_dim']
    num_epochs = kge_config['num_epochs']
    batch_size = kge_config['batch_size']
    learning_rate = kge_config['learning_rate']
    margin = kge_config.get('margin', 1.0)
    num_negatives = kge_config.get('num_negatives', 1)
    
    # Check if model already exists
    save_dir = f'data_files/{dataset_name}/kge/{model_type}'
    model_path = os.path.join(save_dir, f'{split}_model.pth')
    
    if os.path.exists(model_path) and not force_retrain:
        print(f"✅ KGE model already exists at {model_path}")
        print("Loading existing model...")
        
        # Load existing model
        checkpoint = torch.load(model_path, map_location='cpu')
        kge_model = create_kge_model(
            model_type=checkpoint['model_type'],
            num_entities=checkpoint['num_entities'],
            num_relations=checkpoint['num_relations'],
            embedding_dim=checkpoint['embedding_dim']
        )
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create dataset for statistics
        processed_file = f'data_files/{dataset_name}/processed/{split}.pkl'
        with open(processed_file, 'rb') as f:
            processed_data = pickle.load(f)
        kge_dataset = KGEDataset(processed_data, num_negatives=num_negatives)
        
        return kge_model, kge_dataset
    
    print(f"🔄 Training new KGE model for {dataset_name}/{model_type}/{split}")
    
    # Load processed data
    processed_file = f'data_files/{dataset_name}/processed/{split}.pkl'
    with open(processed_file, 'rb') as f:
        processed_data = pickle.load(f)
    
    # Create KGE dataset
    kge_dataset = KGEDataset(processed_data, num_negatives=num_negatives)
    
    # Create KGE model
    kge_model = create_kge_model(
        model_type=model_type,
        num_entities=kge_dataset.num_entities,
        num_relations=kge_dataset.num_relations,
        embedding_dim=embedding_dim,
        margin=margin
    )
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kge_model = kge_model.to(device)
    
    criterion = KGELoss(margin=margin)
    optimizer = optim.Adam(kge_model.parameters(), lr=learning_rate)
    
    # Training loop
    kge_model.train()
    total_loss = 0
    
    # Main training loop with overall progress
    from tqdm import tqdm
    epoch_pbar = tqdm(range(num_epochs), desc='Training KGE Model', position=0, leave=False)
    for epoch in epoch_pbar:
        epoch_loss = 0
        num_batches = 0
        
        # Create batches
        indices = list(range(len(kge_dataset)))
        random.shuffle(indices)
        
        # Batch training (不再用 tqdm)
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            
            pos_h, pos_r, pos_t = [], [], []
            neg_h, neg_r, neg_t = [], [], []
            
            for idx in batch_indices:
                (h, r, t), neg_samples = kge_dataset[idx]
                pos_h.append(h)
                pos_r.append(r)
                pos_t.append(t)
                
                for neg_h_i, neg_r_i, neg_t_i in neg_samples:
                    neg_h.append(neg_h_i)
                    neg_r.append(neg_r_i)
                    neg_t.append(neg_t_i)
            
            # Convert to tensors
            pos_h = torch.LongTensor(pos_h).to(device)
            pos_r = torch.LongTensor(pos_r).to(device)
            pos_t = torch.LongTensor(pos_t).to(device)
            neg_h = torch.LongTensor(neg_h).to(device)
            neg_r = torch.LongTensor(neg_r).to(device)
            neg_t = torch.LongTensor(neg_t).to(device)
            
            # Forward pass
            pos_score, neg_score = kge_model(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)
            
            # Compute loss
            loss = criterion(pos_score, neg_score)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches
        total_loss += avg_epoch_loss
        
        # Update progress bar description with loss info
        epoch_pbar.set_postfix({
            'Epoch': f'{epoch+1}/{num_epochs}',
            'Loss': f'{avg_epoch_loss:.4f}',
            'Avg Loss': f'{total_loss/(epoch+1):.4f}'
        })
    
    epoch_pbar.close()
    print(f'\nTraining completed. Average loss: {total_loss/num_epochs:.4f}')
    
    # Save model and mappings
    save_dir = f'data_files/{dataset_name}/kge/{model_type}'
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save({
        'model_state_dict': kge_model.state_dict(),
        'entity_to_id': kge_dataset.entity_to_id,
        'relation_to_id': kge_dataset.relation_to_id,
        'num_entities': kge_dataset.num_entities,
        'num_relations': kge_dataset.num_relations,
        'embedding_dim': embedding_dim,
        'model_type': model_type
    }, os.path.join(save_dir, f'{split}_model.pth'))
    
    return kge_model, kge_dataset

def main(args):
    # Load configuration
    config_file = f'configs/retriever/{args.dataset}.yaml'
    config = load_yaml(config_file)
    
    # Train KGE model
    # 若有指定--model_type則覆蓋config，支援多種KGE模型 (Override config if --model_type is set)
    if args.model_type is not None:
        config['kge']['model_type'] = args.model_type
    kge_model, kge_dataset = train_kge_model(config, args.dataset, args.split, args.force_retrain)
    
    print(f"KGE model ready for {args.dataset} dataset")
    print(f"Number of entities: {kge_dataset.num_entities}")
    print(f"Number of relations: {kge_dataset.num_relations}")
    print(f"Number of triples: {len(kge_dataset)}")

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser('Train KGE Model')
    parser.add_argument('-d', '--dataset', type=str, required=True, 
                        choices=['webqsp', 'cwq', 'kgqagen'], help='Dataset name')
    parser.add_argument('-s', '--split', type=str, default='train',
                        choices=['train', 'validation', 'test'], help='Data split')
    parser.add_argument('--force_retrain', action='store_true',
                        help='Force retrain even if model exists')
    parser.add_argument('--model_type', type=str, default=None,
                        choices=['transe', 'distmult', 'ptranse', 'rotate', 'complex', 'simple', 'interht', 'cmkge', 'cake', 'kgeprisma', 'rdf2vec'],
                        help='KGE model type (可選: transe, distmult, ptranse, rotate, complex, simple, interht, cmkge, cake, kgeprisma, rdf2vec)')
    args = parser.parse_args()
    
    main(args) 