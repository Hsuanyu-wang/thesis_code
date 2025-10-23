import argparse
import os.path as osp
import os

import torch
import torch.optim as optim
import wandb
import time
import pickle
from collections import defaultdict
from torch.amp import autocast, GradScaler

from torch_geometric.datasets import FB15k_237, WebQSPDataset
# CWQDataset
from torch_geometric.nn import ComplEx, DistMult, RotatE, TransE
from tqdm import tqdm

model_map = {
    'transe': TransE,
    'complex': ComplEx,
    'distmult': DistMult,
    'rotate': RotatE,
}

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=model_map.keys(), type=str.lower,
                    required=True)
parser.add_argument('-d', '--dataset', choices=['fb15k237', 'webqsp', 'cwq'],
                    type=str.lower, required=True)
parser.add_argument('--from_processed', action='store_true', default=None,
                    help='Use processed PKL from retriever pipeline to build KGE graph (align IDs). Auto-enabled for webqsp/cwq.')
parser.add_argument('--processed_root', type=str,
                    default=None,
                    help='Root folder for processed PKL files when --from_processed is set. If omitted, use /home/YX_thesis/retrieve/data_files/<dataset>/processed')
parser.add_argument('--batch_size', type=int, default=1000,
                    help='Batch size for training (default: 1000)')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of workers for data loading (default: 4)')
parser.add_argument('--mixed_precision', action='store_true',
                    help='Use mixed precision training for speed')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help='Number of gradient accumulation steps (default: 1)')
parser.add_argument('--lr_scheduler', choices=['cosine', 'step', 'exponential', 'none'], 
                    default='none', help='Learning rate scheduler (default: none)')
parser.add_argument('--warmup_epochs', type=int, default=0,
                    help='Number of warmup epochs (default: 0)')
parser.add_argument('--auto_optimize', action='store_true', default=True,
                    help='Auto-optimize settings for webqsp/cwq datasets (default: True)')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_dir = osp.dirname(osp.realpath(__file__))

# Data preparation - Auto-enable processed data for webqsp and cwq
if args.from_processed is None:
    use_processed = args.dataset in ['webqsp', 'cwq']
else:
    use_processed = args.from_processed

if use_processed and (args.processed_root is None):
    args.processed_root = f"/home/YX_thesis/retrieve/data_files/{args.dataset}/processed"

# Auto-optimize settings for webqsp and cwq datasets
if args.auto_optimize and args.dataset in ['webqsp', 'cwq']:
    print(f"Auto-optimizing settings for {args.dataset} dataset...")
    
    # Optimize batch size for webqsp/cwq (larger datasets can handle bigger batches)
    if args.batch_size == 1000:  # Only change if using default
        args.batch_size = 2000
        print(f"  - Increased batch_size to {args.batch_size}")
    
    # Enable mixed precision for speed
    if not args.mixed_precision:
        args.mixed_precision = True
        print("  - Enabled mixed precision training")
    
    # Use cosine learning rate scheduler for better convergence
    if args.lr_scheduler == 'none':
        args.lr_scheduler = 'cosine'
        print("  - Enabled cosine learning rate scheduler")
    
    # Increase num_workers for better data loading (but not for CUDA due to multiprocessing issues)
    if args.num_workers == 4:  # Only change if using default
        if device == 'cuda':
            args.num_workers = 0  # CUDA doesn't work well with multiprocessing
            print(f"  - Set num_workers to 0 (CUDA multiprocessing issue)")
        else:
            args.num_workers = 8
            print(f"  - Increased num_workers to {args.num_workers}")
    
    print("Auto-optimization complete!")

if use_processed:
    # Build triples from processed PKLs to align entity/relation IDs with retriever pipeline
    def load_triples_from_pkl(pkl_path):
        with open(pkl_path, 'rb') as f:
            items = pickle.load(f)
        triples = []  # list of (h_label, r_label, t_label)
        for ex in items:
            entity_labels = list(ex['text_entity_list']) + list(ex['non_text_entity_list'])
            rel_labels = list(ex['relation_list'])
            for h_id, r_id, t_id in zip(ex['h_id_list'], ex['r_id_list'], ex['t_id_list']):
                h = entity_labels[h_id]
                r = rel_labels[r_id]
                t = entity_labels[t_id]
                triples.append((h, r, t))
        return triples

    train_triples = load_triples_from_pkl(osp.join(args.processed_root, 'train.pkl'))
    val_triples   = load_triples_from_pkl(osp.join(args.processed_root, 'val.pkl'))
    test_triples  = load_triples_from_pkl(osp.join(args.processed_root, 'test.pkl'))

    # Build global mappings
    all_entities = set()
    all_relations = set()
    for h, r, t in train_triples + val_triples + test_triples:
        all_entities.add(h); all_entities.add(t); all_relations.add(r)
    entity_to_id = {e: i for i, e in enumerate(sorted(all_entities))}
    relation_to_id = {r: i for i, r in enumerate(sorted(all_relations))}

    def triples_to_tensors(triples):
        heads = torch.tensor([entity_to_id[h] for h, _, _ in triples], dtype=torch.long)  # Keep on CPU for DataLoader
        rels  = torch.tensor([relation_to_id[r] for _, r, _ in triples], dtype=torch.long)  # Keep on CPU for DataLoader
        tails = torch.tensor([entity_to_id[t] for _, _, t in triples], dtype=torch.long)  # Keep on CPU for DataLoader
        return heads, rels, tails

    head_tr, rel_tr, tail_tr = triples_to_tensors(train_triples)
    head_va, rel_va, tail_va = triples_to_tensors(val_triples)
    head_te, rel_te, tail_te = triples_to_tensors(test_triples)

    num_nodes = len(entity_to_id)
    actual_num_relations = len(relation_to_id)
else:
    dataset_config = {
        'fb15k237': {
            'cls': FB15k_237,
            'path': osp.join(base_dir, 'data', 'FB15k'),
        },
        'webqsp': {
            'cls': WebQSPDataset,
            'path': osp.join(base_dir, 'data', 'webqsp'),
        },
        # 'cwq': {
        #     'cls': CWQDataset,
        #     'path': osp.join(base_dir, 'data', 'cwq'),
        # },
    }

    dataset_entry = dataset_config[args.dataset]
    path = dataset_entry['path']
    DatasetCls = dataset_entry['cls']

    train_data = DatasetCls(path, split='train')[0].to(device)
    val_data = DatasetCls(path, split='val')[0].to(device)
    test_data = DatasetCls(path, split='test')[0].to(device)

    # For WebQSPDataset, we need to create discrete relation types from edge_attr
    if args.dataset == 'webqsp':
        # Extract relation names from the description
        desc_lines = train_data.desc.split('\n')
        relation_names = []
        for line in desc_lines:
            if ',' in line and not line.startswith('node_id') and not line.startswith('src'):
                parts = line.split(',')
                if len(parts) >= 2:
                    relation_name = parts[1].strip()
                    if relation_name not in relation_names:
                        relation_names.append(relation_name)
        
        # Create mapping from edge index to relation type (dummy mapping)
        num_unique_relations = len(set(relation_names)) if relation_names else train_data.edge_index.size(1)
        rel_type_mapping = torch.zeros(train_data.edge_index.size(1), dtype=torch.long, device=device)
        for i in range(train_data.edge_index.size(1)):
            rel_type_mapping[i] = i % num_unique_relations
        actual_num_relations = num_unique_relations
    else:
        rel_type_mapping = train_data.edge_type
        actual_num_relations = train_data.num_edge_types

model_arg_map = {'rotate': {'margin': 9.0}}
# model = model_map[args.model](
#     num_nodes=train_data.num_nodes,
#     num_relations=actual_num_relations,
#     hidden_channels=50,
#     **model_arg_map.get(args.model, {}),
# ).to(device)
if use_processed:
    # Build model with sizes from processed mappings
    model = model_map[args.model](
        num_nodes=num_nodes,
        num_relations=actual_num_relations,
        hidden_channels=50,
        **model_arg_map.get(args.model, {}),
    ).to(device)
else:
    model = model_map[args.model](
        num_nodes=train_data.num_nodes,
        num_relations=actual_num_relations,
        hidden_channels=50,
        **model_arg_map.get(args.model, {}),
    ).to(device)

if use_processed:
    loader = model.loader(
        head_index=head_tr,
        rel_type=rel_tr,
        tail_index=tail_tr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device == 'cuda' else False,  # Enable pin_memory for CPU tensors
    )
else:
    loader = model.loader(
        head_index=train_data.edge_index[0],
        rel_type=rel_type_mapping,
        tail_index=train_data.edge_index[1],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device == 'cuda' else False,  # Enable pin_memory for CPU tensors
    )

optimizer_map = {
    'transe': optim.Adam(model.parameters(), lr=0.01),
    'complex': optim.Adagrad(model.parameters(), lr=0.001, weight_decay=1e-6),
    'distmult': optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6),
    'rotate': optim.Adam(model.parameters(), lr=1e-3),
}
optimizer = optimizer_map[args.model]

# Initialize mixed precision scaler if enabled
scaler = GradScaler('cuda') if args.mixed_precision and device == 'cuda' else None

# Initialize learning rate scheduler
scheduler = None
if args.lr_scheduler != 'none':
    if args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    elif args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    elif args.lr_scheduler == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Initialize Weights & Biases (style inspired by train_main.py)
ts = time.strftime('%b%d-%H:%M:%S', time.localtime(time.time() + 8 * 3600))
exp_name = f'{args.dataset}_{args.model}_{ts}'

# Prepare config payload with richer metadata (align to actual sizes)
wb_config = {
    'dataset': args.dataset,
    'model': args.model,
    'device': device,
    'num_nodes': (num_nodes if use_processed else (train_data.num_nodes if 'train_data' in globals() else None)),
    'num_relations': actual_num_relations,
    'hidden_channels': 50,
    'optimizer': optimizer.__class__.__name__,
    'lr': optimizer.param_groups[0].get('lr', None),
    'batch_size': args.batch_size,
    'num_workers': args.num_workers,
    'mixed_precision': args.mixed_precision,
    'gradient_accumulation_steps': args.gradient_accumulation_steps,
    'lr_scheduler': args.lr_scheduler,
    'auto_optimized': args.auto_optimize and args.dataset in ['webqsp', 'cwq'],
    'use_processed': use_processed
}

wandb.init(
    project='pyg_kge',
    name=exp_name,
    config=wb_config
)


def train():
    model.train()
    total_loss = total_examples = 0
    train_pbar = tqdm(loader, desc='Train', leave=False)
    
    for step, (head_index, rel_type, tail_index) in enumerate(train_pbar):
        # Move data to device if not already there
        head_index = head_index.to(device)
        rel_type = rel_type.to(device)
        tail_index = tail_index.to(device)
        
        optimizer.zero_grad()
        
        # Use mixed precision if enabled
        if args.mixed_precision and device == 'cuda':
            with autocast(device_type='cuda'):
                loss = model.loss(head_index, rel_type, tail_index)
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
        else:
            loss = model.loss(head_index, rel_type, tail_index)
            loss.backward()
            
            # Gradient accumulation
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
        
        total_loss += float(loss) * head_index.numel()
        total_examples += head_index.numel()
        try:
            train_pbar.set_postfix({'loss': f'{float(loss):.4f}'})
        except Exception:
            pass
    return total_loss / total_examples


@torch.no_grad()
def eval_split(split: str):
    model.eval()
    if use_processed:
        if split == 'val':
            # Ensure tensors are on the same device as the model
            return model.test(
                head_index=head_va.to(device),
                rel_type=rel_va.to(device),
                tail_index=tail_va.to(device),
                batch_size=20000,
                k=10,
            )
        else:
            # Ensure tensors are on the same device as the model
            return model.test(
                head_index=head_te.to(device),
                rel_type=rel_te.to(device),
                tail_index=tail_te.to(device),
                batch_size=20000,
                k=10,
            )
    else:
        data = val_data if split == 'val' else test_data
        if args.dataset == 'webqsp':
            test_rel_type_mapping = torch.zeros(data.edge_index.size(1), dtype=torch.long, device=device)
            for i in range(data.edge_index.size(1)):
                test_rel_type_mapping[i] = i % actual_num_relations
            test_rel_type = test_rel_type_mapping.to(device)
        else:
            test_rel_type = data.edge_type.to(device)
        return model.test(
            head_index=data.edge_index[0].to(device),
            rel_type=test_rel_type,
            tail_index=data.edge_index[1].to(device),
            batch_size=20000,
            k=10,
        )


best_mrr = float('-inf')
best_loss = float('inf')  # Initialize to positive infinity for loss minimization
epochs_no_improve = 0

# Create model save directory
save_dir = osp.join(base_dir, 'model')
os.makedirs(save_dir, exist_ok=True)
os.makedirs(osp.join(save_dir, args.dataset), exist_ok=True)

def save_checkpoint(epoch, loss, is_best=False):
    """Save model checkpoint"""
    checkpoint_path = osp.join(save_dir, args.dataset, f"{args.model}_{args.dataset}_epoch_{epoch}.pt")
    best_path = osp.join(save_dir, args.dataset, f"{args.model}_{args.dataset}_min.pt")
    
    save_payload = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_name': args.model,
        'dataset': args.dataset,
        'num_nodes': (num_nodes if use_processed else (train_data.num_nodes if 'train_data' in globals() else None)),
        'num_relations': actual_num_relations,
        'hidden_channels': 50,
        'epoch': epoch,
        'loss': loss,
        'best_mrr': best_mrr,
    }
    
    torch.save(save_payload, checkpoint_path)
    if is_best:
        torch.save(save_payload, best_path)
        print(f"Best model (minimum loss) saved at epoch {epoch} with loss {loss:.4f}")
    
    print(f"Checkpoint saved at epoch {epoch}")

try:
    loss = 0.0  # Initialize loss variable
    for epoch in tqdm(range(1, 501), desc='Epochs'):
        # 訓練並更新 loss
        loss = train()

        # 更新學習率 scheduler 並記錄
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({'epoch': epoch, 'train/loss': loss, 'learning_rate': current_lr})
        else:
            wandb.log({'epoch': epoch, 'train/loss': loss})

        # Save checkpoint only when loss reaches new minimum
        if loss < best_loss:
            best_loss = loss
            save_checkpoint(epoch, loss, is_best=True)
            wandb.log({'epoch': epoch, 'train/best_loss': best_loss})
        else:
            # Only save regular checkpoint every 50 epochs to avoid too many files
            if epoch % 50 == 0:
                save_checkpoint(epoch, loss)
        
        # # 每 10 epochs 做 validation
        # if epoch % 10 == 0:
        #     rank, mrr, hits = eval_split('val')
        #     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Mean Rank: {rank:.2f}, '
        #           f'Val MRR: {mrr:.4f}, Val Hits@10: {hits:.4f}')
        #     wandb.log({'epoch': epoch, 'val/mean_rank': rank, 'val/mrr': mrr, 'val/hits@10': hits})

        #     # 檢查是否最好的 mrr
        #     if mrr > best_mrr:
        #         best_mrr = mrr
        #         epochs_no_improve = 0
        #         save_checkpoint(epoch, loss, is_best=True)
        #         wandb.log({'epoch': epoch, 'val/best_mrr': best_mrr})
        #     else:
        #         epochs_no_improve += 1

        #     # early stopping
        #     if epochs_no_improve >= 10:
        #         print('Early stopping triggered (no improvement in 10 consecutive epochs).')
        #         break

except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving current model...")
    save_checkpoint(epoch, loss)
    print("Model saved successfully. You can resume training later.")
    wandb.finish()
    exit(0)
except Exception as e:
    print(f"\nTraining interrupted by error: {e}")
    print("Saving current model...")
    save_checkpoint(epoch, loss)
    print("Model saved successfully.")
    wandb.finish()
    raise e

rank, mrr, hits_at_10 = eval_split('test')
print(f'Test Mean Rank: {rank:.2f}, Test MRR: {mrr:.4f}, '
      f'Test Hits@10: {hits_at_10:.4f}')
wandb.log({'test/mean_rank': rank, 'test/mrr': mrr, 'test/hits@10': hits_at_10})
wandb.finish()