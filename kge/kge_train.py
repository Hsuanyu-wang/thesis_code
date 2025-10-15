import argparse
import os.path as osp

import torch
import torch.optim as optim
from time import time
from tqdm import tqdm
import os
from typing import List, Tuple, Dict
import csv
from datetime import datetime
import wandb

from torch_geometric.datasets import FB15k_237
from torch_geometric.nn import ComplEx, DistMult, RotatE, TransE
from kge_model import TransE_filtered_negative_sampling, TransH, TransR, TransD
from torch_geometric.data import Data

model_map = {
    'transe': TransE,
    'transe_f_n': TransE_filtered_negative_sampling,
    'transh': TransH,
    'transr': TransR,
    'transd': TransD,
    'complex': ComplEx,
    'distmult': DistMult,
    'rotate': RotatE,
}

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=model_map.keys(), type=str.lower,
                    required=True)
parser.add_argument('--dataset_dir', type=str, default=None,
                    help='Custom dataset directory containing train.txt, valid.txt (or val.txt), test.txt with tab-separated head\trel\ttail.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of training epochs')
parser.add_argument('--log_dir', type=str, default='/home/SubgraphRAG/kge/logs',
                    help='Directory to save per-experiment CSV logs')
parser.add_argument('--eval_every', type=int, default=25,
                    help='Evaluate on validation set every N epochs')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _read_triples(file_path: str) -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t') if ('\t' in line) else line.split()
            if len(parts) < 3:
                continue
            # forcibly only take first three columns and strip
            h, r, t = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if h == '' or r == '' or t == '':
                continue
            triples.append((h, r, t))
    return triples


def _build_mappings(*triple_lists: List[Tuple[str, str, str]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    ent2id: Dict[str, int] = {}
    rel2id: Dict[str, int] = {}
    for triples in triple_lists:
        for h, r, t in triples:
            if h not in ent2id:
                ent2id[h] = len(ent2id)
            if t not in ent2id:
                ent2id[t] = len(ent2id)
            if r not in rel2id:
                rel2id[r] = len(rel2id)
    return ent2id, rel2id


def _to_data(triples: List[Tuple[str, str, str]], ent2id: Dict[str, int], rel2id: Dict[str, int]) -> Data:
    if len(triples) == 0:
        raise ValueError('No triples found to build Data object')
    heads = []
    tails = []
    rels = []
    for h, r, t in triples:
        heads.append(ent2id[h])
        tails.append(ent2id[t])
        rels.append(rel2id[r])
    edge_index = torch.tensor([heads, tails], dtype=torch.long, device=device)
    edge_type = torch.tensor(rels, dtype=torch.long, device=device)
    data = Data(edge_index=edge_index)
    data.edge_type = edge_type
    data.num_nodes = len(ent2id)
    data.num_edge_types = len(rel2id)
    return data


def _validate_data(data: Data, split_name: str):
    if data.edge_index.numel() == 0:
        raise ValueError(f'{split_name}: empty edge_index')
    if data.edge_type.numel() == 0:
        raise ValueError(f'{split_name}: empty edge_type')
    if data.edge_index.dtype != torch.long or data.edge_type.dtype != torch.long:
        raise TypeError(f'{split_name}: indices must be torch.long')
    if data.edge_index.min().item() < 0 or data.edge_type.min().item() < 0:
        raise ValueError(f'{split_name}: found negative indices')
    max_h = int(data.edge_index[0].max().item())
    max_t = int(data.edge_index[1].max().item())
    max_r = int(data.edge_type.max().item())
    if max(max_h, max_t) >= int(data.num_nodes) or max_r >= int(data.num_edge_types):
        raise ValueError(
            f'{split_name}: index out of range. '
            f'max_h={max_h}, max_t={max_t}, max_r={max_r}, '
            f'num_nodes={int(data.num_nodes)}, num_relations={int(data.num_edge_types)}')


# Build datasets and capture global counts for model initialization
num_nodes_total = None
num_rel_total = None
if args.dataset_dir is not None:
    ds_dir = args.dataset_dir
    train_path = osp.join(ds_dir, 'train.txt')
    valid_path = osp.join(ds_dir, 'valid.txt')
    if not osp.exists(valid_path):
        valid_path = osp.join(ds_dir, 'val.txt')
    test_path = osp.join(ds_dir, 'test.txt')
    assert osp.exists(train_path) and osp.exists(valid_path) and osp.exists(test_path), 'Missing one of train/valid(test)/test txt files.'

    train_triples = _read_triples(train_path)
    val_triples = _read_triples(valid_path)
    test_triples = _read_triples(test_path)
    ent2id, rel2id = _build_mappings(train_triples, val_triples, test_triples)
    train_data = _to_data(train_triples, ent2id, rel2id)
    val_data = _to_data(val_triples, ent2id, rel2id)
    test_data = _to_data(test_triples, ent2id, rel2id)
    _validate_data(train_data, 'train')
    _validate_data(val_data, 'valid')
    _validate_data(test_data, 'test')
    num_nodes_total = len(ent2id)
    num_rel_total = len(rel2id)
else:
    path = osp.join(osp.dirname(osp.realpath(__file__)), './', 'data', 'FB15k_237')
    train_data = FB15k_237(path, split='train')[0].to(device)
    val_data = FB15k_237(path, split='val')[0].to(device)
    test_data = FB15k_237(path, split='test')[0].to(device)
    num_nodes_total = int(train_data.num_nodes)
    num_rel_total = int(train_data.num_edge_types)

model_arg_map = {'rotate': {'margin': 9.0}}
model = model_map[args.model](
    num_nodes=num_nodes_total,
    num_relations=num_rel_total,
    hidden_channels=50,
    **model_arg_map.get(args.model, {}),
).to(device)

loader = model.loader(
    head_index=train_data.edge_index[0],
    rel_type=train_data.edge_type,
    tail_index=train_data.edge_index[1],
    batch_size=1000,
    shuffle=True,
)

optimizer_map = {
    'transe': optim.Adam(model.parameters(), lr=0.01),
    'transe_f_n': optim.Adam(model.parameters(), lr=0.01),
    'transh': optim.Adam(model.parameters(), lr=0.01),
    'transr': optim.Adam(model.parameters(), lr=0.01),
    'transd': optim.Adam(model.parameters(), lr=0.01),
    'complex': optim.Adagrad(model.parameters(), lr=0.001, weight_decay=1e-6),
    'distmult': optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6),
    'rotate': optim.Adam(model.parameters(), lr=1e-3),
}
optimizer = optimizer_map[args.model]


def train():
    model.train()
    total_loss = total_examples = 0
    for head_index, rel_type, tail_index in loader:
        optimizer.zero_grad()
        loss = model.loss(head_index, rel_type, tail_index)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * head_index.numel()
        total_examples += head_index.numel()
    return total_loss / total_examples


@torch.no_grad()
def compute_val_loss(data):
    # Build a deterministic loader over validation edges
    val_loader = model.loader(
        head_index=data.edge_index[0],
        rel_type=data.edge_type,
        tail_index=data.edge_index[1],
        batch_size=5000,
        shuffle=False,
    )
    model.eval()
    total_loss = 0.0
    total_examples = 0
    for h, r, t in val_loader:
        loss = model.loss(h, r, t)
        total_loss += float(loss) * h.numel()
        total_examples += h.numel()
    return total_loss / max(total_examples, 1)


@torch.no_grad()
def test(data):
    model.eval()
    return model.test(
        head_index=data.edge_index[0],
        rel_type=data.edge_type,
        tail_index=data.edge_index[1],
        batch_size=5000,
        k=10,
    )


start_time = time()
result_dir = '/home/SubgraphRAG/kge/result'
os.makedirs(result_dir, exist_ok=True)

# Init Weights & Biases
dataset_name = osp.basename(args.dataset_dir) if args.dataset_dir is not None else 'FB15k_237'
wandb.init(project='pyg_kge', name=f'{args.model}-{dataset_name}-{datetime.now().strftime("%m%d_%H%M%S")}',
           config={
               'model': args.model,
               'dataset': dataset_name,
               'epochs': args.epochs,
               'eval_every': args.eval_every,
               'num_nodes': num_nodes_total,
               'num_relations': num_rel_total,
           })

# Per-experiment CSV logger
os.makedirs(args.log_dir, exist_ok=True)
experiment_id = f"{args.model}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
exp_log_path = osp.join(args.log_dir, f"{experiment_id}.csv")
with open(exp_log_path, 'w', newline='', encoding='utf-8') as f_log:
    writer = csv.writer(f_log)
    writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_mean_rank', 'val_mrr', 'val_hits@10', 'elapsed_sec'])

    best_mrr = -1.0
    best_epoch = -1
    best_vals = {'rank': None, 'mrr': None, 'hits': None, 'val_loss': None}
    best_ckpt_path = None

    for epoch in tqdm(range(1, args.epochs + 1), desc='Training epochs', ncols=100):
        loss = train()
        elapsed = time() - start_time
        wandb.log({'train/loss': loss, 'epoch': epoch})

        val_rank = val_mrr = val_hits = None
        val_loss = None

        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if do_eval:
            # metrics
            rank, mrr, hits = test(val_data)
            val_rank, val_mrr, val_hits = float(rank), float(mrr), float(hits)
            # loss
            val_loss = compute_val_loss(val_data)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, ValLoss: {val_loss:.4f}, Val Mean Rank: {rank:.2f}, Val MRR: {mrr:.4f}, Val Hits@10: {hits:.4f}')
            wandb.log({'val/loss': val_loss, 'val/mean_rank': val_rank, 'val/mrr': val_mrr, 'val/hits@10': val_hits, 'epoch': epoch})
            # Save only the best model by MRR
            if val_mrr > best_mrr:
                best_mrr = val_mrr
                best_epoch = epoch
                best_vals = {'rank': val_rank, 'mrr': val_mrr, 'hits': val_hits, 'val_loss': val_loss}
                best_ckpt_path = osp.join(result_dir, f'{experiment_id}-best.pt')
                torch.save({
                    'epoch': epoch,
                    'model': args.model,
                    'state_dict': model.state_dict(),
                    'val_mrr': best_mrr,
                    'val_mean_rank': val_rank,
                    'val_hits@10': val_hits,
                    'val_loss': val_loss,
                    'experiment_id': experiment_id,
                }, best_ckpt_path)
        else:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

        writer.writerow([epoch, loss, val_loss, val_rank, val_mrr, val_hits, elapsed])
        f_log.flush()

elapsed = time() - start_time
print(f'Total training time: {elapsed:.2f}s')
if best_ckpt_path is not None:
    print(f'Best model saved: {best_ckpt_path} (epoch={best_epoch}, val_mrr={best_mrr:.4f})')

# Append one-row experiment summary to result CSV
summary_csv = osp.join(result_dir, 'results.csv')
summary_new = not osp.exists(summary_csv)
with open(summary_csv, 'a', newline='', encoding='utf-8') as f_sum:
    w = csv.writer(f_sum)
    if summary_new:
        w.writerow(['experiment_id', 'model', 'dataset', 'best_epoch', 'best_val_mrr', 'best_val_mean_rank', 'best_val_hits@10', 'best_val_loss', 'elapsed_sec', 'best_ckpt_path', 'exp_log_csv'])
    w.writerow([experiment_id, args.model, dataset_name, best_epoch, best_vals['mrr'], best_vals['rank'], best_vals['hits'], best_vals['val_loss'], elapsed, best_ckpt_path, exp_log_path])

rank, mrr, hits_at_10 = test(test_data)
print(f'Test Mean Rank: {rank:.2f}, Test MRR: {mrr:.4f}, '
      f'Test Hits@10: {hits_at_10:.4f}')
wandb.log({'test/mean_rank': float(rank), 'test/mrr': float(mrr), 'test/hits@10': float(hits_at_10)})
wandb.finish()

# python /home/SubgraphRAG/kge/kge_train.py 
# --model transe 
# --dataset_dir /home/SubgraphRAG/kge/data/webqsp_triples 
# --epochs 120
# --log_dir /home/SubgraphRAG/kge/logs