import argparse
import os.path as osp
import os

import torch
import torch.optim as optim

from torch_geometric.datasets import FB15k_237, WebQSPDataset, CWQDataset
from torch_geometric.nn import ComplEx, DistMult, RotatE, TransE

model_map = {
    'transe': TransE,
    'complex': ComplEx,
    'distmult': DistMult,
    'rotate': RotatE,
}

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=model_map.keys(), type=str.lower,
                    required=True)
parser.add_argument('--dataset', choices=['fb15k237', 'webqsp', 'cwq'],
                    type=str.lower, required=True)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_dir = osp.dirname(osp.realpath(__file__))
dataset_config = {
    'fb15k237': {
        'cls': FB15k_237,
        'path': osp.join(base_dir, '..', 'data', 'FB15k'),
    },
    'webqsp': {
        'cls': WebQSPDataset,
        'path': osp.join(base_dir, 'data', 'webqsp'),
    },
    'cwq': {
        'cls': CWQDataset,
        'path': osp.join(base_dir, 'data', 'cwq'),
    },
}

dataset_entry = dataset_config[args.dataset]
path = dataset_entry['path']
DatasetCls = dataset_entry['cls']

train_data = DatasetCls(path, split='train')[0].to(device)
val_data = DatasetCls(path, split='val')[0].to(device)
test_data = DatasetCls(path, split='test')[0].to(device)

model_arg_map = {'rotate': {'margin': 9.0}}
model = model_map[args.model](
    num_nodes=train_data.num_nodes,
    num_relations=train_data.num_edge_types,
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
def test(data):
    model.eval()
    return model.test(
        head_index=data.edge_index[0],
        rel_type=data.edge_type,
        tail_index=data.edge_index[1],
        batch_size=20000,
        k=10,
    )


best_mrr = float('-inf')
epochs_no_improve = 0

for epoch in range(1, 501):
    loss = train()
    rank, mrr, hits = test(val_data)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Mean Rank: {rank:.2f}, '
          f'Val MRR: {mrr:.4f}, Val Hits@10: {hits:.4f}')

    if mrr > best_mrr:
        best_mrr = mrr
        epochs_no_improve = 0
        save_dir = osp.join(base_dir, 'model')
        os.makedirs(save_dir, exist_ok=True)
        save_path = osp.join(save_dir, f"{args.model}_{args.dataset}_best.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': args.model,
            'dataset': args.dataset,
            'num_nodes': train_data.num_nodes,
            'num_relations': train_data.num_edge_types,
            'hidden_channels': 50,
            'best_val_mrr': best_mrr,
            'epoch': epoch,
        }, save_path)
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= 10:
        print('Early stopping triggered (no improvement in 10 consecutive epochs).')
        break

rank, mrr, hits_at_10 = test(test_data)
print(f'Test Mean Rank: {rank:.2f}, Test MRR: {mrr:.4f}, '
      f'Test Hits@10: {hits_at_10:.4f}')