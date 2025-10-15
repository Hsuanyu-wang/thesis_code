import os
import pickle
import numpy as np
from typing import Iterable, Tuple, Set
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

def triples_from_pkl(pkl_path: str) -> np.ndarray:
    with open(pkl_path, 'rb') as f:
        items = pickle.load(f)

    triples_set: Set[Tuple[str, str, str]] = set()
    for ex in items:
        # 還原 entity / relation 標籤
        entity_labels = list(ex['text_entity_list']) + list(ex['non_text_entity_list'])
        rel_labels = list(ex['relation_list'])
        for h_id, r_id, t_id in zip(ex['h_id_list'], ex['r_id_list'], ex['t_id_list']):
            h = entity_labels[h_id]
            r = rel_labels[r_id]
            t = entity_labels[t_id]
            triples_set.add((h, r, t))  # 去重

    triples = np.asarray(list(triples_set), dtype=object)
    return triples

dataset_name = 'webqsp'

train_pkl = f'/home/YX_thesis/retrieve/data_files/{dataset_name}/processed/train.pkl'
val_pkl   = f'/home/YX_thesis/retrieve/data_files/{dataset_name}/processed/val.pkl'
test_pkl  = f'/home/YX_thesis/retrieve/data_files/{dataset_name}/processed/test.pkl'

train_triples = triples_from_pkl(train_pkl)
val_triples   = triples_from_pkl(val_pkl)
test_triples  = triples_from_pkl(test_pkl)

training = TriplesFactory.from_labeled_triples(train_triples)
validation = TriplesFactory.from_labeled_triples(
    val_triples,
    entity_to_id=training.entity_to_id,
    relation_to_id=training.relation_to_id,
)
testing = TriplesFactory.from_labeled_triples(
    test_triples,
    entity_to_id=training.entity_to_id,
    relation_to_id=training.relation_to_id,
)

checkpoint_root = '/home/YX_thesis/pykeen/pykeen_result'
os.makedirs(checkpoint_root, exist_ok=True)

model_name = "TransE"

pipeline_result = pipeline(
    random_seed=42,
    training=training,
    validation=validation,
    testing=testing,
    model=model_name,
    training_loop='sLCWA',
    negative_sampler='basic',
    stopper='early',
    training_kwargs=dict(
        num_epochs=200,  # 減少 epochs，early stopping 會自動停止
        batch_size=1024,  # 增大 batch size 加速訓練
        checkpoint_name=f'{model_name}_{dataset_name}.pt',
        checkpoint_directory=checkpoint_root,
        checkpoint_frequency=200,
        checkpoint_on_failure=True,
    ),
    model_kwargs=dict(
        # scoring_fct_norm=2,  # L2 norm，TransE 標準設定
        embedding_dim=1024,
    ),
    optimizer_kwargs=dict(
        lr=0.01,  # 較高的學習率
    ),
    stopper_kwargs=dict(
        patience=10,  # 10 epochs 無改善就停止
        relative_delta=0.001,  # 0.1% 改善閾值
    ),
)
# pipeline_result.plot()
# pipeline_result.plot_loss()
# pipeline_result.plot_losses()

save_dir = os.path.join(checkpoint_root, f'{model_name}_{dataset_name}')
os.makedirs(save_dir, exist_ok=True)

# 保存 PyKEEN 標準模型檔案（包含 trained_model.pkl）
pipeline_result.save_model(save_dir)

# 另存完整結果（配置與指標）
pipeline_result.save_to_directory(save_dir)

# 顯示保存的內容
print(f'=== 模型與結果已保存到: {save_dir} ===')
print(f'- 主要模型檔: {os.path.join(save_dir, "trained_model.pkl")}')
print(f'- 最佳驗證指標: {pipeline_result.metric_results}')
print(f'- 訓練配置: {pipeline_result.config}')

# 檢查 checkpoint 是否存在
checkpoint_path = os.path.join(checkpoint_root, f'{model_name}_{dataset_name}.pt')
if os.path.exists(checkpoint_path):
    print(f'✓ Checkpoint 已保存: {checkpoint_path}')
else:
    print(f'✗ Checkpoint 未找到: {checkpoint_path}')