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

    triples = np.array(sorted(triples_set), dtype=str)
    return triples

train_pkl = '/home/YX_thesis/retrieve/data_files/webqsp/processed/train.pkl'
val_pkl   = '/home/YX_thesis/retrieve/data_files/webqsp/processed/val.pkl'
test_pkl  = '/home/YX_thesis/retrieve/data_files/webqsp/processed/test.pkl'

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

checkpoint_dir = '/home/pykeen_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

pipeline_result = pipeline(
    training=training,
    validation=validation,
    testing=testing,
    model='Rotate',
    training_loop='sLCWA',
    negative_sampler='basic',
    stopper='early',
    training_kwargs=dict(
        num_epochs=1000,
        checkpoint_name='webqsp_transe.pt',
        checkpoint_directory=checkpoint_dir,
        checkpoint_frequency=5,
        checkpoint_on_failure=True,
    ),
)

save_dir = '/home/pykeen_result/webqsp_transe_from_pkl'
os.makedirs(save_dir, exist_ok=True)
pipeline_result.save_to_directory(save_dir)
print(f'Saved to {save_dir}')