import os
from typing import List, Tuple
import pickle

# This script exports triples ONLY from locally processed PKL files
# expected under a directory containing: train.pkl / val.pkl / test.pkl.


def _write_tsv(triples: List[Tuple[str, str, str]], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")


def _export_split_local(processed_dir: str, split: str, out_dir: str, dedup: bool = True) -> int:
    split_map = {'validation': 'val', 'valid': 'val'}
    split_key = split_map.get(split, split)
    pkl_path = os.path.join(processed_dir, f'{split_key}.pkl')
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f'Processed pkl not found: {pkl_path}')
    with open(pkl_path, 'rb') as f:
        items = pickle.load(f)

    triples_set = set() if dedup else None
    triples: List[Tuple[str, str, str]] = []
    for ex in items:
        entity_labels = list(ex.get('text_entity_list', [])) + list(ex.get('non_text_entity_list', []))
        rel_labels = list(ex.get('relation_list', []))
        h_ids = ex.get('h_id_list', [])
        r_ids = ex.get('r_id_list', [])
        t_ids = ex.get('t_id_list', [])
        for h_id, r_id, t_id in zip(h_ids, r_ids, t_ids):
            h = str(entity_labels[h_id])
            r = str(rel_labels[r_id])
            t = str(entity_labels[t_id])
            if dedup:
                key = (h, r, t)
                if key in triples_set:
                    continue
                triples_set.add(key)
                triples.append(key)
            else:
                triples.append((h, r, t))

    split_name = 'train' if split_key == 'train' else ('valid' if split_key == 'val' else 'test')
    out_path = os.path.join(out_dir, f'{split_name}.txt')
    _write_tsv(triples, out_path)
    return len(triples)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_processed_dir', type=str, required=True,
                        help='Directory with train.pkl / val.pkl / test.pkl produced by local preprocessing')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory to write train.txt, valid.txt, test.txt')
    parser.add_argument('--splits', type=str, default='train,validation,test',
                        help='Comma-separated splits to export (train/validation/test)')
    parser.add_argument('--no_dedup', action='store_true', help='Do not deduplicate triples within each split')
    args = parser.parse_args()

    processed_dir = args.local_processed_dir
    out_dir = args.out_dir
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]

    total = 0
    for split in splits:
        total += _export_split_local(processed_dir, split, out_dir, dedup=not args.no_dedup)
    print(f'Export (local processed) finished. Total triples: {total}. Files written under: {out_dir}')


if __name__ == '__main__':
    main() 
    
# python /home/SubgraphRAG/kge/export_webqsp_triples.py \
#   --local_processed_dir /home/SubgraphRAG/retrieve/data_files/webqsp/processed \
#   --out_dir /home/SubgraphRAG/kge/data/webqsp_triples \
#   --splits train,validation,test