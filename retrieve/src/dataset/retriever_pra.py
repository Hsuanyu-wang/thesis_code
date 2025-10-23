import os
import pickle
from collections import defaultdict

import networkx as nx
import torch
import torch.nn.functional as F
from tqdm import tqdm


class RetrieverDatasetPRA:
    """
    PRA-based weak supervision dataset.
    - Builds a per-sample graph and extracts paths up to a cutoff using PRA-style
      all-simple-paths enumeration, optionally weighting edges by relation frequency.
    - Produces `target_triple_probs` compatible with the existing training pipeline.
    """

    def __init__(
        self,
        config,
        split,
        skip_no_path=True,
        use_pra=True,
        pra_max_path_length=3,
        pra_max_paths=100,
        pra_use_freq_weight=False,
    ):
        dataset_name = config['dataset']['name']
        self.pra_max_path_length = int(pra_max_path_length)
        self.pra_max_paths = int(pra_max_paths)
        self.pra_use_freq_weight = bool(pra_use_freq_weight)
        self.skip_no_path = bool(skip_no_path)

        processed_dict_list = self._load_processed(dataset_name, split)
        triple_score_dict = self._get_triple_scores(
            dataset_name,
            split,
            processed_dict_list,
            self.pra_max_path_length,
            self.pra_max_paths,
            self.pra_use_freq_weight,
        )

        emb_dict = self._load_emb(
            dataset_name, config['dataset']['text_encoder_name'], split
        )

        self._assembly(processed_dict_list, triple_score_dict, emb_dict, self.skip_no_path)

    def _load_processed(self, dataset_name, split):
        processed_file = os.path.join(
            f'data_files/{dataset_name}/processed/{split}.pkl'
        )
        with open(processed_file, 'rb') as f:
            return pickle.load(f)

    def _get_triple_scores(
        self,
        dataset_name,
        split,
        processed_dict_list,
        pra_max_path_length,
        pra_max_paths,
        pra_use_freq_weight,
    ):
        # Save under a dedicated PRA directory; add a suffix when freq weighting is used
        suffix = 'freq' if pra_use_freq_weight else 'plain'
        save_dir = os.path.join('data_files', dataset_name, f'triple_scores_pra_{suffix}')
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, f'{split}.pth')

        if os.path.exists(save_file):
            return torch.load(save_file)

        triple_score_dict = {}
        for sample in tqdm(range(len(processed_dict_list))):
            s = processed_dict_list[sample]
            sample_id = s['id']
            triple_scores, max_path_len = self._extract_pra_paths_and_score(
                s, pra_max_path_length, pra_max_paths, pra_use_freq_weight
            )
            triple_score_dict[sample_id] = {
                'triple_scores': triple_scores,
                'max_path_length': max_path_len,
            }

        torch.save(triple_score_dict, save_file)
        return triple_score_dict

    def _build_graph(self, sample, use_freq_weight=False):
        h_list = sample['h_id_list']
        r_list = sample['r_id_list']
        t_list = sample['t_id_list']
        nx_g = nx.DiGraph()
        if use_freq_weight:
            relation_counts = defaultdict(int)
            for r in r_list:
                relation_counts[r] += 1
        for i, (h, r, t) in enumerate(zip(h_list, r_list, t_list)):
            if use_freq_weight:
                # Higher frequency => lower cost along the path
                w = 1.0 / float(relation_counts[r] + 1)
                nx_g.add_edge(h, t, triple_id=i, relation_id=r, weight=w)
            else:
                nx_g.add_edge(h, t, triple_id=i, relation_id=r)
        return nx_g

    def _extract_pra_paths_and_score(self, sample, max_len, max_paths, use_freq_weight):
        nx_g = self._build_graph(sample, use_freq_weight)

        # Enumerate simple paths up to cutoff between any q and any a
        raw_paths = []  # list of node sequences
        for q_id in sample['q_entity_id_list']:
            for a_id in sample['a_entity_id_list']:
                try:
                    # cutoff is path length in nodes; PRA uses small lengths (<=3 typically)
                    paths_iter = nx.all_simple_paths(nx_g, source=q_id, target=a_id, cutoff=max_len)
                    for p in paths_iter:
                        raw_paths.append(p)
                        if len(raw_paths) >= max_paths:
                            break
                except Exception:
                    pass
                if len(raw_paths) >= max_paths:
                    break
            if len(raw_paths) >= max_paths:
                break

        if not raw_paths:
            return torch.zeros(len(sample['h_id_list'])), None

        # Convert node paths to triple-id paths and compute path scores
        scored_paths = []  # (triple_id_list, score)
        for nodes in raw_paths:
            if len(nodes) < 2:
                continue
            triple_ids = []
            score = 1.0
            valid = True
            for i in range(len(nodes) - 1):
                u, v = nodes[i], nodes[i + 1]
                if not nx_g.has_edge(u, v):
                    valid = False
                    break
                e = nx_g[u][v]
                triple_ids.append(int(e['triple_id']))
                if self.pra_use_freq_weight:
                    score *= float(e.get('weight', 1.0))
            if not valid:
                continue
            scored_paths.append((triple_ids, score))

        if not scored_paths:
            return torch.zeros(len(sample['h_id_list'])), None

        # Score triples by summing path scores (or counts if no weighting)
        num_triples = len(sample['h_id_list'])
        triple_scores = torch.zeros(num_triples)
        for tids, s in scored_paths:
            path_weight = float(s if self.pra_use_freq_weight else 1.0)
            for tid in tids:
                triple_scores[tid] += path_weight

        # Max path length among discovered paths (in number of triples)
        max_path_length = max((len(tids) for tids, _ in scored_paths), default=0)
        return triple_scores, (None if max_path_length == 0 else max_path_length)

    def _load_emb(self, dataset_name, text_encoder_name, split):
        file_path = f'data_files/{dataset_name}/emb/{text_encoder_name}/{split}.pth'
        if os.path.exists(file_path):
            return torch.load(file_path)
        batch_dir = f'data_files/{dataset_name}/emb/{text_encoder_name}'
        if os.path.exists(batch_dir):
            return torch.load(os.path.join(batch_dir, f'{split}.pth'))
        raise FileNotFoundError(f"No embeddings found for {dataset_name}/{text_encoder_name}/{split}")

    def _assembly(self, processed_dict_list, triple_score_dict, emb_dict, skip_no_path):
        self.processed_dict_list = []
        num_relevant_triples = []
        num_skipped = 0

        for i in tqdm(range(len(processed_dict_list))):
            sample = processed_dict_list[i]
            sample_id = sample['id']
            assert sample_id in triple_score_dict

            triple_score_info = triple_score_dict[sample_id]
            triple_scores = triple_score_info['triple_scores']
            max_path_length = triple_score_info['max_path_length']

            num_relevant_triples_i = len(triple_scores.nonzero())
            num_relevant_triples.append(num_relevant_triples_i)

            sample['target_triple_probs'] = triple_scores
            sample['max_path_length'] = max_path_length

            if skip_no_path and (max_path_length in [None, 0]):
                num_skipped += 1
                continue

            sample.update(emb_dict[sample_id])

            sample['a_entity'] = list(set(sample['a_entity']))
            sample['a_entity_id_list'] = list(set(sample['a_entity_id_list']))

            num_entities = len(sample['text_entity_list']) + len(sample['non_text_entity_list'])
            topic_entity_mask = torch.zeros(num_entities)
            topic_entity_mask[sample['q_entity_id_list']] = 1.
            topic_entity_one_hot = F.one_hot(topic_entity_mask.long(), num_classes=2)
            sample['topic_entity_one_hot'] = topic_entity_one_hot.float()

            self.processed_dict_list.append(sample)

        median_num_relevant = int(torch.tensor(num_relevant_triples).median().item()) if num_relevant_triples else 0
        mean_num_relevant = int(torch.tensor(num_relevant_triples).float().mean().item()) if num_relevant_triples else 0
        max_num_relevant = int(max(num_relevant_triples)) if num_relevant_triples else 0

        self.num_skipped = num_skipped
        self.median_num_relevant = median_num_relevant
        self.mean_num_relevant = mean_num_relevant
        self.max_num_relevant = max_num_relevant

        print(f'# skipped samples: {num_skipped}')
        print(f'# relevant triples | median: {median_num_relevant} | mean: {mean_num_relevant} | max: {max_num_relevant}')

    def __len__(self):
        return len(self.processed_dict_list)

    def __getitem__(self, i):
        return self.processed_dict_list[i]


