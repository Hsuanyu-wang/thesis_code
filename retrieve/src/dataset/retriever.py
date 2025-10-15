import networkx as nx
import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import gc
import random
import threading
from tqdm import tqdm

# =============================================================================
# 原始類別 - 保留不變
# =============================================================================
"""
模組層級輔助函數：圖操作與評分
"""
import networkx as nx
import torch
from collections import defaultdict

def build_nx_g(h_id_list, r_id_list, t_id_list):
    """建立標準 NetworkX 圖"""
    nx_g = nx.DiGraph()
    num_triples = len(h_id_list)
    for i in range(num_triples):
        h_i = h_id_list[i]
        r_i = r_id_list[i]
        t_i = t_id_list[i]
        nx_g.add_edge(h_i, t_i, triple_id=i, relation_id=r_i)
    return nx_g

def build_nx_g_with_weights(h_id_list, r_id_list, t_id_list):
    """建立帶頻率權重的 NetworkX 圖（出現越多，路徑代價越低）"""
    nx_g = nx.DiGraph()
    edge_counts = defaultdict(int)
    for h, r, t in zip(h_id_list, r_id_list, t_id_list):
        edge_counts[(h, r, t)] += 1
    edge_weights = {}
    for (h, r, t), count in edge_counts.items():
        weight = 1.0 / (count + 1)
        edge_weights[(h, r, t)] = weight
    num_triples = len(h_id_list)
    for i in range(num_triples):
        h_i = h_id_list[i]
        r_i = r_id_list[i]
        t_i = t_id_list[i]
        nx_g.add_edge(h_i, t_i, triple_id=i, relation_id=r_i, weight=edge_weights[(h_i, r_i, t_i)])
    return nx_g

def build_nx_g_with_inverse_weights(h_id_list, r_id_list, t_id_list):
    """建立逆頻率權重的 NetworkX 圖（出現越多，路徑代價越高）"""
    nx_g = nx.DiGraph()
    edge_counts = defaultdict(int)
    for h, r, t in zip(h_id_list, r_id_list, t_id_list):
        edge_counts[(h, r, t)] += 1
    edge_weights = {}
    for (h, r, t), count in edge_counts.items():
        # 出現次數作為代價，越常見越貴
        weight = float(max(1, count))
        edge_weights[(h, r, t)] = weight
    num_triples = len(h_id_list)
    for i in range(num_triples):
        h_i = h_id_list[i]
        r_i = r_id_list[i]
        t_i = t_id_list[i]
        nx_g.add_edge(h_i, t_i, triple_id=i, relation_id=r_i, weight=edge_weights[(h_i, r_i, t_i)])
    return nx_g

def find_shortest_paths(nx_g, q_entity_id, a_entity_id):
    """找最短路徑（標準版）"""
    try:
        forward_paths = list(nx.all_shortest_paths(nx_g, q_entity_id, a_entity_id))
    except Exception:
        forward_paths = []
    try:
        backward_paths = list(nx.all_shortest_paths(nx_g, a_entity_id, q_entity_id))
    except Exception:
        backward_paths = []
    full_paths = forward_paths + backward_paths
    if (len(forward_paths) == 0) or (len(backward_paths) == 0):
        return full_paths
    min_path_len = min([len(path) for path in full_paths])
    refined_paths = []
    for path in full_paths:
        if len(path) == min_path_len:
            refined_paths.append(path)
    return refined_paths

def find_weighted_shortest_paths(nx_g, q_entity_id, a_entity_id):
    """找加權最短路徑"""
    try:
        forward_paths = list(nx.all_shortest_paths(nx_g, q_entity_id, a_entity_id, weight='weight'))
    except Exception:
        forward_paths = []
    try:
        backward_paths = list(nx.all_shortest_paths(nx_g, a_entity_id, q_entity_id, weight='weight'))
    except Exception:
        backward_paths = []
    full_paths = forward_paths + backward_paths
    if (len(forward_paths) == 0) or (len(backward_paths) == 0):
        return full_paths
    weighted_paths = []
    for path in full_paths:
        weight_sum = 0
        for i in range(len(path) - 1):
            weight_sum += nx_g[path[i]][path[i+1]].get('weight', 1.0)
        weighted_paths.append((path, weight_sum))
    if not weighted_paths:
        return []
    min_weight = min([wp[1] for wp in weighted_paths])
    refined_paths = [wp[0] for wp in weighted_paths if wp[1] == min_weight]
    return refined_paths

def score_triples(path_list, num_triples):
    """標準 triple 評分"""
    triple_scores = torch.zeros(num_triples)
    for path in path_list:
        for triple_id_list in path:
            triple_scores[triple_id_list] = 1.
    return triple_scores

def score_triples_with_weights(path_list, num_triples, nx_g):
    """加權 triple 評分（出現越多，加分越多）"""
    triple_scores = torch.zeros(num_triples)
    for path in path_list:
        for triple_id_list in path:
            for triple_id in triple_id_list:
                triple_scores[triple_id] += 1.0
    return triple_scores

def score_triples_with_inverse_weights(path_list, num_triples, triple_id_to_count):
    """逆頻率 triple 評分（出現越多，加分越少）"""
    triple_scores = torch.zeros(num_triples)
    for path in path_list:
        for triple_id_list in path:
            for triple_id in triple_id_list:
                count = max(1, int(triple_id_to_count.get(triple_id, 1)))
                triple_scores[triple_id] += 1.0 / float(count)
    return triple_scores

class LazyEmbeddingDict:
    """
    懶加載的 embedding 字典：
    - 啟動時建立 sample_id -> batch_idx 的映射（一次掃描）
    - 使用 LRU 緩存載入的批次，節省記憶體
    - 提供 keys()/__contains__/__len__ 介面
    """
    def __init__(self, batch_dir, batch_files, cache_size=8):
        self.batch_dir = batch_dir
        self.batch_files = batch_files
        self._cache = {}
        self._cache_size = cache_size
        self._cache_access_order = []
        # 建立樣本映射
        self._sample_to_batch = {}
        self._build_sample_mapping()

    def _build_sample_mapping(self):
        total_samples = 0
        for batch_idx, batch_file in enumerate(tqdm(self.batch_files, desc="Mapping sample IDs", unit="batch")):
            batch_path = os.path.join(self.batch_dir, batch_file)
            try:
                batch_data = torch.load(batch_path, map_location='cpu')
                for sample_id in batch_data.keys():
                    self._sample_to_batch[sample_id] = batch_idx
                total_samples += len(batch_data)
            except Exception as e:
                tqdm.write(f"Error loading {batch_file}: {e}")
            finally:
                del batch_data
                gc.collect()
        print(f"✅ Created mapping for {len(self._sample_to_batch)} samples from {len(self.batch_files)} batches (avg {total_samples / max(1, len(self.batch_files)):.1f}/batch)")

    def _load_batch_to_cache(self, batch_idx):
        if batch_idx in self._cache:
            # 更新 LRU
            if batch_idx in self._cache_access_order:
                self._cache_access_order.remove(batch_idx)
            self._cache_access_order.append(batch_idx)
            return self._cache[batch_idx]
        batch_file = self.batch_files[batch_idx]
        batch_path = os.path.join(self.batch_dir, batch_file)
        batch_data = torch.load(batch_path, map_location='cpu')
        # LRU 驅逐
        if len(self._cache) >= self._cache_size:
            oldest = self._cache_access_order.pop(0)
            if oldest in self._cache:
                del self._cache[oldest]
        self._cache[batch_idx] = batch_data
        self._cache_access_order.append(batch_idx)
        return batch_data

    def __getitem__(self, sample_id):
        if sample_id not in self._sample_to_batch:
            raise KeyError(f"Sample ID {sample_id} not found in any batch")
        batch_idx = self._sample_to_batch[sample_id]
        batch_data = self._load_batch_to_cache(batch_idx)
        if sample_id not in batch_data:
            raise KeyError(f"Sample {sample_id} not found in batch {batch_idx}")
        return batch_data[sample_id]

    def __contains__(self, sample_id):
        return sample_id in self._sample_to_batch

    def keys(self):
        return self._sample_to_batch.keys()

    def __len__(self):
        return len(self._sample_to_batch)

# =============================================================================
# 原始 RetrieverDataset - 保留不變
# =============================================================================

class RetrieverDataset:
    def __init__(
        self,
        config,
        split,
        skip_no_path=True,
        freq_weight=False,
        freq_weight_inv=False,
        kge_shortest_path=False,
        path_weight=False,
        path_weight_inv=False,
        kge_scorer=None
    ):
        # Load pre-processed data.
        dataset_name = config['dataset']['name']
        processed_dict_list = self._load_processed(dataset_name, split)

        # Extract directed shortest paths from topic entities to answer
        # entities or vice versa as weak supervision signals for triple scoring.
        triple_score_dict = self._get_triple_scores(
            dataset_name, split, processed_dict_list,
            freq_weight, freq_weight_inv,
            kge_shortest_path,
            path_weight, path_weight_inv,
            kge_scorer)

        # Load pre-computed embeddings.
        emb_dict = self._load_emb(
            dataset_name, config['dataset']['text_encoder_name'], split)

        # Put everything together.
        self._assembly(
            processed_dict_list, triple_score_dict, emb_dict, skip_no_path)

    def _load_processed(self, dataset_name, split):
        processed_file = os.path.join(
            f'data_files/{dataset_name}/processed/{split}.pkl')
        with open(processed_file, 'rb') as f:
            return pickle.load(f)

    def _get_triple_scores(
        self,
        dataset_name,
        split,
        processed_dict_list,
        freq_weight=False,
        freq_weight_inv=False,
        kge_shortest_path=False,
        path_weight=False,
        path_weight_inv=False,
        kge_scorer=None
    ):
        # 根據是否使用頻率權重選擇不同的保存目錄
        if path_weight:
            suffix = 'inv' if path_weight_inv else 'count'
            save_dir = os.path.join('data_files', dataset_name, f'triple_scores_path_weight_{suffix}')
        elif kge_shortest_path:
            save_dir = os.path.join('data_files', dataset_name, 'triple_scores_kge_freq_weight')
        elif freq_weight_inv:
            save_dir = os.path.join('data_files', dataset_name, 'triple_scores_freq_weight_inv')
        elif freq_weight:
            save_dir = os.path.join('data_files', dataset_name, 'triple_scores_freq_weight')
        else:
            save_dir = os.path.join('data_files', dataset_name, 'triple_scores')
        
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, f'{split}.pth')

        if os.path.exists(save_file):
            return torch.load(save_file)

        triple_score_dict = dict()
        for i in tqdm(range(len(processed_dict_list))):
            sample_i = processed_dict_list[i]
            sample_i_id = sample_i['id']
            triple_scores_i, max_path_length_i = self._extract_paths_and_score(
                sample_i, freq_weight, freq_weight_inv, kge_shortest_path, path_weight, path_weight_inv, kge_scorer)

            triple_score_dict[sample_i_id] = {
                'triple_scores': triple_scores_i,
                'max_path_length': max_path_length_i
            }

        torch.save(triple_score_dict, save_file)
        return triple_score_dict

    def _extract_paths_and_score(self, sample, freq_weight=False, freq_weight_inv=False, kge_shortest_path=False, path_weight=False, path_weight_inv=False, kge_scorer=None):
        if kge_shortest_path:
            nx_g = self._get_nx_g_with_kge_weights(
                sample['h_id_list'],
                sample['r_id_list'],
                sample['t_id_list'],
                kge_scorer
            )
        elif freq_weight_inv:
            nx_g = self._get_nx_g_with_inverse_weights(
                sample['h_id_list'],
                sample['r_id_list'],
                sample['t_id_list']
            )
        elif freq_weight:
            nx_g = self._get_nx_g_with_weights(
                sample['h_id_list'],
                sample['r_id_list'],
                sample['t_id_list']
            )
        else:
            nx_g = self._get_nx_g(
                sample['h_id_list'],
                sample['r_id_list'],
                sample['t_id_list']
            )

        # Each raw path is a list of entity IDs.
        path_list_ = []
        for q_entity_id in sample['q_entity_id_list']:
            for a_entity_id in sample['a_entity_id_list']:
                if freq_weight or freq_weight_inv or kge_shortest_path or path_weight:
                    paths_q_a = self._weighted_shortest_path(nx_g, q_entity_id, a_entity_id)
                else:
                    paths_q_a = self._shortest_path(nx_g, q_entity_id, a_entity_id)
                if len(paths_q_a) > 0:
                    path_list_.extend(paths_q_a)

        if len(path_list_) == 0:
            max_path_length = None
        else:
            max_path_length = 0

        # Each processed path is a list of triple IDs.
        path_list = []
        for path in path_list_:
            num_triples_path = len(path) - 1
            max_path_length = max(max_path_length, num_triples_path)
            triples_path = []

            for i in range(num_triples_path):
                h_id_i = path[i]
                t_id_i = path[i+1]
                triple_id_i_list = [nx_g[h_id_i][t_id_i]['triple_id']]
                triples_path.append(triple_id_i_list)

            path_list.append(triples_path)

        # Create triple score dict.
        num_triples = len(sample['h_id_list'])
        if path_weight:
            triple_scores = self._score_triples_with_path_weight(path_list, num_triples, inverse=path_weight_inv)
        elif freq_weight or freq_weight_inv:
            triple_scores = self._score_triples_with_weights(
                path_list, num_triples, nx_g)
        else:
            triple_scores = self._score_triples(path_list, num_triples)
        
        return triple_scores, max_path_length

    def _get_nx_g(self, h_id_list, r_id_list, t_id_list):
        nx_g = nx.DiGraph()
        num_triples = len(h_id_list)
        for i in range(num_triples):
            h_i = h_id_list[i]
            r_i = r_id_list[i]
            t_i = t_id_list[i]
            nx_g.add_edge(h_i, t_i, triple_id=i, relation_id=r_i)
        return nx_g

    def _get_nx_g_with_weights(self, h_id_list, r_id_list, t_id_list):
        """建立帶頻率權重的 NetworkX 圖（使用 relation 出現頻率作為權重）"""
        nx_g = nx.DiGraph()
        
        # 先計算每個 relation 的出現頻率
        relation_counts = defaultdict(int)
        for r in r_id_list:
            relation_counts[r] += 1
        
        # relation 出現越多，邊的代價越低（以 1/(count+1) 當作代價）
        num_triples = len(h_id_list)
        for i in range(num_triples):
            h_i = h_id_list[i]
            r_i = r_id_list[i]
            t_i = t_id_list[i]
            weight = 1.0 / float(relation_counts[r_i] + 1)
            nx_g.add_edge(
                h_i,
                t_i,
                triple_id=i,
                relation_id=r_i,
                weight=weight
            )
        
        return nx_g

    def _shortest_path(self, nx_g, q_entity_id, a_entity_id):
        try:
            forward_paths = list(nx.all_shortest_paths(nx_g, q_entity_id, a_entity_id))
        except:
            forward_paths = []
        
        try:
            backward_paths = list(nx.all_shortest_paths(nx_g, a_entity_id, q_entity_id))
        except:
            backward_paths = []
        
        full_paths = forward_paths + backward_paths
        if (len(forward_paths) == 0) or (len(backward_paths) == 0):
            return full_paths
        
        min_path_len = min([len(path) for path in full_paths])
        refined_paths = []
        for path in full_paths:
            if len(path) == min_path_len:
                refined_paths.append(path)
        
        return refined_paths

    def _weighted_shortest_path(self, nx_g, q_entity_id, a_entity_id):
        """使用加權最短路徑"""
        try:
            forward_paths = list(nx.all_shortest_paths(
                nx_g, q_entity_id, a_entity_id, weight='weight'))
        except:
            forward_paths = []
        
        try:
            backward_paths = list(nx.all_shortest_paths(
                nx_g, a_entity_id, q_entity_id, weight='weight'))
        except:
            backward_paths = []
        
        full_paths = forward_paths + backward_paths
        if (len(forward_paths) == 0) or (len(backward_paths) == 0):
            return full_paths
        
        # 計算加權路徑長度
        weighted_paths = []
        for path in full_paths:
            weight_sum = 0
            for i in range(len(path) - 1):
                weight_sum += nx_g[path[i]][path[i+1]]['weight']
            weighted_paths.append((path, weight_sum))
        
        # 選擇權重最小的路徑
        min_weight = min([wp[1] for wp in weighted_paths]) if weighted_paths else float('inf')
        refined_paths = [wp[0] for wp in weighted_paths if wp[1] == min_weight]
        
        return refined_paths

    def _score_triples(self, path_list, num_triples):
        triple_scores = torch.zeros(num_triples)
        for path in path_list:
            for triple_id_list in path:
                triple_scores[triple_id_list] = 1.
        return triple_scores

    def _score_triples_with_weights(self, path_list, num_triples, nx_g):
        """基於路徑權重給 triple 評分"""
        triple_scores = torch.zeros(num_triples)
        for path in path_list:
            for triple_id_list in path:
                for triple_id in triple_id_list:
                    triple_scores[triple_id] += 1.0
        return triple_scores

    def _score_triples_with_path_weight(self, path_list, num_triples, inverse=False):
        """
        Path weight (no KGE dependency): 如果某個 triple t1 出現在多條最短路徑中（例如 a,b,c），
        則把這個計數作為一個路徑權重，並把該權重同時加到這些路徑上的所有 triple。
        具體作法：
        1) 先統計每個 triple 出現於多少條 path：triple_to_path_count。
        2) 對於每條 path，計算該 path 上任意 triple 的最大出現次數 max_count（或其他聚合方式）。
        3) 將 max_count 加到此 path 上所有 triple 的分數上。
        若 inverse=True，則使用 1/(max_count+1) 作為 path 權重（越常見越小）。
        """
        triple_scores = torch.zeros(num_triples)
        # 1) 計數每個 triple 出現於多少條 path
        triple_to_path_count = torch.zeros(num_triples)
        for path in path_list:
            # 這條 path 上包含的 triple set
            triples_in_path = set()
            for triple_id_list in path:
                for triple_id in triple_id_list:
                    triples_in_path.add(int(triple_id))
            for tid in triples_in_path:
                triple_to_path_count[tid] += 1
        # 2) 對每條 path 計算 max_count，並 3) 分配到該 path 的所有 triples
        for path in path_list:
            triples_in_path = []
            for triple_id_list in path:
                for triple_id in triple_id_list:
                    triples_in_path.append(int(triple_id))
            if not triples_in_path:
                continue
            counts = triple_to_path_count[triples_in_path]
            max_count = counts.max().item() if counts.numel() > 0 else 0.0
            if max_count <= 0:
                path_weight = 0.0
            else:
                path_weight = (1.0 / float(max_count + 1.0)) if inverse else float(max_count)
            if path_weight <= 0:
                continue
            for tid in triples_in_path:
                triple_scores[tid] += float(path_weight)
        return triple_scores

    def _load_emb(self, dataset_name, text_encoder_name, split):
        # 首先嘗試載入合併後的文件
        file_path = f'data_files/{dataset_name}/emb/{text_encoder_name}/{split}.pth'
        
        if os.path.exists(file_path):
            print(f"Loading merged embeddings from: {file_path}")
            return torch.load(file_path)
        
        # 如果合併文件不存在，嘗試載入批次文件
        batch_dir = f'data_files/{dataset_name}/emb/{text_encoder_name}'
        if os.path.exists(batch_dir):
            print(f"Loading batch embeddings from: {batch_dir}")
            return self._load_batch_embeddings(batch_dir, split)
        
        raise FileNotFoundError(f"No embeddings found for {dataset_name}/{text_encoder_name}/{split}")

    def _load_batch_embeddings(self, batch_dir, split):
        """使用懶加載方式載入批次 embedding 文件"""
        batch_files = []
        for filename in os.listdir(batch_dir):
            if filename.startswith(f'{split}_batch_') and filename.endswith('.pth'):
                batch_files.append(filename)
        
        if not batch_files:
            raise FileNotFoundError(f"No batch files found for {split} in {batch_dir}")
        
        batch_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        print(f"Found {len(batch_files)} batch files")
        print("Using lazy loading to avoid memory issues")
        
        return LazyEmbeddingDict(batch_dir, batch_files)

    def _assembly(self, processed_dict_list, triple_score_dict, emb_dict, skip_no_path):
        self.processed_dict_list = []
        num_relevant_triples = []
        num_skipped = 0
        
        for i in tqdm(range(len(processed_dict_list))):
            sample_i = processed_dict_list[i]
            sample_i_id = sample_i['id']
            assert sample_i_id in triple_score_dict

            triple_score_i = triple_score_dict[sample_i_id]['triple_scores']
            max_path_length_i = triple_score_dict[sample_i_id]['max_path_length']

            num_relevant_triples_i = len(triple_score_i.nonzero())
            num_relevant_triples.append(num_relevant_triples_i)

            sample_i['target_triple_probs'] = triple_score_i
            sample_i['max_path_length'] = max_path_length_i

            if skip_no_path and (max_path_length_i in [None, 0]):
                num_skipped += 1
                continue

            sample_i.update(emb_dict[sample_i_id])

            sample_i['a_entity'] = list(set(sample_i['a_entity']))
            sample_i['a_entity_id_list'] = list(set(sample_i['a_entity_id_list']))

            # PE for topic entities.
            num_entities_i = len(sample_i['text_entity_list']) + len(sample_i['non_text_entity_list'])
            topic_entity_mask = torch.zeros(num_entities_i)
            topic_entity_mask[sample_i['q_entity_id_list']] = 1.
            topic_entity_one_hot = F.one_hot(topic_entity_mask.long(), num_classes=2)
            sample_i['topic_entity_one_hot'] = topic_entity_one_hot.float()

            self.processed_dict_list.append(sample_i)

        median_num_relevant = int(np.median(num_relevant_triples))
        mean_num_relevant = int(np.mean(num_relevant_triples))
        max_num_relevant = int(np.max(num_relevant_triples))

        # 保存統計信息到實例變量
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

    def _get_nx_g_with_kge_weights(self, h_id_list, r_id_list, t_id_list, kge_scorer):
        """建立帶 KGE 權重的 NetworkX 圖"""
        nx_g = nx.DiGraph()
        
        if kge_scorer is None:
            # 如果沒有 KGE scorer，回退到標準圖
            return self._get_nx_g(h_id_list, r_id_list, t_id_list)
        
        # 計算每個 triple 的 KGE 權重
        num_triples = len(h_id_list)
        for i in range(num_triples):
            h_i = h_id_list[i]
            r_i = r_id_list[i]
            t_i = t_id_list[i]
            
            # 計算這個 triple 的 KGE 權重
            kge_weights = kge_scorer.compute_triple_weights(
                torch.tensor([h_i]), torch.tensor([r_i]), torch.tensor([t_i])
            )
            kge_weight = kge_weights[0].item()
            
            # 將 KGE 權重轉換為邊權重（KGE 分數越高，邊權重越低）
            # 使用 1 / (kge_weight + epsilon) 來避免除零
            epsilon = 1e-6
            edge_weight = 1.0 / (kge_weight + epsilon)
            
            nx_g.add_edge(
                h_i,
                t_i,
                triple_id=i,
                relation_id=r_i,
                weight=edge_weight
            )
        
        return nx_g

    def _get_nx_g_with_inverse_weights(self, h_id_list, r_id_list, t_id_list):
        """建立逆頻率權重的 NetworkX 圖（使用 relation 出現頻率作為代價，越常見越高）"""
        nx_g = nx.DiGraph()
        
        # 計算每個 relation 的出現頻率
        relation_counts = defaultdict(int)
        for r in r_id_list:
            relation_counts[r] += 1
        
        # relation 出現越多，邊的代價越高（以 count 或 count+1 當作代價）
        num_triples = len(h_id_list)
        for i in range(num_triples):
            h_i = h_id_list[i]
            r_i = r_id_list[i]
            t_i = t_id_list[i]
            weight = float(relation_counts[r_i] + 1)
            nx_g.add_edge(
                h_i,
                t_i,
                triple_id=i,
                relation_id=r_i,
                weight=weight
            )
        
        return nx_g

# =============================================================================
# 原始 collate_retriever - 保留不變
# =============================================================================

def collate_retriever(data):
    sample = data[0]
    
    h_id_list = sample['h_id_list']
    h_id_tensor = torch.tensor(h_id_list)
    
    r_id_list = sample['r_id_list']
    r_id_tensor = torch.tensor(r_id_list)
    
    t_id_list = sample['t_id_list']
    t_id_tensor = torch.tensor(t_id_list)
    
    num_non_text_entities = len(sample['non_text_entity_list'])
    
    return h_id_tensor, r_id_tensor, t_id_tensor, sample['q_emb'],\
        sample['entity_embs'], num_non_text_entities, sample['relation_embs'],\
        sample['topic_entity_one_hot'], sample['target_triple_probs'], sample['a_entity_id_list']

# =============================================================================
# 優化的 RandomBatchRetrieverDataset
# =============================================================================

class RandomBatchRetrieverDataset:
    """
    優化的隨機批次採樣 Dataset，使用批次緩存和智能預取
    整合了 RetrieverDataset 的所有功能，但使用批次載入策略
    """
    def __init__(
        self,
        config,
        split,
        skip_no_path=True,
        freq_weight=False,
        samples_per_epoch=20000,
        cache_size=8,  # 緩存8個批次
        prefetch_batches=4  # 預取4個批次
    ):
        self.config = config
        self.split = split
        self.skip_no_path = skip_no_path
        self.freq_weight = freq_weight
        self.samples_per_epoch = samples_per_epoch
        self.cache_size = cache_size
        self.prefetch_batches = prefetch_batches
        
        # 批次緩存
        self._batch_cache = {}  # {batch_idx: batch_data}
        self._cache_access_order = []  # LRU 順序
        
        # 預取相關
        self._prefetch_queue = []
        self._prefetch_lock = threading.Lock()
        
        dataset_name = config['dataset']['name']
        
        # 載入輕量級資料（與 RetrieverDataset 相同）
        self.processed_dict_list = self._load_processed(dataset_name, split)
        self.triple_score_dict = self._get_triple_scores(
            dataset_name, split, self.processed_dict_list, freq_weight)
        
        # 載入嵌入資料（使用統一的載入邏輯）
        self.emb_dict = self._load_emb(
            dataset_name, config['dataset']['text_encoder_name'], split)
        
        # 根據嵌入類型決定載入策略
        if isinstance(self.emb_dict, LazyEmbeddingDict):
            # 使用批次載入
            self.batch_files = self.emb_dict.batch_files
            self.sample_to_batch = self.emb_dict._sample_to_batch
            self.emb_batch_dir = self.emb_dict.batch_dir
        else:
            # 使用完整載入（回退到原始 RetrieverDataset 行為）
            self._create_sample_mapping()
        
        # 過濾有效樣本
        self._filter_valid_samples()
        
        # 統計資訊
        self._compute_statistics()
        
        # 初始化緩存（僅在批次模式下）
        if isinstance(self.emb_dict, LazyEmbeddingDict):
            self._initialize_cache()
    
    def _load_processed(self, dataset_name, split):
        """與 RetrieverDataset 相同的載入邏輯"""
        processed_file = os.path.join(f'data_files/{dataset_name}/processed/{split}.pkl')
        with open(processed_file, 'rb') as f:
            return pickle.load(f)
    
    def _get_triple_scores(self, dataset_name, split, processed_dict_list, freq_weight=False):
        """與 RetrieverDataset 相同的 triple scores 載入邏輯"""
        if freq_weight:
            save_dir = os.path.join('data_files', dataset_name, 'triple_scores_freq_weight')
        else:
            save_dir = os.path.join('data_files', dataset_name, 'triple_scores')
        
        save_file = os.path.join(save_dir, f'{split}.pth')
        if os.path.exists(save_file):
            return torch.load(save_file)
        
        # 如果不存在，則需要計算（這裡複用原始類別的邏輯）
        raise FileNotFoundError(f"Triple scores not found: {save_file}")
    
    def _load_emb(self, dataset_name, text_encoder_name, split):
        """與 RetrieverDataset 相同的嵌入載入邏輯"""
        # 首先嘗試載入合併後的文件
        file_path = f'data_files/{dataset_name}/emb/{text_encoder_name}/{split}.pth'
        
        if os.path.exists(file_path):
            print(f"Loading merged embeddings from: {file_path}")
            return torch.load(file_path)
        
        # 如果合併文件不存在，嘗試載入批次文件
        batch_dir = f'data_files/{dataset_name}/emb/{text_encoder_name}'
        if os.path.exists(batch_dir):
            print(f"Loading batch embeddings from: {batch_dir}")
            return self._load_batch_embeddings(batch_dir, split)
        
        raise FileNotFoundError(f"No embeddings found for {dataset_name}/{text_encoder_name}/{split}")
    
    def _load_batch_embeddings(self, batch_dir, split):
        """與 RetrieverDataset 相同的批次嵌入載入邏輯"""
        batch_files = []
        for filename in os.listdir(batch_dir):
            if filename.startswith(f'{split}_batch_') and filename.endswith('.pth'):
                batch_files.append(filename)
        
        if not batch_files:
            raise FileNotFoundError(f"No batch files found for {split} in {batch_dir}")
        
        batch_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        print(f"Found {len(batch_files)} batch files")
        print("Using lazy loading to avoid memory issues")
        
        return LazyEmbeddingDict(batch_dir, batch_files, cache_size=self.cache_size)
    
    def _create_sample_mapping(self):
        """創建樣本映射（用於完整載入模式）"""
        print("Creating sample ID to batch mapping (lightweight)...")
        self.sample_to_batch = {}
        
        for batch_idx, batch_file in enumerate(self.batch_files):
            batch_path = os.path.join(self.emb_batch_dir, batch_file)
            try:
                batch_data = torch.load(batch_path, map_location='cpu')
                for sample_id in batch_data.keys():
                    self.sample_to_batch[sample_id] = batch_idx
                del batch_data
                gc.collect()
            except Exception as e:
                print(f"Error loading {batch_file} for mapping: {e}")
                continue
        
        print(f"Created lightweight mapping for {len(self.sample_to_batch)} samples")
    
    def _filter_valid_samples(self):
        """過濾掉無效的樣本"""
        self.valid_sample_ids = []
        
        for sample in self.processed_dict_list:
            sample_id = sample['id']
            
            # 檢查是否有對應的嵌入
            if sample_id not in self.sample_to_batch:
                continue
            
            # 檢查是否有有效的路徑
            if sample_id not in self.triple_score_dict:
                continue
                
            max_path_length = self.triple_score_dict[sample_id]['max_path_length']
            if self.skip_no_path and (max_path_length in [None, 0]):
                continue
            
            self.valid_sample_ids.append(sample_id)
        
        print(f"Found {len(self.valid_sample_ids)} valid samples")
    
    def _compute_statistics(self):
        """計算統計資訊"""
        num_relevant_triples = []
        for sample_id in self.valid_sample_ids:
            triple_score = self.triple_score_dict[sample_id]['triple_scores']
            num_relevant_triples_i = len(triple_score.nonzero())
            num_relevant_triples.append(num_relevant_triples_i)
        
        if num_relevant_triples:
            self.median_num_relevant = int(np.median(num_relevant_triples))
            self.mean_num_relevant = int(np.mean(num_relevant_triples))
            self.max_num_relevant = int(np.max(num_relevant_triples))
        else:
            self.median_num_relevant = 0
            self.mean_num_relevant = 0
            self.max_num_relevant = 0
        
        self.num_skipped = len(self.processed_dict_list) - len(self.valid_sample_ids)
        
        print(f'# skipped samples: {self.num_skipped}')
        print(f'# relevant triples | median: {self.median_num_relevant} | mean: {self.mean_num_relevant} | max: {self.max_num_relevant}')
    
    def _initialize_cache(self):
        """初始化緩存，載入前幾個批次"""
        print("Initializing batch cache...")
        
        # 隨機選擇要緩存的批次
        available_batches = list(range(len(self.batch_files)))
        random.shuffle(available_batches)
        
        for i in range(min(self.cache_size, len(available_batches))):
            batch_idx = available_batches[i]
            self._load_batch_to_cache(batch_idx)
        
        print(f"Initialized cache with {len(self._batch_cache)} batches")
    
    def _load_batch_to_cache(self, batch_idx):
        """載入批次到緩存"""
        if batch_idx in self._batch_cache:
            # 更新LRU順序
            if batch_idx in self._cache_access_order:
                self._cache_access_order.remove(batch_idx)
            self._cache_access_order.append(batch_idx)
            return
        
        batch_file = self.batch_files[batch_idx]
        batch_path = os.path.join(self.emb_batch_dir, batch_file)
        
        try:
            batch_data = torch.load(batch_path, map_location='cpu')
            self._batch_cache[batch_idx] = batch_data
            
            # 更新LRU順序
            if batch_idx in self._cache_access_order:
                self._cache_access_order.remove(batch_idx)
            self._cache_access_order.append(batch_idx)
            
            # 如果緩存滿了，移除最舊的
            if len(self._batch_cache) > self.cache_size:
                oldest_batch = self._cache_access_order.pop(0)
                del self._batch_cache[oldest_batch]
                
        except Exception as e:
            print(f"Error loading batch {batch_idx}: {e}")
    
    def _get_random_sample_from_cache(self):
        """從緩存中隨機選擇樣本"""
        # 隨機選擇一個緩存的批次
        cached_batches = list(self._batch_cache.keys())
        if not cached_batches:
            # 如果緩存為空，隨機載入一個批次
            batch_idx = random.randint(0, len(self.batch_files) - 1)
            self._load_batch_to_cache(batch_idx)
            cached_batches = [batch_idx]
        
        batch_idx = random.choice(cached_batches)
        batch_data = self._batch_cache[batch_idx]
        
        # 從批次中隨機選擇樣本
        available_samples = [sid for sid in batch_data.keys() 
                           if sid in self.valid_sample_ids]
        
        if not available_samples:
            # 如果批次中沒有有效樣本，遞歸調用
            return self._get_random_sample_from_cache()
        
        sample_id = random.choice(available_samples)
        return sample_id, batch_data[sample_id]
    
    def _prepare_sample(self, sample_id, sample_emb_data):
        """準備樣本資料（與 RetrieverDataset 相同的邏輯）"""
        # 找到對應的 processed sample
        sample = None
        for s in self.processed_dict_list:
            if s['id'] == sample_id:
                sample = s.copy()
                break
        
        if sample is None:
            raise KeyError(f"Sample {sample_id} not found in processed data")
        
        # 添加 triple scores
        triple_score_info = self.triple_score_dict[sample_id]
        sample['target_triple_probs'] = triple_score_info['triple_scores']
        sample['max_path_length'] = triple_score_info['max_path_length']
        
        # 添加嵌入資料
        sample.update(sample_emb_data)
        
        # 清理重複的 answer entities
        sample['a_entity'] = list(set(sample['a_entity']))
        sample['a_entity_id_list'] = list(set(sample['a_entity_id_list']))
        
        # PE for topic entities
        num_entities = len(sample['text_entity_list']) + len(sample['non_text_entity_list'])
        topic_entity_mask = torch.zeros(num_entities)
        topic_entity_mask[sample['q_entity_id_list']] = 1.
        topic_entity_one_hot = F.one_hot(topic_entity_mask.long(), num_classes=2)
        sample['topic_entity_one_hot'] = topic_entity_one_hot.float()
        
        return sample
    
    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, idx):
        if isinstance(self.emb_dict, LazyEmbeddingDict):
            # 使用批次緩存模式
            sample_id, sample_emb_data = self._get_random_sample_from_cache()
            sample = self._prepare_sample(sample_id, sample_emb_data)
            
            # 異步預取新批次（可選）
            if random.random() < 0.1:  # 10% 概率觸發預取
                self._async_prefetch()
            
            return sample
        else:
            # 回退到完整載入模式（與 RetrieverDataset 相同）
            sample_id = random.choice(self.valid_sample_ids)
            sample_emb_data = self.emb_dict[sample_id]
            return self._prepare_sample(sample_id, sample_emb_data)
    
    def _async_prefetch(self):
        """異步預取新批次"""
        if len(self._batch_cache) < self.cache_size:
            # 隨機選擇一個未緩存的批次
            all_batches = set(range(len(self.batch_files)))
            cached_batches = set(self._batch_cache.keys())
            uncached_batches = list(all_batches - cached_batches)
            
            if uncached_batches:
                batch_idx = random.choice(uncached_batches)
                self._load_batch_to_cache(batch_idx)

class SmartBatchRetrieverDataset:
    """
    智能批次Dataset：根據資料格式自動選擇最佳策略
    保持與RetrieverDataset完全相同的行為
    """
    def __init__(
        self,
        config,
        split,
        skip_no_path=True,
        freq_weight=False,
        cache_size=8
    ):
        self.config = config
        self.split = split
        self.skip_no_path = skip_no_path
        self.freq_weight = freq_weight
        self.cache_size = cache_size
        
        dataset_name = config['dataset']['name']
        
        # 載入輕量級資料
        self.processed_dict_list = self._load_processed(dataset_name, split)
        self.triple_score_dict = self._get_triple_scores(
            dataset_name, split, self.processed_dict_list, freq_weight)
        
        # 智能嵌入載入策略
        self.emb_dict = self._smart_load_embeddings(
            dataset_name, config['dataset']['text_encoder_name'], split)
        
        # 根據載入結果決定行為模式
        if isinstance(self.emb_dict, LazyEmbeddingDict):
            # 批次模式：需要特殊處理
            self._setup_batch_mode()
        else:
            # 完整模式：與RetrieverDataset相同
            self._setup_full_mode()
        
        # 組裝資料
        self._assembly(processed_dict_list, triple_score_dict, emb_dict, skip_no_path)
    
    def _smart_load_embeddings(self, dataset_name, text_encoder_name, split):
        """智能載入策略：自動檢測最佳載入方式"""
        # 1. 檢查是否有合併檔案
        merged_file = f'data_files/{dataset_name}/emb/{text_encoder_name}/{split}.pth'
        if os.path.exists(merged_file):
            print(f"✅ Found merged file, using full loading: {merged_file}")
            return torch.load(merged_file)
        
        # 2. 檢查批次檔案
        batch_dir = f'data_files/{dataset_name}/emb/{text_encoder_name}'
        if os.path.exists(batch_dir):
            batch_files = [f for f in os.listdir(batch_dir) 
                          if f.startswith(f'{split}_batch_') and f.endswith('.pth')]
            if batch_files:
                print(f"✅ Found {len(batch_files)} batch files, using batch loading")
                return LazyEmbeddingDict(batch_dir, batch_files, cache_size=self.cache_size)
        
        raise FileNotFoundError(f"No embeddings found for {dataset_name}/{text_encoder_name}/{split}")
    
    def _setup_batch_mode(self):
        """設置批次模式：模擬DataLoader的shuffle行為"""
        # 建立樣本索引映射
        self.sample_to_batch = self.emb_dict._sample_to_batch
        self.valid_sample_ids = self._filter_valid_samples()
        
        # 建立樣本索引列表（用於DataLoader shuffle）
        self.sample_indices = list(range(len(self.valid_sample_ids)))
        
        print(f"Batch mode: {len(self.valid_sample_ids)} valid samples in {len(self.emb_dict.batch_files)} batches")
    
    def _setup_full_mode(self):
        """設置完整模式：與RetrieverDataset相同"""
        self.valid_sample_ids = self._filter_valid_samples()
        print(f"Full mode: {len(self.valid_sample_ids)} valid samples loaded")
    
    def _filter_valid_samples(self):
        """過濾有效樣本"""
        valid_ids = []
        for sample in self.processed_dict_list:
            sample_id = sample['id']
            
            # 檢查嵌入是否存在
            if sample_id not in self.sample_to_batch:
                continue
            
            # 檢查路徑是否有效
            if sample_id not in self.triple_score_dict:
                continue
                
            max_path_length = self.triple_score_dict[sample_id]['max_path_length']
            if self.skip_no_path and (max_path_length in [None, 0]):
                continue
            
            valid_ids.append(sample_id)
        
        return valid_ids
    
    def __len__(self):
        """返回有效樣本數量"""
        return len(self.valid_sample_ids)
    
    def __getitem__(self, idx):
        """按索引存取樣本，讓DataLoader處理shuffle"""
        if isinstance(self.emb_dict, LazyEmbeddingDict):
            # 批次模式：通過索引取得樣本ID
            sample_id = self.valid_sample_ids[idx]
            sample_emb_data = self.emb_dict[sample_id]
        else:
            # 完整模式：直接存取
            sample_id = self.valid_sample_ids[idx]
            sample_emb_data = self.emb_dict[sample_id]
        
        return self._prepare_sample(sample_id, sample_emb_data)
    
    def _prepare_sample(self, sample_id, sample_emb_data):
        """準備樣本資料（與RetrieverDataset相同）"""
        # 找到對應的processed sample
        sample = None
        for s in self.processed_dict_list:
            if s['id'] == sample_id:
                sample = s.copy()
                break
        
        if sample is None:
            raise KeyError(f"Sample {sample_id} not found")
        
        # 添加triple scores
        triple_score_info = self.triple_score_dict[sample_id]
        sample['target_triple_probs'] = triple_score_info['triple_scores']
        sample['max_path_length'] = triple_score_info['max_path_length']
        
        # 添加嵌入資料
        sample.update(sample_emb_data)
        
        # 清理和格式化
        sample['a_entity'] = list(set(sample['a_entity']))
        sample['a_entity_id_list'] = list(set(sample['a_entity_id_list']))
        
        # PE for topic entities
        num_entities = len(sample['text_entity_list']) + len(sample['non_text_entity_list'])
        topic_entity_mask = torch.zeros(num_entities)
        topic_entity_mask[sample['q_entity_id_list']] = 1.
        topic_entity_one_hot = F.one_hot(topic_entity_mask.long(), num_classes=2)
        sample['topic_entity_one_hot'] = topic_entity_one_hot.float()
        
        return sample

# =============================================================================
# 優化的 collate 函數
# =============================================================================

def collate_retriever_batch(data):
    """支持真正批次處理的 collate 函數"""
    batch_size = len(data)
    
    if batch_size == 1:
        # 單樣本模式，直接返回
        sample = data[0]
        h_id_tensor = torch.tensor(sample['h_id_list'])
        r_id_tensor = torch.tensor(sample['r_id_list'])
        t_id_tensor = torch.tensor(sample['t_id_list'])
        num_non_text_entities = len(sample['non_text_entity_list'])
        
        return (h_id_tensor, r_id_tensor, t_id_tensor, 
                sample['q_emb'],
                sample['entity_embs'], 
                num_non_text_entities,
                sample['relation_embs'],
                sample['topic_entity_one_hot'],
                sample['target_triple_probs'], 
                sample['a_entity_id_list'])
    
    else:
        # 多樣本批次模式
        h_id_tensors = []
        r_id_tensors = []
        t_id_tensors = []
        q_embs = []
        entity_embs_list = []
        relation_embs_list = []
        topic_entity_one_hots = []
        target_triple_probs_list = []
        a_entity_id_lists = []
        num_non_text_entities_list = []
        
        for sample in data:
            h_id_tensors.append(torch.tensor(sample['h_id_list']))
            r_id_tensors.append(torch.tensor(sample['r_id_list']))
            t_id_tensors.append(torch.tensor(sample['t_id_list']))
            q_embs.append(sample['q_emb'])
            entity_embs_list.append(sample['entity_embs'])
            relation_embs_list.append(sample['relation_embs'])
            topic_entity_one_hots.append(sample['topic_entity_one_hot'])
            target_triple_probs_list.append(sample['target_triple_probs'])
            a_entity_id_lists.append(sample['a_entity_id_list'])
            num_non_text_entities_list.append(len(sample['non_text_entity_list']))
        
        return {
            'h_id_tensors': h_id_tensors,
            'r_id_tensors': r_id_tensors,
            't_id_tensors': t_id_tensors,
            'q_embs': q_embs,
            'entity_embs_list': entity_embs_list,
            'relation_embs_list': relation_embs_list,
            'topic_entity_one_hots': topic_entity_one_hots,
            'target_triple_probs_list': target_triple_probs_list,
            'a_entity_id_lists': a_entity_id_lists,
            'num_non_text_entities_list': num_non_text_entities_list
        }


# =============================================================================
# 改進的 RandomBatchRetrieverDataset - 解決三個關鍵問題
# =============================================================================

class ImprovedRandomBatchRetrieverDataset:
    """
    改進的隨機批次採樣 Dataset，解決以下問題：
    1. 確保每個epoch覆蓋所有數據
    2. 提高批次載入效率（一次載入多個樣本）
    3. 保留RetrieverDataset的所有功能
    """
    def __init__(
        self,
        config,
        split,
        skip_no_path=True,
        freq_weight=False,
        samples_per_epoch=None,  # 如果為None，則使用所有有效樣本
        cache_size=8,
        batch_loading_size=32  # 每次從磁盤載入的樣本數量
    ):
        self.config = config
        self.split = split
        self.skip_no_path = skip_no_path
        self.freq_weight = freq_weight
        self.cache_size = cache_size
        self.batch_loading_size = batch_loading_size
        
        dataset_name = config['dataset']['name']
        
        # 載入輕量級資料（與 RetrieverDataset 相同）
        self.processed_dict_list = self._load_processed(dataset_name, split)
        self.triple_score_dict = self._get_triple_scores(
            dataset_name, split, self.processed_dict_list, freq_weight)
        
        # 載入嵌入資料（使用統一的載入邏輯）
        self.emb_dict = self._load_emb(
            dataset_name, config['dataset']['text_encoder_name'], split)
        
        # 根據嵌入類型決定載入策略
        if isinstance(self.emb_dict, LazyEmbeddingDict):
            # 使用批次載入
            self.batch_files = self.emb_dict.batch_files
            self.sample_to_batch = self.emb_dict._sample_to_batch
            self.emb_batch_dir = self.emb_dict.batch_dir
        else:
            # 使用完整載入（回退到原始 RetrieverDataset 行為）
            self._create_sample_mapping()
        
        # 過濾有效樣本
        self.valid_sample_ids = self._filter_valid_samples()
        
        # 計算統計資訊
        self._compute_statistics()
        
        # 設置epoch樣本數量
        if samples_per_epoch is None:
            self.samples_per_epoch = len(self.valid_sample_ids)
        else:
            self.samples_per_epoch = min(samples_per_epoch, len(self.valid_sample_ids))
        
        # 初始化epoch追蹤
        self._current_epoch = 0
        self._epoch_sample_indices = []
        self._epoch_sample_counter = 0
        
        # 批次緩存
        self._batch_cache = {}  # {batch_idx: batch_data}
        self._cache_access_order = []  # LRU 順序
        
        # 樣本緩存（用於批次載入）
        self._sample_cache = {}  # {sample_id: sample_data}
        self._sample_cache_order = []  # LRU 順序
        self._max_sample_cache_size = cache_size * batch_loading_size
        
        # 初始化緩存（僅在批次模式下）
        if isinstance(self.emb_dict, LazyEmbeddingDict):
            self._initialize_cache()
        
        print(f"✅ ImprovedRandomBatchRetrieverDataset initialized")
        print(f"📊 Valid samples: {len(self.valid_sample_ids)}")
        print(f"📊 Samples per epoch: {self.samples_per_epoch}")
        print(f"📊 Batch loading size: {self.batch_loading_size}")
    
    def _load_processed(self, dataset_name, split):
        """與 RetrieverDataset 相同的載入邏輯"""
        processed_file = os.path.join(f'data_files/{dataset_name}/processed/{split}.pkl')
        with open(processed_file, 'rb') as f:
            return pickle.load(f)
    
    def _get_triple_scores(self, dataset_name, split, processed_dict_list, freq_weight=False):
        """與 RetrieverDataset 相同的 triple scores 載入邏輯"""
        if freq_weight:
            save_dir = os.path.join('data_files', dataset_name, 'triple_scores_freq_weight')
        else:
            save_dir = os.path.join('data_files', dataset_name, 'triple_scores')
        
        save_file = os.path.join(save_dir, f'{split}.pth')
        if os.path.exists(save_file):
            return torch.load(save_file)
        
        # 如果不存在，則需要計算（這裡複用原始類別的邏輯）
        raise FileNotFoundError(f"Triple scores not found: {save_file}")
    
    def _load_emb(self, dataset_name, text_encoder_name, split):
        """與 RetrieverDataset 相同的嵌入載入邏輯"""
        # 首先嘗試載入合併後的文件
        file_path = f'data_files/{dataset_name}/emb/{text_encoder_name}/{split}.pth'
        
        if os.path.exists(file_path):
            print(f"Loading merged embeddings from: {file_path}")
            return torch.load(file_path)
        
        # 如果合併文件不存在，嘗試載入批次文件
        batch_dir = f'data_files/{dataset_name}/emb/{text_encoder_name}'
        if os.path.exists(batch_dir):
            print(f"Loading batch embeddings from: {batch_dir}")
            return self._load_batch_embeddings(batch_dir, split)
        
        raise FileNotFoundError(f"No embeddings found for {dataset_name}/{text_encoder_name}/{split}")
    
    def _load_batch_embeddings(self, batch_dir, split):
        """與 RetrieverDataset 相同的批次嵌入載入邏輯"""
        batch_files = []
        for filename in os.listdir(batch_dir):
            if filename.startswith(f'{split}_batch_') and filename.endswith('.pth'):
                batch_files.append(filename)
        
        if not batch_files:
            raise FileNotFoundError(f"No batch files found for {split} in {batch_dir}")
        
        batch_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        print(f"Found {len(batch_files)} batch files")
        print("Using lazy loading to avoid memory issues")
        
        return LazyEmbeddingDict(batch_dir, batch_files, cache_size=self.cache_size)
    
    def _create_sample_mapping(self):
        """創建樣本映射（用於完整載入模式）"""
        print("Creating sample ID to batch mapping (lightweight)...")
        self.sample_to_batch = {}
        
        for batch_idx, batch_file in enumerate(self.batch_files):
            batch_path = os.path.join(self.emb_batch_dir, batch_file)
            try:
                batch_data = torch.load(batch_path, map_location='cpu')
                for sample_id in batch_data.keys():
                    self.sample_to_batch[sample_id] = batch_idx
                del batch_data
                gc.collect()
            except Exception as e:
                print(f"Error loading {batch_file} for mapping: {e}")
                continue
        
        print(f"Created lightweight mapping for {len(self.sample_to_batch)} samples")
    
    def _filter_valid_samples(self):
        """過濾掉無效的樣本"""
        valid_ids = []
        
        for sample in self.processed_dict_list:
            sample_id = sample['id']
            
            # 檢查是否有對應的嵌入
            if sample_id not in self.sample_to_batch:
                continue
            
            # 檢查是否有有效的路徑
            if sample_id not in self.triple_score_dict:
                continue
                
            max_path_length = self.triple_score_dict[sample_id]['max_path_length']
            if self.skip_no_path and (max_path_length in [None, 0]):
                continue
            
            valid_ids.append(sample_id)
        
        print(f"Found {len(valid_ids)} valid samples")
        return valid_ids
    
    def _compute_statistics(self):
        """計算統計資訊"""
        num_relevant_triples = []
        for sample_id in self.valid_sample_ids:
            triple_score = self.triple_score_dict[sample_id]['triple_scores']
            num_relevant_triples_i = len(triple_score.nonzero())
            num_relevant_triples.append(num_relevant_triples_i)
        
        if num_relevant_triples:
            self.median_num_relevant = int(np.median(num_relevant_triples))
            self.mean_num_relevant = int(np.mean(num_relevant_triples))
            self.max_num_relevant = int(np.max(num_relevant_triples))
        else:
            self.median_num_relevant = 0
            self.mean_num_relevant = 0
            self.max_num_relevant = 0
        
        self.num_skipped = len(self.processed_dict_list) - len(self.valid_sample_ids)
        
        print(f'# skipped samples: {self.num_skipped}')
        print(f'# relevant triples | median: {self.median_num_relevant} | mean: {self.mean_num_relevant} | max: {self.max_num_relevant}')
    
    def _initialize_cache(self):
        """初始化緩存，載入前幾個批次"""
        print("Initializing batch cache...")
        
        # 隨機選擇要緩存的批次
        available_batches = list(range(len(self.batch_files)))
        random.shuffle(available_batches)
        
        for i in range(min(self.cache_size, len(available_batches))):
            batch_idx = available_batches[i]
            self._load_batch_to_cache(batch_idx)
        
        print(f"Initialized cache with {len(self._batch_cache)} batches")
    
    def _load_batch_to_cache(self, batch_idx):
        """載入批次到緩存"""
        if batch_idx in self._batch_cache:
            # 更新LRU順序
            if batch_idx in self._cache_access_order:
                self._cache_access_order.remove(batch_idx)
            self._cache_access_order.append(batch_idx)
            return
        
        batch_file = self.batch_files[batch_idx]
        batch_path = os.path.join(self.emb_batch_dir, batch_file)
        
        try:
            batch_data = torch.load(batch_path, map_location='cpu')
            self._batch_cache[batch_idx] = batch_data
            
            # 更新LRU順序
            if batch_idx in self._cache_access_order:
                self._cache_access_order.remove(batch_idx)
            self._cache_access_order.append(batch_idx)
            
            # 如果緩存滿了，移除最舊的
            if len(self._batch_cache) > self.cache_size:
                oldest_batch = self._cache_access_order.pop(0)
                del self._batch_cache[oldest_batch]
                
        except Exception as e:
            print(f"Error loading batch {batch_idx}: {e}")
    
    def _start_new_epoch(self):
        """開始新的epoch，確保覆蓋所有數據"""
        self._current_epoch += 1
        self._epoch_sample_counter = 0
        
        # 創建本epoch的樣本索引列表，確保覆蓋所有有效樣本
        if self.samples_per_epoch >= len(self.valid_sample_ids):
            # 如果需要的樣本數 >= 有效樣本數，則使用所有樣本並重複
            base_indices = list(range(len(self.valid_sample_ids)))
            # 重複直到達到samples_per_epoch
            self._epoch_sample_indices = []
            while len(self._epoch_sample_indices) < self.samples_per_epoch:
                remaining = self.samples_per_epoch - len(self._epoch_sample_indices)
                self._epoch_sample_indices.extend(base_indices[:remaining])
        else:
            # 如果需要的樣本數 < 有效樣本數，則隨機選擇
            self._epoch_sample_indices = random.sample(
                list(range(len(self.valid_sample_ids))), 
                self.samples_per_epoch
            )
        
        # 打亂順序
        random.shuffle(self._epoch_sample_indices)
        
        print(f"🔄 Started epoch {self._current_epoch} with {len(self._epoch_sample_indices)} samples")
    
    def _load_samples_batch(self, sample_ids):
        """批次載入多個樣本，提高效率"""
        loaded_samples = {}
        
        if isinstance(self.emb_dict, LazyEmbeddingDict):
            # 批次模式：按批次分組載入
            batch_groups = defaultdict(list)
            for sample_id in sample_ids:
                if sample_id in self.sample_to_batch:
                    batch_idx = self.sample_to_batch[sample_id]
                    batch_groups[batch_idx].append(sample_id)
            
            # 載入每個批次
            for batch_idx, batch_sample_ids in batch_groups.items():
                if batch_idx not in self._batch_cache:
                    self._load_batch_to_cache(batch_idx)
                
                batch_data = self._batch_cache[batch_idx]
                for sample_id in batch_sample_ids:
                    if sample_id in batch_data:
                        loaded_samples[sample_id] = batch_data[sample_id]
        else:
            # 完整模式：直接載入
            for sample_id in sample_ids:
                if sample_id in self.emb_dict:
                    loaded_samples[sample_id] = self.emb_dict[sample_id]
        
        return loaded_samples
    
    def _get_samples_efficiently(self, num_samples):
        """高效獲取多個樣本"""
        # 確定需要載入的樣本ID
        sample_ids_to_load = []
        samples_to_return = []
        
        # 首先檢查樣本緩存
        for i in range(num_samples):
            if self._epoch_sample_counter >= len(self._epoch_sample_indices):
                self._start_new_epoch()
            
            sample_idx = self._epoch_sample_indices[self._epoch_sample_counter]
            sample_id = self.valid_sample_ids[sample_idx]
            self._epoch_sample_counter += 1
            
            if sample_id in self._sample_cache:
                # 從緩存中獲取
                samples_to_return.append((sample_id, self._sample_cache[sample_id]))
                # 更新LRU順序
                if sample_id in self._sample_cache_order:
                    self._sample_cache_order.remove(sample_id)
                self._sample_cache_order.append(sample_id)
            else:
                # 需要載入
                sample_ids_to_load.append(sample_id)
        
        # 批次載入需要的樣本
        if sample_ids_to_load:
            loaded_samples = self._load_samples_batch(sample_ids_to_load)
            
            # 將載入的樣本加入緩存
            for sample_id, sample_data in loaded_samples.items():
                # 更新緩存
                if len(self._sample_cache) >= self._max_sample_cache_size:
                    # 移除最舊的緩存項
                    oldest_sample = self._sample_cache_order.pop(0)
                    del self._sample_cache[oldest_sample]
                
                self._sample_cache[sample_id] = sample_data
                self._sample_cache_order.append(sample_id)
            
            # 重新構建返回列表
            samples_to_return = []
            for i in range(num_samples):
                sample_idx = self._epoch_sample_indices[self._epoch_sample_counter - num_samples + i]
                sample_id = self.valid_sample_ids[sample_idx]
                if sample_id in self._sample_cache:
                    samples_to_return.append((sample_id, self._sample_cache[sample_id]))
        
        return samples_to_return
    
    def _prepare_sample(self, sample_id, sample_emb_data):
        """準備樣本資料（與 RetrieverDataset 相同的邏輯）"""
        # 找到對應的 processed sample
        sample = None
        for s in self.processed_dict_list:
            if s['id'] == sample_id:
                sample = s.copy()
                break
        
        if sample is None:
            raise KeyError(f"Sample {sample_id} not found in processed data")
        
        # 添加 triple scores
        triple_score_info = self.triple_score_dict[sample_id]
        sample['target_triple_probs'] = triple_score_info['triple_scores']
        sample['max_path_length'] = triple_score_info['max_path_length']
        
        # 添加嵌入資料
        sample.update(sample_emb_data)
        
        # 清理重複的 answer entities
        sample['a_entity'] = list(set(sample['a_entity']))
        sample['a_entity_id_list'] = list(set(sample['a_entity_id_list']))
        
        # PE for topic entities
        num_entities = len(sample['text_entity_list']) + len(sample['non_text_entity_list'])
        topic_entity_mask = torch.zeros(num_entities)
        topic_entity_mask[sample['q_entity_id_list']] = 1.
        topic_entity_one_hot = F.one_hot(topic_entity_mask.long(), num_classes=2)
        sample['topic_entity_one_hot'] = topic_entity_one_hot.float()
        
        return sample
    
    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, idx):
        # 如果這是新的epoch開始，重新初始化
        if self._epoch_sample_counter == 0 and len(self._epoch_sample_indices) == 0:
            self._start_new_epoch()
        
        # 高效獲取樣本（一次載入多個以提高效率）
        samples = self._get_samples_efficiently(1)
        sample_id, sample_emb_data = samples[0]
        
        return self._prepare_sample(sample_id, sample_emb_data)
    
    def get_batch_samples(self, batch_size):
        """獲取批次樣本，進一步提高效率"""
        if self._epoch_sample_counter == 0 and len(self._epoch_sample_indices) == 0:
            self._start_new_epoch()
        
        # 高效獲取多個樣本
        samples = self._get_samples_efficiently(batch_size)
        
        # 準備所有樣本
        prepared_samples = []
        for sample_id, sample_emb_data in samples:
            prepared_samples.append(self._prepare_sample(sample_id, sample_emb_data))
        
        return prepared_samples

# =============================================================================
# latest issue
# =============================================================================

class OptimizedRetrieverDataset:
    """
    真正符合 PyTorch 設計理念的優化版 Dataset。
    - __init__: 只做最輕量級的初始化 (讀取 metadata)。
    - __getitem__: 負責載入並準備一個完整的樣本。
    將批次處理、Shuffle、多線程預取等工作完全交給 DataLoader。
    """
    def __init__(self, config, split, skip_no_path=True, freq_weight=False, weight_mode='none'):
        # --- 初始化階段：只載入輕量級 metadata ---
        print(f"Initializing {split} set (lightweight)...")
        self.config = config
        self.split = split
        self.skip_no_path = skip_no_path
        # 兼容舊參數：若 weight_mode 未指定，根據 freq_weight 推導
        if weight_mode == 'none' and freq_weight:
            weight_mode = 'freq'
        self.weight_mode = weight_mode  # 'none' | 'freq' | 'inv' | 'spcount' | 'spcount_inv'
        self.freq_weight = (self.weight_mode == 'freq')
        dataset_name = config['dataset']['name']

        # 1. 載入 processed data (通常是 .pkl，很快)
        processed_file = os.path.join(f'data_files/{dataset_name}/processed/{split}.pkl')
        with open(processed_file, 'rb') as f:
            self.processed_dict_list = pickle.load(f)

        # 建立即時查找的 processed_map，並做可前置的預處理（去重與 one_hot）
        self.processed_map = {}
        for s in self.processed_dict_list:
            sample = s.copy()
            # 去重答案實體
            if 'a_entity_id_list' in sample:
                sample['a_entity_id_list'] = list(set(sample['a_entity_id_list']))
            # 預先計算 topic_entity_one_hot（不依賴 embeddings）
            if 'text_entity_list' in sample and 'non_text_entity_list' in sample and 'q_entity_id_list' in sample:
                num_entities = len(sample['text_entity_list']) + len(sample['non_text_entity_list'])
                topic_entity_mask = torch.zeros(num_entities)
                if sample['q_entity_id_list']:
                    topic_entity_mask[sample['q_entity_id_list']] = 1.
                sample['topic_entity_one_hot'] = F.one_hot(topic_entity_mask.long(), num_classes=2).float()
            self.processed_map[s['id']] = sample

        # 2. 載入 triple scores (通常是 .pth，很快)
        self.triple_score_dict = self._load_triple_scores(dataset_name, split, self.weight_mode)

        # 3. 懶加載 Embeddings (只建立 LazyEmbeddingDict 對象，不讀取數據)
        self.emb_dict = self._setup_lazy_embeddings(dataset_name, config['dataset']['text_encoder_name'], split)
        
        # 4. 過濾出有效的樣本 ID 列表
        # 這個列表的順序將是 DataLoader 索引的依據
        self.valid_sample_ids = self._filter_valid_samples()
        
        # 5. 【新增】計算並儲存統計數據
        self._compute_statistics()
        
        print(f"✅ Initialized {split} set with {len(self.valid_sample_ids)} valid samples.")
        print(f"   - Skipped samples: {self.num_skipped}")
        print(f"   - Relevant triples (median): {self.median_num_relevant}")
        print(f"   Embeddings will be loaded on-demand by DataLoader workers.")

    def __len__(self):
        """返回有效樣本的總數"""
        return len(self.valid_sample_ids)

    def __getitem__(self, idx):
        """
        DataLoader 的核心！根據索引 idx 獲取單一完整樣本。
        這個函數會被多個 worker 並行調用。
        """
        # 1. 從有效 ID 列表中獲取 sample_id
        sample_id = self.valid_sample_ids[idx]

        # 2. 快速取得預處理後的 metadata（拷貝以保留不可變性）
        sample = self.processed_map[sample_id].copy()

        # 3. 添加 triple scores (從記憶體中，很快)
        triple_score_info = self.triple_score_dict[sample_id]
        sample['target_triple_probs'] = triple_score_info['triple_scores']
        
        # 4. **執行 I/O 操作**: 使用 LazyEmbeddingDict 懶加載此樣本的 embedding
        sample_emb_data = self.emb_dict[sample_id]
        sample.update(sample_emb_data)

        return sample

    # --- 以下是輔助函數 ---

    def _load_triple_scores(self, dataset_name, split, weight_mode):
        save_dir_map = {
            'none': 'triple_scores',
            'freq': 'triple_scores_freq_weight',
            'inv': 'triple_scores_inv_freq_weight',
            'spcount': 'triple_scores_spcount',
            'spcount_inv': 'triple_scores_spcount_inv'
        }
        save_dir = os.path.join('data_files', dataset_name, save_dir_map.get(weight_mode, 'triple_scores'))
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, f'{split}.pth')
        if os.path.exists(save_file):
            return torch.load(save_file)
        # 檔案不存在時，現場計算並儲存（避免因缺檔而中止）
        print(f"⚠️ Triple scores not found. Computing on-the-fly: {save_file}")
        triple_score_dict = {}
        for sample in tqdm(self.processed_dict_list, desc=f"Computing triple scores ({weight_mode})"):
            sample_id = sample['id']
            h_list = sample['h_id_list']
            r_list = sample['r_id_list']
            t_list = sample['t_id_list']
            # 選擇圖構建
            if weight_mode == 'freq':
                nx_g = build_nx_g_with_weights(h_list, r_list, t_list)
            elif weight_mode == 'inv':
                nx_g = build_nx_g_with_inverse_weights(h_list, r_list, t_list)
            else:
                nx_g = self._get_nx_g(h_list, r_list, t_list)
            
            # 蒐集最短路徑（spcount 模式下總是用未加權最短路徑）
            path_list_ = []
            for q_entity_id in sample['q_entity_id_list']:
                for a_entity_id in sample['a_entity_id_list']:
                    if weight_mode in ['freq', 'inv']:
                        paths_q_a = self._weighted_shortest_path(nx_g, q_entity_id, a_entity_id)
                    else:
                        paths_q_a = self._shortest_path(nx_g, q_entity_id, a_entity_id)
                    if len(paths_q_a) > 0:
                        path_list_.extend(paths_q_a)
            # 轉為 triple 路徑並計分
            if len(path_list_) == 0:
                max_path_length = None
            else:
                max_path_length = 0
            path_list = []
            for path in path_list_:
                num_triples_path = len(path) - 1
                max_path_length = max(max_path_length, num_triples_path)
                triples_path = []
                for i in range(num_triples_path):
                    h_i = path[i]
                    t_i = path[i+1]
                    triple_id_i_list = [nx_g[h_i][t_i]['triple_id']]
                    triples_path.append(triple_id_i_list)
                path_list.append(triples_path)
            num_triples = len(h_list)

            # 依不同模式產生目標與正樣本權重因子
            if weight_mode == 'freq':
                triple_scores = self._score_triples_with_weights(path_list, num_triples, nx_g)
                pos_weight_factors = torch.ones(num_triples)
            elif weight_mode == 'inv':
                # 需要 triple_id -> count
                edge_counts = defaultdict(int)
                for h, r, t in zip(h_list, r_list, t_list):
                    edge_counts[(h, r, t)] += 1
                triple_id_to_count = {i: edge_counts[(h, r, t)] for i, (h, r, t) in enumerate(zip(h_list, r_list, t_list))}
                triple_scores = score_triples_with_inverse_weights(path_list, num_triples, triple_id_to_count)
                pos_weight_factors = torch.ones(num_triples)
            elif weight_mode in ['spcount', 'spcount_inv']:
                # 統計每個 triple 出現在多少條最短路徑中（未加權 graph）
                triple_counts = torch.zeros(num_triples)
                for path in path_list:
                    for triple_id_list in path:
                        for triple_id in triple_id_list:
                            triple_counts[triple_id] += 1
                # 目標為是否在任一最短路徑上
                triple_scores = (triple_counts > 0).float()
                # 正樣本權重因子：spcount -> 次數；spcount_inv -> 1/次數
                pos_weight_factors = torch.ones(num_triples)
                positive_idx = triple_counts > 0
                if weight_mode == 'spcount':
                    pos_weight_factors[positive_idx] = triple_counts[positive_idx]
                else:
                    pos_weight_factors[positive_idx] = 1.0 / triple_counts[positive_idx]
            else:
                triple_scores = self._score_triples(path_list, num_triples)
                pos_weight_factors = torch.ones(num_triples)

            triple_score_dict[sample_id] = {
                'triple_scores': triple_scores,
                'pos_weight_factors': pos_weight_factors,
                'max_path_length': max_path_length
            }
        torch.save(triple_score_dict, save_file)
        print(f"✅ Saved computed triple scores to {save_file}")
        return triple_score_dict

    def _setup_lazy_embeddings(self, dataset_name, text_encoder_name, split):
        # 先嘗試合併檔（webqsp 等小數據集）
        merged_file = f'data_files/{dataset_name}/emb/{text_encoder_name}/{split}.pth'
        if os.path.exists(merged_file):
            print(f"✅ Found merged embedding file: {merged_file}")
            return torch.load(merged_file, map_location='cpu')
        
        # 否則使用分批檔（cwq 等大數據集）
        batch_dir = f'data_files/{dataset_name}/emb/{text_encoder_name}'
        batch_files = sorted([f for f in os.listdir(batch_dir)
                              if f.startswith(f'{split}_batch_') and f.endswith('.pth')],
                             key=lambda x: int(x.split('_')[-1].split('.')[0]))
        if not batch_files:
            raise FileNotFoundError(f"No embeddings found for split {split}: neither merged file nor batch files present in {batch_dir}")
        
        # 建議 cache_size 與 num_workers 數量相關，預設較小以避免多 worker 累積占用
        cfg_cache = None
        try:
            cfg_cache = int(self.config.get('dataset', {}).get('emb_cache_size', 0))
        except Exception:
            cfg_cache = None
        env_cache = os.environ.get('EMB_CACHE_SIZE')
        cache_size = 4  # 預設 4，避免多進程下佔用過大
        if cfg_cache and cfg_cache > 0:
            cache_size = cfg_cache
        if env_cache and env_cache.isdigit() and int(env_cache) > 0:
            cache_size = int(env_cache)
        print(f"🔧 LazyEmbeddingDict cache_size={cache_size}")
        return LazyEmbeddingDict(batch_dir, batch_files, cache_size=cache_size)

    def _filter_valid_samples(self):
        """過濾出在所有資料源中都存在的樣本ID"""
        valid_ids = []
        processed_ids = {s['id'] for s in self.processed_dict_list}

        for sample_id in tqdm(self.emb_dict.keys(), desc="Filtering valid samples"):
            if sample_id not in processed_ids:
                continue
            if sample_id not in self.triple_score_dict:
                continue
            
            # skip_no_path 邏輯
            max_path_length = self.triple_score_dict[sample_id].get('max_path_length')
            if self.skip_no_path and (max_path_length is None or max_path_length == 0):
                continue
            
            valid_ids.append(sample_id)
        return valid_ids
    
    def _compute_statistics(self):
        """
        在初始化結束時計算一次統計數據。
        """
        print("📊 Computing statistics for the dataset...")
        
        # 1. 計算跳過的樣本數
        total_processed = len(self.processed_dict_list)
        valid_samples = len(self.valid_sample_ids)
        self.num_skipped = total_processed - valid_samples

        # 2. 計算相關 triple 的統計
        num_relevant_triples = []
        if not self.valid_sample_ids:
            # 如果沒有有效樣本，設定預設值
            self.median_num_relevant = 0
            self.mean_num_relevant = 0
            self.max_num_relevant = 0
            return

        for sample_id in self.valid_sample_ids:
            # 確保 triple_score_dict 中有這個 key
            if sample_id in self.triple_score_dict:
                triple_score = self.triple_score_dict[sample_id]['triple_scores']
                num_relevant_triples.append(len(triple_score.nonzero()))
        
        if num_relevant_triples:
            self.median_num_relevant = int(np.median(num_relevant_triples))
            self.mean_num_relevant = int(np.mean(num_relevant_triples))
            self.max_num_relevant = int(np.max(num_relevant_triples))
        else:
            self.median_num_relevant = 0
            self.mean_num_relevant = 0
            self.max_num_relevant = 0
    

    def _get_nx_g(self, h_id_list, r_id_list, t_id_list):
        """建立標準 NetworkX 圖"""
        nx_g = nx.DiGraph()
        num_triples = len(h_id_list)
        for i in range(num_triples):
            h_i = h_id_list[i]
            r_i = r_id_list[i]
            t_i = t_id_list[i]
            nx_g.add_edge(h_i, t_i, triple_id=i, relation_id=r_i)
        return nx_g

    def _shortest_path(self, nx_g, q_entity_id, a_entity_id):
        """找最短路徑（標準版）"""
        try:
            forward_paths = list(nx.all_shortest_paths(nx_g, q_entity_id, a_entity_id))
        except Exception:
            forward_paths = []
        try:
            backward_paths = list(nx.all_shortest_paths(nx_g, a_entity_id, q_entity_id))
        except Exception:
            backward_paths = []
        full_paths = forward_paths + backward_paths
        if (len(forward_paths) == 0) or (len(backward_paths) == 0):
            return full_paths
        min_path_len = min([len(path) for path in full_paths])
        refined_paths = []
        for path in full_paths:
            if len(path) == min_path_len:
                refined_paths.append(path)
        return refined_paths

    def _weighted_shortest_path(self, nx_g, q_entity_id, a_entity_id):
        """找加權最短路徑"""
        try:
            forward_paths = list(nx.all_shortest_paths(nx_g, q_entity_id, a_entity_id, weight='weight'))
        except Exception:
            forward_paths = []
        try:
            backward_paths = list(nx.all_shortest_paths(nx_g, a_entity_id, q_entity_id, weight='weight'))
        except Exception:
            backward_paths = []
        full_paths = forward_paths + backward_paths
        if (len(forward_paths) == 0) or (len(backward_paths) == 0):
            return full_paths
        weighted_paths = []
        for path in full_paths:
            weight_sum = 0
            for i in range(len(path) - 1):
                weight_sum += nx_g[path[i]][path[i+1]].get('weight', 1.0)
            weighted_paths.append((path, weight_sum))
        if not weighted_paths:
            return []
        min_weight = min([wp[1] for wp in weighted_paths])
        refined_paths = [wp[0] for wp in weighted_paths if wp[1] == min_weight]
        return refined_paths

    def _score_triples(self, path_list, num_triples):
        """標準 triple 評分"""
        triple_scores = torch.zeros(num_triples)
        for path in path_list:
            for triple_id_list in path:
                triple_scores[triple_id_list] = 1.
        return triple_scores

    def _score_triples_with_weights(self, path_list, num_triples, nx_g):
        """加權 triple 評分"""
        triple_scores = torch.zeros(num_triples)
        for path in path_list:
            for triple_id_list in path:
                for triple_id in triple_id_list:
                    triple_scores[triple_id] += 1.0
        return triple_scores

def optimized_collate_retriever(batch_samples):
    """
    高效的 collate_fn，將 list of samples 正確打包成一個批次。
    - 對於可變長度的 tensor，使用 pad_sequence。
    - 對於固定長度的 tensor，使用 stack。
    """
    # 提取各個欄位
    q_embs = torch.stack([s['q_emb'] for s in batch_samples])
    
    # 不要對實體相關張量使用 pad_sequence，保持為 list
    entity_embs_list = [s['entity_embs'] for s in batch_samples]
    relation_embs_list = [s['relation_embs'] for s in batch_samples]
    topic_entity_one_hots_list = [s['topic_entity_one_hot'] for s in batch_samples]
    target_triple_probs_list = [s['target_triple_probs'] for s in batch_samples]
    
    # 處理 ID 列表
    h_id_tensors = [torch.tensor(s['h_id_list']) for s in batch_samples]
    r_id_tensors = [torch.tensor(s['r_id_list']) for s in batch_samples]
    t_id_tensors = [torch.tensor(s['t_id_list']) for s in batch_samples]

    # 其他非 tensor 資訊
    a_entity_id_lists = [s['a_entity_id_list'] for s in batch_samples]
    num_non_text_entities = [len(s['non_text_entity_list']) for s in batch_samples]

    return {
        'q_emb': q_embs,
        'entity_embs_list': entity_embs_list,  # 改為 list
        'relation_embs_list': relation_embs_list,  # 改為 list
        'topic_entity_one_hot_list': topic_entity_one_hots_list,  # 改為 list
        'target_triple_probs_list': target_triple_probs_list,  # 改為 list
        'h_id_tensors': h_id_tensors,
        'r_id_tensors': r_id_tensors,
        't_id_tensors': t_id_tensors,
        'a_entity_id_lists': a_entity_id_lists,
        'num_non_text_entities': num_non_text_entities,
    }

# =============================================================================
# 分組取樣器：按檔案分桶以提高 I/O 局部性
# =============================================================================

class GroupedByFileBatchSampler:
    """
    將同一 embedding 檔案中的樣本盡量分配到同一個 batch，提升 I/O 快取命中。
    - 若 dataset.emb_dict 具有 `_sample_to_batch`（LazyEmbeddingDict），則依據該映射分桶。
    - 否則所有樣本視為同一桶，退化為一般打散後組 batch。
    """
    def __init__(self, dataset, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)

        # 準備索引與分桶
        self.indices = list(range(len(dataset)))
        # 建立 sample_id -> batch_idx 的映射（若可用）
        self._sample_to_batch = getattr(getattr(dataset, 'emb_dict', None), '_sample_to_batch', None)

        # 預先建立桶: bucket_id -> [dataset_index]
        self._buckets = self._build_buckets()

    def _build_buckets(self):
        buckets = defaultdict(list)
        if self._sample_to_batch is None:
            # 單一桶退化情況
            buckets[0] = self.indices
            return buckets
        # 根據 sample_id -> batch_idx 分桶
        for ds_idx in self.indices:
            sample_id = self.dataset.valid_sample_ids[ds_idx]
            bucket_id = self._sample_to_batch.get(sample_id, -1)
            buckets[bucket_id].append(ds_idx)
        return buckets

    def __iter__(self):
        # 取得桶列表
        bucket_items = list(self._buckets.items())
        if self.shuffle:
            random.shuffle(bucket_items)
        # 逐桶產生 batch
        for _, idx_list in bucket_items:
            if self.shuffle:
                random.shuffle(idx_list)
            # 依序切分 batch
            for start in range(0, len(idx_list), self.batch_size):
                end = start + self.batch_size
                if end > len(idx_list) and self.drop_last:
                    break
                yield idx_list[start:end]

    def __len__(self):
        # 估算 batch 數量
        num_batches = 0
        for _, idx_list in self._buckets.items():
            if self.drop_last:
                num_batches += len(idx_list) // self.batch_size
            else:
                num_batches += (len(idx_list) + self.batch_size - 1) // self.batch_size
        return num_batches
