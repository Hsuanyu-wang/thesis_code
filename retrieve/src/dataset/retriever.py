########################################
# Optimal Subgraph Gold 標註法具體實現
# 1. 限制長度內所有路徑 (All Paths within a Length Threshold)
def all_paths_within_length(nx_g, q_entity_id_list, a_entity_id_list, max_length=None, max_paths=None):
    """
    取得所有主題實體到答案實體間的路徑
    Args:
        nx_g: networkx 圖物件
        q_entity_id_list: 查詢實體 ID 列表
        a_entity_id_list: 答案實體 ID 列表
        max_length: 最大路徑長度，若為 None 則不限制長度
        max_paths: 最大路徑數量，若為 None 則取所有路徑，若為 'medium' 則取中等數量
    Returns: List[List[entity_id]]
    """
    path_list_ = []
    for q_entity_id in q_entity_id_list:
        for a_entity_id in a_entity_id_list:
            try:
                # 如果指定了 max_length，使用 cutoff 參數
                if max_length is not None:
                    paths = list(nx.all_simple_paths(nx_g, q_entity_id, a_entity_id, cutoff=max_length))
                else:
                    # 不限制長度，取得所有路徑
                    paths = list(nx.all_simple_paths(nx_g, q_entity_id, a_entity_id))
                path_list_.extend(paths)
            except:
                continue
    
    # 如果指定了 max_paths，限制路徑數量
    if max_paths is not None:
        if max_paths == 'medium':
            # 取中等數量的路徑
            total_paths = len(path_list_)
            if total_paths > 0:
                # 取中間 50% 的路徑
                start_idx = total_paths // 4
                end_idx = 3 * total_paths // 4
                path_list_ = path_list_[start_idx:end_idx]
        elif isinstance(max_paths, int):
            # 取指定數量的路徑
            path_list_ = path_list_[:max_paths]
    
    return path_list_

# 2. 隨機遊走子圖 (Random Walk-based Subgraph)
import random

def random_walk_paths(nx_g, q_entity_id_list, a_entity_id_list, num_walks=20, walk_length=4):
    """
    以主題實體為起點，進行多次隨機遊走，收集能到達答案的路徑
    Returns: List[List[entity_id]]
    """
    path_list_ = []
    for q_entity_id in q_entity_id_list:
        for _ in range(num_walks):
            path = [q_entity_id]
            current = q_entity_id
            for _ in range(walk_length):
                neighbors = list(nx_g.successors(current))
                if not neighbors:
                    break
                next_node = random.choice(neighbors)
                path.append(next_node)
                current = next_node
                if current in a_entity_id_list:
                    path_list_.append(path.copy())
                    break
    return path_list_

# 3. Personalized PageRank 子圖 (Personalized PageRank Subgraph)
def pagerank_topk_edges(nx_g, q_entity_id_list, topk=10):
    """
    以主題實體為 seed，計算 Personalized PageRank，取分數最高的 topk 條邊
    Returns: List[Tuple(h_id, t_id)]
    """
    # 多個主題實體平均分配權重
    personalization = {n: 0 for n in nx_g.nodes()}
    for q in q_entity_id_list:
        personalization[q] = 1  # 將 float 改為 int，避免 linter 錯誤
    pr = nx.pagerank(nx_g, personalization=personalization)
    # 取分數最高的 topk 節點，然後找出這些節點相關的邊
    top_nodes = sorted(pr, key=pr.get, reverse=True)[:topk]
    edge_set = set()
    for h in top_nodes:
        for t in nx_g.successors(h):
            edge_set.add((h, t))
    return list(edge_set)
########################################
import networkx as nx
import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F

# from tqdm import tqdm  # 移除 tqdm

class RetrieverDataset:
    def __init__(
        self,
        config,
        split,
        skip_no_path=True
    ):
        # Load pre-processed data.
        # 載入預先處理好的資料
        dataset_name = config['dataset']['name']
        processed_dict_list = self._load_processed(dataset_name, split)

        # Extract directed shortest paths from topic entities to answer
        # entities or vice versa as weak supervision signals for triple scoring.
        # 從主題實體到答案實體（或反向）提取有向最短路徑，作為三元組評分的弱監督訊號
        triple_score_dict = self._get_triple_scores(
            dataset_name, split, processed_dict_list)

        # Load pre-computed embeddings.
        # 載入預先計算好的嵌入向量
        emb_dict = self._load_emb(
            dataset_name, config['dataset']['text_encoder_name'], split)

        # Put everything together.
        # 將所有資料組合起來
        self._assembly(
            processed_dict_list, triple_score_dict, emb_dict, skip_no_path)

    def _load_processed(
        self,
        dataset_name,
        split
    ):
        # 載入已處理的資料檔案
        processed_file = os.path.join(
            f'data_files/{dataset_name}/processed/{split}.pkl')
        with open(processed_file, 'rb') as f:
            return pickle.load(f)

    def _get_triple_scores(
        self,
        dataset_name,
        split,
        processed_dict_list
    ):
        # 取得三元組分數，若已存在則直接載入，否則計算後儲存
        save_dir = os.path.join('data_files', dataset_name, 'triple_scores')
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, f'{split}.pth')

        if os.path.exists(save_file):
            return torch.load(save_file)

        triple_score_dict = dict()
        for i in range(len(processed_dict_list)):
            sample_i = processed_dict_list[i]
            sample_i_id = sample_i['id']
            triple_scores_i, max_path_length_i = self._extract_paths_and_score(
                sample_i)

            triple_score_dict[sample_i_id] = {
                'triple_scores': triple_scores_i,
                'max_path_length': max_path_length_i
            }

        torch.save(triple_score_dict, save_file)
        
        return triple_score_dict

    def _extract_paths_and_score(
        self,
        sample
    ):
        # 取得 networkx 圖物件
        nx_g = self._get_nx_g(
            sample['h_id_list'],
            sample['r_id_list'],
            sample['t_id_list']
        )

        # Each raw path is a list of entity IDs.
        # 每條原始路徑是一串實體 ID
        path_list_ = []
        for q_entity_id in sample['q_entity_id_list']:
            for a_entity_id in sample['a_entity_id_list']:
                # paths_q_a = self._shortest_path(nx_g, q_entity_id, a_entity_id)
                ##############################################################
                paths_q_a = pagerank_topk_edges(nx_g, q_entity_id, topk=10)
                # paths_q_a = random_walk_paths(nx_g, q_entity_id, a_entity_id)
                # 可以選擇以下幾種方式：
                # 1. 指定最大長度
                # paths_q_a = all_paths_within_length(nx_g, q_entity_id, a_entity_id, max_length=3)
                # 2. 不限制長度，取所有路徑
                # paths_q_a = all_paths_within_length(nx_g, q_entity_id, a_entity_id)
                # 3. 不限制長度，取中等數量的路徑
                # paths_q_a = all_paths_within_length(nx_g, q_entity_id, a_entity_id, max_paths='medium')
                # 4. 不限制長度，取指定數量的路徑
                # paths_q_a = all_paths_within_length(nx_g, q_entity_id, a_entity_id, max_paths=10)
                ##############################################################
                if len(paths_q_a) > 0:
                    path_list_.extend(paths_q_a)

        if len(path_list_) == 0:
            max_path_length = None  # 沒有路徑
        else:
            max_path_length = 0  # 有路徑，初始化最大長度

        # Each processed path is a list of triple IDs.
        # 每條處理後的路徑是一串三元組 ID
        path_list = []

        for path in path_list_:
            num_triples_path = len(path) - 1
            # 取最大路徑長度，若 max_path_length 為 None 則設為 num_triples_path
            max_path_length = max(num_triples_path, max_path_length) if max_path_length is not None else num_triples_path
            triples_path = []

            for i in range(num_triples_path):
                h_id_i = path[i]
                t_id_i = path[i+1]
                triple_id_i_list = [
                    nx_g[h_id_i][t_id_i]['triple_id']
                ]              
                triples_path.append(triple_id_i_list)

            path_list.append(triples_path)

        num_triples = len(sample['h_id_list'])
        triple_scores = self._score_triples(
            path_list,
            num_triples
        )
        
        return triple_scores, max_path_length

    def _get_nx_g(
        self,
        h_id_list,
        r_id_list,
        t_id_list
    ):
        # 建立 networkx 有向圖，節點為實體，邊為三元組
        nx_g = nx.DiGraph()
        num_triples = len(h_id_list)
        for i in range(num_triples):
            h_i = h_id_list[i]
            r_i = r_id_list[i]
            t_i = t_id_list[i]
            nx_g.add_edge(h_i, t_i, triple_id=i, relation_id=r_i)

        return nx_g

    def _shortest_path(
        self,
        nx_g,
        q_entity_id,
        a_entity_id
    ):
        # 嘗試找正向與反向的最短路徑
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

    def _score_triples(
        self,
        path_list,
        num_triples
    ):
        # 根據路徑標記三元組分數
        triple_scores = torch.zeros(num_triples)
        
        for path in path_list:
            for triple_id_list in path:
                triple_scores[triple_id_list] = 1.

        return triple_scores

    def _load_emb(
        self,
        dataset_name,
        text_encoder_name,
        split
    ):
        # 載入嵌入向量檔案
        file_path = f'data_files/{dataset_name}/emb/{text_encoder_name}/{split}.pth'
        dict_file = torch.load(file_path)
        
        return dict_file

    def _assembly(
        self,
        processed_dict_list,
        triple_score_dict,
        emb_dict,
        skip_no_path,
    ):
        # 組合所有資料，並處理每個樣本
        self.processed_dict_list = []

        num_relevant_triples = []
        num_skipped = 0
        for i in range(len(processed_dict_list)):
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
            # 主題實體的 positional encoding
            num_entities_i = len(sample_i['text_entity_list']) + len(sample_i['non_text_entity_list'])
            topic_entity_mask = torch.zeros(num_entities_i)
            topic_entity_mask[sample_i['q_entity_id_list']] = 1.
            topic_entity_one_hot = F.one_hot(topic_entity_mask.long(), num_classes=2)
            sample_i['topic_entity_one_hot'] = topic_entity_one_hot.float()

            self.processed_dict_list.append(sample_i)

        median_num_relevant = int(np.median(num_relevant_triples))
        mean_num_relevant = int(np.mean(num_relevant_triples))
        max_num_relevant = int(np.max(num_relevant_triples))

        print(f'# skipped samples: {num_skipped}')
        print(f'# relevant triples | median: {median_num_relevant} | mean: {mean_num_relevant} | max: {max_num_relevant}')

    def __len__(self):
        # 回傳資料集長度
        return len(self.processed_dict_list)
    
    def __getitem__(self, i):
        # 取得第 i 筆資料
        return self.processed_dict_list[i]

def collate_retriever(data):
    # 將多筆資料組合成 batch，回傳 tensor 與必要欄位
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
