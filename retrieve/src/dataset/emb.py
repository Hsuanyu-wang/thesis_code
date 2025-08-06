# 匯入必要的套件
import os
import pickle

import numpy as np
import torch
# from tqdm import tqdm  # 移除 tqdm

# EmbInferDataset 用於將原始資料集處理成適合嵌入推論的格式
class EmbInferDataset:
    def __init__(
        self,
        raw_set,
        entity_identifiers,
        save_path,
        skip_no_topic=True,
        skip_no_ans=True
    ):
        """
        參數說明
        ----------
        entity_identifiers : set
            不適合做文本嵌入的實體 id 集合，例如 m.06w2sn5。
        skip_no_topic : bool
            是否跳過圖中沒有主題實體的樣本。
        skip_no_ans : bool
            是否跳過圖中沒有答案實體的樣本。
        """
        # 處理原始資料集，轉成 processed_dict_list
        self.processed_dict_list = self._process(
            raw_set,
            entity_identifiers,
            save_path)
        
        self.skip_no_topic = skip_no_topic
        self.skip_no_ans = skip_no_ans
        
        # 根據設定，過濾掉沒有主題或答案實體的樣本
        processed_dict_list = []
        for processed_dict_i in self.processed_dict_list:
            if (len(processed_dict_i['q_entity_id_list']) == 0) and skip_no_topic:
                continue
            
            if (len(processed_dict_i['a_entity_id_list']) == 0) and skip_no_ans:
                continue
            
            processed_dict_list.append(processed_dict_i)
        self.processed_dict_list = processed_dict_list
        
        print(f'# raw samples: {len(raw_set)} | # processed samples: {len(self.processed_dict_list)}')

    # 將原始資料集處理成嵌入推論格式，並快取到 pickle 檔
    def _process(
        self,
        raw_set,
        entity_identifiers,
        save_path
    ):
        # 若已存在快取檔，直接讀取
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                return pickle.load(f)
        
        processed_dict_list = []
        for i in range(len(raw_set)):
            sample_i = raw_set[i]
            processed_dict_i = self._process_sample(
                sample_i, 
                entity_identifiers)
            # if processed_dict_i is not None:
            processed_dict_list.append(processed_dict_i)

        # 儲存處理後的資料到 pickle 檔
        with open(save_path, 'wb') as f:
            pickle.dump(processed_dict_list, f)
        
        return processed_dict_list

    # 處理單一樣本，轉成模型需要的格式
    def _process_sample(
        self,
        sample,
        entity_identifiers
    ):
        # 取得問題文本
        question = sample['question']
        
        # 處理不同資料集格式
        if 'graph' in sample:
            # WebQSP/CWQ 格式
            triples = sample['graph']
            q_entity = sample.get('q_entity', [])
            a_entity = sample.get('a_entity', sample.get('answer', []))
        elif 'proof' in sample:
            # KGQAGen-10k 格式
            triples = sample['proof']
            # 從 proof 中提取主題實體（通常是第一個三元組的頭實體）
            q_entity = [triples[0][0]] if triples else []
            # 答案實體從 answer 欄位獲取，但需要轉換為實體 ID 格式
            a_entity = sample.get('answer', [])
            # 將答案實體名稱轉換為實體 ID 格式
            a_entity_ids = []
            for answer_name in a_entity:
                # 在 proof 中尋找對應的實體 ID
                for triple in triples:
                    if len(triple) >= 3:  # 確保三元組格式正確
                        if answer_name in triple[0]:  # 檢查頭實體
                            a_entity_ids.append(triple[0])
                            break
                        elif answer_name in triple[2]:  # 檢查尾實體
                            a_entity_ids.append(triple[2])
                            break
            a_entity = a_entity_ids
        else:
            raise ValueError(f"Unknown dataset format: {sample.keys()}")

        # 收集所有實體與關係
        all_entities = set()
        all_relations = set()
        for triple in triples:
            if len(triple) >= 3:  # 確保三元組格式正確
                h, r, t = triple[0], triple[1], triple[2]
                if h is not None and r is not None and t is not None:
                    all_entities.add(h)
                    all_relations.add(r)
                    all_entities.add(t)
        
        # 排序以確保實體 id 一致性
        entity_list = sorted(all_entities)
        # 根據是否有對應文本，分成 text_entity 與 non_text_entity
        text_entity_list = []
        non_text_entity_list = []
        for entity in entity_list:
            if entity in entity_identifiers:
                non_text_entity_list.append(entity)
            else:
                text_entity_list.append(entity)

        # 建立 entity2id 映射
        entity2id = dict()
        entity_id = 0
        for entity in text_entity_list:
            entity2id[entity] = entity_id
            entity_id += 1
        for entity in non_text_entity_list:
            entity2id[entity] = entity_id
            entity_id += 1

        # 處理關係，建立 rel2id
        relation_list = sorted(all_relations)
        rel2id = dict()
        rel_id = 0
        for rel in relation_list:
            rel2id[rel] = rel_id
            rel_id += 1

        # 將三元組轉成 id 空間，方便後續嵌入與圖建構
        h_id_list = []
        r_id_list = []
        t_id_list = []
        for triple in triples:
            if len(triple) >= 3:  # 確保三元組格式正確
                h, r, t = triple[0], triple[1], triple[2]
                if h is not None and r is not None and t is not None:
                    if h in entity2id and r in rel2id and t in entity2id:
                        h_id_list.append(entity2id[h])
                        r_id_list.append(rel2id[r])
                        t_id_list.append(entity2id[t])

        # 問題實體 id 列表
        q_entity_id_list = []
        for entity in q_entity:
            if entity in entity2id:
                q_entity_id_list.append(entity2id[entity])

        # 答案實體 id 列表
        a_entity_id_list = []
        for entity in a_entity:
            entity_id = entity2id.get(entity, None)
            if entity_id is not None:
                a_entity_id_list.append(entity_id)

        # 封裝所有處理後資訊
        processed_dict = {
            'id': sample['id'],
            'question': question,
            'q_entity': q_entity,
            'q_entity_id_list': q_entity_id_list,
            'text_entity_list': text_entity_list,
            'non_text_entity_list': non_text_entity_list,
            'relation_list': relation_list,
            'h_id_list': h_id_list,
            'r_id_list': r_id_list,
            't_id_list': t_id_list,
            'a_entity': a_entity,
            'a_entity_id_list': a_entity_id_list
        }

        return processed_dict

    # 取得資料集長度
    def __len__(self):
        return len(self.processed_dict_list)
    
    # 取得第 i 筆資料，回傳 (id, 問題文本, 可嵌入實體列表, 關係列表)
    def __getitem__(self, i):
        sample = self.processed_dict_list[i]
        
        id = sample['id']
        q_text = sample['question']
        text_entity_list = sample['text_entity_list']
        relation_list = sample['relation_list']
        
        return id, q_text, text_entity_list, relation_list
