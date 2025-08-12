# 匯入必要的套件
import os
import pickle

from tqdm import tqdm

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
        初始化嵌入推論資料集
        
        Parameters / 參數說明
        ----------
        entity_identifiers : set
            Set of entity identifiers, e.g., m.06w2sn5, for which we cannot
            get meaningful text embeddings.
            不適合做文本嵌入的實體 id 集合，例如 m.06w2sn5。
        skip_no_topic : bool
            Whether to skip samples without topic entities in the graph.
            是否跳過圖中沒有主題實體的樣本。
        skip_no_ans : bool
            Whether to skip samples without answer entities in the graph.
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
            # 如果設定跳過沒有主題實體的樣本，且當前樣本沒有主題實體，則跳過
            if (len(processed_dict_i['q_entity_id_list']) == 0) and skip_no_topic:
                continue
            
            # 如果設定跳過沒有答案實體的樣本，且當前樣本沒有答案實體，則跳過
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
        # 使用進度條逐筆處理原始資料集
        for i in tqdm(range(len(raw_set))):
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
        # Model input (0) question / 模型輸入 (0) 問題文本
        question = sample['question']
        
        # 處理圖中的三元組資料
        triples = sample['graph']

        # 收集所有實體與關係
        all_entities = set()
        all_relations = set()
        for (h, r, t) in triples:
            all_entities.add(h)  # 頭實體
            all_relations.add(r)  # 關係
            all_entities.add(t)   # 尾實體
        
        # Sort for deterministic entity IDs. / 排序以確保實體 id 一致性
        entity_list = sorted(all_entities)
        # Parition the entities based on if the associated text is meaningful.
        # 根據是否有對應文本，分成 text_entity 與 non_text_entity
        # Model input (1) text of entities / 模型輸入 (1) 實體文本
        #             (2) number of entities without text / (2) 無文本實體數量
        text_entity_list = []      # 需要文本嵌入的實體列表
        non_text_entity_list = []  # 不需要文本嵌入的實體列表
        for entity in entity_list:
            if entity in entity_identifiers:
                # 如果實體在識別符集合中，歸類為 non_text_entity
                non_text_entity_list.append(entity)
            else:
                # 否則歸類為 text_entity，需要進行文本嵌入
                text_entity_list.append(entity)

        # Create entity IDs. / 建立 entity2id 映射
        entity2id = dict()
        entity_id = 0
        # 先為需要文本嵌入的實體分配 ID
        for entity in text_entity_list:
            entity2id[entity] = entity_id
            entity_id += 1
        # 再為不需要文本嵌入的實體分配 ID
        for entity in non_text_entity_list:
            entity2id[entity] = entity_id
            entity_id += 1

        # Model input (3) text of relations / 模型輸入 (3) 關係文本
        relation_list = sorted(all_relations)
        # Create relation IDs. / 建立 rel2id 映射
        rel2id = dict()
        rel_id = 0
        for rel in relation_list:
            rel2id[rel] = rel_id
            rel_id += 1

        # Convert triples to entity and relation IDs.
        # 將三元組轉成 id 空間，方便後續嵌入與圖建構
        # Model input (4) triples in th ID space for
        # graph construction and embedding indexing
        # 模型輸入 (4) ID 空間中的三元組，用於圖建構和嵌入索引
        h_id_list = []  # 頭實體 ID 列表
        r_id_list = []  # 關係 ID 列表
        t_id_list = []  # 尾實體 ID 列表
        for (h, r, t) in triples:
            h_id_list.append(entity2id[h])
            r_id_list.append(rel2id[r])
            t_id_list.append(entity2id[t])

        # Model input (5) list of question entity IDs / 模型輸入 (5) 問題實體 ID 列表
        q_entity_id_list = []
        for entity in sample['q_entity']:
            if entity in entity2id:
                q_entity_id_list.append(entity2id[entity])

        # Prepare output labels. / 準備輸出標籤
        assert sample['a_entity'] == sample['answer']
        # 答案實體 ID 列表
        a_entity_id_list = []
        for entity in sample['a_entity']:
            entity_id = entity2id.get(entity, None)
            if entity_id is not None:
                a_entity_id_list.append(entity_id)

        # 封裝所有處理後資訊
        processed_dict = {
            'id': sample['id'],                          # 樣本 ID
            'question': question,                        # 問題文本
            'q_entity': sample['q_entity'],             # 問題實體（原始名稱）
            'q_entity_id_list': q_entity_id_list,       # 問題實體 ID 列表
            'text_entity_list': text_entity_list,       # 需要文本嵌入的實體列表
            'non_text_entity_list': non_text_entity_list, # 不需要文本嵌入的實體列表
            'relation_list': relation_list,             # 關係列表
            'h_id_list': h_id_list,                     # 頭實體 ID 列表
            'r_id_list': r_id_list,                     # 關係 ID 列表
            't_id_list': t_id_list,                     # 尾實體 ID 列表
            'a_entity': sample['a_entity'],             # 答案實體（原始名稱）
            'a_entity_id_list': a_entity_id_list        # 答案實體 ID 列表
        }

        return processed_dict

    # 取得資料集長度
    def __len__(self):
        return len(self.processed_dict_list)
    
    # 取得第 i 筆資料，回傳 (id, 問題文本, 可嵌入實體列表, 關係列表)
    def __getitem__(self, i):
        sample = self.processed_dict_list[i]
        
        id = sample['id']                        # 樣本 ID
        q_text = sample['question']             # 問題文本
        text_entity_list = sample['text_entity_list']  # 需要嵌入的實體列表
        relation_list = sample['relation_list']         # 關係列表
        
        return id, q_text, text_entity_list, relation_list
