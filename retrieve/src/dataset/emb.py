import os
import pickle
import torch

from tqdm import tqdm

class EmbInferDataset:
    def __init__(
        self,
        raw_set,
        entity_identifiers,
        save_path,
        skip_no_topic=True,
        skip_no_ans=True,
        load_embeddings=False,
        embedding_dir=None
    ):
        """
        Parameters
        ----------
        entity_identifiers : set
            Set of entity identifiers, e.g., m.06w2sn5, for which we cannot
            get meaningful text embeddings.
        skip_no_topic : bool
            Whether to skip samples without topic entities in the graph.
        skip_no_ans : bool
            Whether to skip samples without answer entities in the graph.
        load_embeddings : bool
            Whether to load pre-computed embeddings.
        embedding_dir : str
            Directory containing pre-computed embeddings.
        """
        self.processed_dict_list = self._process(
            raw_set,
            entity_identifiers,
            save_path)
        
        self.skip_no_topic = skip_no_topic
        self.skip_no_ans = skip_no_ans
        self.load_embeddings = load_embeddings
        self.embedding_dir = embedding_dir
        
        processed_dict_list = []
        for processed_dict_i in self.processed_dict_list:
            if (len(processed_dict_i['q_entity_id_list']) == 0) and skip_no_topic:
                continue
            
            if (len(processed_dict_i['a_entity_id_list']) == 0) and skip_no_ans:
                continue
            
            processed_dict_list.append(processed_dict_i)
        self.processed_dict_list = processed_dict_list
        
        print(f'# raw samples: {len(raw_set)} | # processed samples: {len(self.processed_dict_list)}')
        
        # 如果啟用 embedding 載入，預載入所有 embedding 文件
        if self.load_embeddings and self.embedding_dir:
            self.embedding_files = self._load_embedding_files()
            print(f'Loaded {len(self.embedding_files)} embedding files')

    def _load_embedding_files(self):
        """載入所有 embedding 文件的路徑"""
        embedding_files = {}
        if os.path.exists(self.embedding_dir):
            for filename in os.listdir(self.embedding_dir):
                if filename.startswith('train_batch_') and filename.endswith('.pth'):
                    batch_idx = int(filename.split('_')[-1].split('.')[0])
                    embedding_files[batch_idx] = os.path.join(self.embedding_dir, filename)
        return embedding_files

    def _load_sample_embeddings(self, sample_id):
        """載入特定樣本的 embeddings"""
        if not self.load_embeddings or not self.embedding_dir:
            return None, None, None
        
        # 計算樣本屬於哪個批次
        batch_idx = sample_id // 64  # 假設批次大小為64
        
        if batch_idx not in self.embedding_files:
            print(f"Warning: Embedding file for batch {batch_idx} not found")
            return None, None, None
        
        try:
            # 載入批次文件
            batch_data = torch.load(self.embedding_files[batch_idx])
            
            # 獲取樣本在批次中的位置
            sample_in_batch = sample_id % 64
            
            # 找到對應的樣本ID
            sample_ids = list(batch_data.keys())
            if sample_in_batch < len(sample_ids):
                sample_key = sample_ids[sample_in_batch]
                sample_embeddings = batch_data[sample_key]
                
                return (sample_embeddings['q_emb'], 
                       sample_embeddings['entity_embs'], 
                       sample_embeddings['relation_embs'])
            else:
                return None, None, None
                
        except Exception as e:
            print(f"Error loading embeddings for sample {sample_id}: {e}")
            return None, None, None

    def _process(
        self,
        raw_set,
        entity_identifiers,
        save_path
    ):
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                return pickle.load(f)
        
        processed_dict_list = []
        for i in tqdm(range(len(raw_set))):
            sample_i = raw_set[i]
            processed_dict_i = self._process_sample(
                sample_i, 
                entity_identifiers)
            # if processed_dict_i is not None:
            processed_dict_list.append(processed_dict_i)

        with open(save_path, 'wb') as f:
            pickle.dump(processed_dict_list, f)
        
        return processed_dict_list

    def _process_sample(
        self,
        sample,
        entity_identifiers
    ):
        # Model input (0) question
        question = sample['question']
        
        triples = sample['graph']

        all_entities = set()
        all_relations = set()
        for (h, r, t) in triples:
            all_entities.add(h)
            all_relations.add(r)
            all_entities.add(t)
        
        # Sort for deterministic entity IDs.
        entity_list = sorted(all_entities)
        # Parition the entities based on if the associated text is meaningful.
        # Model input (1) text of entities
        #             (2) number of entities without text
        text_entity_list = []
        non_text_entity_list = []
        for entity in entity_list:
            if entity in entity_identifiers:
                non_text_entity_list.append(entity)
            else:
                text_entity_list.append(entity)

        # Create entity IDs.
        entity2id = dict()
        entity_id = 0
        for entity in text_entity_list:
            entity2id[entity] = entity_id
            entity_id += 1
        for entity in non_text_entity_list:
            entity2id[entity] = entity_id
            entity_id += 1

        # Model input (3) text of relations
        relation_list = sorted(all_relations)
        # Create relation IDs.
        rel2id = dict()
        rel_id = 0
        for rel in relation_list:
            rel2id[rel] = rel_id
            rel_id += 1

        # Convert triples to entity and relation IDs.
        # Model input (4) triples in th ID space for
        # graph construction and embedding indexing
        h_id_list = []
        r_id_list = []
        t_id_list = []
        for (h, r, t) in triples:
            h_id_list.append(entity2id[h])
            r_id_list.append(rel2id[r])
            t_id_list.append(entity2id[t])

        # Model input (5) list of question entity IDs
        q_entity_id_list = []
        for entity in sample['q_entity']:
            if entity in entity2id:
                q_entity_id_list.append(entity2id[entity])

        # Prepare output labels.
        assert sample['a_entity'] == sample['answer']
        a_entity_id_list = []
        for entity in sample['a_entity']:
            entity_id = entity2id.get(entity, None)
            if entity_id is not None:
                a_entity_id_list.append(entity_id)

        processed_dict = {
            'id': sample['id'],
            'question': question,
            'q_entity': sample['q_entity'],
            'q_entity_id_list': q_entity_id_list,
            'text_entity_list': text_entity_list,
            'non_text_entity_list': non_text_entity_list,
            'relation_list': relation_list,
            'h_id_list': h_id_list,
            'r_id_list': r_id_list,
            't_id_list': t_id_list,
            'a_entity': sample['a_entity'],
            'a_entity_id_list': a_entity_id_list
        }

        return processed_dict

    def __len__(self):
        return len(self.processed_dict_list)
    
    def __getitem__(self, i):
        sample = self.processed_dict_list[i]
        
        id = sample['id']
        q_text = sample['question']
        text_entity_list = sample['text_entity_list']
        relation_list = sample['relation_list']
        
        # 如果啟用 embedding 載入，返回預計算的 embeddings
        if self.load_embeddings:
            q_emb, entity_embs, relation_embs = self._load_sample_embeddings(i)
            if q_emb is not None:
                return {
                    'id': id,
                    'q_emb': q_emb,
                    'entity_embs': entity_embs,
                    'relation_embs': relation_embs,
                    'h_id_list': sample['h_id_list'],
                    'r_id_list': sample['r_id_list'],
                    't_id_list': sample['t_id_list'],
                    'q_entity_id_list': sample['q_entity_id_list'],
                    'a_entity_id_list': sample['a_entity_id_list'],
                    'non_text_entity_list': sample['non_text_entity_list']
                }
        
        # 否則返回原始格式（用於 embedding 生成）
        return id, q_text, text_entity_list, relation_list
