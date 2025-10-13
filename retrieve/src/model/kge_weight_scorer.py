"""
KGE-based triple weight scorer using PyKEEN TransE model
"""
import torch
import torch.nn as nn
import os
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from pykeen.models import TransE, DistMult, ComplEx
    from pykeen.triples import TriplesFactory
    from pykeen.utils import resolve_device
    import numpy as np
    _HAS_PYKEEN = True
except ImportError:
    _HAS_PYKEEN = False
    logging.warning("PyKEEN not available. KGE weight scoring will be disabled.")

class KGEWeightScorer:
    """
    使用 PyKEEN 預訓練模型為 triples 計算權重
    支持 TransE, DistMult, ComplEx 等模型
    """
    
    def __init__(self, 
                 kge_model_path: str,
                 kge_model_type: str = 'TransE',
                 device: str = 'cuda',
                 freeze_kge: bool = True,
                 weight_mode: str = 'score'):
        """
        Args:
            kge_model_path: PyKEEN 模型路徑
            kge_model_type: 模型類型 ('TransE', 'DistMult', 'ComplEx')
            device: 計算設備
            freeze_kge: 是否凍結 KGE 模型參數
            weight_mode: 權重計算模式 ('score', 'score_inv', 'prob', 'prob_inv')
        """
        if not _HAS_PYKEEN:
            raise ImportError("PyKEEN is required for KGE weight scoring. Please install it.")
        
        self.kge_model_path = kge_model_path
        self.kge_model_type = kge_model_type
        self.device = resolve_device(device)
        self.freeze_kge = freeze_kge
        self.weight_mode = weight_mode
        
        # 加載KGE模型
        self.kge_model = self._load_kge_model()
        
        # 實體和關係映射
        self.entity_to_id = {}
        self.relation_to_id = {}
        self.id_to_entity = {}
        self.id_to_relation = {}
        
        # 是否啟用KGE
        self.enabled = True
        
        print(f"✅ KGE Weight Scorer initialized:")
        print(f"   Model: {kge_model_type}")
        print(f"   Path: {kge_model_path}")
        print(f"   Weight mode: {weight_mode}")
        print(f"   Device: {self.device}")
        
    def _load_kge_model(self):
        """加載預訓練的KGE模型"""
        if not os.path.exists(self.kge_model_path):
            raise FileNotFoundError(f"KGE model not found at {self.kge_model_path}")
        
        # 加載模型狀態
        try:
            checkpoint = torch.load(self.kge_model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            # 如果 weights_only=False 失敗，嘗試其他方法
            print(f"Warning: Failed to load with weights_only=False: {e}")
            try:
                checkpoint = torch.load(self.kge_model_path, map_location=self.device)
            except Exception as e2:
                raise RuntimeError(f"Failed to load KGE model from {self.kge_model_path}: {e2}")
        
        # 根據模型類型創建模型實例
        if self.kge_model_type == 'TransE':
            model_class = TransE
        elif self.kge_model_type == 'DistMult':
            model_class = DistMult
        elif self.kge_model_type == 'ComplEx':
            model_class = ComplEx
        else:
            raise ValueError(f"Unsupported KGE model type: {self.kge_model_type}")
        
        # 從checkpoint中提取模型參數
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 創建模型實例（需要實體和關係數量）
        num_entities = self._get_num_entities(state_dict)
        num_relations = self._get_num_relations(state_dict)
        embedding_dim = self._get_embedding_dim(state_dict)
        
        # 創建虛擬的 triples factory 來滿足 PyKEEN 的要求
        dummy_triples = np.array([['entity_0', 'relation_0', 'entity_0']])  # 單個虛擬三元組
        triples_factory = TriplesFactory.from_labeled_triples(
            dummy_triples,
            entity_to_id={f'entity_{i}': i for i in range(num_entities)},
            relation_to_id={f'relation_{i}': i for i in range(num_relations)}
        )
        
        kge_model = model_class(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim
        )
        
        # 加載權重
        kge_model.load_state_dict(state_dict)
        kge_model.to(self.device)
        
        if self.freeze_kge:
            for param in kge_model.parameters():
                param.requires_grad = False
        
        return kge_model
    
    def _get_num_entities(self, state_dict: Dict) -> int:
        """從狀態字典中推斷實體數量"""
        # PyKEEN 格式
        if 'entity_representations.0._embeddings.weight' in state_dict:
            return state_dict['entity_representations.0._embeddings.weight'].shape[0]
        # 標準格式
        elif 'entity_embeddings.weight' in state_dict:
            return state_dict['entity_embeddings.weight'].shape[0]
        else:
            raise ValueError("Cannot determine number of entities from state dict")
    
    def _get_num_relations(self, state_dict: Dict) -> int:
        """從狀態字典中推斷關係數量"""
        # PyKEEN 格式
        if 'relation_representations.0._embeddings.weight' in state_dict:
            return state_dict['relation_representations.0._embeddings.weight'].shape[0]
        # 標準格式
        elif 'relation_embeddings.weight' in state_dict:
            return state_dict['relation_embeddings.weight'].shape[0]
        else:
            raise ValueError("Cannot determine number of relations from state dict")
    
    def _get_embedding_dim(self, state_dict: Dict) -> int:
        """從狀態字典中推斷嵌入維度"""
        # PyKEEN 格式
        if 'entity_representations.0._embeddings.weight' in state_dict:
            return state_dict['entity_representations.0._embeddings.weight'].shape[1]
        # 標準格式
        elif 'entity_embeddings.weight' in state_dict:
            return state_dict['entity_embeddings.weight'].shape[1]
        else:
            raise ValueError("Cannot determine embedding dimension from state dict")
    
    def set_entity_mapping(self, entity_to_id: Dict[str, int]):
        """設置實體名稱到ID的映射"""
        self.entity_to_id = entity_to_id
        self.id_to_entity = {v: k for k, v in entity_to_id.items()}
    
    def set_relation_mapping(self, relation_to_id: Dict[str, int]):
        """設置關係名稱到ID的映射"""
        self.relation_to_id = relation_to_id
        self.id_to_relation = {v: k for k, v in relation_to_id.items()}
    
    def compute_triple_weights(self, 
                              h_ids: torch.Tensor, 
                              r_ids: torch.Tensor, 
                              t_ids: torch.Tensor) -> torch.Tensor:
        """
        計算三元組的KGE權重
        
        Args:
            h_ids: 頭實體ID張量 [num_triples]
            r_ids: 關係ID張量 [num_triples] 
            t_ids: 尾實體ID張量 [num_triples]
            
        Returns:
            weights: KGE權重張量 [num_triples]
        """
        if not self.enabled:
            return torch.ones_like(h_ids, dtype=torch.float32, device=h_ids.device)
        
        device = h_ids.device
        
        # 將ID張量移動到KGE模型設備
        h_ids_kge = h_ids.to(self.device)
        r_ids_kge = r_ids.to(self.device)
        t_ids_kge = t_ids.to(self.device)
        
        # 計算KGE分數
        with torch.no_grad() if self.freeze_kge else torch.enable_grad():
            # 將 h, r, t 合併為 hrt_batch 格式 [num_triples, 3]
            hrt_batch = torch.stack([h_ids_kge, r_ids_kge, t_ids_kge], dim=1)
            # 計算三元組分數
            kge_scores = self.kge_model.score_hrt(hrt_batch)
            
            # 根據權重模式轉換分數為權重
            if self.weight_mode == 'score':
                # 直接使用分數作為權重（需要歸一化）
                weights = torch.sigmoid(kge_scores)  # 轉換到 [0, 1]
                weights = weights * 2.0  # 縮放到 [0, 2]
                
            elif self.weight_mode == 'score_inv':
                # 使用分數的倒數作為權重
                weights = torch.sigmoid(-kge_scores)  # 分數越低權重越高
                weights = weights * 2.0  # 縮放到 [0, 2]
                
            elif self.weight_mode == 'prob':
                # 使用概率作為權重
                weights = torch.sigmoid(kge_scores)
                
            elif self.weight_mode == 'prob_inv':
                # 使用概率的倒數作為權重
                weights = torch.sigmoid(-kge_scores)
                
            else:
                raise ValueError(f"Unsupported weight mode: {self.weight_mode}")
        
        # 移動回原始設備
        return weights.to(device)
    
    def compute_batch_triple_weights(self, 
                                   h_id_tensors: List[torch.Tensor],
                                   r_id_tensors: List[torch.Tensor], 
                                   t_id_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        計算批次中每個樣本的三元組權重
        
        Args:
            h_id_tensors: 頭實體ID張量列表
            r_id_tensors: 關係ID張量列表
            t_id_tensors: 尾實體ID張量列表
            
        Returns:
            weights_list: 權重張量列表
        """
        if not self.enabled:
            return [torch.ones_like(h, dtype=torch.float32) for h in h_id_tensors]
        
        weights_list = []
        
        for h_ids, r_ids, t_ids in zip(h_id_tensors, r_id_tensors, t_id_tensors):
            if len(h_ids) == 0:
                weights_list.append(torch.tensor([], dtype=torch.float32, device=h_ids.device))
                continue
            
            weights = self.compute_triple_weights(h_ids, r_ids, t_ids)
            weights_list.append(weights)
        
        return weights_list
    
    def disable(self):
        """禁用KGE權重計算"""
        self.enabled = False
    
    def enable(self):
        """啟用KGE權重計算"""
        self.enabled = True


def create_kge_weight_scorer(kge_model_path: str, 
                           kge_model_type: str = 'TransE',
                           device: str = 'cuda',
                           weight_mode: str = 'score') -> Optional[KGEWeightScorer]:
    """
    創建 KGE 權重計算器的便捷函數
    
    Args:
        kge_model_path: PyKEEN 模型路徑
        kge_model_type: 模型類型
        device: 計算設備
        weight_mode: 權重計算模式
        
    Returns:
        KGEWeightScorer 實例或 None（如果 PyKEEN 不可用）
    """
    if not _HAS_PYKEEN:
        logging.warning("PyKEEN not available. KGE weight scoring disabled.")
        return None
    
    try:
        return KGEWeightScorer(
            kge_model_path=kge_model_path,
            kge_model_type=kge_model_type,
            device=device,
            weight_mode=weight_mode
        )
    except Exception as e:
        logging.error(f"Failed to create KGE weight scorer: {e}")
        return None
