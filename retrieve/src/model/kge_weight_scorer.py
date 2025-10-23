"""
KGE-based triple weight scorer using PyG KGE models
"""
import torch
import torch.nn as nn
import os
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from torch_geometric.nn import TransE, DistMult, ComplEx, RotatE
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    logging.warning("PyG not available. KGE weight scoring will be disabled.")

class KGEWeightScorer:
    """
    使用 PyG 預訓練模型為 triples 計算權重
    支持 TransE, DistMult, ComplEx, RotatE 等模型
    """
    
    def __init__(self, 
                 kge_model_path: str,
                 kge_model_type: str = 'TransE',
                 device: str = 'cuda',
                 freeze_kge: bool = True,
                 weight_mode: str = 'score'):
        """
        Args:
            kge_model_path: PyG 模型路徑
            kge_model_type: 模型類型 ('TransE', 'DistMult', 'ComplEx', 'RotatE')
            device: 計算設備
            freeze_kge: 是否凍結 KGE 模型參數
            weight_mode: 權重計算模式 ('score', 'score_inv', 'prob', 'prob_inv', 'raw', 'raw_inv')
        """
        if not _HAS_PYG:
            raise ImportError("PyG is required for KGE weight scoring. Please install it.")
        
        self.kge_model_path = kge_model_path
        self.kge_model_type = kge_model_type
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
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
        """加載預訓練的PyG KGE模型"""
        if not os.path.exists(self.kge_model_path):
            raise FileNotFoundError(f"KGE model not found at {self.kge_model_path}")
        
        # 加載模型檢查點
        try:
            checkpoint = torch.load(self.kge_model_path, map_location=self.device)
        except Exception as e:
            try:
                checkpoint = torch.load(self.kge_model_path, map_location='cpu')
            except Exception as e2:
                raise RuntimeError(f"Failed to load KGE model from {self.kge_model_path}: {e2}")
        
        # 如果是已載入的 PyG 模型物件，直接返回
        try:
            if hasattr(checkpoint, 'score') and hasattr(checkpoint, 'state_dict'):
                kge_model = checkpoint
                try:
                    kge_model.to(self.device)
                except Exception:
                    pass
                return kge_model
        except Exception:
            pass
        
        # 從檢查點加載狀態字典
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint)
        else:
            state_dict = checkpoint
        
        # 根據模型類型創建模型實例
        model_class_map = {
            'TransE': TransE,
            'DistMult': DistMult,
            'ComplEx': ComplEx,
            'RotatE': RotatE
        }
        
        model_class = model_class_map.get(self.kge_model_type)
        if model_class is None:
            raise ValueError(f"Unsupported KGE model type: {self.kge_model_type}")
        
        # 從狀態字典推斷模型參數
        num_entities = self._get_num_entities(state_dict)
        num_relations = self._get_num_relations(state_dict)
        embedding_dim = self._get_embedding_dim(state_dict)
        
        # 創建 PyG KGE 模型實例
        model_kwargs = {}
        if self.kge_model_type == 'RotatE':
            model_kwargs['margin'] = 9.0  # RotatE 默認 margin
        
        kge_model = model_class(
            num_nodes=num_entities,
            num_relations=num_relations,
            hidden_channels=embedding_dim,
            **model_kwargs
        )
        
        # 修復 ComplEx 和 RotatE 模型的權重加載問題
        if self.kge_model_type in ['ComplEx', 'RotatE']:
            # ComplEx 和 RotatE 都需要實部和虛部權重，如果只有實部，複製一份作為虛部
            if 'node_emb.weight' in state_dict and 'node_emb_im.weight' not in state_dict:
                state_dict['node_emb_im.weight'] = state_dict['node_emb.weight'].clone()
            if 'rel_emb.weight' in state_dict and 'rel_emb_im.weight' not in state_dict:
                state_dict['rel_emb_im.weight'] = state_dict['rel_emb.weight'].clone()
        
        # 加載權重
        try:
            kge_model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning: Failed to load complete state dict: {e}")
            # 嘗試只加載存在的權重
            model_dict = kge_model.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(filtered_dict)
            kge_model.load_state_dict(model_dict)
        
        kge_model.to(self.device)
        
        if self.freeze_kge:
            for param in kge_model.parameters():
                param.requires_grad = False
        
        return kge_model
    
    def _get_num_entities(self, state_dict: Dict) -> int:
        """從狀態字典中推斷實體數量"""
        # PyG 格式 - 查找實體嵌入層
        for key in state_dict.keys():
            if 'node_emb' in key and 'weight' in key:
                return state_dict[key].shape[0]
            elif 'entity_emb' in key and 'weight' in key:
                return state_dict[key].shape[0]
            elif 'node_emb_re' in key and 'weight' in key:  # ComplEx 實部
                return state_dict[key].shape[0]
        # 回退到 PyKEEN 格式
        if 'entity_representations.0._embeddings.weight' in state_dict:
            return state_dict['entity_representations.0._embeddings.weight'].shape[0]
        # 標準格式
        elif 'entity_embeddings.weight' in state_dict:
            return state_dict['entity_embeddings.weight'].shape[0]
        else:
            raise ValueError("Cannot determine number of entities from state dict")
    
    def _get_num_relations(self, state_dict: Dict) -> int:
        """從狀態字典中推斷關係數量"""
        # PyG 格式 - 查找關係嵌入層
        for key in state_dict.keys():
            if 'rel_emb' in key and 'weight' in key:
                return state_dict[key].shape[0]
            elif 'relation_emb' in key and 'weight' in key:
                return state_dict[key].shape[0]
            elif 'rel_emb_re' in key and 'weight' in key:  # ComplEx 實部
                return state_dict[key].shape[0]
        # 回退到 PyKEEN 格式
        if 'relation_representations.0._embeddings.weight' in state_dict:
            return state_dict['relation_representations.0._embeddings.weight'].shape[0]
        # 標準格式
        elif 'relation_embeddings.weight' in state_dict:
            return state_dict['relation_embeddings.weight'].shape[0]
        else:
            raise ValueError("Cannot determine number of relations from state dict")
    
    def _get_embedding_dim(self, state_dict: Dict) -> int:
        """從狀態字典中推斷嵌入維度"""
        # PyG 格式 - 查找實體嵌入層
        for key in state_dict.keys():
            if 'node_emb' in key and 'weight' in key:
                return state_dict[key].shape[1]
            elif 'entity_emb' in key and 'weight' in key:
                return state_dict[key].shape[1]
            elif 'node_emb_re' in key and 'weight' in key:  # ComplEx 實部
                return state_dict[key].shape[1]
        # 回退到 PyKEEN 格式
        if 'entity_representations.0._embeddings.weight' in state_dict:
            return state_dict['entity_representations.0._embeddings.weight'].shape[1]
        # 標準格式
        elif 'entity_embeddings.weight' in state_dict:
            return state_dict['entity_embeddings.weight'].shape[1]
        else:
            raise ValueError("Cannot determine embedding dimension from state dict")
    
    def compute_triple_weights(self, 
                              h_ids: torch.Tensor, 
                              r_ids: torch.Tensor, 
                              t_ids: torch.Tensor) -> torch.Tensor:
        """
        計算三元組的權重
        
        Args:
            h_ids: 頭實體ID張量 [num_triples]
            r_ids: 關係ID張量 [num_triples] 
            t_ids: 尾實體ID張量 [num_triples]
            
        Returns:
            weights: 三元組權重張量 [num_triples]
        """
        if not self.enabled:
            return torch.ones(len(h_ids), device=h_ids.device)
        
        device = h_ids.device
        
        # 將ID張量移動到KGE模型設備
        h_ids_kge = h_ids.to(self.device)
        r_ids_kge = r_ids.to(self.device)
        t_ids_kge = t_ids.to(self.device)
        
        # 取得 KGE 模型的實體/關係大小
        num_entities = getattr(self.kge_model, 'num_nodes', None)
        num_relations = getattr(self.kge_model, 'num_relations', None)
        try:
            if num_entities is None and hasattr(self.kge_model, 'entity_representations'):
                emb = self.kge_model.entity_representations[0]._embeddings
                num_entities = getattr(emb, 'num_embeddings', None)
            if num_relations is None and hasattr(self.kge_model, 'relation_representations'):
                remb = self.kge_model.relation_representations[0]._embeddings
                num_relations = getattr(remb, 'num_embeddings', None)
        except Exception:
            pass
        
        # 建立預設權重為1
        weights = torch.ones(len(h_ids), device=device)
        
        # 篩選合法索引
        if (num_entities is not None) and (num_relations is not None):
            valid_mask = (h_ids_kge >= 0) & (h_ids_kge < num_entities) \
                         & (t_ids_kge >= 0) & (t_ids_kge < num_entities) \
                         & (r_ids_kge >= 0) & (r_ids_kge < num_relations)
        else:
            valid_mask = torch.ones_like(h_ids_kge, dtype=torch.bool)
        
        if valid_mask.any():
            with torch.no_grad() if self.freeze_kge else torch.enable_grad():
                # 使用 PyG 的 loss 方法的負值作為分數
                try:
                    kge_loss = self.kge_model.loss(
                        head_index=h_ids_kge[valid_mask],
                        rel_type=r_ids_kge[valid_mask],
                        tail_index=t_ids_kge[valid_mask]
                    )
                    kge_scores = -kge_loss  # 負損失作為分數
                except Exception as e:
                    print(f"Warning: Failed to compute loss with PyG model: {e}")
                    # 如果損失計算失敗，使用隨機分數
                    kge_scores = torch.zeros(len(h_ids_kge[valid_mask]), device=self.device)
            
            # 根據權重模式轉換分數為權重
            if self.weight_mode == 'score':
                weights[valid_mask] = kge_scores
            elif self.weight_mode == 'score_inv':
                weights[valid_mask] = 1.0 / (kge_scores + 1e-8)
            elif self.weight_mode == 'prob':
                # 使用 sigmoid 將分數轉換為概率
                probs = torch.sigmoid(kge_scores)
                weights[valid_mask] = probs
            elif self.weight_mode == 'prob_inv':
                probs = torch.sigmoid(kge_scores)
                weights[valid_mask] = 1.0 / (probs + 1e-8)
            elif self.weight_mode == 'raw':
                # 直接使用原始分數（可能為負）
                weights[valid_mask] = kge_scores
            elif self.weight_mode == 'raw_inv':
                weights[valid_mask] = 1.0 / (kge_scores + 1e-8)
            else:
                # 默認使用分數
                weights[valid_mask] = kge_scores
        
        return weights
    
    def enable(self):
        """啟用KGE權重計算"""
        self.enabled = True
    
    def compute_regularization_loss(self, 
                                  h_ids: torch.Tensor, 
                                  r_ids: torch.Tensor, 
                                  t_ids: torch.Tensor) -> torch.Tensor:
        """
        計算 KGE regularization loss
        
        Args:
            h_ids: 頭實體ID張量 [num_triples]
            r_ids: 關係ID張量 [num_triples] 
            t_ids: 尾實體ID張量 [num_triples]
            
        Returns:
            reg_loss: KGE regularization loss 標量
        """
        if not self.enabled:
            return torch.tensor(0.0, device=h_ids.device, requires_grad=True)
        
        device = h_ids.device
        
        # 將ID張量移動到KGE模型設備
        h_ids_kge = h_ids.to(self.device)
        r_ids_kge = r_ids.to(self.device)
        t_ids_kge = t_ids.to(self.device)
        
        # 取得 KGE 模型的實體/關係大小
        num_entities = getattr(self.kge_model, 'num_nodes', None)
        num_relations = getattr(self.kge_model, 'num_relations', None)

        # 篩選合法索引
        if (num_entities is not None) and (num_relations is not None):
            valid_mask = (h_ids_kge >= 0) & (h_ids_kge < num_entities) \
                         & (t_ids_kge >= 0) & (t_ids_kge < num_entities) \
                         & (r_ids_kge >= 0) & (r_ids_kge < num_relations)
        else:
            valid_mask = torch.ones_like(h_ids_kge, dtype=torch.bool)

        if not valid_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 計算 KGE loss 作為 regularization
        with torch.enable_grad():
            try:
                kge_loss = self.kge_model.loss(
                    head_index=h_ids_kge[valid_mask],
                    rel_type=r_ids_kge[valid_mask],
                    tail_index=t_ids_kge[valid_mask]
                )
                # 將 KGE loss 作為 regularization term
                # 使用負值，因為我們希望 KGE 分數越高越好（loss 越低越好）
                reg_loss = -kge_loss.mean()
                
            except Exception as e:
                print(f"Warning: Failed to compute KGE regularization loss: {e}")
                reg_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return reg_loss


def create_kge_weight_scorer(kge_model_path: str,
                           kge_model_type: str,
                           device: str = 'cuda',
                           weight_mode: str = 'score') -> Optional[KGEWeightScorer]:
    """
    創建 KGE 權重計算器
    
    Args:
        kge_model_path: PyG 模型路徑
        kge_model_type: 模型類型
        device: 計算設備
        weight_mode: 權重計算模式
        
    Returns:
        KGEWeightScorer 實例或 None（如果 PyG 不可用）
    """
    if not _HAS_PYG:
        logging.warning("PyG not available. KGE weight scoring disabled.")
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