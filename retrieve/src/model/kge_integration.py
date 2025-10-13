import torch
import torch.nn as nn
import os
from typing import Dict, List, Optional, Tuple
import logging

try:
    from pykeen.models import TransE, DistMult, ComplEx
    from pykeen.triples import TriplesFactory
    from pykeen.utils import resolve_device
    _HAS_PYKEEN = True
except ImportError:
    _HAS_PYKEEN = False
    logging.warning("PyKEEN not available. KGE integration will be disabled.")

class KGEIntegration(nn.Module):
    """
    整合PyKEEN KGE模型到訓練pipeline中
    支持TransE, DistMult, ComplEx等模型
    """
    
    def __init__(self, 
                 kge_model_path: str,
                 kge_model_type: str = 'TransE',
                 device: str = 'cuda',
                 freeze_kge: bool = True,
                 kge_weight: float = 0.1):
        super().__init__()
        
        if not _HAS_PYKEEN:
            raise ImportError("PyKEEN is required for KGE integration. Please install it.")
        
        self.kge_model_path = kge_model_path
        self.kge_model_type = kge_model_type
        self.device = resolve_device(device)
        self.freeze_kge = freeze_kge
        self.kge_weight = kge_weight
        
        # 加載KGE模型
        self.kge_model = self._load_kge_model()
        
        # 創建實體和關係映射
        self.entity_to_id = {}
        self.relation_to_id = {}
        self.id_to_entity = {}
        self.id_to_relation = {}
        
        # 是否啟用KGE
        self.enabled = True
        
    def _load_kge_model(self):
        """加載預訓練的KGE模型"""
        if not os.path.exists(self.kge_model_path):
            raise FileNotFoundError(f"KGE model not found at {self.kge_model_path}")
        
        # 加載模型狀態
        checkpoint = torch.load(self.kge_model_path, map_location=self.device)
        
        # 根據模型類型創建模型實例
        if self.kge_model_type == 'TransE':
            model = TransE
        elif self.kge_model_type == 'DistMult':
            model = DistMult
        elif self.kge_model_type == 'ComplEx':
            model = ComplEx
        else:
            raise ValueError(f"Unsupported KGE model type: {self.kge_model_type}")
        
        # 從checkpoint中提取模型參數
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 創建模型實例（需要實體和關係數量）
        # 這裡需要從checkpoint或配置中獲取這些信息
        num_entities = self._get_num_entities(state_dict)
        num_relations = self._get_num_relations(state_dict)
        
        kge_model = model(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=state_dict['entity_embeddings.weight'].shape[1]
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
        if 'entity_embeddings.weight' in state_dict:
            return state_dict['entity_embeddings.weight'].shape[0]
        elif 'entity_representations.0._embeddings.weight' in state_dict:
            return state_dict['entity_representations.0._embeddings.weight'].shape[0]
        else:
            raise ValueError("Cannot determine number of entities from state dict")
    
    def _get_num_relations(self, state_dict: Dict) -> int:
        """從狀態字典中推斷關係數量"""
        if 'relation_embeddings.weight' in state_dict:
            return state_dict['relation_embeddings.weight'].shape[0]
        elif 'relation_representations.0._embeddings.weight' in state_dict:
            return state_dict['relation_representations.0._embeddings.weight'].shape[0]
        else:
            raise ValueError("Cannot determine number of relations from state dict")
    
    def set_entity_mapping(self, entity_to_id: Dict[str, int]):
        """設置實體名稱到ID的映射"""
        self.entity_to_id = entity_to_id
        self.id_to_entity = {v: k for k, v in entity_to_id.items()}
    
    def set_relation_mapping(self, relation_to_id: Dict[str, int]):
        """設置關係名稱到ID的映射"""
        self.relation_to_id = relation_to_id
        self.id_to_relation = {v: k for k, v in relation_to_id.items()}
    
    def compute_kge_scores(self, 
                          h_ids: torch.Tensor, 
                          r_ids: torch.Tensor, 
                          t_ids: torch.Tensor) -> torch.Tensor:
        """
        計算三元組的KGE分數
        
        Args:
            h_ids: 頭實體ID張量 [batch_size, num_triples]
            r_ids: 關係ID張量 [batch_size, num_triples] 
            t_ids: 尾實體ID張量 [batch_size, num_triples]
            
        Returns:
            kge_scores: KGE分數張量 [batch_size, num_triples]
        """
        if not self.enabled:
            return torch.zeros_like(h_ids, dtype=torch.float32, device=h_ids.device)
        
        batch_size, num_triples = h_ids.shape
        device = h_ids.device
        
        # 將ID張量移動到KGE模型設備
        h_ids_kge = h_ids.to(self.device)
        r_ids_kge = r_ids.to(self.device)
        t_ids_kge = t_ids.to(self.device)
        
        # 計算KGE分數
        with torch.no_grad() if self.freeze_kge else torch.enable_grad():
            # 重塑為 [batch_size * num_triples]
            h_flat = h_ids_kge.view(-1)
            r_flat = r_ids_kge.view(-1)
            t_flat = t_ids_kge.view(-1)
            
            # 計算三元組分數
            kge_scores_flat = self.kge_model.score_hrt(h_flat, r_flat, t_flat)
            
            # 重塑回 [batch_size, num_triples]
            kge_scores = kge_scores_flat.view(batch_size, num_triples)
        
        # 移動回原始設備
        return kge_scores.to(device)
    
    def compute_kge_scores_list(self, 
                               h_id_tensors: List[torch.Tensor],
                               r_id_tensors: List[torch.Tensor], 
                               t_id_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        計算批次中每個樣本的KGE分數（支持不同長度的三元組列表）
        
        Args:
            h_id_tensors: 頭實體ID張量列表
            r_id_tensors: 關係ID張量列表
            t_id_tensors: 尾實體ID張量列表
            
        Returns:
            kge_scores_list: KGE分數張量列表
        """
        if not self.enabled:
            return [torch.zeros_like(h, dtype=torch.float32) for h in h_id_tensors]
        
        kge_scores_list = []
        
        for h_ids, r_ids, t_ids in zip(h_id_tensors, r_id_tensors, t_id_tensors):
            if len(h_ids) == 0:
                kge_scores_list.append(torch.tensor([], dtype=torch.float32, device=h_ids.device))
                continue
            
            # 移動到KGE模型設備
            h_ids_kge = h_ids.to(self.device)
            r_ids_kge = r_ids.to(self.device)
            t_ids_kge = t_ids.to(self.device)
            
            # 計算KGE分數
            with torch.no_grad() if self.freeze_kge else torch.enable_grad():
                kge_scores = self.kge_model.score_hrt(h_ids_kge, r_ids_kge, t_ids_kge)
            
            # 移動回原始設備
            kge_scores_list.append(kge_scores.to(h_ids.device))
        
        return kge_scores_list
    
    def forward(self, batch_data: Dict) -> Dict[str, torch.Tensor]:
        """
        計算批次數據的KGE分數
        
        Args:
            batch_data: 包含三元組ID的批次數據
            
        Returns:
            包含KGE分數的字典
        """
        if 'h_id_tensors' in batch_data and 'r_id_tensors' in batch_data and 't_id_tensors' in batch_data:
            # 列表格式
            kge_scores_list = self.compute_kge_scores_list(
                batch_data['h_id_tensors'],
                batch_data['r_id_tensors'], 
                batch_data['t_id_tensors']
            )
            return {'kge_scores_list': kge_scores_list}
        else:
            # 張量格式（如果有的話）
            raise NotImplementedError("Tensor format KGE scoring not implemented yet")
    
    def disable(self):
        """禁用KGE計算"""
        self.enabled = False
    
    def enable(self):
        """啟用KGE計算"""
        self.enabled = True


class KGEEnhancedRetriever(nn.Module):
    """
    增強版Retriever，整合KGE分數
    """
    
    def __init__(self, 
                 base_retriever: nn.Module,
                 kge_integration: KGEIntegration,
                 kge_weight: float = 0.1):
        super().__init__()
        
        self.base_retriever = base_retriever
        self.kge_integration = kge_integration
        self.kge_weight = kge_weight
        
    def forward(self, batch_data: Dict) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        前向傳播，返回原始預測和KGE分數
        
        Returns:
            base_predictions: 基礎模型預測
            kge_outputs: KGE相關輸出
        """
        # 基礎模型預測
        base_predictions = self.base_retriever(batch_data)
        
        # KGE分數計算
        kge_outputs = self.kge_integration(batch_data)
        
        return base_predictions, kge_outputs
    
    def compute_combined_loss(self, 
                            base_predictions: List[torch.Tensor],
                            kge_scores_list: List[torch.Tensor],
                            targets: List[torch.Tensor]) -> torch.Tensor:
        """
        計算結合KGE分數的損失
        
        Args:
            base_predictions: 基礎模型預測
            kge_scores_list: KGE分數列表
            targets: 目標標籤列表
            
        Returns:
            combined_loss: 結合損失
        """
        base_loss = 0.0
        kge_loss = 0.0
        valid_samples = 0
        
        for i, (pred, kge_score, target) in enumerate(zip(base_predictions, kge_scores_list, targets)):
            if len(pred) == 0:
                continue
                
            # 對齊長度
            pred = pred.reshape(-1)
            target = target.reshape(-1)
            
            if len(kge_score) > 0:
                kge_score = kge_score.reshape(-1)
                min_len = min(len(pred), len(target), len(kge_score))
                pred = pred[:min_len]
                target = target[:min_len]
                kge_score = kge_score[:min_len]
            else:
                min_len = min(len(pred), len(target))
                pred = pred[:min_len]
                target = target[:min_len]
                kge_score = torch.zeros_like(pred)
            
            if min_len == 0:
                continue
            
            # 基礎損失
            num_positive = target.sum().item()
            num_total = len(target)
            pos_weight = torch.tensor([(num_total - num_positive) / num_total if num_positive > 0 else 1.0], 
                                    device=pred.device)
            base_loss += torch.nn.functional.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)
            
            # KGE損失（使用KGE分數作為額外的正則化項）
            if self.kge_integration.enabled and len(kge_score) > 0:
                # 將KGE分數轉換為概率並與目標對齊
                kge_prob = torch.sigmoid(kge_score)
                kge_loss += torch.nn.functional.binary_cross_entropy(kge_prob, target)
            
            valid_samples += 1
        
        if valid_samples == 0:
            return torch.tensor(0.0, requires_grad=True)
        
        base_loss = base_loss / valid_samples
        kge_loss = kge_loss / valid_samples if self.kge_integration.enabled else 0.0
        
        # 結合損失
        combined_loss = base_loss + self.kge_weight * kge_loss
        
        return combined_loss
