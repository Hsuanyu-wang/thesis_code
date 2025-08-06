import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing
from .kge_models import create_kge_model

# PEConv 類別：用於圖神經網路的訊息傳遞，繼承自 MessagePassing
class PEConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')  # 使用 mean 聚合方式

    def forward(self, edge_index, x):
        """
        前向傳播函數。
        參數：
            edge_index: 圖的邊索引 (2, num_edges)
            x: 節點特徵 (num_nodes, 特徵維度)
        回傳：
            傳遞後的節點特徵
        """
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        """
        訊息函數，這裡直接回傳鄰居節點的特徵。
        參數：
            x_j: 鄰居節點的特徵
        回傳：
            x_j
        """
        return x_j

# DDE 類別：雙向訊息傳遞的圖神經網路模組
class DDE(nn.Module):
    def __init__(
        self,
        num_rounds,           # 正向傳遞的層數
        num_reverse_rounds    # 反向傳遞的層數
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for _ in range(num_rounds):
            self.layers.append(PEConv())  # 正向 PEConv 層
        
        self.reverse_layers = nn.ModuleList()
        for _ in range(num_reverse_rounds):
            self.reverse_layers.append(PEConv())  # 反向 PEConv 層
    
    def forward(
        self,
        topic_entity_one_hot,   # 主題實體的 one-hot 向量
        edge_index,             # 正向邊索引
        reverse_edge_index      # 反向邊索引
    ):
        """
        前向傳播，依序經過正向與反向 PEConv 層，並收集每層的輸出。
        參數：
            topic_entity_one_hot: 主題實體 one-hot 向量
            edge_index: 正向邊索引
            reverse_edge_index: 反向邊索引
        回傳：
            每一層輸出的 list
        """
        result_list = []
        
        h_pe = topic_entity_one_hot
        for layer in self.layers:
            h_pe = layer(edge_index, h_pe)
            result_list.append(h_pe)
        
        h_pe_rev = topic_entity_one_hot
        for layer in self.reverse_layers:
            h_pe_rev = layer(reverse_edge_index, h_pe_rev)
            result_list.append(h_pe_rev)
        
        return result_list

# Retriever 類別：檢索器主體，結合圖神經網路與 KGE
class Retriever(nn.Module):
    def __init__(
        self,
        emb_size,        # 嵌入維度
        topic_pe,        # 是否使用主題實體位置編碼
        DDE_kwargs,      # DDE 初始化參數 (dict)
        kge_config=None  # KGE 配置 (dict, 可選)
    ):
        super().__init__()
        
        self.non_text_entity_emb = nn.Embedding(1, emb_size)  # 非文本實體嵌入
        self.topic_pe = topic_pe  # 是否使用主題實體位置編碼
        self.dde = DDE(**DDE_kwargs)  # DDE 模組
        
        # KGE 整合 - 僅用於分數計算
        self.use_kge = kge_config is not None
        if self.use_kge:
            self.kge_model = create_kge_model(
                model_type=kge_config['model_type'],
                num_entities=kge_config['num_entities'],
                num_relations=kge_config['num_relations'],
                embedding_dim=kge_config['embedding_dim'],
                margin=kge_config.get('margin', 1.0),
                norm=kge_config.get('norm', 1)
            )
        
        pred_in_size = 4 * emb_size  # 預測器輸入維度
        if topic_pe:
            pred_in_size += 2 * 2
        pred_in_size += 2 * 2 * (DDE_kwargs['num_rounds'] + DDE_kwargs['num_reverse_rounds'])
        
        # 不需額外 KGE 嵌入維度

        self.pred = nn.Sequential(
            nn.Linear(pred_in_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1)
        )  # MLP 預測器

    def forward(
        self,
        h_id_tensor,           # 頭實體 id 張量
        r_id_tensor,           # 關係 id 張量
        t_id_tensor,           # 尾實體 id 張量
        q_emb,                 # 查詢嵌入
        entity_embs,           # 實體嵌入 (文本)
        num_non_text_entities, # 非文本實體數量
        relation_embs,         # 關係嵌入
        topic_entity_one_hot   # 主題實體 one-hot
    ):
        """
        前向傳播，計算檢索分數與（可選）KGE 分數。
        參數：
            h_id_tensor: 頭實體 id
            r_id_tensor: 關係 id
            t_id_tensor: 尾實體 id
            q_emb: 查詢嵌入
            entity_embs: 文本實體嵌入
            num_non_text_entities: 非文本實體數
            relation_embs: 關係嵌入
            topic_entity_one_hot: 主題實體 one-hot
        回傳：
            mlp_logits: MLP 預測分數
            kge_score: KGE 分數（如有）
        """
        device = entity_embs.device
        
        h_e = torch.cat(
            [
                entity_embs,  # 文本實體嵌入
                self.non_text_entity_emb(
                    torch.LongTensor([0]).to(device)).expand(num_non_text_entities, -1)
            ]
        , dim=0)
        h_e_list = [h_e]
        if self.topic_pe:
            h_e_list.append(topic_entity_one_hot)

        edge_index = torch.stack([
            h_id_tensor,
            t_id_tensor
        ], dim=0)  # 正向邊索引
        reverse_edge_index = torch.stack([
            t_id_tensor,
            h_id_tensor
        ], dim=0)  # 反向邊索引
        dde_list = self.dde(topic_entity_one_hot, edge_index, reverse_edge_index)
        h_e_list.extend(dde_list)
        h_e = torch.cat(h_e_list, dim=1)

        h_q = q_emb  # 查詢嵌入
        # 可能有記憶體問題
        h_r = relation_embs[r_id_tensor]  # 關係嵌入

        # 準備預測輸入，保持原本 GTE 拼接方式
        h_triple = torch.cat([
            h_q.expand(len(h_r), -1),
            h_e[h_id_tensor],
            h_r,
            h_e[t_id_tensor]
        ], dim=1)
        
        # MLP 預測
        mlp_logits = self.pred(h_triple)
        
        # 若有 KGE，計算 KGE 分數
        kge_score = None
        if self.use_kge and len(h_id_tensor) > 0:
            kge_score = self.kge_model.predict(h_id_tensor, r_id_tensor, t_id_tensor)
        
        return mlp_logits, kge_score
