import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class PEConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')

    def forward(self, edge_index, x):
        # x: [N, C]，edge_index: [2, E]
        # 輸出維度同 x（聚合不改變通道數 C）
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

class DDE(nn.Module):
    def __init__(self, num_rounds, num_reverse_rounds):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for _ in range(num_rounds):
            self.layers.append(PEConv())
        
        self.reverse_layers = nn.ModuleList()
        for _ in range(num_reverse_rounds):
            self.reverse_layers.append(PEConv())
    
    def forward(self, topic_entity_one_hot, edge_index, reverse_edge_index):
        # topic_entity_one_hot: [N, 2]
        # edge_index / reverse_edge_index: [2, E]
        # 回傳 list，長度 K = num_rounds + num_reverse_rounds，
        # 其中每個元素形狀皆為 [N, 2]
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

class Retriever(nn.Module):
    def __init__(self, emb_size, topic_pe, DDE_kwargs, use_dropout: bool = False, dropout_p: float = 0.2):
        super().__init__()
        
        self.non_text_entity_emb = nn.Embedding(1, emb_size)
        self.topic_pe = topic_pe
        self.dde = DDE(**DDE_kwargs)
        
        # 三元組特徵寬度計算：
        # 基礎來自 (q, h, r, t) 共 4*d
        # 若啟用 topic_pe，額外來自 head/tail 的 topic one-hot：2 個位置 × 2 維 × 1 = 4
        # DDE 產生 K = num_rounds + num_reverse_rounds 組 2 維訊號，
        # head/tail 兩個位置共 2 × 2 × K = 4K
        pred_in_size = 4 * emb_size
        if topic_pe:
            pred_in_size += 2 * 2  # = 4
        pred_in_size += 2 * 2 * (DDE_kwargs['num_rounds'] + DDE_kwargs['num_reverse_rounds'])  # = 4K

        dropout_layer = nn.Dropout(dropout_p) if use_dropout else nn.Identity()

        self.pred = nn.Sequential(
            nn.Linear(pred_in_size, emb_size),
            nn.ReLU(),
            dropout_layer,
            nn.Linear(emb_size, 1)
        )

    def forward(self, batch_data):
        """
        支持單樣本和多樣本批次處理
        batch_data 可以是：
        1. 元組格式 (單樣本，來自 collate_retriever)
        2. 字典格式 (多樣本，來自 collate_retriever_batch)
        """
        if isinstance(batch_data, tuple):
            # 單樣本模式 (向後兼容)
            return self._forward_single(batch_data)
        elif isinstance(batch_data, dict):
            # 多樣本模式
            return self._forward_batch(batch_data)
        else:
            raise ValueError(f"Unsupported batch_data type: {type(batch_data)}")

    def _forward_single(self, batch_data):
        """單樣本前向傳播 (原有邏輯)"""
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs, \
        num_non_text_entities, relation_embs, topic_entity_one_hot = batch_data
        
        device = entity_embs.device
        
        # 構建實體嵌入
        # entity_embs: [N_text, d]
        # 非文本複製後: [N_non_text, d]
        # h_e: [N_text + N_non_text, d] = [N, d]
        h_e = torch.cat([
            entity_embs,
            # self.non_text_entity_emb(torch.LongTensor([0]).to(device))
            # .expand((num_non_text_entities, -1))
            self.non_text_entity_emb(torch.tensor([0], dtype=torch.long, device=device)).expand((num_non_text_entities, -1))
        ], dim=0)
        
        # 構建特徵列表
        # 若啟用 topic_pe，附加 topic one-hot: [N, 2]
        h_e_list = [h_e]
        if self.topic_pe:
            h_e_list.append(topic_entity_one_hot)

        # 構建邊索引
        # edge_index / reverse_edge_index: [2, E]
        edge_index = torch.stack([h_id_tensor, t_id_tensor], dim=0)
        reverse_edge_index = torch.stack([t_id_tensor, h_id_tensor], dim=0)
        
        # DDE 處理
        # dde_list: 長度 K，每個 [N, 2]
        dde_list = self.dde(topic_entity_one_hot, edge_index, reverse_edge_index)
        h_e_list.extend(dde_list)
        
        # 拼接特徵
        # h_e 最終形狀: [N, d + (topic?2:0) + 2K]
        h_e = torch.cat(h_e_list, dim=1)

        # 構建三元組特徵
        h_q = q_emb              # [d]
        h_r = relation_embs[r_id_tensor]  # [E, d]

        h_triple = torch.cat([
            h_q.expand(len(h_r), -1),   # [E, d]
            h_e[h_id_tensor],           # [E, d + (topic?2:0) + 2K]
            h_r,                        # [E, d]
            h_e[t_id_tensor]            # [E, d + (topic?2:0) + 2K]
        ], dim=1)
        
        # h_triple: [E, 4d + (topic?4:0) + 4K]
        return self.pred(h_triple)

    def _forward_batch(self, batch_data):
        """多樣本批次前向傳播"""
        h_id_tensors = batch_data['h_id_tensors']
        r_id_tensors = batch_data['r_id_tensors']
        t_id_tensors = batch_data['t_id_tensors']
        # 支援 q_embs 與 q_emb 兩種鍵名
        q_embs = batch_data['q_embs'] if 'q_embs' in batch_data else batch_data['q_emb']
        # 使用 list 版欄位，避免因 padding 造成的尺寸不一致
        entity_embs_list = batch_data['entity_embs_list'] if 'entity_embs_list' in batch_data else batch_data['entity_embs']
        relation_embs_list = batch_data['relation_embs_list'] if 'relation_embs_list' in batch_data else batch_data['relation_embs']
        topic_entity_one_hots = batch_data['topic_entity_one_hot_list'] if 'topic_entity_one_hot_list' in batch_data else batch_data['topic_entity_one_hot']
        num_non_text_entities_list = batch_data['num_non_text_entities']
        
        batch_size = q_embs.shape[0]
        device = q_embs.device
        
        # 分別處理每個樣本
        batch_results = []
        
        for i in range(batch_size):
            # 將當前樣本所需張量移動到 device，確保裝置一致
            entity_embs = entity_embs_list[i].to(device)
            relation_embs = relation_embs_list[i].to(device)
            topic_entity_one_hot = topic_entity_one_hots[i].to(device)
            h_id = h_id_tensors[i].to(device)
            t_id = t_id_tensors[i].to(device)
            r_id = r_id_tensors[i].to(device)
            
            # 構建實體嵌入（文本 + 非文本）
            # h_e: [N_i, d]
            non_text_emb = self.non_text_entity_emb(torch.LongTensor([0]).to(device))\
                .expand((num_non_text_entities_list[i], -1))
            h_e = torch.cat([entity_embs, non_text_emb], dim=0)
            
            # 構建特徵列表，並確保 topic one-hot 與 h_e 在第0維對齊
            h_e_list = [h_e]
            if self.topic_pe:
                if topic_entity_one_hot.shape[0] != h_e.shape[0]:
                    # 對齊長度：優先截斷，必要時再做零填充
                    if topic_entity_one_hot.shape[0] > h_e.shape[0]:
                        topic_entity_one_hot = topic_entity_one_hot[:h_e.shape[0]]
                    else:
                        pad_len = h_e.shape[0] - topic_entity_one_hot.shape[0]
                        pad = torch.zeros((pad_len, topic_entity_one_hot.shape[1]), device=topic_entity_one_hot.device, dtype=topic_entity_one_hot.dtype)
                        topic_entity_one_hot = torch.cat([topic_entity_one_hot, pad], dim=0)
                h_e_list.append(topic_entity_one_hot)
            
            # 構建邊索引
            # edge_index / reverse_edge_index: [2, E_i]
            edge_index = torch.stack([h_id, t_id], dim=0)
            reverse_edge_index = torch.stack([t_id, h_id], dim=0)
            
            # DDE 處理
            # 若未啟用 topic_pe，則以零張量替代，形狀仍為 [N_i, 2]
            dde_list = self.dde(topic_entity_one_hot if self.topic_pe else torch.zeros_like(h_e[:, :2]), edge_index, reverse_edge_index)
            h_e_list.extend(dde_list)
            
            # 拼接特徵
            # h_e: [N_i, d + (topic?2:0) + 2K]
            h_e = torch.cat(h_e_list, dim=1)

            # 構建三元組特徵
            h_q = q_embs[i]                 # [d]
            h_r = relation_embs[r_id]       # [E_i, d]

            h_triple = torch.cat([
                h_q.expand((len(h_r), -1)),  # [E_i, d]
                h_e[h_id],                   # [E_i, d + (topic?2:0) + 2K]
                h_r,                          # [E_i, d]
                h_e[t_id]                    # [E_i, d + (topic?2:0) + 2K]
            ], dim=1)
            
            # 預測
            # 輸出形狀: [E_i, 1]
            pred_result = self.pred(h_triple)
            batch_results.append(pred_result)
        
        return batch_results

    # 向後兼容的方法
    def forward_legacy(self, h_id_tensor, r_id_tensor, t_id_tensor, q_emb, 
                      entity_embs, num_non_text_entities, relation_embs, 
                      topic_entity_one_hot):
        """向後兼容的單樣本前向傳播"""
        batch_data = (h_id_tensor, r_id_tensor, t_id_tensor, q_emb, 
                     entity_embs, num_non_text_entities, relation_embs, 
                     topic_entity_one_hot)
        return self._forward_single(batch_data)