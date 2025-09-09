import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class PEConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')

    def forward(self, edge_index, x):
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
    def __init__(self, emb_size, topic_pe, DDE_kwargs):
        super().__init__()
        
        self.non_text_entity_emb = nn.Embedding(1, emb_size)
        self.topic_pe = topic_pe
        self.dde = DDE(**DDE_kwargs)
        
        pred_in_size = 4 * emb_size
        if topic_pe:
            pred_in_size += 2 * 2
        pred_in_size += 2 * 2 * (DDE_kwargs['num_rounds'] + DDE_kwargs['num_reverse_rounds'])

        self.pred = nn.Sequential(
            nn.Linear(pred_in_size, emb_size),
            nn.ReLU(),
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
        h_e = torch.cat([
            entity_embs,
            self.non_text_entity_emb(torch.LongTensor([0]).to(device))
            .expand(num_non_text_entities, -1)
        ], dim=0)
        
        # 構建特徵列表
        h_e_list = [h_e]
        if self.topic_pe:
            h_e_list.append(topic_entity_one_hot)

        # 構建邊索引
        edge_index = torch.stack([h_id_tensor, t_id_tensor], dim=0)
        reverse_edge_index = torch.stack([t_id_tensor, h_id_tensor], dim=0)
        
        # DDE 處理
        dde_list = self.dde(topic_entity_one_hot, edge_index, reverse_edge_index)
        h_e_list.extend(dde_list)
        
        # 拼接特徵
        h_e = torch.cat(h_e_list, dim=1)

        # 構建三元組特徵
        h_q = q_emb
        h_r = relation_embs[r_id_tensor]

        h_triple = torch.cat([
            h_q.expand(len(h_r), -1),
            h_e[h_id_tensor],
            h_r,
            h_e[t_id_tensor]
        ], dim=1)
        
        return self.pred(h_triple)

    def _forward_batch(self, batch_data):
        """多樣本批次前向傳播"""
        h_id_tensors = batch_data['h_id_tensors']
        r_id_tensors = batch_data['r_id_tensors']
        t_id_tensors = batch_data['t_id_tensors']
        q_embs = batch_data['q_embs']
        entity_embs_list = batch_data['entity_embs_list']
        num_non_text_entities_list = batch_data['num_non_text_entities_list']
        relation_embs_list = batch_data['relation_embs_list']
        topic_entity_one_hots = batch_data['topic_entity_one_hots']
        
        batch_size = len(h_id_tensors)
        device = entity_embs_list[0].device
        
        # 分別處理每個樣本
        batch_results = []
        
        for i in range(batch_size):
            # 構建實體嵌入
            h_e = torch.cat([
                entity_embs_list[i],
                self.non_text_entity_emb(torch.LongTensor([0]).to(device))
                .expand(num_non_text_entities_list[i], -1)
            ], dim=0)
            
            # 構建特徵列表
            h_e_list = [h_e]
            if self.topic_pe:
                h_e_list.append(topic_entity_one_hots[i])

            # 構建邊索引
            edge_index = torch.stack([h_id_tensors[i], t_id_tensors[i]], dim=0)
            reverse_edge_index = torch.stack([t_id_tensors[i], h_id_tensors[i]], dim=0)
            
            # DDE 處理
            dde_list = self.dde(topic_entity_one_hots[i], edge_index, reverse_edge_index)
            h_e_list.extend(dde_list)
            
            # 拼接特徵
            h_e = torch.cat(h_e_list, dim=1)

            # 構建三元組特徵
            h_q = q_embs[i]
            h_r = relation_embs_list[i][r_id_tensors[i]]

            h_triple = torch.cat([
                h_q.expand(len(h_r), -1),
                h_e[h_id_tensors[i]],
                h_r,
                h_e[t_id_tensors[i]]
            ], dim=1)
            
            # 預測
            pred_result = self.pred(h_triple)
            batch_results.append(pred_result)
        
        # 拼接所有樣本的結果
        return torch.cat(batch_results, dim=0)

    # 向後兼容的方法
    def forward_legacy(self, h_id_tensor, r_id_tensor, t_id_tensor, q_emb, 
                      entity_embs, num_non_text_entities, relation_embs, 
                      topic_entity_one_hot):
        """向後兼容的單樣本前向傳播"""
        batch_data = (h_id_tensor, r_id_tensor, t_id_tensor, q_emb, 
                     entity_embs, num_non_text_entities, relation_embs, 
                     topic_entity_one_hot)
        return self._forward_single(batch_data)