# KGE Integration for SubgraphRAG

本文件說明如何在SubgraphRAG中整合知識圖譜嵌入（Knowledge Graph Embedding, KGE）方法，包括TransE、DistMult和PTransE。

## 概述

SubgraphRAG原本只考慮語意嵌入（GTE embedding model）和距離嵌入（DDE）。我們現在加入了KGE作為第三種嵌入方式，以更好地捕捉知識圖譜中的結構信息。

### 新增的KGE方法

1. **TransE**: 將關係視為實體間的平移向量
2. **DistMult**: 使用三向張量分解來建模實體和關係
3. **PTransE**: TransE的改進版本，考慮路徑信息

## 文件結構

```
retrieve/
├── src/model/
│   ├── kge_models.py          # KGE模型實現
│   ├── kge_utils.py           # KGE工具函數
│   └── retriever.py           # 修改後的檢索器（整合KGE）
├── train_kge.py               # KGE模型訓練腳本
├── run_kge_integration.py     # 完整整合流程腳本
└── KGE_INTEGRATION_README.md  # 本文件
```

## 快速開始

### 1. 訓練KGE模型

```bash
# 訓練TransE模型
python train_kge.py --dataset webqsp --split train

# 訓練DistMult模型
python train_kge.py --dataset cwq --split train
```

### 2. 運行完整流程

```bash
# 使用TransE進行完整訓練
python run_kge_integration.py --dataset webqsp --kge_model transe --full_pipeline

# 使用DistMult進行完整訓練
python run_kge_integration.py --dataset cwq --kge_model distmult --full_pipeline
```

### 3. 分步驟執行

```bash
# 步驟1: 訓練KGE模型
python run_kge_integration.py --dataset webqsp --kge_model transe --train_kge

# 步驟2: 計算文本嵌入
python run_kge_integration.py --dataset webqsp --train_embeddings

# 步驟3: 訓練檢索器（整合KGE）
python run_kge_integration.py --dataset webqsp --train_retriever
```

## 配置說明

### KGE配置

在 `configs/retriever/{dataset}.yaml` 中添加KGE配置：

```yaml
kge:
  enabled: true
  model_type: 'transe'  # 選項: 'transe', 'distmult', 'ptranse'
  embedding_dim: 256
  num_epochs: 100
  batch_size: 1024
  learning_rate: 0.001
  margin: 1.0
  num_negatives: 1
  norm: 1  # TransE/PTransE的範數類型
```

### 參數說明

- `enabled`: 是否啟用KGE
- `model_type`: KGE模型類型
- `embedding_dim`: KGE嵌入維度
- `num_epochs`: KGE訓練輪數
- `batch_size`: 批次大小
- `learning_rate`: 學習率
- `margin`: 邊際損失的邊際值
- `num_negatives`: 負樣本數量
- `norm`: 範數類型（僅適用於TransE/PTransE）

## 模型架構

### 原始SubgraphRAG架構

```
輸入: [q_emb, h_emb, r_emb, t_emb] + DDE + PE
```

### 整合KGE後的架構

```
輸入: [q_emb, h_emb, r_emb, t_emb, kge_h_emb, kge_t_emb] + DDE + PE
```

其中：
- `q_emb`: 問題嵌入（GTE）
- `h_emb, t_emb`: 頭尾實體嵌入（GTE）
- `r_emb`: 關係嵌入（GTE）
- `kge_h_emb, kge_t_emb`: 頭尾實體的KGE嵌入
- `DDE`: 距離嵌入
- `PE`: 位置編碼

## 訓練流程

### 1. KGE模型訓練

```python
# 從知識圖譜三元組訓練KGE模型
kge_model = create_kge_model('transe', num_entities, num_relations, embedding_dim)
# 使用負採樣和邊際損失進行訓練
```

### 2. 檢索器整合

```python
# 在Retriever中整合KGE嵌入
retriever = Retriever(emb_size, kge_config=kge_config)
# KGE嵌入會與文本嵌入一起輸入到預測層
```

## 性能比較

### 評估指標

- `triple_recall@k`: 三元組召回率
- `ans_recall@k`: 答案實體召回率

### 預期改進

1. **結構感知**: KGE能更好地捕捉知識圖譜的結構信息
2. **關係建模**: 更好地建模實體間的關係
3. **泛化能力**: 提高對未見實體和關係的泛化能力

## 實驗建議

### 1. 模型選擇

- **TransE**: 適合簡單的關係類型
- **DistMult**: 適合對稱關係
- **PTransE**: 適合複雜的關係路徑

### 2. 超參數調優

- 調整 `embedding_dim` 以平衡性能和計算成本
- 調整 `margin` 以優化損失函數
- 調整 `num_negatives` 以改善負採樣

### 3. 消融研究

比較以下配置的性能：
- 僅GTE + DDE（原始）
- GTE + DDE + TransE
- GTE + DDE + DistMult
- GTE + DDE + PTransE

## 故障排除

### 常見問題

1. **KGE模型未找到**
   ```
   Warning: KGE is enabled but no trained model found
   ```
   解決方案：先運行 `train_kge.py` 訓練KGE模型

2. **內存不足**
   ```
   CUDA out of memory
   ```
   解決方案：減少 `batch_size` 或 `embedding_dim`

3. **配置錯誤**
   ```
   Unknown KGE model type
   ```
   解決方案：檢查 `model_type` 是否為 'transe', 'distmult', 或 'ptranse'

### 調試技巧

1. 檢查模型文件是否正確保存：
   ```bash
   ls data_files/{dataset}/kge/{model_type}/
   ```

2. 驗證配置文件：
   ```bash
   cat configs/retriever/{dataset}.yaml
   ```

3. 檢查訓練日誌：
   ```bash
   tail -f wandb/latest-run/logs/debug.log
   ```

## 進階用法

### 自定義KGE模型

可以通過繼承基礎類別來實現自定義KGE模型：

```python
class CustomKGE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super().__init__()
        # 實現自定義邏輯
    
    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        # 實現前向傳播
```

### 多模型集成

可以同時使用多個KGE模型：

```python
# 在配置中指定多個模型
kge:
  models: ['transe', 'distmult']
  ensemble_method: 'concat'  # 或 'average'
```

## 參考文獻

1. Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (2013). Translating embeddings for modeling multi-relational data. NIPS.
2. Yang, B., Yih, W., He, X., Gao, J., & Deng, L. (2015). Embedding entities and relations for learning and inference in knowledge bases. ICLR.
3. Lin, Y., Liu, Z., Sun, M., Liu, Y., & Zhu, X. (2015). Learning entity and relation embeddings for knowledge graph completion. AAAI.

## 聯繫方式

如有問題或建議，請聯繫開發團隊或提交issue。 