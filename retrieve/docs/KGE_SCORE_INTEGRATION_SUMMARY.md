# KGE Score Integration Summary

## 🎯 修改目標

根據你的需求，我們將原本的 KGE embedding 整合方式改為 KGE score 整合方式：

**原本的做法**：將 KGE 的 h, r, t embedding 分別加到 GTE 的 h, r, t embedding 上（element-wise add）

**新的做法**：
1. **保持原本的 GTE embedding concatenation**：`[Zq||Zh||Zr||Zt||Ztau]`
2. **KGE 只提供 score**：將 h, r, t 通過 KGE model 計算出三元組的 plausibility score
3. **KGE score 作為額外的 loss 項**：使用 margin ranking loss，與原本的 BCE loss 相加

## 📋 修改的文件

### 1. `src/model/retriever.py`
- **修改前**：KGE embedding 與 GTE embedding 相加後 concat
- **修改後**：保持原本的 GTE embedding concat，KGE 只計算 score
- **主要變更**：
  - 移除了 KGE embedding projection layer
  - 移除了 KGE embedding 的 concat
  - forward 方法現在返回 `(mlp_logits, kge_score)`

### 2. `train.py`
- **修改前**：只有 BCE loss
- **修改後**：BCE loss + λ * KGE margin ranking loss
- **主要變更**：
  - `train_epoch` 函數加入 config 參數
  - 計算 KGE margin ranking loss
  - 總 loss = BCE loss + λ * KGE loss
  - 返回詳細的 loss 資訊

### 3. `eval_epoch` 函數
- **修改**：處理模型現在返回 `(mlp_logits, kge_score)` 的情況

### 4. `inference.py`
- **修改**：處理模型現在返回 `(mlp_logits, kge_score)` 的情況

### 5. 配置文件
- **`configs/retriever/webqsp.yaml`** 和 **`configs/retriever/cwq.yaml`**
- **新增參數**：`kge.loss_weight: 1.0` - 控制 KGE loss 的權重

### 6. `src/model/kge_models.py`
- **修改**：所有 KGE 模型的 `predict` 方法文檔，說明支持批次輸入

## 🏗️ 新的架構

### 模型 Forward 流程
```
輸入: [Zq||Zh||Zr||Zt||Ztau] (GTE + DDE + PE)
    ↓
MLP → mlp_logits
    ↓
KGE Model → kge_score
    ↓
返回: (mlp_logits, kge_score)
```

### 訓練 Loss 計算
```
BCE Loss = binary_cross_entropy_with_logits(mlp_logits, target_triple_probs)

KGE Margin Ranking Loss = max(0, pos_kge_scores - neg_kge_scores + margin)

Total Loss = BCE Loss + λ * KGE Margin Ranking Loss
```

## 🚀 使用方法

### 1. 配置 KGE
在配置文件中設置：
```yaml
kge:
  enabled: true
  model_type: 'transe'  # 或其他 KGE 模型
  embedding_dim: 256
  margin: 1.0
  loss_weight: 1.0  # 新增：KGE loss 權重
```

### 2. 訓練
```bash
python train.py --dataset webqsp
```

### 3. 推論
```bash
python inference.py --path path/to/checkpoint.pth
```

## ✅ 驗證

創建了測試腳本 `test_kge_score_integration.py` 來驗證：
- KGE 模型能正確計算 score
- Retriever 模型能正確整合 KGE score
- Margin ranking loss 能正確計算
- 所有組件能正常協作

測試結果：✅ 所有測試通過

## 🔧 技術細節

### KGE Score 計算
- 使用 KGE 模型的 `predict` 方法
- 支持批次輸入
- 返回三元組的 plausibility score

### Margin Ranking Loss
- 正樣本：`target_triple_probs > 0.5`
- 負樣本：`target_triple_probs <= 0.5`
- 計算：`max(0, pos_scores - neg_scores + margin)`

### Loss 權重
- 可通過 `kge.loss_weight` 調整 KGE loss 的影響
- 預設值：1.0
- 建議範圍：0.1 - 2.0

## 📊 預期效果

1. **更純粹的 KGE 整合**：KGE 作為獨立的監督信號，不干擾 GTE embedding
2. **更好的可解釋性**：可以分別觀察 MLP 和 KGE 的表現
3. **更靈活的調優**：可以獨立調整 KGE loss 的權重
4. **更穩定的訓練**：避免 embedding 維度不匹配的問題

## 🎉 總結

成功實現了你要求的 KGE score 整合方式：
- ✅ 保持原本的 GTE embedding concatenation
- ✅ KGE 只提供 score，不改變 embedding 結構
- ✅ 使用 margin ranking loss 作為額外的監督信號
- ✅ 支持可配置的 loss 權重
- ✅ 所有組件經過測試驗證 