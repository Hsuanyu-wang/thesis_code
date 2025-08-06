# PyKEEN 整合說明

## 概述

本專案現在支援使用 [PyKEEN](https://pykeen.readthedocs.io/) 套件中的現成 KGE 模型，同時保留原有的自定義實現。這提供了更好的模型穩定性和更多的模型選擇。

## 支援的 PyKEEN 模型

### 1. TransE
- **PyKEEN 版本**: `pykeen_transe`
- **自定義版本**: `transe`
- **論文**: Translating Embeddings for Modeling Multi-relational Data
- **特點**: 使用翻譯向量表示關係

### 2. DistMult
- **PyKEEN 版本**: `pykeen_distmult`
- **自定義版本**: `distmult`
- **論文**: Embedding Entities and Relations for Learning and Inference in Knowledge Bases
- **特點**: 使用對角矩陣表示關係，計算效率高

### 3. ComplEx
- **PyKEEN 版本**: `pykeen_complex`
- **自定義版本**: `complex`
- **論文**: Complex Embeddings for Simple Link Prediction
- **特點**: 使用複數嵌入處理非對稱關係

### 4. RotatE
- **PyKEEN 版本**: `pykeen_rotate`
- **自定義版本**: `rotate`
- **論文**: RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space
- **特點**: 在複數空間中使用旋轉表示關係

### 5. SimplE
- **PyKEEN 版本**: `pykeen_simple`
- **自定義版本**: `simple`
- **論文**: SimplE: Embedding for Simple Link Prediction
- **特點**: 為頭尾實體使用不同的嵌入

## 安裝 PyKEEN

```bash
pip install pykeen
```

## 使用方法

### 1. 訓練 KGE 模型

使用 PyKEEN 模型訓練：

```bash
# 使用 PyKEEN DistMult
python train_kge.py -d webqsp --model_type pykeen_distmult

# 使用 PyKEEN TransE
python train_kge.py -d webqsp --model_type pykeen_transe

# 使用 PyKEEN ComplEx
python train_kge.py -d webqsp --model_type pykeen_complex

# 使用 PyKEEN RotatE
python train_kge.py -d webqsp --model_type pykeen_rotate

# 使用 PyKEEN SimplE
python train_kge.py -d webqsp --model_type pykeen_simple
```

### 2. 配置檔案

在 `configs/retriever/{dataset}.yaml` 中設定：

```yaml
kge:
  model_type: pykeen_distmult  # 或 pykeen_transe, pykeen_complex 等
  embedding_dim: 100
  num_epochs: 100
  batch_size: 1024
  learning_rate: 0.001
  margin: 1.0  # 僅適用於 TransE 和 RotatE
```

### 3. 程式碼中使用

```python
from src.model.kge_models import create_kge_model

# 創建 PyKEEN 模型
model = create_kge_model(
    model_type='pykeen_distmult',
    num_entities=1000,
    num_relations=100,
    embedding_dim=100
)

# 使用方式與自定義模型相同
pos_score, neg_score = model(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)
```

## 優勢

### PyKEEN 模型的優勢

1. **穩定性**: PyKEEN 是經過廣泛測試的成熟套件
2. **優化**: 使用 PyKEEN 的優化實現，通常比自定義實現更快
3. **維護**: 由社群維護，bug 修復和功能更新更及時
4. **標準化**: 遵循學術標準的實現

### 自定義模型的優勢

1. **靈活性**: 可以根據特定需求自定義實現
2. **控制**: 對模型內部運作有完全控制
3. **實驗性**: 可以實現實驗性的新方法

## 測試

運行測試腳本驗證整合：

```bash
python test_pykeen_integration.py
```

測試包括：
- 基本功能測試
- 模型比較測試
- 損失函數兼容性測試

## 向後兼容性

- 所有現有的自定義模型仍然可用
- 現有的訓練腳本和配置檔案無需修改
- 可以無縫切換 between PyKEEN 和自定義模型

## 故障排除

### 1. PyKEEN 未安裝

如果遇到 `ImportError`，請安裝 PyKEEN：

```bash
pip install pykeen
```

### 2. 模型不支援

如果指定的 PyKEEN 模型類型不支援，系統會自動回退到自定義實現或拋出錯誤。

### 3. 記憶體問題

PyKEEN 模型可能需要更多記憶體，可以嘗試：
- 減少 `batch_size`
- 減少 `embedding_dim`
- 使用較小的資料集進行測試

## 參考資料

- [PyKEEN 官方文檔](https://pykeen.readthedocs.io/)
- [PyKEEN GitHub](https://github.com/pykeen/pykeen)
- [DistMult 論文](https://arxiv.org/abs/1412.6575)
- [TransE 論文](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data)
- [ComplEx 論文](https://arxiv.org/abs/1606.06357)
- [RotatE 論文](https://arxiv.org/abs/1902.10197)
- [SimplE 論文](https://arxiv.org/abs/1802.04868) 