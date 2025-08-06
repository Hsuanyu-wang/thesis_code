# SubgraphRAG 檔案說明

## 📁 檔案結構總覽

```
SubgraphRAG_backup/retrieve/
├── README.md                           # 主要說明文件
├── FILES_DESCRIPTION.md               # 本檔案說明文件
├── pipeline.sh                        # 完整流程腳本
├── USAGE.md                          # 簡短使用說明
├── KGE_SCORE_INTEGRATION_SUMMARY.md  # KGE Score 整合總結
├── KGE_INTEGRATION_README.md         # KGE 整合說明
├── README_V2.md                      # 舊版說明文件
├── USAGE.md                          # 簡短使用說明
│
├── 主要腳本/
│   ├── train.py                      # 訓練檢索器主程式
│   ├── inference.py                  # 推論與檢索程式
│   ├── eval.py                       # 評估檢索結果程式
│   ├── emb.py                        # 預計算嵌入程式
│   ├── train_kge.py                  # 訓練 KGE 模型程式
│   └── run_kge_integration.py       # KGE 完整流程程式
│
├── 測試腳本/
│   ├── test_kge_integration.py      # KGE 整合測試
│   ├── test_kge_score_integration.py # KGE Score 整合測試
│   └── demo_progress_bars.py        # 進度條展示
│
├── 配置檔案/
│   ├── configs/retriever/
│   │   ├── webqsp.yaml              # WebQSP 檢索器配置
│   │   └── cwq.yaml                 # CWQ 檢索器配置
│   └── configs/emb/
│       └── gte-large-en-v1.5/       # GTE 嵌入配置
│
├── 核心模組/
│   └── src/
│       ├── model/
│       │   ├── retriever.py         # 檢索器模型主體
│       │   ├── kge_models.py        # KGE 模型實現
│       │   ├── kge_utils.py         # KGE 工具函數
│       │   └── text_encoders/       # 語意嵌入模型
│       ├── dataset/
│       │   ├── retriever.py         # 檢索器資料集處理
│       │   └── emb.py               # 嵌入資料集處理
│       └── config/
│           ├── retriever.py         # 檢索器配置
│           └── emb.py               # 嵌入配置
│
├── 資料檔案/
│   ├── data_files/                   # 資料集和預處理檔案
│   ├── training result/              # 訓練結果
│   ├── retrieve_result/              # 評估結果
│   └── wandb/                       # 實驗追蹤
│
└── 環境需求/
    └── requirements/
        ├── gte_large_en_v1-5.txt    # GTE 嵌入環境需求
        └── retriever.txt             # 檢索器環境需求
```

## 📄 詳細檔案說明

### 🎯 主要腳本檔案

#### `train.py` - 訓練檢索器主程式
- **功能**: 訓練 SubgraphRAG 檢索器模型
- **支援**: GTE、DDE、KGE 整合
- **輸入**: 資料集名稱 (webqsp/cwq)
- **輸出**: 訓練好的模型檢查點
- **執行**: `python train.py -d webqsp`
- **特色**: 
  - 支援 KGE score 整合
  - 使用 margin ranking loss
  - wandb 實驗追蹤
  - 早停機制

#### `inference.py` - 推論與檢索程式
- **功能**: 使用訓練好的模型進行推論
- **輸入**: 模型檢查點路徑
- **輸出**: 檢索結果檔案
- **執行**: `python inference.py -p "path/to/cpt.pth"`
- **特色**:
  - 支援批次處理
  - 可設定最大檢索數量
  - 自動保存結果

#### `eval.py` - 評估檢索結果程式
- **功能**: 評估檢索結果的品質
- **輸入**: 資料集名稱和檢索結果路徑
- **輸出**: 評估指標和結果表格
- **執行**: `python eval.py -d webqsp -p "path/to/result.pth"`
- **指標**:
  - triple_recall@k
  - ans_recall@k

#### `emb.py` - 預計算嵌入程式
- **功能**: 預計算實體和關係的語意嵌入
- **目的**: 加速後續訓練和推論
- **執行**: `python emb.py -d webqsp`
- **特色**: 使用 GTE-large-en-v1.5 模型

#### `train_kge.py` - 訓練 KGE 模型程式
- **功能**: 訓練知識圖譜嵌入模型
- **支援**: TransE、DistMult、PTransE
- **執行**: `python train_kge.py --dataset webqsp`
- **特色**:
  - 負採樣訓練
  - margin ranking loss
  - 自動保存模型

#### `run_kge_integration.py` - KGE 完整流程程式
- **功能**: 一鍵執行 KGE 完整流程
- **流程**: KGE 訓練 → 嵌入計算 → 檢索器訓練
- **執行**: `python run_kge_integration.py --dataset webqsp`

### 🔧 測試腳本檔案

#### `test_kge_integration.py` - KGE 整合測試
- **功能**: 測試 KGE 模型和整合功能
- **測試項目**:
  - KGE 模型創建
  - 前向傳播
  - 預測功能
  - 嵌入獲取
- **執行**: `python test_kge_integration.py`

#### `test_kge_score_integration.py` - KGE Score 整合測試
- **功能**: 測試 KGE score 整合方式
- **測試項目**:
  - Retriever 模型整合
  - KGE score 計算
  - Margin ranking loss
  - 批次處理
- **執行**: `python test_kge_score_integration.py`

#### `demo_progress_bars.py` - 進度條展示
- **功能**: 展示進度條功能
- **特色**: 多層級進度條顯示

### 📂 核心模組檔案

#### `src/model/retriever.py` - 檢索器模型主體
- **功能**: 實現 SubgraphRAG 檢索器
- **架構**:
  - GTE embedding concatenation
  - DDE (Distance-based Dynamic Embedding)
  - KGE score integration
  - MLP 預測層
- **輸入**: [Zq||Zh||Zr||Zt||Ztau]
- **輸出**: (mlp_logits, kge_score)

#### `src/model/kge_models.py` - KGE 模型實現
- **功能**: 實現多種 KGE 模型
- **支援模型**:
  - TransE: 平移嵌入
  - DistMult: 三向張量分解
  - PTransE: 路徑感知 TransE
  - RotatE: 複數旋轉嵌入
  - ComplEx: 複數嵌入
  - SimplE: 簡單嵌入
- **特色**: 統一的工廠函數接口

#### `src/model/kge_utils.py` - KGE 工具函數
- **功能**: KGE 相關工具函數
- **主要功能**:
  - 模型載入
  - 配置創建
  - 嵌入查詢
  - 錯誤處理

#### `src/dataset/retriever.py` - 檢索器資料集處理
- **功能**: 檢索器訓練和推論的資料集處理
- **主要功能**:
  - 路徑提取
  - 三元組標註
  - 嵌入整合
  - 批次處理

#### `src/dataset/emb.py` - 嵌入資料集處理
- **功能**: 嵌入預計算的資料集處理
- **特色**: 支援大規模資料處理

### ⚙️ 配置檔案

#### `configs/retriever/webqsp.yaml` - WebQSP 檢索器配置
```yaml
# 基本配置
task: 'retriever'
env:
  num_threads: 16
  seed: 42

# 資料集配置
dataset:
  name: 'webqsp'
  text_encoder_name: 'gte-large-en-v1.5'

# 檢索器配置
retriever:
  topic_pe: true
  DDE_kwargs:
    num_rounds: 2
    num_reverse_rounds: 2

# KGE 配置
kge:
  enabled: true
  model_type: 'transe'
  embedding_dim: 256
  margin: 1.0
  loss_weight: 1.0
  num_epochs: 100
  batch_size: 1024
  learning_rate: 0.001
  num_negatives: 1
  norm: 1

# 訓練配置
train:
  num_epochs: 10000
  patience: 10
  save_prefix: 'webqsp'

# 評估配置
eval:
  k_list: '100'
```

#### `configs/retriever/cwq.yaml` - CWQ 檢索器配置
- 類似 WebQSP 配置，但針對 CWQ 資料集優化

### 📊 資料檔案

#### `data_files/` - 資料集和預處理檔案
- **結構**:
  ```
  data_files/
  ├── webqsp/
  │   ├── processed/          # 預處理資料
  │   ├── emb/               # 嵌入檔案
  │   └── kge/               # KGE 模型檔案
  └── cwq/
      ├── processed/
      ├── emb/
      └── kge/
  ```

#### `training result/` - 訓練結果
- **內容**: 模型檢查點、配置檔案
- **命名**: `{dataset}_{timestamp}/`

#### `retrieve_result/` - 評估結果
- **內容**: 評估指標、結果表格
- **格式**: CSV、JSON

#### `wandb/` - 實驗追蹤
- **內容**: 訓練日誌、實驗配置
- **功能**: 實驗管理和可視化

### 🔧 環境需求檔案

#### `requirements/gte_large_en_v1-5.txt` - GTE 嵌入環境需求
- **用途**: 嵌入預計算環境
- **主要套件**: transformers、torch、xformers

#### `requirements/retriever.txt` - 檢索器環境需求
- **用途**: 檢索器訓練和推論環境
- **主要套件**: torch、torch_geometric、wandb

## 🎯 設計流程

### 1. 資料預處理流程
```
原始資料 → 路徑提取 → 三元組標註 → 嵌入預計算
```

### 2. 模型訓練流程
```
嵌入資料 → Retriever 模型 → KGE 整合 → 聯合訓練
```

### 3. 推論評估流程
```
訓練模型 → 檢索推論 → 結果評估 → 指標計算
```

## 🔗 檔案相互關聯

### 依賴關係
- `train.py` 依賴 `src/model/retriever.py`、`src/dataset/retriever.py`、`src/config/retriever.py`
- `inference.py` 依賴 `src/model/retriever.py`、`src/dataset/retriever.py`
- `eval.py` 依賴 `src/dataset/retriever.py`
- `emb.py` 依賴 `src/dataset/emb.py`、`src/config/emb.py`
- `train_kge.py` 依賴 `src/model/kge_models.py`、`src/model/kge_utils.py`

### 配置關係
- 所有腳本都使用 `configs/` 下的配置檔案
- KGE 相關腳本使用 `configs/retriever/{dataset}.yaml` 中的 KGE 配置
- 嵌入腳本使用 `configs/emb/gte-large-en-v1.5/` 下的配置

### 資料流關係
- `emb.py` 產生嵌入檔案 → `train.py` 使用嵌入檔案
- `train_kge.py` 產生 KGE 模型 → `train.py` 整合 KGE 模型
- `train.py` 產生模型檢查點 → `inference.py` 使用檢查點
- `inference.py` 產生檢索結果 → `eval.py` 評估結果

## 📝 使用建議

### 新用戶
1. 閱讀 `README.md` 了解整體架構
2. 使用 `pipeline.sh` 執行完整流程
3. 查看 `FILES_DESCRIPTION.md` 了解檔案功能

### 進階用戶
1. 修改 `configs/retriever/{dataset}.yaml` 調整參數
2. 使用測試腳本驗證功能
3. 查看 wandb 日誌分析實驗

### 開發者
1. 修改 `src/model/` 下的模型檔案
2. 更新 `src/dataset/` 下的資料處理
3. 添加新的配置檔案到 `configs/` 