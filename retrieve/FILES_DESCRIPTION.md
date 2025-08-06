# SubgraphRAG 檔案說明與設計流程

## 📁 檔案結構總覽

```
SubgraphRAG_backup/retrieve/
├── 📄 主要腳本 (Core Scripts)
│   ├── train.py                    # 訓練檢索器主程式
│   ├── inference.py                # 推論與檢索程式
│   ├── eval.py                     # 評估檢索結果程式
│   ├── emb.py                      # 預計算嵌入程式
│   ├── train_kge.py               # 訓練 KGE 模型程式
│   └── run_kge_integration.py     # KGE 完整流程程式
│
├── 🧪 測試腳本 (Test Scripts)
│   ├── test_kge_score_integration.py  # KGE score 整合測試
│   └── test_kge_integration.py        # KGE 整合測試
│
├── ⚙️ 配置檔案 (Configuration)
│   ├── configs/retriever/
│   │   ├── webqsp.yaml            # WebQSP 檢索器配置
│   │   └── cwq.yaml               # CWQ 檢索器配置
│   └── configs/emb/
│       └── gte-large-en-v1.5/     # GTE 嵌入配置
│
├── 🔧 核心模組 (Source Code)
│   └── src/
│       ├── model/
│       │   ├── retriever.py       # 檢索器模型主體
│       │   ├── kge_models.py      # KGE 模型實現
│       │   ├── kge_utils.py       # KGE 工具函數
│       │   └── text_encoders/     # 語意嵌入模型
│       ├── dataset/
│       │   ├── retriever.py       # 檢索器資料集處理
│       │   └── emb.py             # 嵌入資料集處理
│       └── config/
│           ├── retriever.py       # 檢索器配置
│           └── emb.py             # 嵌入配置
│
├── 📊 結果目錄 (Results)
│   ├── training result/            # 訓練結果
│   ├── retrieve_result/            # 評估結果
│   └── wandb/                     # 實驗追蹤
│
└── 📚 文檔 (Documentation)
    ├── README.md                   # 主要說明文件
    ├── FILES_DESCRIPTION.md        # 本檔案說明
    ├── KGE_SCORE_INTEGRATION_SUMMARY.md  # KGE 整合總結
    └── docs/                       # 額外文檔
```

## 📄 詳細檔案說明

### 🎯 主要腳本檔案

#### `train.py` - 訓練檢索器主程式
**功能**: 訓練 SubgraphRAG 檢索器模型，整合 GTE、DDE、KGE 和 PE
**設計流程**:
1. 載入配置和資料集
2. 初始化模型（包含 KGE 整合）
3. 訓練迴圈（BCE loss + KGE margin ranking loss）
4. 驗證和早停機制
5. 保存最佳模型

**關鍵特性**:
- 支援 KGE score 整合
- 使用 margin ranking loss 作為額外監督信號
- Wandb 實驗追蹤
- 早停機制避免過擬合
- 可配置的 KGE loss 權重

**使用方式**:
```bash
python train.py -d webqsp
```

#### `inference.py` - 推論與檢索程式
**功能**: 使用訓練好的模型進行推論，檢索相關三元組
**設計流程**:
1. 載入訓練好的模型檢查點
2. 載入測試資料集
3. 批次推論處理
4. 排序和篩選 top-K 結果
5. 保存檢索結果

**關鍵特性**:
- 支援批次處理
- 可設定最大檢索數量 (max_K)
- 自動保存結果到對應目錄
- 處理空樣本情況

**使用方式**:
```bash
python inference.py -p "path/to/cpt.pth" --max_K 500
```

#### `eval.py` - 評估檢索結果程式
**功能**: 評估檢索結果的品質，計算各種召回率指標
**設計流程**:
1. 載入檢索結果和真實標籤
2. 計算 triple_recall@k 指標
3. 計算 ans_recall@k 指標
4. 生成詳細評估報告
5. 保存評估結果

**評估指標**:
- `triple_recall@k`: 三元組召回率
- `ans_recall@k`: 答案實體召回率
- 支援多個 k 值評估

**使用方式**:
```bash
python eval.py -d webqsp -p "path/to/retrieval_result.pth"
```

#### `emb.py` - 預計算嵌入程式
**功能**: 預計算實體和關係的語意嵌入，加速後續訓練和推論
**設計流程**:
1. 載入 GTE 文本編碼器
2. 處理所有實體和關係文本
3. 批次計算嵌入向量
4. 保存到快取檔案

**目的**: 避免在訓練時重複計算嵌入，大幅提升效率

**使用方式**:
```bash
python emb.py -d webqsp
```

#### `train_kge.py` - 訓練 KGE 模型程式
**功能**: 訓練知識圖譜嵌入模型，提供結構知識
**設計流程**:
1. 從資料集提取三元組
2. 建立負樣本
3. 訓練 KGE 模型（TransE、DistMult 等）
4. 保存訓練好的模型

**支援的 KGE 模型**:
- TransE: 平移嵌入
- DistMult: 對角矩陣分解
- PTransE: 路徑感知的 TransE

**使用方式**:
```bash
python train_kge.py --dataset webqsp --split train
```

#### `run_kge_integration.py` - KGE 完整流程程式
**功能**: 執行完整的 KGE 整合流程
**設計流程**:
1. 訓練 KGE 模型
2. 計算文本嵌入
3. 訓練檢索器（整合 KGE）
4. 可選的推論和評估

**使用方式**:
```bash
python run_kge_integration.py --dataset webqsp --kge_model transe --full_pipeline
```

### 🧪 測試腳本檔案

#### `test_kge_score_integration.py` - KGE Score 整合測試
**功能**: 測試 KGE score 整合的正確性
**測試內容**:
- KGE 模型創建和前向傳播
- Retriever 模型整合 KGE score
- Margin ranking loss 計算
- 所有組件的協作

**使用方式**:
```bash
python test_kge_score_integration.py
```

#### `test_kge_integration.py` - KGE 整合測試
**功能**: 測試 KGE 整合功能
**測試內容**:
- KGE 模型載入
- 配置創建
- 模型前向傳播

### ⚙️ 配置檔案

#### `configs/retriever/webqsp.yaml` - WebQSP 檢索器配置
**內容**:
- 環境設定（執行緒數、隨機種子）
- 資料集設定
- 檢索器模型參數
- KGE 整合設定
- 訓練參數
- 評估設定

#### `configs/retriever/cwq.yaml` - CWQ 檢索器配置
**內容**: 類似 WebQSP，但針對 CWQ 資料集優化

### 🔧 核心模組

#### `src/model/retriever.py` - 檢索器模型主體
**功能**: 實現 SubgraphRAG 檢索器模型
**架構**:
```
Input: [Zq||Zh||Zr||Zt||Ztau] (GTE + DDE + PE)
    ↓
MLP → mlp_logits
    ↓
KGE Model → kge_score
    ↓
Output: (mlp_logits, kge_score)
```

**關鍵組件**:
- GTE embedding concatenation
- DDE (Distance-based Dynamic Embedding)
- PE (Positional Encoding)
- KGE score computation
- MLP prediction layer

#### `src/model/kge_models.py` - KGE 模型實現
**功能**: 實現各種知識圖譜嵌入模型
**支援模型**:
- TransE: 平移嵌入
- DistMult: 對角矩陣分解
- PTransE: 路徑感知的 TransE
- RotatE: 旋轉嵌入
- ComplEx: 複數嵌入
- SimplE: 簡單嵌入

**共同接口**:
- `forward()`: 訓練時的前向傳播
- `predict()`: 推論時的預測
- `_score()`: 計算三元組分數

#### `src/model/kge_utils.py` - KGE 工具函數
**功能**: KGE 相關的工具函數
**主要函數**:
- `load_kge_model()`: 載入訓練好的 KGE 模型
- `create_kge_config_from_model()`: 從模型創建配置

#### `src/dataset/retriever.py` - 檢索器資料集處理
**功能**: 處理檢索器訓練和評估的資料集
**主要類別**:
- `RetrieverDataset`: 檢索器資料集類別
- `collate_retriever`: 批次整理函數

**處理流程**:
1. 載入預處理的資料
2. 提取路徑和計算三元組分數
3. 載入預計算的嵌入
4. 整合所有資料

#### `src/config/retriever.py` - 檢索器配置
**功能**: 載入和管理檢索器配置
**主要函數**:
- `load_yaml()`: 載入 YAML 配置文件

### 📊 結果目錄

#### `training result/` - 訓練結果
**內容**:
- 模型檢查點 (`cpt.pth`)
- 訓練日誌
- 配置備份

#### `retrieve_result/` - 評估結果
**內容**:
- 評估報告
- 指標表格
- 詳細結果

#### `wandb/` - 實驗追蹤
**內容**:
- 實驗日誌
- 學習曲線
- 超參數記錄

### 📚 文檔檔案

#### `README.md` - 主要說明文件
**內容**:
- 系統概述
- 安裝指南
- 快速開始
- 詳細使用說明
- 故障排除

#### `KGE_SCORE_INTEGRATION_SUMMARY.md` - KGE 整合總結
**內容**:
- KGE score 整合的詳細說明
- 修改的文件清單
- 新的架構設計
- 使用方法和驗證

## 🔄 設計流程

### 1. 資料預處理流程
```
原始資料 → 預處理 → 三元組提取 → 路徑計算 → 嵌入預計算
```

### 2. 訓練流程
```
嵌入預計算 → KGE 訓練 → 檢索器訓練 → 驗證 → 模型保存
```

### 3. 推論流程
```
模型載入 → 批次推論 → 結果排序 → 結果保存
```

### 4. 評估流程
```
結果載入 → 指標計算 → 報告生成 → 結果保存
```

## 🚀 Pipeline 腳本

### `pipeline.sh` - 完整流程腳本
**功能**: 執行完整的訓練和評估流程
**步驟**:
1. 環境檢查（Python、GPU、依賴）
2. 資料檔案驗證
3. 嵌入預計算
4. KGE 模型訓練
5. 檢索器訓練
6. 推論
7. 評估
8. 生成總結報告

**特性**:
- 彩色輸出和進度顯示
- 錯誤處理和恢復
- 智能跳過已完成的步驟
- 詳細的日誌記錄

**使用方式**:
```bash
./pipeline.sh webqsp          # 基本使用
./pipeline.sh cwq distmult    # 指定 KGE 模型
./pipeline.sh webqsp transe 5000  # 自定義最大 epoch
```

## 🔧 開發指南

### 添加新的 KGE 模型
1. 在 `src/model/kge_models.py` 中實現新模型
2. 在 `create_kge_model()` 函數中添加支援
3. 更新配置文件
4. 添加測試

### 修改模型架構
1. 更新 `src/model/retriever.py`
2. 修改配置文件
3. 更新訓練腳本
4. 重新訓練模型

### 添加新的評估指標
1. 在 `eval.py` 中實現新指標
2. 更新評估報告格式
3. 添加測試

## 📈 監控和調試

### Wandb 整合
- 自動實驗追蹤
- 學習曲線可視化
- 超參數記錄
- 模型檢查點管理

### 日誌記錄
- 訓練進度（tqdm）
- Loss 組件（BCE、KGE、Total）
- 評估指標
- 早停信息

### 測試
- 單元測試：`test_kge_score_integration.py`
- 整合測試：`test_kge_integration.py`
- 模型導入測試

## 🔍 故障排除

### 常見問題
1. **CUDA 記憶體不足**: 減少批次大小
2. **KGE 模型未找到**: 先運行 `python train_kge.py`
3. **形狀不匹配**: 確保模型檢查點與當前架構匹配

### 調試技巧
1. 使用測試腳本驗證組件
2. 檢查配置檔案
3. 查看 Wandb 日誌
4. 驗證資料檔案完整性 