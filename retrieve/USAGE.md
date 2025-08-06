# SubgraphRAG 快速使用指南

## 🚀 快速開始

### 1. 完整流程（推薦）
```bash
# 執行完整的訓練和評估流程
./pipeline.sh webqsp
```

### 2. 逐步執行
```bash
# 步驟 1: 預計算嵌入
python emb.py -d webqsp

# 步驟 2: 訓練檢索器
python train.py -d webqsp

# 步驟 3: 執行推論
python inference.py -p "training result/webqsp_xxx/cpt.pth"

# 步驟 4: 評估結果
python eval.py -d webqsp -p "training result/webqsp_xxx/retrieval_result.pth"
```

## 📊 支援的資料集

- `webqsp`: WebQuestionsSP 資料集
- `cwq`: Complex Web Questions 資料集

## ⚙️ 配置選項

### KGE 模型選擇
```bash
./pipeline.sh webqsp transe    # TransE 模型
./pipeline.sh webqsp distmult  # DistMult 模型
./pipeline.sh webqsp ptranse   # PTransE 模型
```

### 自定義訓練參數
```bash
./pipeline.sh webqsp transe 5000  # 最大 5000 epochs
```

## 📈 監控訓練

### Wandb 追蹤
- 自動實驗追蹤
- 學習曲線可視化
- 超參數記錄

### 本地日誌
- 訓練進度顯示
- Loss 組件分解
- 評估指標記錄

## 🔍 常見問題

### 1. 記憶體不足
```bash
# 減少批次大小
# 編輯 configs/retriever/webqsp.yaml
# 修改 batch_size 參數
```

### 2. KGE 模型未找到
```bash
# 先訓練 KGE 模型
python train_kge.py --dataset webqsp --split train
```

### 3. 測試整合
```bash
# 測試 KGE 整合
python test_kge_score_integration.py
```

## 📁 結果檔案

### 訓練結果
- `training result/webqsp_xxx/cpt.pth`: 模型檢查點
- `training result/webqsp_xxx/retrieval_result.pth`: 推論結果

### 評估結果
- `retrieve_result/`: 評估報告和指標

## 🛠️ 進階使用

### 自定義配置
```bash
# 編輯配置文件
vim configs/retriever/webqsp.yaml
```

### 單獨測試組件
```bash
# 測試 KGE 模型
python -c "from src.model.kge_models import TransE; print('KGE OK')"

# 測試檢索器模型
python -c "from src.model.retriever import Retriever; print('Retriever OK')"
```

### 批次處理多個資料集
```bash
# 處理 WebQSP
./pipeline.sh webqsp

# 處理 CWQ
./pipeline.sh cwq
```

## 📞 支援

- 詳細文檔: `README.md`
- 檔案說明: `FILES_DESCRIPTION.md`
- KGE 整合: `KGE_SCORE_INTEGRATION_SUMMARY.md`