# README_NEW

## 專案簡介

本專案為多跳知識圖譜問答（KGQA）檢索系統，支援語意嵌入（GTE）、距離嵌入（DDE）及知識圖譜嵌入（KGE, 如 TransE/DistMult/PTransE）等多種檢索特徵。可用於訓練、推論、評估 KGQA 檢索器，並支援 KGE 模型的整合與測試。

---

## 目錄結構與主要檔案說明

### 頂層腳本

- **train.py**  
  訓練 Retriever（檢索器）主程式。讀取資料集、配置檔，進行模型訓練，支援 KGE 整合。  
  執行：`python train.py -d DATASET`

- **inference.py**  
  使用訓練好的檢索器模型進行推論，產生檢索結果。  
  執行：`python inference.py -p MODEL_PATH`

- **eval.py**  
  對檢索結果進行評估，計算 recall 等指標，並可輸出結果表格。  
  執行：`python eval.py -d DATASET -p RESULT_PATH`

- **emb.py**  
  預先計算並儲存所有實體與關係的語意嵌入（GTE），加速後續訓練與推論。  
  執行：`python emb.py -d DATASET`

- **train_kge.py**  
  訓練 KGE（知識圖譜嵌入）模型（如 TransE/DistMult/PTransE）。  
  執行：`python train_kge.py --dataset DATASET --split train`

- **run_kge_integration.py**  
  一鍵執行 KGE 訓練、嵌入計算、檢索器訓練等完整流程。  
  執行：`python run_kge_integration.py --dataset DATASET --kge_model transe --full_pipeline`

- **test_kge_integration.py**  
  測試 KGE 模型與整合流程的單元測試腳本。

- **demo_progress_bars.py**  
  展示 KGE 訓練與資料處理進度條的 demo 腳本。

- **pipeline.sh**  
  Shell 腳本，串接多步驟流程（如訓練、推論、評估等）。

---

### 主要資料夾

- **src/model/**  
  - `retriever.py`：Retriever 檢索器模型主體，支援 KGE 整合。  
  - `kge_models.py`：KGE 模型（TransE、DistMult、PTransE）與損失函數定義。  
  - `kge_utils.py`：KGE 模型載入、配置、嵌入查詢等工具函數。  
  - `text_encoders/`：語意嵌入模型（如 GTE）相關程式（如 gte_large_en.py）。

- **src/dataset/**  
  - `retriever.py`：Retriever 訓練/推論用資料集處理，含資料載入、路徑標註、嵌入整合等。  
  - `emb.py`：語意嵌入資料集處理（EmbInferDataset），將原始資料轉為嵌入格式。

- **src/config/**  
  - `retriever.py`、`emb.py`：配置檔載入工具。  
  - `base.py`：基礎設定工具。

- **requirements/**  
  - `gte_large_en_v1-5.txt`：GTE 嵌入計算環境需求。  
  - `retriever.txt`：Retriever 訓練/推論環境需求。

- **data_files/**  
  - 存放資料集（webqsp/cwq）、預處理資料、嵌入、KGE 模型等。

- **retrieve_result/**  
  - 儲存評估結果表格、推論結果（如 .csv/.json 檔）。  
  - 依資料集分子資料夾（如 webqsp/）儲存多次實驗結果。

- **configs/**  
  - `retriever/`：各資料集（webqsp/cwq）Retriever 設定檔（.yaml）。  
  - `emb/`：嵌入計算設定檔（如 gte-large-en-v1.5/ 內含 .yaml）。

- **wandb/**  
  - 實驗追蹤與日誌資料夾（自動產生，含多次 run 子目錄與 log 檔）。

- **webqsp_*/cwq_*/**  
  - 以時間戳命名的資料夾，儲存每次推論/訓練的中間結果（如 .pth、.json、.csv）。

---

## 各檔案相互關聯

- `train.py`、`inference.py`、`eval.py`、`emb.py` 皆依賴 `src/model/retriever.py`（模型）、`src/dataset/retriever.py`（資料集）、`src/config/retriever.py`（配置）。
- `train_kge.py`、`run_kge_integration.py`、`test_kge_integration.py` 依賴 `src/model/kge_models.py`、`src/model/kge_utils.py`。
- Retriever 可選擇是否整合 KGE，相關設定與模型由 `kge_utils.py` 載入。
- 嵌入計算（`emb.py`）與 Retriever 訓練/推論皆需先產生嵌入檔案。
- `run_kge_integration.py` 串接 KGE 訓練、嵌入計算、Retriever 訓練等流程。
- `pipeline.sh` 可自動化多步驟流程。
- `wandb/` 用於追蹤訓練過程與結果。

---

## 安裝與使用方法

### 1. 安裝環境

#### (1) 嵌入計算（GTE）

```bash
conda create -n gte_large_en_v1-5 python=3.10 -y
conda activate gte_large_en_v1-5
pip install -r requirements/gte_large_en_v1-5.txt
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
```

#### (2) Retriever 訓練/推論

```bash
conda create -n retriever python=3.10 -y
conda activate retriever
pip install -r requirements/retriever.txt
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.5.3
pip install pyg_lib==0.3.1 torch_scatter==2.1.2 torch_sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

---

### 2. 嵌入計算

```bash
python emb.py -d DATASET
```
產生語意嵌入檔案，供後續訓練/推論使用。

---

### 3. Retriever 訓練

```bash
python train.py -d DATASET
```
訓練檢索器，支援 KGE 整合（需於 config 啟用）。

---

### 4. Retriever 推論

```bash
python inference.py -p MODEL_PATH
```
使用訓練好的模型進行檢索，產生檢索結果。

---

### 5. 檢索評估

```bash
python eval.py -d DATASET -p RESULT_PATH
```
評估檢索結果，輸出 recall 等指標。

---

### 6. KGE 訓練與整合

#### (1) 單獨訓練 KGE

```bash
python train_kge.py --dataset DATASET --split train
```

#### (2) 一鍵整合流程

```bash
python run_kge_integration.py --dataset DATASET --kge_model transe --full_pipeline
```

---

### 7. 測試與 Demo

- `python test_kge_integration.py`：單元測試 KGE 與整合流程。
- `python demo_progress_bars.py`：展示進度條效果。
- `bash pipeline.sh`：執行自動化流程 demo。

---

## 支援資料集

- `webqsp`
- `cwq`

---

## 進階設定與說明

- Retriever 設定請見 `configs/retriever/{dataset}.yaml`。
- 嵌入設定請見 `configs/emb/gte-large-en-v1.5/{dataset}.yaml`。
- 更多細節請參考 `KGE_INTEGRATION_README.md`。

---

如需進一步協助，請參考原始 README 或聯絡開發團隊。

---

如需更細緻的檔案/流程圖或有特定細節需求，請告知！ 