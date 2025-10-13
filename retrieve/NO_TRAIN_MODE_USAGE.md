# No-Train Mode Usage Guide

這個文檔說明如何使用新添加的 no-train 模式，該模式可以跳過 retriever 訓練，直接使用 shortest path triplets 進行不同的 ranking 方法。

## 功能概述

No-train 模式提供了 5 種不同的 ranking 方法來選擇 shortest path 中的 triplets：

1. **random**: 隨機選擇 K 個 triplets
2. **sp_count**: 根據 SP count 排序（出現次數越多排名越高）
3. **sp_count_inv**: 根據 inverse SP count 排序（出現次數越少排名越高）
4. **freq_weight**: 類似 freq_weight 的方法（頻率權重）
5. **freq_weight_inv**: 類似 freq_weight_inv 的方法（反向頻率權重）

## 使用方法

### 使用 train_main.py

```bash
# 基本用法
python train_main.py -d webqsp --no_train --ranking_method random --top_k 100

# 使用 SP count 方法
python train_main.py -d webqsp --no_train --ranking_method sp_count --top_k 200

# 使用 inverse SP count 方法
python train_main.py -d webqsp --no_train --ranking_method sp_count_inv --top_k 150

# 使用頻率權重方法
python train_main.py -d webqsp --no_train --ranking_method freq_weight --top_k 100

# 使用反向頻率權重方法
python train_main.py -d webqsp --no_train --ranking_method freq_weight_inv --top_k 100
```

### 使用 inference.py

```bash
# 基本用法
python inference.py --no_train --dataset webqsp --ranking_method random --top_k 100

# 使用 SP count 方法
python inference.py --no_train --dataset webqsp --ranking_method sp_count --top_k 200

# 使用其他方法
python inference.py --no_train --dataset webqsp --ranking_method sp_count_inv --top_k 150
```

## 參數說明

### train_main.py 參數

- `-d, --dataset`: 數據集名稱（webqsp 或 cwq）
- `--no_train`: 啟用 no-train 模式
- `--ranking_method`: ranking 方法，可選值：
  - `random`: 隨機選擇
  - `sp_count`: SP count 排序
  - `sp_count_inv`: 反向 SP count 排序
  - `freq_weight`: 頻率權重
  - `freq_weight_inv`: 反向頻率權重
- `--top_k`: 選擇的 top-K triplets 數量

### inference.py 參數

- `--no_train`: 啟用 no-train 模式
- `--dataset`: 數據集名稱（webqsp 或 cwq）
- `--ranking_method`: ranking 方法（同上）
- `--top_k`: 選擇的 top-K triplets 數量
- `--max_K`: 最大 K 值（預設 500）

## 輸出結果

No-train 模式會生成以下文件：

1. **retrieval_result.pth**: 包含所有測試樣本的推理結果
2. **training_info.txt**: 包含配置信息和執行統計（與訓練模型格式一致）

結果保存在：
```
/home/YX_thesis/retrieve/results/training/{dataset}/{experiment_name}/
```

其中 `experiment_name` 格式為：
```
no_train_{ranking_method}_{top_k}_{timestamp}
```

**重要**: 結果直接保存在 training 目錄下，這樣 `eval.py` 可以使用原有的 batch 評估功能自動處理。

## 結果格式

每個樣本的結果包含：

- `question`: 問題文本
- `scored_triples`: 按排名排序的 triplets 列表，每個 triplet 包含：
  - 頭實體
  - 關係
  - 尾實體
  - 分數
- `q_entity`: 問題實體
- `q_entity_in_graph`: 圖中的問題實體列表
- `a_entity`: 答案實體
- `a_entity_in_graph`: 圖中的答案實體列表
- `max_path_length`: 最大路徑長度
- `target_relevant_triples`: 目標相關 triplets

## 評估結果

生成的結果可以直接使用 `eval.py` 進行評估：

```bash
# 評估所有結果（包括 no-train 結果）
python eval.py -d webqsp --batch_dir /home/YX_thesis/retrieve/results/training/webqsp

# 評估單個 no-train 結果
python eval.py -d webqsp -p /home/YX_thesis/retrieve/results/training/webqsp/no_train_sp_count_100_Oct09-02:15:45/retrieval_result.pth
```

評估結果會自動添加到現有的 `evaluation_results.csv` 和 `evaluation_results_table_format.csv` 文件中。

## 注意事項

1. No-train 模式不需要訓練模型，因此執行速度很快
2. 結果直接基於 shortest path 中的 triplets，不涉及神經網路推理
3. 不同的 ranking 方法可能會產生不同的結果，建議比較多種方法的效果
4. 確保數據集配置文件存在於 `configs/retriever/` 目錄中
5. **結果直接整合到現有評估流程**：no-train 結果會自動與訓練模型結果一起評估和比較

## 示例輸出

執行成功時會看到類似以下的輸出：

```
========== Start inference retriever 2025-10-09 02:46:00 ==========
🚀 Running no-train inference...
   Dataset: webqsp
   Ranking method: sp_count
   Top-K triplets: 50
Loading merged embeddings from: data_files/webqsp/emb/gte-large-en-v1.5/test.pth
# skipped samples: 0
# relevant triples | median: 4 | mean: 20 | max: 699
🚀 Processing 1639 test samples...
✅ No-train inference completed!
🎯 Results saved to: /home/YX_thesis/retrieve/results/training/webqsp/no_train_sp_count_50_Oct09-02:46:06/retrieval_result.pth
========== End inference retriever 2025-10-09 02:46:06 ==========
Retriever inference time: 5.91 seconds
```

然後可以使用 `eval.py` 評估結果：

```
========== Start evaluation retriever 2025-10-09 02:46:11 ==========
Evaluated: no_train_sp_count_50_Oct09-02:46:06
Appended 1 rows to: /home/YX_thesis/retrieve/results/evaluation/webqsp/evaluation_results.csv
Table format saved to: /home/YX_thesis/retrieve/results/evaluation/webqsp/evaluation_results_table_format.csv
========== End evaluation retriever 2025-10-09 02:46:11 ==========
```
