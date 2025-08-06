# SubgraphRAG Stage 2: Reasoning

## 目錄

- [安裝說明](#安裝說明)
- [可重現性：預處理檔案](#可重現性預處理檔案)
- [推論流程](#推論流程)
- [自訂檢索結果用法](#自訂檢索結果用法)
- [設定檔說明](#設定檔說明)

---

## 安裝說明

建議使用 Conda 建立獨立環境：

```bash
conda create -n reasoner python=3.10.14 -y
conda activate reasoner
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.5.5 openai==1.50.2 wandb
```

---

## 可重現性：預處理檔案

我們提供已處理好的檢索與推理結果，方便直接重現論文實驗。下載方式：

```bash
huggingface-cli download siqim311/SubgraphRAG --revision main --local-dir ./
```

- `scored_triples/`：存放檢索階段的結果
- `results/KGQA/`：存放推理階段的結果

---

## 推論流程

下載好預處理結果後，可直接執行主程式：

```bash
python main.py -d webqsp --prompt_mode scored_100
python main.py -d cwq --prompt_mode scored_100
```

---

## 自訂檢索結果用法

若要使用自訂的檢索結果，請加上 `-p` 參數指定路徑：

```bash
python main.py -d webqsp --prompt_mode scored_100 -p <檢索結果路徑>
```
範例：`../retrieve/webqsp_Nov08-01:14:47/retrieval_result.pth`

---

## 設定檔說明

各資料集的詳細設定可參考 `./configs/` 目錄。

---

如需更多細節，請參閱原始碼或聯絡作者。 