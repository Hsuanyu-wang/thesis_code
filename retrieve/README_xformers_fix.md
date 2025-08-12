# 解決 xformers 兼容性問題

## 問題描述

在使用 `emb.py -d cwq` 時遇到 xformers 相關錯誤，這通常是由於以下原因：

1. **版本不兼容**：PyTorch、transformers 和 xformers 版本不匹配
2. **安裝順序問題**：xformers 需要在 PyTorch 之後安裝
3. **CUDA 版本不匹配**：xformers 需要與 PyTorch 的 CUDA 版本一致

## 解決方案

### 方法 1：使用自動安裝腳本（推薦）

```bash
# 1. 確保在正確的 conda 環境中
conda activate gte_large_en_v1-5

# 2. 運行自動安裝腳本
chmod +x install_emb_deps.sh
bash install_emb_deps.sh
```

### 方法 2：手動安裝

```bash
# 1. 清理現有環境（可選）
pip uninstall torch torchvision torchaudio xformers transformers -y

# 2. 按順序安裝依賴
# 首先安裝 PyTorch
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 然後安裝 xformers
pip install xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu121

# 最後安裝其他依賴
pip install transformers==4.43.2
pip install accelerate==0.32.1
pip install datasets==2.20.0
pip install pydantic==2.8.2
pip install numpy==1.24.2
pip install tqdm==4.66.4
pip install pyyaml==6.0.1
```

### 方法 3：使用 conda 安裝（替代方案）

```bash
# 如果 pip 安裝有問題，可以嘗試 conda
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.43.2
```

## 驗證安裝

運行環境檢查腳本：

```bash
python check_environment.py
```

這個腳本會檢查：
- ✅ 所有必需套件的版本
- ✅ CUDA 可用性
- ✅ xformers 功能測試
- ✅ GTE 模型載入測試

## 推薦的版本組合

| 套件 | 版本 | 說明 |
|------|------|------|
| PyTorch | 2.1.0 | 核心深度學習框架 |
| xformers | 0.0.23 | 記憶體效率注意力機制 |
| transformers | 4.43.2 | Hugging Face 模型庫 |
| accelerate | 0.32.1 | 模型加速 |
| datasets | 2.20.0 | 資料集處理 |
| CUDA | 12.1 | GPU 加速支援 |

## 常見錯誤及解決方案

### 錯誤 1：xformers 安裝失敗
```
ERROR: Could not find a version that satisfies the requirement xformers
```
**解決方案**：確保使用正確的 PyTorch 版本和 CUDA 版本

### 錯誤 2：記憶體效率注意力不可用
```
RuntimeError: xformers is not available
```
**解決方案**：重新安裝 xformers，確保版本匹配

### 錯誤 3：CUDA 版本不匹配
```
RuntimeError: CUDA version mismatch
```
**解決方案**：檢查 CUDA 版本，確保與 PyTorch 版本一致

## 運行嵌入計算

安裝完成後，運行：

```bash
python emb.py -d cwq
```

## 故障排除

如果仍然遇到問題：

1. **檢查 CUDA 版本**：
   ```bash
   nvidia-smi
   ```

2. **檢查 PyTorch CUDA 支援**：
   ```python
   import torch
   print(torch.version.cuda)
   print(torch.cuda.is_available())
   ```

3. **重新安裝特定版本**：
   ```bash
   pip uninstall xformers -y
   pip install xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu121
   ```

4. **使用 CPU 版本（不推薦，速度慢）**：
   ```bash
   pip install torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu
   ```

## 注意事項

- 確保有足夠的 GPU 記憶體（建議至少 8GB）
- 首次運行會下載 GTE 模型（約 1.5GB）
- 如果 GPU 記憶體不足，可以考慮使用較小的模型或批次大小 