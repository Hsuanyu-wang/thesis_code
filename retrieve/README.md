# SubgraphRAG Retrieval System

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Supported Datasets](#supported-datasets)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [File Structure](#file-structure)
- [Detailed Usage](#detailed-usage)
- [KGE Integration](#kge-integration)
- [Pipeline Scripts](#pipeline-scripts)

## 🎯 Overview

SubgraphRAG is a knowledge graph question answering system that combines:
- **GTE (Graph-aware Text Embedding)**: Semantic embeddings for entities and relations
- **DDE (Distance-based Dynamic Embedding)**: Graph structure information
- **KGE (Knowledge Graph Embedding)**: Structural knowledge from KG embeddings
- **PE (Positional Encoding)**: Topic entity positional information

The system retrieves relevant triples from knowledge graphs to answer complex multi-hop questions.

## 🏗️ Architecture

### Model Components
```
Input: [Zq||Zh||Zr||Zt||Ztau] (GTE + DDE + PE)
    ↓
MLP → mlp_logits
    ↓
KGE Model → kge_score
    ↓
Output: (mlp_logits, kge_score)
```

### Loss Function
```
BCE Loss = binary_cross_entropy_with_logits(mlp_logits, target_triple_probs)
KGE Margin Ranking Loss = max(0, pos_kge_scores - neg_kge_scores + margin)
Total Loss = BCE Loss + λ * KGE Margin Ranking Loss
```

## 📊 Supported Datasets

- **WebQSP**: WebQuestionsSP dataset
- **CWQ**: Complex Web Questions dataset
- **KGQAGen**: KGQAGen-10k dataset (HuggingFace: lianglz/KGQAGen-10k)

## 🚀 Installation

### Environment Setup

```bash
# Create and activate environment
conda create -n subgraphrag python=3.10 -y
conda activate subgraphrag

# Install dependencies
pip install -r requirements/retriever.txt
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.5.3
pip install pyg_lib==0.3.1 torch_scatter==2.1.2 torch_sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

## ⚡ Quick Start

### 1. Complete Pipeline (Recommended)
```bash
# Run complete training and evaluation pipeline
./pipeline.sh webqsp
```

### 2. Step-by-Step Execution
```bash
# Step 1: Pre-compute embeddings
python emb.py -d webqsp

# Step 2: Train retriever
python train.py -d webqsp

# Step 3: Run inference
python inference.py -p "training result/webqsp_xxx/cpt.pth"

# Step 4: Evaluate results
python eval.py -d webqsp -p "training result/webqsp_xxx/retrieval_result.pth"
```

## 📁 File Structure

```
retrieve/
├── 📄 Core Scripts
│   ├── train.py                    # Main training script
│   ├── inference.py                # Inference and retrieval
│   ├── eval.py                     # Evaluation script
│   ├── emb.py                      # Embedding pre-computation
│   └── train_kge.py               # KGE model training
│
├── 🧪 Test Scripts
│   ├── test_kge_score_integration.py  # KGE score integration test
│   └── test_kge_integration.py        # KGE integration test
│
├── ⚙️ Configuration
│   ├── configs/retriever/
│   │   ├── webqsp.yaml            # WebQSP configuration
│   │   └── cwq.yaml               # CWQ configuration
│   └── configs/emb/
│       └── gte-large-en-v1.5/     # GTE embedding config
│
├── 🔧 Source Code
│   └── src/
│       ├── model/
│       │   ├── retriever.py       # Main retriever model
│       │   ├── kge_models.py      # KGE models (TransE, DistMult, etc.)
│       │   └── kge_utils.py       # KGE utilities
│       ├── dataset/
│       │   └── retriever.py       # Dataset processing
│       └── config/
│           └── retriever.py       # Configuration loading
│
├── 📊 Results
│   ├── training result/            # Training checkpoints
│   ├── retrieve_result/            # Evaluation results
│   └── wandb/                     # Experiment tracking
│
└── 📚 Documentation
    ├── README.md                   # This file
    ├── KGE_SCORE_INTEGRATION_SUMMARY.md  # KGE integration details
    └── docs/                       # Additional documentation
```

## 🔧 Detailed Usage

### 1. Embedding Pre-computation
```bash
python emb.py -d webqsp
```
**Purpose**: Pre-compute entity and relation embeddings using GTE model
**Output**: Cached embeddings in `data_files/webqsp/emb/`

### 2. Training
```bash
python train.py -d webqsp
```
**Features**:
- KGE score integration with margin ranking loss
- Wandb experiment tracking
- Early stopping mechanism
- Configurable KGE loss weight

**Output**: Model checkpoint in `training result/webqsp_xxx/cpt.pth`

### 3. Inference
```bash
python inference.py -p "path/to/cpt.pth" --max_K 500
```
**Features**:
- Batch processing
- Configurable top-K retrieval
- Automatic result saving

### 4. Evaluation
```bash
python eval.py -d webqsp -p "path/to/retrieval_result.pth"
```
**Metrics**:
- `triple_recall@k`: Triple retrieval accuracy
- `ans_recall@k`: Answer entity retrieval accuracy

## 🧠 KGE Integration

### Supported KGE Models
- **TransE**: Translating embeddings for multi-relational data
- **DistMult**: Diagonal matrix factorization
- **PTransE**: Path-based TransE
- **RotatE**: Rotation-based embeddings
- **ComplEx**: Complex embeddings
- **SimplE**: Simple embedding model

### Configuration
```yaml
kge:
  enabled: true
  model_type: 'transe'  # Options: 'transe', 'distmult', 'ptranse'
  embedding_dim: 256
  margin: 1.0
  loss_weight: 1.0      # Weight for KGE margin ranking loss
```

### KGE Score Integration
- KGE provides structural knowledge as additional supervision
- Margin ranking loss encourages positive triples to have higher scores than negative ones
- Configurable loss weight allows balancing between semantic and structural information

## 🚀 Pipeline Scripts

### Complete Pipeline
```bash
./pipeline.sh webqsp
```
**Steps**:
1. Pre-compute embeddings
2. Train retriever with KGE integration
3. Run inference
4. Evaluate results
5. Generate summary report

### KGE Training Pipeline
```bash
python run_kge_integration.py --dataset webqsp --kge_model transe --full_pipeline
```

### Testing Pipeline
```bash
python test_kge_score_integration.py
```

## 📈 Monitoring

### Wandb Integration
- Automatic experiment tracking
- Learning curves visualization
- Hyperparameter logging
- Model checkpoint management

### Logging
- Training progress with tqdm
- Loss components (BCE, KGE, Total)
- Evaluation metrics
- Early stopping information

## 🔍 Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size in config
2. **KGE model not found**: Run `python train_kge.py` first
3. **Shape mismatch**: Ensure model checkpoint matches current architecture

### Testing
```bash
# Test KGE integration
python test_kge_score_integration.py

# Test individual components
python -c "from src.model.retriever import Retriever; print('Model import successful')"
```

## 📚 References

- SubgraphRAG: Retrieval-Augmented Generation for Complex Multi-hop Questions
- Knowledge Graph Embeddings: A Survey
- TransE: Translating Embeddings for Modeling Multi-relational Data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.
