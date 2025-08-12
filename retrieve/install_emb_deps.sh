#!/bin/bash

# Installation script for embedding computation dependencies
# This script ensures proper installation order to avoid xformers compatibility issues

echo "Installing dependencies for GTE-Large-EN-v1.5 embedding computation..."

# Step 1: Install PyTorch with CUDA support first
echo "Installing PyTorch 2.1.0 with CUDA 12.1 support..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Step 2: Install xformers (critical for memory efficient attention)
echo "Installing xformers 0.0.23..."
pip install xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu121

# Step 3: Install transformers and other dependencies
echo "Installing transformers and other dependencies..."
pip install transformers==4.43.2
pip install accelerate==0.32.1
pip install datasets==2.20.0
pip install pydantic==2.8.2
pip install numpy==1.24.2
pip install tqdm==4.66.4
pip install pyyaml==6.0.1

# Step 4: Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import xformers; print(f'xformers version: {xformers.__version__}')"
python -c "import transformers; print(f'transformers version: {transformers.__version__}')"

echo "Installation completed successfully!"
echo "You can now run: python emb.py -d cwq" 