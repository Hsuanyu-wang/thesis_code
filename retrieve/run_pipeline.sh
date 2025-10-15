#!/bin/bash

# Complete Pipeline Script
# Usage: ./run_pipeline.sh <dataset> [--method <freq_weight|inv_freq_weight|default>] [--freq_weight] [--inv_freq_weight] [--method_sp <none|spcount|spcount_inv>]
# Example: ./run_pipeline.sh webqsp --inv_freq_weight --method_sp spcount

set -e  # Exit on any error

# Check if dataset argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <dataset> [--method <freq_weight|inv_freq_weight|default>] [--freq_weight] [--inv_freq_weight] [--method_sp <none|spcount|spcount_inv>]"
    echo "Example: $0 webqsp --inv_freq_weight --method_sp spcount"
    exit 1
fi

DATASET=$1
shift

# Defaults
METHOD=""
METHOD_SP=""

# Parse optional flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)
            if [[ -n "$2" ]]; then
                METHOD="$2"
                shift 2
            else
                echo "Error: --method requires an argument (freq_weight|inv_freq_weight|default)"
                exit 1
            fi
            ;;
        --freq_weight)
            METHOD="freq_weight"
            shift 1
            ;;
        --inv_freq_weight)
            METHOD="inv_freq_weight"
            shift 1
            ;;
        --method_sp)
            if [[ -n "$2" ]]; then
                METHOD_SP="$2"
                shift 2
            else
                echo "Error: --method_sp requires an argument (none|spcount|spcount_inv)"
                exit 1
            fi
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 <dataset> [--method <freq_weight|inv_freq_weight|default>] [--freq_weight] [--inv_freq_weight] [--method_sp <none|spcount|spcount_inv>]"
            exit 1
            ;;
    esac
done

echo "Starting pipeline for dataset: $DATASET"
if [[ -n "$METHOD" ]]; then
    echo "Training method: $METHOD"
fi
if [[ -n "$METHOD_SP" ]]; then
    echo "Post-SP reweight method: $METHOD_SP"
fi

# Step 1: Training
echo "Step 1: Training..."
TRAIN_CMD=(python train.py -d "$DATASET")
if [[ -n "$METHOD" ]]; then
    TRAIN_CMD+=( -m "$METHOD" )
fi
if [[ -n "$METHOD_SP" ]]; then
    TRAIN_CMD+=( --method_sp "$METHOD_SP" )
fi
"${TRAIN_CMD[@]}"

# Find the latest training result directory
TRAINING_DIR=$(find /home/YX_thesis/retrieve/results/training -name "${DATASET}_*" -type d | sort | tail -1)
if [ -z "$TRAINING_DIR" ]; then
    echo "Error: Could not find training results directory"
    exit 1
fi

CPT_PATH="$TRAINING_DIR/cpt.pth"
echo "Found checkpoint: $CPT_PATH"

# Step 2: Inference
echo "Step 2: Inference..."
python inference.py -p "$CPT_PATH"

# Find the latest inference result
RETRIEVAL_RESULT_PATH="$TRAINING_DIR/retrieval_result.pth"
echo "Using retrieval result: $RETRIEVAL_RESULT_PATH"

# Step 3: Evaluation
echo "Step 3: Evaluation..."
python eval.py -d "$DATASET" -p "$RETRIEVAL_RESULT_PATH"

echo "Pipeline completed successfully!"
echo "Results saved in: $TRAINING_DIR"
