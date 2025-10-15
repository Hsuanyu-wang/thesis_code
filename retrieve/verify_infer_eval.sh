#!/bin/bash

# Verify and Auto-complete Inference & Evaluation Results (Batch-friendly)
# Default root: /home/YX_thesis/retrieve/results
# Usage:
#   ./verify_infer_eval.sh [dataset] [--root <path>] [--exp_dir <path>] [--datasets <ds1,ds2>] [--force-infer] [--force-eval]
# Notes:
#   - Only checks under "$ROOT/training/<dataset>/*" (experiment folders). It will NOT scan "$ROOT/training" directly.
# Examples:
#   ./verify_infer_eval.sh                     # batch over all datasets under default root
#   ./verify_infer_eval.sh webqsp              # batch only webqsp dataset folder
#   ./verify_infer_eval.sh --datasets webqsp,cwq
#   ./verify_infer_eval.sh --exp_dir /home/YX_thesis/retrieve/results/training/webqsp/webqsp_Nov08-01:14:47

set -e

# Defaults
ROOT="/home/YX_thesis/retrieve/results"
DATASET=""
EXP_DIR=""
FORCE_INFER=0
FORCE_EVAL=0
DATASETS_CSV=""

# Collect failures
FAILED_INFER=()
FAILED_EVAL=()

# Parse optional leading positional dataset (if provided)
if [[ $# -gt 0 && "$1" != --* ]]; then
    DATASET="$1"
    shift
fi

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            if [[ -n "$2" ]]; then
                ROOT="$2"
                shift 2
            else
                echo "Error: --root requires a path argument"; exit 1
            fi
            ;;
        --exp_dir)
            if [[ -n "$2" ]]; then
                EXP_DIR="$2"
                shift 2
            else
                echo "Error: --exp_dir requires a path argument"; exit 1
            fi
            ;;
        --datasets)
            if [[ -n "$2" ]]; then
                DATASETS_CSV="$2"
                shift 2
            else
                echo "Error: --datasets requires a comma-separated list"; exit 1
            fi
            ;;
        --force-infer)
            FORCE_INFER=1; shift 1 ;;
        --force-eval)
            FORCE_EVAL=1; shift 1 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

TRAIN_ROOT="$ROOT/training"

if [[ -n "$EXP_DIR" ]]; then
    # Single-experiment mode
    if [[ ! -d "$EXP_DIR" ]]; then
        echo "Error: Experiment directory does not exist: $EXP_DIR"; exit 1
    fi
    # Infer dataset from parent folder if not provided
    if [[ -z "$DATASET" ]]; then
        DATASET=$(basename "$(dirname "$EXP_DIR")")
        if [[ -z "$DATASET" ]]; then
            echo "Error: Unable to infer dataset from parent directory of: $EXP_DIR"; exit 1
        fi
    fi
    CPT_PATH="$EXP_DIR/cpt.pth"
    if [[ ! -f "$CPT_PATH" ]]; then
        echo "Error: Missing checkpoint: $CPT_PATH"; exit 1
    fi
    RETR_PATH="$EXP_DIR/retrieval_result.pth"

    echo "Root: $ROOT"
    echo "Dataset: $DATASET"
    echo "Experiment dir: $EXP_DIR"
    echo "Checkpoint: $CPT_PATH"

    do_infer=0
    if [[ $FORCE_INFER -eq 1 || ! -f "$RETR_PATH" ]]; then
        do_infer=1
    fi
    if [[ $do_infer -eq 1 ]]; then
        echo "[Inference] Folder: $EXP_DIR"
        set +e
        python /home/YX_thesis/retrieve/inference.py -p "$CPT_PATH"
        rc=$?
        set -e
        if [[ $rc -ne 0 ]]; then
            echo "[Inference] Failed: $EXP_DIR (exit $rc)"
            FAILED_INFER+=("$EXP_DIR: inference exit $rc")
        fi
    else
        echo "[Inference] Skipped (exists): $RETR_PATH"
    fi
    if [[ ! -f "$RETR_PATH" ]]; then
        echo "[Inference] Output missing after run: $RETR_PATH"
        FAILED_INFER+=("$EXP_DIR: missing retrieval_result.pth")
    else
        echo "Inference output: $RETR_PATH"
    fi

    # Evaluation with CSV existence check
    if [[ -f "$RETR_PATH" ]]; then
        EXP_ID=$(basename "$EXP_DIR")
        EVAL_DIR="$ROOT/evaluation/$DATASET"
        CSV_PATH="$EVAL_DIR/evaluation_results.csv"
        if [[ -f "$CSV_PATH" && $FORCE_EVAL -ne 1 ]]; then
            if awk -F, 'NR>1 && $1==exp {found=1} END {exit !found}' exp="$EXP_ID" "$CSV_PATH"; then
                echo "[Evaluation] Skipped (exists in CSV): $EXP_ID"
            else
                echo "[Evaluation] Folder: $EXP_DIR"
                set +e
                python /home/YX_thesis/retrieve/eval.py -d "$DATASET" -p "$RETR_PATH"
                rc=$?
                set -e
                if [[ $rc -ne 0 ]]; then
                    echo "[Evaluation] Failed: $EXP_DIR (exit $rc)"
                    FAILED_EVAL+=("$EXP_DIR: eval exit $rc")
                fi
            fi
        else
            echo "[Evaluation] Folder: $EXP_DIR"
            set +e
            python /home/YX_thesis/retrieve/eval.py -d "$DATASET" -p "$RETR_PATH"
            rc=$?
            set -e
            if [[ $rc -ne 0 ]]; then
                echo "[Evaluation] Failed: $EXP_DIR (exit $rc)"
                FAILED_EVAL+=("$EXP_DIR: eval exit $rc")
            fi
        fi
    fi

    # Summary
    if [[ ${#FAILED_INFER[@]} -gt 0 || ${#FAILED_EVAL[@]} -gt 0 ]]; then
        echo "=== Failures Summary ==="
        if [[ ${#FAILED_INFER[@]} -gt 0 ]]; then
            echo "Inference failures:"
            for item in "${FAILED_INFER[@]}"; do echo "  - $item"; done
        fi
        if [[ ${#FAILED_EVAL[@]} -gt 0 ]]; then
            echo "Evaluation failures:"
            for item in "${FAILED_EVAL[@]}"; do echo "  - $item"; done
        fi
    else
        echo "All tasks succeeded for: $EXP_DIR"
    fi
    exit 0
fi

# Batch mode over TRAIN_ROOT/<dataset>/*
if [[ ! -d "$TRAIN_ROOT" ]]; then
    echo "Error: Training root not found: $TRAIN_ROOT"; exit 1
fi

echo "Root: $ROOT"

# Determine dataset list to process (directories immediately under training root)
DATASET_LIST=()
if [[ -n "$DATASETS_CSV" ]]; then
    IFS=',' read -r -a DATASET_LIST <<< "$DATASETS_CSV"
elif [[ -n "$DATASET" ]]; then
    DATASET_LIST=( "$DATASET" )
else
    mapfile -t DATASET_LIST < <(find "$TRAIN_ROOT" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort)
fi

if [[ ${#DATASET_LIST[@]} -eq 0 ]]; then
    echo "Warning: No dataset folders found under: $TRAIN_ROOT"
    exit 0
fi

# 1) Run inference per experiment folder with explicit folder display
for ds in "${DATASET_LIST[@]}"; do
    ds_dir="$TRAIN_ROOT/$ds"
    if [[ ! -d "$ds_dir" ]]; then
        echo "Skip non-directory: $ds_dir"
        continue
    fi
    mapfile -t EXP_DIRS < <(find "$ds_dir" -maxdepth 1 -mindepth 1 -type d | sort)
    for exp in "${EXP_DIRS[@]}"; do
        CPT_PATH="$exp/cpt.pth"
        RETR_PATH="$exp/retrieval_result.pth"
        if [[ ! -f "$CPT_PATH" ]]; then
            echo "[Inference] Skip (no cpt.pth): $exp"
            continue
        fi
        if [[ $FORCE_INFER -eq 1 || ! -f "$RETR_PATH" ]]; then
            echo "[Inference] Folder: $exp"
            set +e
            python /home/YX_thesis/retrieve/inference.py -p "$CPT_PATH"
            rc=$?
            set -e
            if [[ $rc -ne 0 ]]; then
                echo "[Inference] Failed: $exp (exit $rc)"
                FAILED_INFER+=("$exp: inference exit $rc")
                continue
            fi
        else
            echo "[Inference] Skipped (exists): $RETR_PATH"
        fi
        if [[ ! -f "$RETR_PATH" ]]; then
            echo "[Inference] Output missing after run: $RETR_PATH"
            FAILED_INFER+=("$exp: missing retrieval_result.pth")
            continue
        fi
    done
done

# 2) Run evaluation per experiment folder with explicit folder display and CSV check
for ds in "${DATASET_LIST[@]}"; do
    ds_dir="$TRAIN_ROOT/$ds"
    if [[ ! -d "$ds_dir" ]]; then
        continue
    fi
    EVAL_DIR="$ROOT/evaluation/$ds"
    CSV_PATH="$EVAL_DIR/evaluation_results.csv"
    declare -A SEEN_IDS
    if [[ -f "$CSV_PATH" ]]; then
        # Load existing exp_ids into associative array
        while IFS=, read -r exp_id _rest; do
            if [[ "$exp_id" != "exp_id" && -n "$exp_id" ]]; then
                SEEN_IDS["$exp_id"]=1
            fi
        done < "$CSV_PATH"
    fi
    mapfile -t EXP_DIRS < <(find "$ds_dir" -maxdepth 1 -mindepth 1 -type d | sort)
    for exp in "${EXP_DIRS[@]}"; do
        RETR_PATH="$exp/retrieval_result.pth"
        if [[ ! -f "$RETR_PATH" ]]; then
            echo "[Evaluation] Skip (no retrieval_result.pth): $exp"
            continue
        fi
        EXP_ID=$(basename "$exp")
        if [[ $FORCE_EVAL -ne 1 && -n "${SEEN_IDS[$EXP_ID]}" ]]; then
            echo "[Evaluation] Skipped (exists in CSV): $EXP_ID"
            continue
        fi
        echo "[Evaluation] Folder: $exp"
        set +e
        python /home/YX_thesis/retrieve/eval.py -d "$ds" -p "$RETR_PATH"
        rc=$?
        set -e
        if [[ $rc -ne 0 ]]; then
            echo "[Evaluation] Failed: $exp (exit $rc)"
            FAILED_EVAL+=("$exp: eval exit $rc")
            continue
        fi
    done
done

# Final summary
if [[ ${#FAILED_INFER[@]} -gt 0 || ${#FAILED_EVAL[@]} -gt 0 ]]; then
    echo "=== Failures Summary ==="
    if [[ ${#FAILED_INFER[@]} -gt 0 ]]; then
        echo "Inference failures:"
        for item in "${FAILED_INFER[@]}"; do echo "  - $item"; done
    fi
    if [[ ${#FAILED_EVAL[@]} -gt 0 ]]; then
        echo "Evaluation failures:"
        for item in "${FAILED_EVAL[@]}"; do echo "  - $item"; done
    fi
else
    echo "All tasks succeeded."
fi

echo "All done. Evaluation results under: /home/YX_thesis/retrieve/results/evaluation/<dataset>" 