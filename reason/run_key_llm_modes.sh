#!/bin/bash

# Script to run key llm_mode combinations for comparison
# Usage: ./run_key_llm_modes.sh

# Base command components
BASE_CMD="python main.py --force_rerun --reverse_order"
SCORE_PATH="/home/YX_thesis/retrieve/results/training/webqsp/webqsp_Oct08-23:14:02_spcount/retrieval_result.pth"

# Key llm_mode combinations to test (most important ones)
KEY_MODES=(
    "sys"                    # Basic system prompt only
    "sys_icl"               # System + ICL
    "sys_dc"                # System + Decision Check
    "sys_icl_dc"            # System + ICL + Decision Check (default)
    "sys_sys_cot"           # System + COT
    "sys_icl_sys_cot"       # System + ICL + COT
    "sys_icl_dc_sys_cot"    # System + ICL + DC + COT (full pipeline)
)

echo "Running key llm_mode experiments (${#KEY_MODES[@]} combinations)"
echo "Score dict path: $SCORE_PATH"
echo ""

# Track results
SUCCESSFUL=()
FAILED=()

# Run each experiment
for i in "${!KEY_MODES[@]}"; do
    llm_mode="${KEY_MODES[$i]}"
    exp_num=$((i + 1))
    
    echo "===================================================================================================="
    echo "Experiment $exp_num/${#KEY_MODES[@]}: $llm_mode"
    echo "===================================================================================================="
    
    start_time=$(date +%s)
    
    # Run the command
    if $BASE_CMD --llm_mode "$llm_mode" -p "$SCORE_PATH"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo ""
        echo "✓ Successfully completed $llm_mode in ${duration}s"
        SUCCESSFUL+=("$llm_mode")
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo ""
        echo "✗ Failed to run $llm_mode after ${duration}s"
        FAILED+=("$llm_mode")
    fi
    
    echo ""
    sleep 3  # Delay between experiments
done

# Print summary
echo "===================================================================================================="
echo "EXPERIMENT SUMMARY"
echo "===================================================================================================="
echo "Total experiments: ${#KEY_MODES[@]}"
echo "Successful: ${#SUCCESSFUL[@]}"
echo "Failed: ${#FAILED[@]}"

if [ ${#SUCCESSFUL[@]} -gt 0 ]; then
    echo ""
    echo "✓ Successful llm_modes:"
    for mode in "${SUCCESSFUL[@]}"; do
        echo "  - $mode"
    done
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "✗ Failed llm_modes:"
    for mode in "${FAILED[@]}"; do
        echo "  - $mode"
    done
fi

echo "===================================================================================================="
