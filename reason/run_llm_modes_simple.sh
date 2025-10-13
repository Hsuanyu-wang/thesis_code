#!/bin/bash

# Script to iterate through different llm_mode options
# Usage: ./run_llm_modes_simple.sh

# Base command components
BASE_CMD="python main.py --force_rerun"
SCORE_PATH="/home/YX_thesis/retrieve/results/training/webqsp/webqsp_Oct08-23:14:02_spcount/retrieval_result.pth"

# Define all llm_mode options to test
LLM_MODES=(
    "sys"
    "sys_icl" 
    "sys_dc"
    "sys_icl_dc"
    "sys_sys_cot"
    "sys_icl_sys_cot"
    "sys_sys_cot_clear"
    "sys_icl_sys_cot_clear"
    "sys_dc_sys_cot"
    "sys_icl_dc_sys_cot"
    "sys_dc_sys_cot_clear"
    "sys_icl_dc_sys_cot_clear"
)

echo "Starting experiments with ${#LLM_MODES[@]} different llm_mode options"
echo "Score dict path: $SCORE_PATH"
echo ""

# Track results
SUCCESSFUL=()
FAILED=()

# Run each experiment
for i in "${!LLM_MODES[@]}"; do
    llm_mode="${LLM_MODES[$i]}"
    exp_num=$((i + 1))
    
    echo "####################################################################################################"
    echo "Experiment $exp_num/${#LLM_MODES[@]}: $llm_mode"
    echo "####################################################################################################"
    
    # Run the command
    if $BASE_CMD --llm_mode "$llm_mode" -p "$SCORE_PATH"; then
        echo ""
        echo "✓ Successfully completed $llm_mode"
        SUCCESSFUL+=("$llm_mode")
    else
        echo ""
        echo "✗ Failed to run $llm_mode"
        FAILED+=("$llm_mode")
    fi
    
    echo ""
    sleep 2  # Small delay between experiments
done

# Print summary
echo "####################################################################################################"
echo "EXPERIMENT SUMMARY"
echo "####################################################################################################"
echo "Total experiments: ${#LLM_MODES[@]}"
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

echo "####################################################################################################"
