#!/usr/bin/env bash
set -euo pipefail

# Change to the directory of this script so relative paths work
cd "$(dirname "$0")"

# Configurables (can be overridden via env):
#   PYTHON: python interpreter (default: python3)
#   DATASET: dataset name (default: webqsp)
PYTHON=${PYTHON:-python3}
DATASET=${DATASET:-webqsp}

echo "🚀 Running specific KGE commands for dataset: ${DATASET}"
echo "🐍 Using Python interpreter: ${PYTHON}"
echo "=================================================="

# Early-stop validation modes to iterate (from run_webqsp.sh)
esv_modes=("and" "or")

run_cmd() {
  echo "▶️  Running: $*"
  echo "⏰ $(date)"
  eval "$*"
  echo "✅ Completed: $*"
  echo "⏰ $(date)"
  echo "----------------------------------------"
}

# Function to run experiments with different configurations (based on run_webqsp.sh)
run_experiment_variants() {
  local base_cmd="$1"
  local experiment_name="$2"
  
  echo "🔬 Running experiment variants for: ${experiment_name}"
  
  # 1) no-dropout + esv in {and, or}
  for esv in "${esv_modes[@]}"; do
    run_cmd "${base_cmd} -esv ${esv}"
  done
  
  # 2) dropout only
  run_cmd "${base_cmd} -dp"
  
  # 3) dropout + esv in {and, or}
  for esv in "${esv_modes[@]}"; do
    run_cmd "${base_cmd} -dp -esv ${esv}"
  done
}

# Your specific commands
specific_commands=(

    "${PYTHON} train_main.py -d ${DATASET} -kw -km rotate -kwm score"
    "${PYTHON} train_main.py -d ${DATASET} -kw -km rotate -kwm prob"
    "${PYTHON} train_main.py -d ${DATASET} -kw -km complex -kwm score"
    "${PYTHON} train_main.py -d ${DATASET} -kw -km complex -kwm prob"
    "${PYTHON} train_main.py -d ${DATASET} -kgf -km rotate -kwm prob_inv"
    "${PYTHON} train_main.py -d ${DATASET} -kgf -km rotate -kwm score_inv"
    "${PYTHON} train_main.py -d ${DATASET} -kgf -km complex -kwm prob"
    "${PYTHON} train_main.py -d ${DATASET} -kgf -km complex -kwm score"
)

echo "📊 Running ${#specific_commands[@]} specific KGE commands"
echo "📊 Total variants per command: $((${#esv_modes[@]} * 2 + 1))"
echo "📊 Total experiments to run: $((${#specific_commands[@]} * (${#esv_modes[@]} * 2 + 1)))"
echo ""

experiment_count=0
for cmd in "${specific_commands[@]}"; do
  experiment_count=$((experiment_count + 1))
  
  # Extract experiment details for naming
  if [[ $cmd == *"-kw "* ]]; then
    exp_type="kge_weight"
  elif [[ $cmd == *"-kgf "* ]]; then
    exp_type="kge_freq_weight"
  else
    exp_type="unknown"
  fi
  
  model=$(echo $cmd | grep -o '\-km [a-z]*' | cut -d' ' -f2)
  mode=$(echo $cmd | grep -o '\-kwm [a-z_]*' | cut -d' ' -f2)
  
  experiment_name="${exp_type}_${model}_${mode}"
  
  echo "🎯 Command ${experiment_count}/${#specific_commands[@]}: ${experiment_name}"
  echo "   Base command: ${cmd}"
  
  run_experiment_variants "${cmd}" "${experiment_name}"
  
  echo "✅ Completed command group: ${experiment_name}"
  echo "=================================================="
done

echo "🎉 All specific KGE commands completed!"
echo "⏰ Finished at: $(date)"

# Generate summary
echo ""
echo "📋 Command Summary:"
echo "🔍 Results saved in: /home/YX_thesis/retrieve/results/training/${DATASET}/"
echo "📁 Look for directories with patterns:"
echo "   - retriever_${DATASET}_*_kge_rotate_score*"
echo "   - retriever_${DATASET}_*_kge_rotate_prob*"
echo "   - retriever_${DATASET}_*_kge_complex_score*"
echo "   - retriever_${DATASET}_*_kge_complex_prob*"
echo "   - retriever_${DATASET}_*_kge_freq_rotate_prob_inv*"
echo "   - retriever_${DATASET}_*_kge_freq_rotate_score_inv*"
echo "   - retriever_${DATASET}_*_kge_freq_complex_prob*"
echo "   - retriever_${DATASET}_*_kge_freq_complex_score*"


