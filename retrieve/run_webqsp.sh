#!/usr/bin/env bash
set -euo pipefail

# Change to the directory of this script so relative paths work
cd "$(dirname "$0")"

# Configurables (can be overridden via env):
#   PYTHON: python interpreter (default: python3)
#   DATASET: dataset name (default: webqsp)
PYTHON=${PYTHON:-python3}
DATASET=${DATASET:-webqsp}


# Early-stop validation modes to iterate
esv_modes=("and" "or")

run_cmd() {
  echo "$*"
  eval "$*"
}

for fw in "${kge_flags[@]}"; do
  echo "==== Running for weighting ${fw} on dataset ${DATASET} ===="

  # 1) no-dropout + esv in {and, or}
  for esv in "${esv_modes[@]}"; do
    run_cmd "${PYTHON} train_main.py -d ${DATASET} ${fw} -esv ${esv}"
  done

  # 2) dropout only
  run_cmd "${PYTHON} train_main.py -d ${DATASET} ${fw} -dp"

  # 3) dropout + esv in {and, or}
  for esv in "${esv_modes[@]}"; do
    run_cmd "${PYTHON} train_main.py -d ${DATASET} ${fw} -dp -esv ${esv}"
  done

done 