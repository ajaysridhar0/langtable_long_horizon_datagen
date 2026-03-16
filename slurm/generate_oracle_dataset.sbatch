#!/bin/bash
# Generic SLURM array launcher for scripted oracle data generation.
# Customize resources at submit time for your cluster.
#
# Examples:
#   sbatch --array=0-19 \
#     --export=ALL,PRESET=4_color_sort,OUT_ROOT=data/4_color_sort,TOTAL_DEMOS=100,DEMOS_PER_TASK=5,BASE_SEED=0 \
#     slurm/generate_oracle_dataset.sbatch
#
#   sbatch --array=0-19 \
#     --export=ALL,PRESET=4_shape_sort_pairings,OUT_ROOT=data/4_shape_sort_pairings,TOTAL_DEMOS=100,DEMOS_PER_TASK=5,BASE_SEED=7000,COMBO_CYCLE_LENGTH=90 \
#     slurm/generate_oracle_dataset.sbatch
#
#   sbatch --array=0-19 \
#     --export=ALL,PRESET=4_shape_sort_fixedcombo0,OUT_ROOT=data/4_shape_sort_fixedcombo0,TOTAL_DEMOS=100,DEMOS_PER_TASK=5,BASE_SEED=7000 \
#     slurm/generate_oracle_dataset.sbatch

#SBATCH --job-name=oracle_data
#SBATCH --array=0-0
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PRESET="${PRESET:?Set PRESET to a preset name or path.}"
PRESET_BASENAME="$(basename "${PRESET%.json}")"
OUT_ROOT="${OUT_ROOT:-data/${PRESET_BASENAME}}"
TOTAL_DEMOS="${TOTAL_DEMOS:?Set TOTAL_DEMOS to the number of seed folders to generate.}"
DEMOS_PER_TASK="${DEMOS_PER_TASK:-1}"
BASE_SEED="${BASE_SEED:-0}"
COMBO_CYCLE_LENGTH="${COMBO_CYCLE_LENGTH:-0}"
FIXED_COMBO_INDEX="${FIXED_COMBO_INDEX:-}"
NUM_EPISODES="${NUM_EPISODES:-}"
MAX_STEPS="${MAX_STEPS:-}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_START=$(( TASK_ID * DEMOS_PER_TASK ))
TASK_END=$(( TASK_START + DEMOS_PER_TASK - 1 ))

echo "repo_root=${REPO_ROOT}"
echo "task=${TASK_ID} demos=${TASK_START}..${TASK_END} total=${TOTAL_DEMOS}"
echo "preset=${PRESET} out_root=${OUT_ROOT}"

for LOCAL_IDX in $(seq 0 $(( DEMOS_PER_TASK - 1 ))); do
  DEMO_IDX=$(( TASK_START + LOCAL_IDX ))
  if [ "${DEMO_IDX}" -ge "${TOTAL_DEMOS}" ]; then
    continue
  fi

  SEED=$(( BASE_SEED + DEMO_IDX ))
  CMD=(
    "${PYTHON_BIN}"
    "${REPO_ROOT}/scripts/run_oracle_preset.py"
    --preset "${PRESET}"
    --seed "${SEED}"
    --out_root "${OUT_ROOT}"
  )

  if [ -n "${NUM_EPISODES}" ]; then
    CMD+=( --num_episodes "${NUM_EPISODES}" )
  fi
  if [ -n "${MAX_STEPS}" ]; then
    CMD+=( --max_steps "${MAX_STEPS}" )
  fi
  if [ "${SAVE_VIDEO}" = "1" ]; then
    CMD+=( --save_video )
  fi
  if [ "${OVERWRITE}" = "1" ]; then
    CMD+=( --overwrite )
  fi
  if [ "${COMBO_CYCLE_LENGTH}" -gt 0 ]; then
    CMD+=( --combo_index "$(( DEMO_IDX % COMBO_CYCLE_LENGTH ))" )
  elif [ -n "${FIXED_COMBO_INDEX}" ]; then
    CMD+=( --combo_index "${FIXED_COMBO_INDEX}" )
  fi
  if [ "${DRY_RUN}" = "1" ]; then
    CMD+=( --dry_run )
  fi

  echo "[$(date)] demo_idx=${DEMO_IDX} seed=${SEED}"
  "${CMD[@]}"
done
