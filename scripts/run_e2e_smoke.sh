#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv-e2e"
RAW_ROOT="${REPO_ROOT}/artifacts/e2e/raw"
HF_LEROBOT_HOME="${REPO_ROOT}/artifacts/e2e/lerobot_home"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
pip install -r "${REPO_ROOT}/requirements.txt"
pip install -r "${REPO_ROOT}/requirements-lerobot.txt"

export HF_LEROBOT_HOME

rm -rf "${RAW_ROOT}" "${HF_LEROBOT_HOME}"
mkdir -p "${RAW_ROOT}" "${HF_LEROBOT_HOME}"

python "${REPO_ROOT}/scripts/run_oracle_preset.py" \
  --preset 4_color_sort \
  --seed 0 \
  --out_root "${RAW_ROOT}/4_color" \
  --save_video \
  --max_steps 50 \
  --set require_success=false \
  --set require_plan=false \
  --set max_episode_attempts=1

python "${REPO_ROOT}/scripts/assert_raw_dataset.py" \
  --seed_dir "${RAW_ROOT}/4_color/seed_0" \
  --expected_task_mode sort_by_color \
  --require_video

python "${REPO_ROOT}/scripts/run_oracle_preset.py" \
  --preset 4_shape_sort_fixedcombo0 \
  --seed 7000 \
  --out_root "${RAW_ROOT}/4_shape" \
  --save_video \
  --max_steps 50 \
  --set require_success=false \
  --set require_plan=false \
  --set max_episode_attempts=1

python "${REPO_ROOT}/scripts/assert_raw_dataset.py" \
  --seed_dir "${RAW_ROOT}/4_shape/seed_7000" \
  --expected_task_mode sort_by_shape_edge_centers \
  --require_video

python "${REPO_ROOT}/scripts/convert_to_lerobot.py" \
  --input_dir "${RAW_ROOT}/4_color/seed_0" \
  --output_repo local/bundle_smoke_4_color \
  --include_semantic \
  --no_filter_success \
  --overwrite

python "${REPO_ROOT}/scripts/assert_lerobot_dataset.py" \
  --repo_id local/bundle_smoke_4_color \
  --hf_lerobot_home "${HF_LEROBOT_HOME}" \
  --expected_episodes 1 \
  --expect_semantic

python "${REPO_ROOT}/scripts/convert_to_lerobot.py" \
  --input_dir "${RAW_ROOT}/4_shape/seed_7000" \
  --output_repo local/bundle_smoke_4_shape \
  --include_semantic \
  --no_filter_success \
  --overwrite

python "${REPO_ROOT}/scripts/assert_lerobot_dataset.py" \
  --repo_id local/bundle_smoke_4_shape \
  --hf_lerobot_home "${HF_LEROBOT_HOME}" \
  --expected_episodes 1 \
  --expect_semantic

echo "E2E smoke test passed."
echo "4-color video: ${RAW_ROOT}/4_color/seed_0/videos/episode_000000.mp4"
echo "4-shape video: ${RAW_ROOT}/4_shape/seed_7000/videos/episode_000000.mp4"
echo "4-color lerobot: ${HF_LEROBOT_HOME}/local/bundle_smoke_4_color"
echo "4-shape lerobot: ${HF_LEROBOT_HOME}/local/bundle_smoke_4_shape"
