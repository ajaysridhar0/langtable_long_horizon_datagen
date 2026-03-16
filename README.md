# Scripted Language Table Data

This repo is a minimal, shareable subset of Language Table for generating
scripted long-horizon data for two tasks:

- `4_color_sort`
- `4_shape_sort`

It keeps the existing oracle generator and environment code, while trimming
the wrapper layer down to a small set of shareable presets and utilities.

The bundle also includes a small local `tf_agents` compatibility shim so the
raw generator can run without installing the full TensorFlow + tf-agents stack.

## Included

- `scripts/run_oracle_preset.py`: thin preset-based launcher
- `scripts/convert_to_lerobot.py`: bundle-local LeRobot conversion wrapper
- `scripts/assert_raw_dataset.py`: strict raw output assertions
- `scripts/assert_lerobot_dataset.py`: strict LeRobot output assertions
- `scripts/run_e2e_smoke.sh`: one-command local e2e smoke test
- `scripts/validate_dataset.py`: dataset sanity checker
- `presets/*.json`: pinned task presets
- `slurm/generate_oracle_dataset.sbatch`: generic SLURM array wrapper
- `data_gen_scripts/generate_oracle_long_horizon.py`: core generator
- `data_gen_scripts/convert_long_horizon_to_lerobot.py`: LeRobot converter
- `language_table/environments/`: environment, rewards, oracles, assets

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For LeRobot conversion:

```bash
pip install -r requirements-lerobot.txt
```

## Quickstart

Generate one 4-color episode:

```bash
python scripts/run_oracle_preset.py \
  --preset 4_color_sort \
  --seed 0 \
  --out_root data/4_color_sort
```

Generate one 4-shape episode with pairing variation:

```bash
python scripts/run_oracle_preset.py \
  --preset 4_shape_sort_pairings \
  --seed 7000 \
  --combo_index 0 \
  --out_root data/4_shape_sort_pairings
```

Generate one 4-shape episode with the fixed combo used in the preset:

```bash
python scripts/run_oracle_preset.py \
  --preset 4_shape_sort_fixedcombo0 \
  --seed 7000 \
  --out_root data/4_shape_sort_fixedcombo0
```

Videos are off by default. To enable them:

```bash
python scripts/run_oracle_preset.py \
  --preset 4_color_sort \
  --seed 1 \
  --out_root data/4_color_sort_preview \
  --save_video
```

Convert one generated seed directory to LeRobot with semantic labels:

```bash
export HF_LEROBOT_HOME=$PWD/artifacts/lerobot_home
python scripts/convert_to_lerobot.py \
  --input_dir data/4_color_sort/seed_0 \
  --output_repo local/4_color_sort_seed0 \
  --include_semantic \
  --overwrite
```

If you generated quick smoke episodes with `require_success=false`, include
`--no_filter_success` during conversion so the single rollout is not skipped.

## Presets

- `4_color_sort`: deterministic `corner015` noclear/norecover preset used by the 4-color corner pipeline
- `4_color_sort_random_corners`: same `corner015` task with randomized color-to-corner map
- `4_shape_sort_pairings`: shape grouping at edge centers with varying
  `BLOCK_8_PAIRINGS` combos
- `4_shape_sort_fixedcombo0`: same task with a fixed combo baked into the preset

## Output Layout

Each run writes:

```text
<out_root>/
  seed_<seed>/
    config.json
    episodes/
      episode_000000.json
      episode_000000.npz
    videos/                  # only when --save_video is enabled
```

The generator writes the effective config used for the run into `config.json`.

## Validation

Validate a generated dataset root:

```bash
python scripts/validate_dataset.py --root data/4_color_sort
python scripts/validate_dataset.py --root data/4_shape_sort_pairings
```

The validator checks that each seed directory has:

- `config.json`
- `episodes/episode_000000.json`
- `episodes/episode_000000.npz`
- `success: true` in episode metadata by default

For stricter raw-output assertions, including subtask arrays and optional video:

```bash
python scripts/assert_raw_dataset.py \
  --seed_dir data/4_color_sort/seed_0 \
  --expected_task_mode sort_by_color \
  --require_video
```

Validate a converted LeRobot dataset:

```bash
python scripts/assert_lerobot_dataset.py \
  --repo_id local/4_color_sort_seed0 \
  --hf_lerobot_home "$HF_LEROBOT_HOME" \
  --expected_episodes 1 \
  --expect_semantic
```

## E2E Smoke Test

Run the full local smoke test from setup through generation, video saving, raw
assertions, LeRobot conversion, and LeRobot assertions:

```bash
./scripts/run_e2e_smoke.sh
```

This smoke test generates exactly one episode for each task type:

- `4_color_sort`
- `4_shape_sort_fixedcombo0`

The smoke conversion step uses `--no_filter_success` because the quick rollout
settings prioritize speed over guaranteed task completion.

Expected video outputs:

- `artifacts/e2e/raw/4_color/seed_0/videos/episode_000000.mp4`
- `artifacts/e2e/raw/4_shape/seed_7000/videos/episode_000000.mp4`

## Generic SLURM Launch

The included SLURM script is intentionally generic. You should customize
resources at submit time for your cluster.

Example: 100 demos of 4-color sort, 5 seeds per array task:

```bash
sbatch --array=0-19 \
  --export=ALL,PRESET=4_color_sort,OUT_ROOT=data/4_color_sort,TOTAL_DEMOS=100,DEMOS_PER_TASK=5,BASE_SEED=0 \
  slurm/generate_oracle_dataset.sbatch
```

Example: 100 demos of 4-shape pairings cycling through all 90 valid combos:

```bash
sbatch --array=0-19 \
  --export=ALL,PRESET=4_shape_sort_pairings,OUT_ROOT=data/4_shape_sort_pairings,TOTAL_DEMOS=100,DEMOS_PER_TASK=5,BASE_SEED=7000,COMBO_CYCLE_LENGTH=90 \
  slurm/generate_oracle_dataset.sbatch
```

Example: fixed-combo 4-shape generation:

```bash
sbatch --array=0-19 \
  --export=ALL,PRESET=4_shape_sort_fixedcombo0,OUT_ROOT=data/4_shape_sort_fixedcombo0,TOTAL_DEMOS=100,DEMOS_PER_TASK=5,BASE_SEED=7000 \
  slurm/generate_oracle_dataset.sbatch
```

## Notes

- The wrapper injects `PYTHONPATH` automatically, so use the preset launcher
  instead of calling the generator directly.
- The common use pattern is one episode per seed directory.
- Re-running a preset with the same `seed` and `out_root` skips the run when
  `episode_000000.json` already exists, unless `--overwrite` is passed.
