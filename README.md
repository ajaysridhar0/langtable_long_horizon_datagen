# Scripted Language Table Data

This repo is a small, shareable subset of Language Table for generating
scripted long-horizon data for:

- `4_color_sort`
- `4_shape_sort`

The main interface is `scripts/run_oracle_preset.py`. In most cases, you only
need to pick a preset, a seed, and an output directory.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you also want LeRobot export:

```bash
pip install -r requirements-lerobot.txt
```

## Quick Start

Generate one 4-color episode:

```bash
python scripts/run_oracle_preset.py \
  --preset 4_color_sort \
  --seed 0 \
  --out_root data/4_color_sort
```

Generate one 4-shape episode:

```bash
python scripts/run_oracle_preset.py \
  --preset 4_shape_sort_fixedcombo0 \
  --seed 7000 \
  --out_root data/4_shape_sort
```

Enable videos with `--save_video`:

```bash
python scripts/run_oracle_preset.py \
  --preset 4_color_sort \
  --seed 1 \
  --out_root data/4_color_sort_preview \
  --save_video
```

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

## Presets

- `4_color_sort`: deterministic 4-color corner sort
- `4_color_sort_random_corners`: 4-color corner sort with randomized color-to-corner map
- `4_shape_sort_pairings`: 4-shape grouping with varying `BLOCK_8_PAIRINGS` combos
- `4_shape_sort_fixedcombo0`: 4-shape grouping with a fixed combo

## Validate Raw Data

Validate an output root:

```bash
python scripts/validate_dataset.py --root data/4_color_sort
python scripts/validate_dataset.py --root data/4_shape_sort
```

For a stricter check on one seed directory:

```bash
python scripts/assert_raw_dataset.py \
  --seed_dir data/4_color_sort/seed_0 \
  --expected_task_mode sort_by_color
```

Add `--require_video` if you saved videos.

## Convert To LeRobot

Convert one generated seed directory:

```bash
export HF_LEROBOT_HOME=$PWD/artifacts/lerobot_home
python scripts/convert_to_lerobot.py \
  --input_dir data/4_color_sort/seed_0 \
  --output_repo local/4_color_sort_seed0 \
  --include_semantic \
  --overwrite
```

Validate the converted dataset:

```bash
python scripts/assert_lerobot_dataset.py \
  --repo_id local/4_color_sort_seed0 \
  --hf_lerobot_home "$HF_LEROBOT_HOME" \
  --expected_episodes 1 \
  --expect_semantic
```

## Smoke Test

Run the full local smoke test:

```bash
./scripts/run_e2e_smoke.sh
```

This generates one 4-color episode, one 4-shape episode, saves videos, converts
both to LeRobot, and runs the assertion scripts.

## Bulk Generation With SLURM

Example: 100 demos of `4_color_sort`, 5 seeds per array task:

```bash
sbatch --array=0-19 \
  --export=ALL,PRESET=4_color_sort,OUT_ROOT=data/4_color_sort,TOTAL_DEMOS=100,DEMOS_PER_TASK=5,BASE_SEED=0 \
  slurm/generate_oracle_dataset.sbatch
```

Example: 100 demos of `4_shape_sort_pairings`, cycling through all 90 combos:

```bash
sbatch --array=0-19 \
  --export=ALL,PRESET=4_shape_sort_pairings,OUT_ROOT=data/4_shape_sort_pairings,TOTAL_DEMOS=100,DEMOS_PER_TASK=5,BASE_SEED=7000,COMBO_CYCLE_LENGTH=90 \
  slurm/generate_oracle_dataset.sbatch
```

## Notes

- Use `scripts/run_oracle_preset.py` rather than calling the generator directly.
- The common pattern is one episode per seed directory.
- Re-running the same seed skips existing output unless `--overwrite` is passed.
