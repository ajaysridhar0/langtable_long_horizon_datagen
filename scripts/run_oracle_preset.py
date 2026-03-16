#!/usr/bin/env python3
"""Run the long-horizon oracle generator from a pinned preset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def _resolve_layout() -> tuple[Path, Path]:
  script_path = Path(__file__).resolve()
  for candidate in script_path.parents:
    generator = candidate / "data_gen_scripts" / "generate_oracle_long_horizon.py"
    exported_presets = candidate / "presets"
    source_presets = candidate / "scripted_data_repo" / "presets"
    if generator.exists() and exported_presets.is_dir():
      return candidate, exported_presets
    if generator.exists() and source_presets.is_dir():
      return candidate, source_presets
  raise FileNotFoundError("Could not locate repo root and presets directory.")


def _repo_root() -> Path:
  repo_root, _ = _resolve_layout()
  return repo_root


def _presets_dir() -> Path:
  _, presets_dir = _resolve_layout()
  return presets_dir


def _generator_script() -> Path:
  return _repo_root() / "data_gen_scripts" / "generate_oracle_long_horizon.py"


def _parse_value(raw: str):
  lowered = raw.lower()
  if lowered == "true":
    return True
  if lowered == "false":
    return False
  if lowered == "null":
    return None
  try:
    return json.loads(raw)
  except json.JSONDecodeError:
    return raw


def _load_preset(preset_arg: str) -> dict:
  preset_path = Path(preset_arg)
  if not preset_path.exists():
    preset_name = preset_arg
    if not preset_name.endswith(".json"):
      preset_name = f"{preset_name}.json"
    preset_path = _presets_dir() / preset_name
  with preset_path.open("r", encoding="utf-8") as f:
    return json.load(f)


def _format_flag(key: str, value) -> str | None:
  if value is None:
    return None
  if isinstance(value, bool):
    return f"--{key}={'true' if value else 'false'}"
  return f"--{key}={value}"


def _build_output_dir(args: argparse.Namespace) -> Path:
  if args.output_dir:
    return Path(args.output_dir)
  if args.out_root is None:
    raise ValueError("Either --out_root or --output_dir is required.")
  return Path(args.out_root) / f"seed_{args.seed}"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--preset",
      required=True,
      help="Preset name from presets/ or a path to a preset JSON file.",
  )
  parser.add_argument("--seed", type=int, required=True, help="Seed for this run.")
  parser.add_argument(
      "--out_root",
      type=Path,
      help="Root output directory. The run writes into <out_root>/seed_<seed>/.",
  )
  parser.add_argument(
      "--output_dir",
      type=Path,
      help="Explicit output directory. Overrides --out_root when set.",
  )
  parser.add_argument(
      "--num_episodes",
      type=int,
      help="Override the preset's num_episodes value.",
  )
  parser.add_argument(
      "--max_steps",
      type=int,
      help="Override the preset's max_steps value.",
  )
  parser.add_argument(
      "--combo_index",
      type=int,
      help="Override combo_index for BLOCK_8_PAIRINGS presets.",
  )
  parser.add_argument(
      "--save_video",
      action="store_true",
      help="Override the preset to save rollout videos.",
  )
  parser.add_argument(
      "--overwrite",
      action="store_true",
      help="Overwrite an existing single-episode seed directory.",
  )
  parser.add_argument(
      "--set",
      dest="overrides",
      action="append",
      default=[],
      metavar="KEY=VALUE",
      help="Extra generator override. May be passed multiple times.",
  )
  parser.add_argument(
      "--dry_run",
      action="store_true",
      help="Print the resolved generator command without executing it.",
  )
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  preset = _load_preset(args.preset)
  flags = dict(preset.get("generator_flags", {}))

  output_dir = _build_output_dir(args)
  repo_root = _repo_root()
  if not output_dir.is_absolute():
    output_dir = repo_root / output_dir
  if (
      not args.overwrite
      and int(flags.get("num_episodes", 1)) == 1
      and (output_dir / "episodes" / "episode_000000.json").exists()
  ):
    print(f"Skipping existing run: {output_dir}")
    return 0

  flags["output_dir"] = str(output_dir)
  flags["seed"] = args.seed
  if args.num_episodes is not None:
    flags["num_episodes"] = args.num_episodes
  if args.max_steps is not None:
    flags["max_steps"] = args.max_steps
  if args.combo_index is not None:
    flags["combo_index"] = args.combo_index
  if args.save_video:
    flags["save_video"] = True

  for item in args.overrides:
    if "=" not in item:
      raise ValueError(f"Invalid override {item!r}; expected KEY=VALUE.")
    key, raw_value = item.split("=", 1)
    flags[key] = _parse_value(raw_value)

  generator = _generator_script()
  if not generator.exists():
    raise FileNotFoundError(f"Generator not found: {generator}")

  cmd = [sys.executable, str(generator)]
  for key in sorted(flags):
    arg = _format_flag(key, flags[key])
    if arg is not None:
      cmd.append(arg)

  env = os.environ.copy()
  existing_pythonpath = env.get("PYTHONPATH", "")
  env["PYTHONPATH"] = (
      f"{repo_root}{os.pathsep}{existing_pythonpath}"
      if existing_pythonpath else
      str(repo_root)
  )

  print("Preset:", preset.get("name", args.preset))
  print("Output:", output_dir)
  print("Command:")
  print(" ".join(cmd))
  if args.dry_run:
    return 0

  output_dir.mkdir(parents=True, exist_ok=True)
  subprocess.run(cmd, check=True, cwd=repo_root, env=env)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
