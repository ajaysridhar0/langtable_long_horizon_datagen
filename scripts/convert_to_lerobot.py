#!/usr/bin/env python3
"""Run the bundled long-horizon -> LeRobot converter with local defaults."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def _repo_root() -> Path:
  return Path(__file__).resolve().parents[1]


def _converter_script() -> Path:
  return _repo_root() / "data_gen_scripts" / "convert_long_horizon_to_lerobot.py"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input_dir", type=Path, required=True)
  parser.add_argument("--output_repo", required=True)
  parser.add_argument("--hf_lerobot_home", type=Path, default=None)
  parser.add_argument("--fps", type=int, default=10)
  parser.add_argument("--max_episodes", type=int, default=None)
  parser.add_argument("--include_semantic", action="store_true")
  parser.add_argument("--overwrite", action="store_true")
  parser.add_argument("--no_filter_success", action="store_true")
  parser.add_argument("--dry_run", action="store_true")
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  repo_root = _repo_root()
  converter = _converter_script()
  if not converter.exists():
    raise FileNotFoundError(f"Converter not found: {converter}")

  hf_lerobot_home = args.hf_lerobot_home
  if hf_lerobot_home is None:
    hf_lerobot_home = repo_root / "artifacts" / "e2e" / "lerobot_home"
  if not hf_lerobot_home.is_absolute():
    hf_lerobot_home = repo_root / hf_lerobot_home

  input_dir = args.input_dir
  if not input_dir.is_absolute():
    input_dir = repo_root / input_dir

  cmd = [
      sys.executable,
      str(converter),
      "--input_dir",
      str(input_dir),
      "--output_repo",
      args.output_repo,
      "--fps",
      str(args.fps),
  ]
  if args.max_episodes is not None:
    cmd.extend(["--max_episodes", str(args.max_episodes)])
  if args.include_semantic:
    cmd.append("--include_semantic")
  if args.overwrite:
    cmd.append("--overwrite")
  if args.no_filter_success:
    cmd.append("--no_filter_success")

  env = os.environ.copy()
  env["HF_LEROBOT_HOME"] = str(hf_lerobot_home)
  existing_pythonpath = env.get("PYTHONPATH", "")
  env["PYTHONPATH"] = (
      f"{repo_root}{os.pathsep}{existing_pythonpath}"
      if existing_pythonpath else
      str(repo_root)
  )

  print("HF_LEROBOT_HOME:", hf_lerobot_home)
  print("Command:")
  print(" ".join(cmd))
  if args.dry_run:
    return 0

  hf_lerobot_home.mkdir(parents=True, exist_ok=True)
  subprocess.run(cmd, check=True, cwd=repo_root, env=env)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
