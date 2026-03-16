#!/usr/bin/env python3
"""Assert the expected raw scripted-data layout and payload fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED_ARRAYS = (
    "states",
    "actions",
    "subtask_indices",
    "subtask_ids",
    "subtask_types",
    "subtask_blocks",
    "subtask_labels",
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--seed_dir", type=Path, required=True)
  parser.add_argument("--expected_task_mode", required=True)
  parser.add_argument("--require_video", action="store_true")
  return parser.parse_args()


def _fail(message: str) -> None:
  raise AssertionError(message)


def main() -> int:
  args = parse_args()
  seed_dir = args.seed_dir.resolve()
  config_path = seed_dir / "config.json"
  episode_json = seed_dir / "episodes" / "episode_000000.json"
  episode_npz = seed_dir / "episodes" / "episode_000000.npz"
  video_path = seed_dir / "videos" / "episode_000000.mp4"

  if not config_path.exists():
    _fail(f"Missing config: {config_path}")
  if not episode_json.exists():
    _fail(f"Missing episode json: {episode_json}")
  if not episode_npz.exists():
    _fail(f"Missing episode npz: {episode_npz}")
  if args.require_video and not video_path.exists():
    _fail(f"Missing video: {video_path}")

  metadata = json.loads(episode_json.read_text(encoding="utf-8"))
  task_mode = str(metadata.get("task_mode", ""))
  if task_mode != args.expected_task_mode:
    _fail(f"Expected task_mode={args.expected_task_mode}, got {task_mode}")

  task_plan = metadata.get("task_plan", [])
  if not task_plan:
    _fail("Expected non-empty task_plan in episode metadata")

  traj = np.load(episode_npz, allow_pickle=True)
  files = set(traj.files)
  missing = [name for name in REQUIRED_ARRAYS if name not in files]
  if missing:
    _fail(f"Missing arrays in NPZ: {missing}")

  lengths = {}
  for name in REQUIRED_ARRAYS:
    value = traj[name]
    lengths[name] = len(value)

  expected_len = lengths["actions"]
  for name, length in lengths.items():
    if length != expected_len:
      _fail(f"Length mismatch for {name}: expected {expected_len}, got {length}")

  print(f"seed_dir: {seed_dir}")
  print(f"task_mode: {task_mode}")
  print(f"steps: {expected_len}")
  if args.require_video:
    print(f"video: {video_path}")
  print("raw_assertions: ok")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
