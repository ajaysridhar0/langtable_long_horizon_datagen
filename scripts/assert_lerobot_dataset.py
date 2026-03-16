#!/usr/bin/env python3
"""Assert the expected LeRobot dataset structure for a converted smoke run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


REQUIRED_META_FILES = (
    "meta/info.json",
    "meta/tasks.jsonl",
    "meta/episodes.jsonl",
)

REQUIRED_SEMANTIC_COLUMNS = (
    "high_level_instruction",
    "semantic_labels",
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--repo_id", required=True)
  parser.add_argument("--hf_lerobot_home", type=Path, required=True)
  parser.add_argument("--expected_episodes", type=int, required=True)
  parser.add_argument("--expect_semantic", action="store_true")
  return parser.parse_args()


def _fail(message: str) -> None:
  raise AssertionError(message)


def _require_column(column_names: set[str], options: tuple[str, ...]) -> str:
  for name in options:
    if name in column_names:
      return name
  _fail(f"Missing required column. Expected one of: {list(options)}")
  raise AssertionError("unreachable")


def main() -> int:
  args = parse_args()
  dataset_root = (args.hf_lerobot_home / args.repo_id).resolve()
  if not dataset_root.exists():
    _fail(f"Missing dataset root: {dataset_root}")

  for rel_path in REQUIRED_META_FILES:
    path = dataset_root / rel_path
    if not path.exists():
      _fail(f"Missing metadata file: {path}")

  parquet_files = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
  if not parquet_files:
    _fail(f"No parquet data files found under {dataset_root / 'data'}")

  info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
  total_episodes = int(info.get("total_episodes", -1))
  if total_episodes != args.expected_episodes:
    _fail(
        f"Expected total_episodes={args.expected_episodes}, got {total_episodes}"
    )

  table = pq.read_table(parquet_files[0])
  column_names = set(table.column_names)
  for name in ("episode_index", "frame_index", "timestamp", "task_index"):
    if name not in column_names:
      _fail(f"Missing parquet column: {name}")

  image_column = _require_column(
      column_names,
      ("image", "observation.image", "observation.images.image"),
  )
  state_column = _require_column(
      column_names,
      ("state", "observation.state"),
  )
  action_column = _require_column(
      column_names,
      ("action", "actions"),
  )

  if args.expect_semantic:
    missing_semantic = [
        name for name in REQUIRED_SEMANTIC_COLUMNS if name not in column_names
    ]
    if missing_semantic:
      _fail(f"Missing semantic columns: {missing_semantic}")

  print(f"dataset_root: {dataset_root}")
  print(f"episodes: {total_episodes}")
  print(f"parquet_file: {parquet_files[0]}")
  print(f"image_column: {image_column}")
  print(f"state_column: {state_column}")
  print(f"action_column: {action_column}")
  print("lerobot_assertions: ok")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
