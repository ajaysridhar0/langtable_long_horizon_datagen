#!/usr/bin/env python3
"""Validate the standard scripted-data output layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


SEED_DIR_RE = re.compile(r"^seed_(\d+)$")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, required=True, help="Dataset root.")
  parser.add_argument(
      "--expected_count",
      type=int,
      help="Optional expected number of seed directories.",
  )
  parser.add_argument(
      "--allow_failed_success",
      action="store_true",
      help="Do not fail validation when episode metadata has success=false.",
  )
  return parser.parse_args()


def _seed_dirs(root: Path) -> list[Path]:
  if not root.exists():
    raise FileNotFoundError(f"Dataset root not found: {root}")
  return sorted(
      path for path in root.iterdir()
      if path.is_dir() and SEED_DIR_RE.match(path.name)
  )


def main() -> int:
  args = parse_args()
  seed_dirs = _seed_dirs(args.root)

  issues: list[str] = []
  success_count = 0
  for seed_dir in seed_dirs:
    config_path = seed_dir / "config.json"
    episode_json = seed_dir / "episodes" / "episode_000000.json"
    episode_npz = seed_dir / "episodes" / "episode_000000.npz"

    if not config_path.exists():
      issues.append(f"missing config: {config_path}")
      continue
    if not episode_json.exists():
      issues.append(f"missing episode json: {episode_json}")
      continue
    if not episode_npz.exists():
      issues.append(f"missing episode npz: {episode_npz}")
      continue

    with episode_json.open("r", encoding="utf-8") as f:
      payload = json.load(f)
    success = bool(payload.get("success", False))
    if success:
      success_count += 1
    elif not args.allow_failed_success:
      issues.append(f"episode success=false: {episode_json}")

  if args.expected_count is not None and len(seed_dirs) != args.expected_count:
    issues.append(
        f"expected {args.expected_count} seed dirs, found {len(seed_dirs)} at {args.root}"
    )

  print(f"root: {args.root}")
  print(f"seed_dirs: {len(seed_dirs)}")
  print(f"successful_episodes: {success_count}")
  if issues:
    print("issues:")
    for item in issues:
      print(f"- {item}")
    return 1

  print("validation: ok")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
