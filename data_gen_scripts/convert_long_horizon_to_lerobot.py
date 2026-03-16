"""Convert Language-Table oracle rollouts to LeRobot format."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
  from lerobot.common.datasets.lerobot_dataset import (
      HF_LEROBOT_HOME,
      LeRobotDataset,
  )
except ImportError:
  try:
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
  except ImportError as exc:
    raise ImportError(
        "lerobot not installed. Install with:\n"
        "  pip install lerobot\n"
        "Or for v2.x compatibility:\n"
        "  pip install 'lerobot<0.4.0'"
    ) from exc


CREATE_SIGNATURE = inspect.signature(LeRobotDataset.create)
ADD_FRAME_SIGNATURE = inspect.signature(LeRobotDataset.add_frame)


def _resolve_episode_dir(input_dir: Path) -> Path:
  if (input_dir / "episodes").is_dir():
    return input_dir / "episodes"
  return input_dir


def _load_metadata(meta_path: Path) -> dict:
  if not meta_path.exists():
    return {}
  with meta_path.open("r", encoding="utf-8") as f:
    return json.load(f)


def _decode_instruction(encoded: np.ndarray) -> str:
  non_zero = encoded[np.where(encoded != 0)]
  if non_zero.shape[0] == 0:
    return ""
  return bytes(non_zero.tolist()).decode("utf-8")


def _get_task_text(metadata: dict, first_state: dict) -> str:
  instruction = metadata.get("instruction")
  if isinstance(instruction, str) and instruction:
    return instruction
  encoded = first_state.get("instruction")
  if encoded is None:
    return ""
  return _decode_instruction(np.asarray(encoded))


def _coerce_state(state_obj):
  if isinstance(state_obj, dict):
    return state_obj
  try:
    return state_obj.item()
  except Exception:
    return state_obj


SHAPE_NAME_MAP = {
    "cube": "square",
}

CORNER_SEMANTIC_REMAP = {
    "top_right": "bottom_left",
    "bottom_left": "top_right",
}


def _block_color(block_name: str) -> str:
  if not block_name:
    return ""
  return block_name.split("_")[0]


def _block_shape(block_name: str) -> str:
  if not block_name or "_" not in block_name:
    return ""
  return block_name.split("_", 1)[1]


def _format_shape(shape: str) -> str:
  return SHAPE_NAME_MAP.get(shape, shape)


def _format_corner(location: str) -> str:
  if not location:
    return ""
  location = CORNER_SEMANTIC_REMAP.get(location, location)
  if location in ("left_view", "right_view"):
    return location.replace("_view", "")
  return location.replace("_", " ")


def _label_from_task(task: Dict[str, Any]) -> str:
  block_name = task.get("block", "")
  color = task.get("color") or _block_color(block_name)
  shape = _format_shape(_block_shape(block_name))
  block_desc = f"{color} {shape}".strip()
  location = _format_corner(task.get("location", ""))
  subtask_type = task.get("subtask_type", "")

  if subtask_type == "clear":
    blocked = _format_corner(
        task.get("blocked_location", "") or task.get("location", "")
    )
    if blocked:
      return f"move {block_desc} away from {blocked}"
    return f"move {block_desc} away"

  if location:
    return f"move {block_desc} to {location}"
  if block_desc:
    return f"move {block_desc}"
  return ""


def _build_labels(
    task_plan: List[Dict[str, Any]], subtask_indices: np.ndarray
) -> List[str]:
  labels: List[str] = []
  last_label = ""
  for idx in subtask_indices.tolist():
    if idx is None or idx < 0 or idx >= len(task_plan):
      labels.append(last_label)
      continue
    label = _label_from_task(task_plan[idx])
    if label:
      last_label = label
    labels.append(last_label)
  return labels


def _output_home() -> Path:
  return Path(os.environ.get("HF_LEROBOT_HOME", HF_LEROBOT_HOME)).expanduser()


def _create_dataset(output_path: Path, output_repo: str, fps: int, features: dict):
  create_kwargs = {
      "repo_id": output_repo,
      "robot_type": "language_table",
      "fps": fps,
      "features": features,
  }
  if "root" in CREATE_SIGNATURE.parameters:
    create_kwargs["root"] = output_path
  if "image_writer_threads" in CREATE_SIGNATURE.parameters:
    create_kwargs["image_writer_threads"] = 4
  if "image_writer_processes" in CREATE_SIGNATURE.parameters:
    create_kwargs["image_writer_processes"] = 2
  return LeRobotDataset.create(**create_kwargs)


def _add_frame(dataset, frame: dict, task_text: str) -> None:
  if "task" in ADD_FRAME_SIGNATURE.parameters:
    dataset.add_frame(frame, task=task_text)
    return
  legacy_frame = dict(frame)
  legacy_frame["task"] = task_text
  dataset.add_frame(legacy_frame)


def convert_long_horizon_to_lerobot(
    input_dir: str,
    output_repo: str,
    fps: int = 10,
    overwrite: bool = False,
    filter_success: bool = True,
    max_episodes: Optional[int] = None,
    include_semantic: bool = False,
) -> None:
  input_path = Path(input_dir)
  episodes_dir = _resolve_episode_dir(input_path)
  npz_files = sorted(episodes_dir.glob("episode_*.npz"))
  if not npz_files:
    raise ValueError(f"No episode_*.npz found in {episodes_dir}")

  first = np.load(npz_files[0], allow_pickle=True)
  first_states = first["states"]
  if len(first_states) == 0:
    raise ValueError(f"Empty episode: {npz_files[0]}")
  first_state = _coerce_state(first_states[0])
  image_shape = first_state["rgb"].shape
  state_dim = np.asarray(first_state["effector_translation"]).shape[0]
  action_dim = first["actions"].shape[1]

  if state_dim != 2:
    raise ValueError(f"Expected 2D effector state, got {state_dim}D")
  if action_dim != 2:
    raise ValueError(f"Expected 2D actions, got {action_dim}D")

  output_path = _output_home() / output_repo
  if output_path.exists():
    if overwrite:
      shutil.rmtree(output_path)
    else:
      raise ValueError(
          f"Dataset already exists at {output_path}. Use --overwrite to replace it."
      )

  features = {
      "image": {
          "dtype": "image",
          "shape": image_shape,
          "names": ["height", "width", "channel"],
      },
      "state": {
          "dtype": "float32",
          "shape": (state_dim,),
          "names": ["state"],
      },
      "action": {
          "dtype": "float32",
          "shape": (action_dim,),
          "names": ["action"],
      },
  }
  if include_semantic:
    features.update({
        "high_level_instruction": {
            "dtype": "string",
            "shape": (1,),
            "names": None,
        },
        "semantic_labels": {
            "dtype": "string",
            "shape": (1,),
            "names": None,
        },
    })

  dataset = _create_dataset(output_path, output_repo, fps, features)

  total_frames = 0
  total_episodes = 0

  for npz_path in npz_files:
    if max_episodes is not None and total_episodes >= max_episodes:
      break
    meta_path = npz_path.with_suffix(".json")
    metadata = _load_metadata(meta_path)
    if filter_success and metadata.get("success") is False:
      continue

    traj = np.load(npz_path, allow_pickle=True)
    states = traj["states"]
    actions = traj["actions"]
    if len(states) == 0:
      continue
    first_state = _coerce_state(states[0])
    task_text = _get_task_text(metadata, first_state)
    semantic_labels = None
    if include_semantic:
      task_plan = metadata.get("task_plan", [])
      subtask_indices = traj.get("subtask_indices")
      if subtask_indices is not None and task_plan:
        semantic_labels = _build_labels(
            task_plan, np.asarray(subtask_indices)
        )

    num_steps = min(len(states), len(actions))
    for i in range(num_steps):
      state = _coerce_state(states[i])
      frame = {
          "image": state["rgb"],
          "state": np.asarray(state["effector_translation"], dtype=np.float32),
          "action": np.asarray(actions[i], dtype=np.float32),
      }
      if include_semantic:
        frame["high_level_instruction"] = task_text
        label = ""
        if semantic_labels is not None and i < len(semantic_labels):
          label = str(semantic_labels[i])
        frame["semantic_labels"] = label
      _add_frame(dataset, frame, task_text)
    dataset.save_episode()
    total_frames += num_steps
    total_episodes += 1

  print("Conversion complete")
  print(f"  Episodes: {total_episodes}")
  print(f"  Frames: {total_frames}")
  print(f"  Output: {output_path}")


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Convert Language-Table long-horizon rollouts to LeRobot format"
  )
  parser.add_argument(
      "--input_dir",
      type=str,
      required=True,
      help="Directory containing episode_*.npz files or an output_dir with episodes/ subdir",
  )
  parser.add_argument(
      "--output_repo",
      type=str,
      default="local/language_table_long_horizon",
      help="LeRobot dataset repo name (default: local/language_table_long_horizon)",
  )
  parser.add_argument(
      "--fps",
      type=int,
      default=10,
      help="Dataset FPS (default: 10)",
  )
  parser.add_argument(
      "--overwrite",
      action="store_true",
      help="Overwrite existing dataset if it exists",
  )
  parser.add_argument(
      "--no_filter_success",
      action="store_true",
      help="Include unsuccessful episodes",
  )
  parser.add_argument(
      "--max_episodes",
      type=int,
      default=None,
      help="Optional cap on number of episodes to convert",
  )
  parser.add_argument(
      "--include_semantic",
      action="store_true",
      help="Include high_level_instruction and semantic_labels fields",
  )

  args = parser.parse_args()

  convert_long_horizon_to_lerobot(
      input_dir=args.input_dir,
      output_repo=args.output_repo,
      fps=args.fps,
      overwrite=args.overwrite,
      filter_success=not args.no_filter_success,
      max_episodes=args.max_episodes,
      include_semantic=args.include_semantic,
  )


if __name__ == "__main__":
  main()
