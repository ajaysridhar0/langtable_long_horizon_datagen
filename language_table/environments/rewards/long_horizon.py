# coding=utf-8
# Copyright 2024 The Language Tale Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Long-horizon reward utilities for chained block-to-location tasks.

This file contains two planners/reward calculators:

* :class:`ChainedBlock2LocationReward` (baseline): produces an independent set
  of block-to-corner subtasks (optionally shuffled).
* :class:`StrategicChainedBlock2LocationReward` (new): a higher-level planner
  that performs *true* sorting by:

  1) clearing wrong-color blockers from corner slots,
  2) placing blocks in a safer order (farthest-first), and
  3) opportunistically replanning if already-placed blocks get disturbed.

The strategic planner emits explicit subtask labels (`place`, `clear`,
`recover`) and stable `subtask_id`s so that data generation can align
subtasks with actions/observations.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from language_table.environments import constants
from language_table.environments.rewards import constants as reward_constants
from language_table.environments.rewards import reward as base_reward
from language_table.environments.rewards import task_info


COLOR_TO_CORNER = {
    "red": "top_left",
    "blue": "top_right",
    "green": "bottom_left",
    "yellow": "bottom_right",
}

# Camera-view aligned mapping (rotated 90 degrees CCW in world coordinates).
# This makes the visual "top-left/top-right" in the rendered image match the
# labels used in instructions.
COLOR_TO_CORNER_VIEW = {
    "red": "bottom_left",
    "blue": "bottom_right",
    "green": "top_left",
    "yellow": "top_right",
}

COLOR_TO_LEFT_RIGHT = {
    "red": "left",
    "blue": "right",
}

COLOR_TO_LEFT_RIGHT_VIEW = {
    "red": "left_view",
    "blue": "right_view",
}

SHAPE_TO_EDGE_CENTER = {
    # Canonical mapping for shape-grouping edge-center task.
    "moon": "left",
    "crescent": "left",
    "cube": "right",
    "star": "top",
    "pentagon": "bottom",
}


def get_color_to_corner_map(mode: str) -> Dict[str, str]:
  """Return color->corner mapping for sort_by_color."""
  if mode == "default":
    return COLOR_TO_CORNER
  if mode == "view":
    return COLOR_TO_CORNER_VIEW
  if mode in ("red_left_blue_right", "left_right"):
    return COLOR_TO_LEFT_RIGHT
  if mode in ("left_right_view", "red_left_blue_right_view"):
    return COLOR_TO_LEFT_RIGHT_VIEW
  raise ValueError(f"Unknown color_to_corner_map: {mode}")

DEFAULT_INSTRUCTION = "group the blocks by color"
DEFAULT_SHAPE_EDGE_INSTRUCTION = "group the blocks by shape"
DEFAULT_HORIZONTAL_LINE_INSTRUCTION = (
    "arrange the blocks in a horizontal line across the middle of the table"
)
DEFAULT_SINGLE_FILE_INSTRUCTION = (
    "arrange the blocks in single file through the middle of the table"
)
HORIZONTAL_LINE_LOCATION = "horizontal_line"
SINGLE_FILE_LOCATION = "single_file"
_LINE_ASSIGNMENT_SORTED = "sorted"
_LINE_ASSIGNMENT_RANDOM = "random"
_LINE_ASSIGNMENT_FIXED_MIXED = "fixed_mixed"
_LINE_ASSIGNMENT_MODES = (
    _LINE_ASSIGNMENT_SORTED,
    _LINE_ASSIGNMENT_RANDOM,
    _LINE_ASSIGNMENT_FIXED_MIXED,
)
_LINE_LAYOUT_HORIZONTAL = "horizontal"
_LINE_LAYOUT_SINGLE_FILE = "single_file"
_LINE_LAYOUT_MODES = (_LINE_LAYOUT_HORIZONTAL, _LINE_LAYOUT_SINGLE_FILE)
_SUCCESS_MODE_POINT = "point"
_SUCCESS_MODE_REGION = "region"
_SUCCESS_MODE_CLUSTER_REGION = "cluster_region"
_SUCCESS_MODE_CORNER_RADIUS = "corner_radius"
_SUCCESS_MODES = (
    _SUCCESS_MODE_POINT,
    _SUCCESS_MODE_REGION,
    _SUCCESS_MODE_CLUSTER_REGION,
    _SUCCESS_MODE_CORNER_RADIUS,
)
_GOAL_VIS_POINT = "point"
_GOAL_VIS_NONE = "none"
_GOAL_VIS_MODES = (_GOAL_VIS_POINT, _GOAL_VIS_NONE)
_CLEAR_TARGET_STAGING = "staging"
_CLEAR_TARGET_OWN_GOAL = "own_goal"
_CLEAR_TARGET_MODES = (_CLEAR_TARGET_STAGING, _CLEAR_TARGET_OWN_GOAL)


def _normalize_workspace_inset(inset: Optional[float]) -> float:
  if inset is None:
    return float(constants.WORKSPACE_BOUNDS_BUFFER)
  inset_val = float(inset)
  if inset_val < 0:
    return float(constants.WORKSPACE_BOUNDS_BUFFER)
  return max(0.0, inset_val)


def _normalize_clear_target_mode(mode: str) -> str:
  mode = str(mode or "").lower().strip()
  if mode in ("goal", "own_goal", "own-goal", "target"):
    return _CLEAR_TARGET_OWN_GOAL
  if mode in ("staging", "stage", "center"):
    return _CLEAR_TARGET_STAGING
  raise ValueError(f"Unknown clear_target_mode: {mode}; expected one of {_CLEAR_TARGET_MODES}")


def _normalize_line_assignment_mode(mode: Optional[str]) -> str:
  if mode is None:
    return _LINE_ASSIGNMENT_SORTED
  normalized = str(mode).lower().strip()
  aliases = {
      "fixed": _LINE_ASSIGNMENT_FIXED_MIXED,
      "mixed": _LINE_ASSIGNMENT_FIXED_MIXED,
      "randomized": _LINE_ASSIGNMENT_RANDOM,
  }
  normalized = aliases.get(normalized, normalized)
  if normalized not in _LINE_ASSIGNMENT_MODES:
    raise ValueError(
        f"Unknown line_assignment_mode: {mode}; expected one of {_LINE_ASSIGNMENT_MODES}"
    )
  return normalized


def _normalize_line_layout_mode(mode: Optional[str]) -> str:
  if mode is None:
    return _LINE_LAYOUT_HORIZONTAL
  normalized = str(mode).lower().strip()
  aliases = {
      "single": _LINE_LAYOUT_SINGLE_FILE,
      "single-file": _LINE_LAYOUT_SINGLE_FILE,
      "vertical": _LINE_LAYOUT_SINGLE_FILE,
      "hline": _LINE_LAYOUT_HORIZONTAL,
  }
  normalized = aliases.get(normalized, normalized)
  if normalized not in _LINE_LAYOUT_MODES:
    raise ValueError(
        f"Unknown line_layout_mode: {mode}; expected one of {_LINE_LAYOUT_MODES}"
    )
  return normalized


@dataclasses.dataclass
class LongHorizonSubtaskInfo(task_info.Block2LocationTaskInfo):
  """Block2LocationTaskInfo extended with long-horizon subtask labels."""

  # One of: "place", "clear", "recover".
  subtask_type: str = ""
  # Monotonically increasing identifier (unique within an episode).
  subtask_id: int = -1


def _block_color(block_name: str) -> str:
  return block_name.split("_")[0]


def _block_shape(block_name: str) -> str:
  if "_" not in block_name:
    return ""
  return block_name.split("_", 1)[1]


def _pretty_block(block_name: str) -> str:
  return " ".join(block_name.split("_"))


def _normalize_success_mode(mode: Optional[str]) -> str:
  if mode is None:
    return _SUCCESS_MODE_POINT
  mode = str(mode).lower()
  if mode not in _SUCCESS_MODES:
    raise ValueError(f"Unknown success_mode: {mode}")
  return mode


def _normalize_cluster_region_padding(
    padding: Optional[float],
    target_distance: float,
) -> float:
  if padding is None:
    return float(target_distance)
  padding_val = float(padding)
  if padding_val < 0:
    return float(target_distance)
  return max(0.0, padding_val)


def _normalize_corner_success_radius(
    radius: Optional[float],
    target_distance: float,
) -> float:
  if radius is None:
    return float(target_distance)
  radius_val = float(radius)
  if radius_val <= 0:
    return float(target_distance)
  return radius_val


def _normalize_goal_vis(mode: Optional[str]) -> str:
  if mode is None:
    return _GOAL_VIS_POINT
  mode = str(mode).lower()
  if mode not in _GOAL_VIS_MODES:
    raise ValueError(f"Unknown goal_visualization: {mode}")
  return mode


def _block_in_location_region(block_xy: np.ndarray, location: str, margin: float) -> bool:
  if not location:
    return False
  margin = max(0.0, float(margin))
  x, y = float(block_xy[0]), float(block_xy[1])
  cx = float(constants.CENTER_X)
  cy = float(constants.CENTER_Y)
  if location == "left":
    return x <= cx - margin
  if location == "right":
    return x >= cx + margin
  if location == "top":
    return y <= cy - margin
  if location == "bottom":
    return y >= cy + margin
  if location == "left_view":
    return y <= cy - margin
  if location == "right_view":
    return y >= cy + margin
  if location == "top_left":
    return x <= cx - margin and y <= cy - margin
  if location == "top_right":
    return x >= cx + margin and y <= cy - margin
  if location == "bottom_left":
    return x <= cx - margin and y >= cy + margin
  if location == "bottom_right":
    return x >= cx + margin and y >= cy + margin
  return False


def _block_in_corner_radius(
    block_xy: np.ndarray,
    location: str,
    radius: float,
    *,
    corner_centers: Dict[str, np.ndarray],
) -> bool:
  center = corner_centers.get(str(location))
  if center is None:
    return False
  return float(np.linalg.norm(np.asarray(block_xy, dtype=np.float32) - center)) <= float(radius)


def _cluster_regions_from_tasks(
    task_plan: Sequence[Dict[str, Any]],
    padding: float,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
  regions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
  pad = max(0.0, float(padding))
  for task in task_plan:
    location = str(task.get("location", ""))
    if not location or location == "staging":
      continue
    if "target_translation" not in task:
      continue
    target = np.asarray(task["target_translation"], dtype=np.float32)
    if location not in regions:
      regions[location] = (target.copy(), target.copy())
    else:
      lo, hi = regions[location]
      regions[location] = (np.minimum(lo, target), np.maximum(hi, target))
  if pad > 0:
    for location, (lo, hi) in regions.items():
      regions[location] = (lo - pad, hi + pad)
  return regions


def _cluster_regions_from_block_targets(
    block_targets: Dict[str, np.ndarray],
    block_locations: Dict[str, str],
    padding: float,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
  regions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
  pad = max(0.0, float(padding))
  for block, target in block_targets.items():
    location = str(block_locations.get(block, ""))
    if not location or location == "staging":
      continue
    target = np.asarray(target, dtype=np.float32)
    if location not in regions:
      regions[location] = (target.copy(), target.copy())
    else:
      lo, hi = regions[location]
      regions[location] = (np.minimum(lo, target), np.maximum(hi, target))
  if pad > 0:
    for location, (lo, hi) in regions.items():
      regions[location] = (lo - pad, hi + pad)
  return regions


def _block_in_cluster_region(
    block_xy: np.ndarray,
    region: Tuple[np.ndarray, np.ndarray],
) -> bool:
  lo, hi = region
  return bool(
      (block_xy[0] >= lo[0])
      and (block_xy[0] <= hi[0])
      and (block_xy[1] >= lo[1])
      and (block_xy[1] <= hi[1])
  )


def _success_by_cluster_region(
    state: Optional[Dict[str, Any]],
    task_plan: Sequence[Dict[str, Any]],
    padding: float,
) -> bool:
  if state is None:
    return False
  regions = _cluster_regions_from_tasks(task_plan, padding)
  if not regions:
    return False
  block_locations: Dict[str, str] = {}
  for task in task_plan:
    location = str(task.get("location", ""))
    if not location or location == "staging":
      continue
    block = str(task.get("block", ""))
    if block:
      block_locations[block] = location
  if not block_locations:
    return False
  for block, location in block_locations.items():
    key = f"block_{block}_translation"
    if key not in state or location not in regions:
      return False
    block_xy = np.asarray(state[key], dtype=np.float32)
    if not _block_in_cluster_region(block_xy, regions[location]):
      return False
  return True


def _corner_centers(
    workspace_inset: Optional[float] = None,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> Dict[str, np.ndarray]:
  inset = _normalize_workspace_inset(workspace_inset)
  x_min = constants.X_MIN + inset
  x_max = constants.X_MAX - inset
  y_min = constants.Y_MIN + inset
  y_max = constants.Y_MAX - inset
  dx = float(offset_x)
  dy = float(offset_y)
  return {
      # NOTE: The LanguageTable environment's debug markers define:
      #   * left/right via X_MIN/X_MAX
      #   * top/bottom via Y_MIN/Y_MAX ("top" is smaller Y)
      # See LanguageTable._setup_pybullet_scene() boundary markers.
      "top_left": np.array([x_min + dx, y_min + dy], dtype=np.float32),
      "top_right": np.array([x_max + dx, y_min + dy], dtype=np.float32),
      "bottom_left": np.array([x_min + dx, y_max + dy], dtype=np.float32),
      "bottom_right": np.array([x_max + dx, y_max + dy], dtype=np.float32),
  }


def _location_centers(workspace_inset: Optional[float] = None) -> Dict[str, np.ndarray]:
  inset = _normalize_workspace_inset(workspace_inset)
  centers = _corner_centers(workspace_inset=inset)
  x_min = constants.X_MIN + inset
  x_max = constants.X_MAX - inset
  y_min = constants.Y_MIN + inset
  y_max = constants.Y_MAX - inset
  y_center = constants.CENTER_Y
  centers.update({
      "left": np.array([x_min, y_center], dtype=np.float32),
      "right": np.array([x_max, y_center], dtype=np.float32),
      "top": np.array([constants.CENTER_X, y_min], dtype=np.float32),
      "bottom": np.array([constants.CENTER_X, y_max], dtype=np.float32),
      # View-aligned left/right (image left/right correspond to world top/bottom).
      "left_view": np.array([constants.CENTER_X, y_min], dtype=np.float32),
      "right_view": np.array([constants.CENTER_X, y_max], dtype=np.float32),
  })
  return centers


def _location_text(location: str) -> str:
  if location == HORIZONTAL_LINE_LOCATION:
    return "horizontal line through the middle of the table"
  if location == SINGLE_FILE_LOCATION:
    return "single-file line through the middle of the table"
  if location in ("left", "right", "left_view", "right_view"):
    side = location.replace("_view", "")
    return f"{side} side of the table"
  if location in ("top", "bottom"):
    return f"{location} side center of the table"
  return f"{location.replace('_', ' ')} corner of the table"


def _clip_to_bounds(
    xy: np.ndarray,
    workspace_inset: Optional[float] = None,
) -> np.ndarray:
  inset = _normalize_workspace_inset(workspace_inset)
  x_min = constants.X_MIN + inset
  x_max = constants.X_MAX - inset
  y_min = constants.Y_MIN + inset
  y_max = constants.Y_MAX - inset
  return np.array([
      np.clip(xy[0], x_min, x_max),
      np.clip(xy[1], y_min, y_max),
  ], dtype=np.float32)


def _cluster_positions(center: np.ndarray, count: int, spacing: float,
                       rng: np.random.RandomState, jitter: float,
                       workspace_inset: Optional[float] = None) -> List[np.ndarray]:
  offsets = [
      (0.0, 0.0),
      (spacing, 0.0),
      (0.0, spacing),
      (spacing, spacing),
      (-spacing, 0.0),
      (0.0, -spacing),
      (-spacing, -spacing),
      (spacing, -spacing),
  ]
  positions = []
  for i in range(count):
    dx, dy = offsets[i % len(offsets)]
    jitter_delta = (
        rng.uniform(-jitter, jitter, size=(2,)) if jitter > 0 else np.zeros((2,))
    )
    candidate = center + np.array([dx, dy], dtype=np.float32) + jitter_delta
    clipped = _clip_to_bounds(candidate, workspace_inset=workspace_inset)
    if not np.allclose(clipped, candidate):
      clip_mask = np.logical_not(np.isclose(clipped, candidate))
      dx = -dx if clip_mask[0] else dx
      dy = -dy if clip_mask[1] else dy
      candidate = center + np.array([dx, dy], dtype=np.float32) + jitter_delta
      clipped = _clip_to_bounds(candidate, workspace_inset=workspace_inset)
    positions.append(clipped)
  return positions


def _horizontal_line_positions(
    count: int,
    spacing: float,
    rng: np.random.RandomState,
    jitter: float,
    workspace_inset: Optional[float] = None,
) -> List[np.ndarray]:
  """Returns evenly-spaced targets along a horizontal line at table center."""
  if count <= 0:
    return []
  inset = _normalize_workspace_inset(workspace_inset)
  x_min = float(constants.X_MIN + inset)
  x_max = float(constants.X_MAX - inset)
  x_span = max(0.0, x_max - x_min)
  spacing = max(0.0, float(spacing))
  if count <= 1:
    effective_spacing = 0.0
  else:
    effective_spacing = min(spacing, x_span / float(count - 1))
  start_x = float(constants.CENTER_X) - 0.5 * effective_spacing * float(count - 1)
  y_center = float(constants.CENTER_Y)
  positions = []
  for i in range(count):
    x = start_x + effective_spacing * float(i)
    base = np.array([x, y_center], dtype=np.float32)
    jitter_delta = (
        rng.uniform(-jitter, jitter, size=(2,)) if jitter > 0 else np.zeros((2,))
    )
    candidate = base + jitter_delta
    positions.append(_clip_to_bounds(candidate, workspace_inset=workspace_inset))
  return positions


def _single_file_positions(
    count: int,
    spacing: float,
    rng: np.random.RandomState,
    jitter: float,
    workspace_inset: Optional[float] = None,
) -> List[np.ndarray]:
  """Returns evenly-spaced targets along a vertical (single-file) center line."""
  if count <= 0:
    return []
  inset = _normalize_workspace_inset(workspace_inset)
  y_min = float(constants.Y_MIN + inset)
  y_max = float(constants.Y_MAX - inset)
  y_span = max(0.0, y_max - y_min)
  spacing = max(0.0, float(spacing))
  if count <= 1:
    effective_spacing = 0.0
  else:
    effective_spacing = min(spacing, y_span / float(count - 1))
  start_y = float(constants.CENTER_Y) - 0.5 * effective_spacing * float(count - 1)
  x_center = float(constants.CENTER_X)
  positions = []
  for i in range(count):
    y = start_y + effective_spacing * float(i)
    base = np.array([x_center, y], dtype=np.float32)
    jitter_delta = (
        rng.uniform(-jitter, jitter, size=(2,)) if jitter > 0 else np.zeros((2,))
    )
    candidate = base + jitter_delta
    positions.append(_clip_to_bounds(candidate, workspace_inset=workspace_inset))
  return positions


def _line_positions(
    count: int,
    spacing: float,
    rng: np.random.RandomState,
    jitter: float,
    *,
    layout_mode: str,
    workspace_inset: Optional[float] = None,
) -> List[np.ndarray]:
  layout_mode = _normalize_line_layout_mode(layout_mode)
  if layout_mode == _LINE_LAYOUT_SINGLE_FILE:
    return _single_file_positions(count, spacing, rng, jitter, workspace_inset=workspace_inset)
  return _horizontal_line_positions(count, spacing, rng, jitter, workspace_inset=workspace_inset)


def _fixed_mixed_block_order(blocks_on_table: Sequence[str]) -> List[str]:
  """Deterministic mixed ordering (round-robin by color)."""
  colors = sorted({_block_color(block) for block in blocks_on_table})
  buckets: Dict[str, List[str]] = {
      color: sorted([block for block in blocks_on_table if _block_color(block) == color])
      for color in colors
  }
  ordered: List[str] = []
  while True:
    appended = False
    for color in colors:
      bucket = buckets.get(color, [])
      if not bucket:
        continue
      ordered.append(bucket.pop(0))
      appended = True
    if not appended:
      break
  return ordered


def _ordered_line_blocks(
    blocks_on_table: Sequence[str],
    *,
    assignment_mode: str,
    rng: np.random.RandomState,
) -> List[str]:
  assignment_mode = _normalize_line_assignment_mode(assignment_mode)
  blocks_sorted = sorted(list(blocks_on_table))
  if assignment_mode == _LINE_ASSIGNMENT_SORTED:
    return blocks_sorted
  if assignment_mode == _LINE_ASSIGNMENT_RANDOM:
    if not blocks_sorted:
      return []
    order = rng.permutation(len(blocks_sorted)).tolist()
    return [blocks_sorted[idx] for idx in order]
  return _fixed_mixed_block_order(blocks_sorted)


def _line_location_name(layout_mode: str) -> str:
  if _normalize_line_layout_mode(layout_mode) == _LINE_LAYOUT_SINGLE_FILE:
    return SINGLE_FILE_LOCATION
  return HORIZONTAL_LINE_LOCATION


def _line_instruction(layout_mode: str) -> str:
  if _normalize_line_layout_mode(layout_mode) == _LINE_LAYOUT_SINGLE_FILE:
    return DEFAULT_SINGLE_FILE_INSTRUCTION
  return DEFAULT_HORIZONTAL_LINE_INSTRUCTION


class ChainedBlock2LocationReward(base_reward.LanguageTableReward):
  """Chained long-horizon tasks using block-to-absolute-location subtasks."""

  def __init__(
      self,
      goal_reward,
      rng,
      delay_reward_steps,
      block_mode,
      task_mode="sort_by_color",
      target_color: Optional[str] = None,
      target_corner="top_left",
      instruction_style="high_level",
      shuffle_tasks=True,
      max_subtasks: Optional[int] = None,
      cluster_spacing=0.03,
      target_jitter=0.0,
      target_distance=0.1,
      cluster_region_padding: Optional[float] = None,
      color_to_corner_map: Optional[Dict[str, str]] = None,
      success_mode: str = _SUCCESS_MODE_POINT,
      region_margin: float = 0.0,
      target_workspace_inset: Optional[float] = None,
      corner_success_radius: Optional[float] = None,
      corner_anchor_offset_x: float = 0.0,
      corner_anchor_offset_y: float = 0.0,
      line_assignment_mode: str = _LINE_ASSIGNMENT_SORTED,
      line_layout_mode: str = _LINE_LAYOUT_HORIZONTAL,
  ):
    super(ChainedBlock2LocationReward, self).__init__(
        goal_reward=goal_reward,
        rng=rng,
        delay_reward_steps=delay_reward_steps,
        block_mode=block_mode,
    )
    self._task_mode = task_mode
    self._target_color = target_color
    self._target_corner = target_corner
    self._instruction_style = instruction_style
    self._shuffle_tasks = shuffle_tasks
    self._max_subtasks = max_subtasks
    self._cluster_spacing = cluster_spacing
    self._target_jitter = target_jitter
    self._target_distance = target_distance
    self._cluster_region_padding = _normalize_cluster_region_padding(
        cluster_region_padding, target_distance
    )
    self._target_workspace_inset = _normalize_workspace_inset(target_workspace_inset)
    self._color_to_corner_map = color_to_corner_map or COLOR_TO_CORNER
    self._success_mode = _normalize_success_mode(success_mode)
    self._region_margin = max(0.0, float(region_margin))
    self._corner_success_radius = _normalize_corner_success_radius(
        corner_success_radius, target_distance
    )
    self._corner_anchor_offset_x = float(corner_anchor_offset_x)
    self._corner_anchor_offset_y = float(corner_anchor_offset_y)
    self._line_assignment_mode = _normalize_line_assignment_mode(line_assignment_mode)
    self._line_layout_mode = _normalize_line_layout_mode(line_layout_mode)
    # Corner-radius success should be anchored to the physical board corners,
    # not the (possibly inset) target placement corners.
    self._corner_centers = _corner_centers(
        workspace_inset=0.0,
        offset_x=self._corner_anchor_offset_x,
        offset_y=self._corner_anchor_offset_y,
    )

    self._instruction = None
    self._task_plan: List[Dict[str, object]] = []
    self._cluster_regions_by_location: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    self._current_task_index = 0

  @property
  def task_mode(self) -> str:
    return self._task_mode

  @property
  def instruction(self) -> str:
    return self._instruction or ""

  @property
  def current_task_index(self) -> int:
    return self._current_task_index

  @property
  def num_tasks(self) -> int:
    return len(self._task_plan)

  def get_task_plan(self) -> List[Dict[str, object]]:
    plan = []
    for task in self._task_plan:
      entry = {
          "block": task["block"],
          "location": task["location"],
          "target_translation": task["target_translation"].tolist(),
          "color": task.get("color"),
      }
      if "shape" in task:
        entry["shape"] = task.get("shape")
      plan.append(entry)
    return plan

  def _build_instruction(self, task: Dict[str, object]) -> str:
    if self._instruction_style == "high_level_plus_subtask":
      block_text = _pretty_block(task["block"])
      # Add punctuation between the high-level instruction and the subtask.
      prefix = (self._instruction or "").rstrip()
      if prefix and prefix[-1] not in ".!?":
        prefix = prefix + "."
      location = task.get("location", "")
      location_text = _location_text(location) if location else "target position"
      subtask = f"Then move the {block_text} to the {location_text}."
      return (prefix + " " + subtask).strip()
    return self._instruction

  def _in_goal_region(self, state, block, target_translation) -> bool:
    current_translation, _ = self._get_pose_for_block(block, state)
    dist = np.linalg.norm(
        np.array(current_translation) - np.array(target_translation))
    return dist < self._target_distance

  def _block_xy(self, state, block: str) -> np.ndarray:
    current_translation, _ = self._get_pose_for_block(block, state)
    return np.asarray(current_translation, dtype=np.float32)

  def _task_satisfied(self, state, task: Dict[str, object]) -> bool:
    # Used for subtask progression. Match success_mode when possible.
    return self._success_satisfied(state, task)

  def _success_satisfied(self, state, task: Dict[str, object]) -> bool:
    location = task.get("location", "")
    block_xy = self._block_xy(state, task["block"])
    if self._success_mode == _SUCCESS_MODE_CORNER_RADIUS and location:
      if str(location) in self._corner_centers:
        return _block_in_corner_radius(
            block_xy,
            str(location),
            self._corner_success_radius,
            corner_centers=self._corner_centers,
        )
      return _block_in_location_region(
          block_xy,
          str(location),
          self._region_margin,
      )
    if self._success_mode == _SUCCESS_MODE_REGION and location:
      return _block_in_location_region(
          block_xy,
          location,
          self._region_margin,
      )
    if self._success_mode == _SUCCESS_MODE_CLUSTER_REGION and location:
      region = self._cluster_regions_by_location.get(str(location))
      if region is None:
        return False
      return _block_in_cluster_region(block_xy, region)
    return self._in_goal_region(state, task["block"], task["target_translation"])

  def _filter_satisfied_tasks(self, state):
    remaining = []
    for task in self._task_plan:
      if not self._task_satisfied(state, task):
        remaining.append(task)
    self._task_plan = remaining

  def _build_sort_by_color_plan(self, blocks_on_table):
    self._instruction = DEFAULT_INSTRUCTION
    centers = _location_centers(workspace_inset=self._target_workspace_inset)
    tasks = []
    blocks_by_color = {}
    for block in blocks_on_table:
      color = _block_color(block)
      blocks_by_color.setdefault(color, []).append(block)

    for color, blocks in blocks_by_color.items():
      location = self._color_to_corner_map.get(color)
      if location is None:
        continue
      center = centers[location]
      positions = _cluster_positions(
          center,
          len(blocks),
          self._cluster_spacing,
          self._rng,
          self._target_jitter,
          workspace_inset=self._target_workspace_inset,
      )
      for block, pos in zip(blocks, positions):
        tasks.append({
            "block": block,
            "location": location,
            "target_translation": pos,
            "color": color,
        })
    return tasks

  def _build_sort_by_shape_edge_centers_plan(self, blocks_on_table):
    self._instruction = DEFAULT_SHAPE_EDGE_INSTRUCTION
    centers = _location_centers(workspace_inset=self._target_workspace_inset)
    tasks = []
    blocks_by_shape = {}
    for block in blocks_on_table:
      shape = _block_shape(block)
      blocks_by_shape.setdefault(shape, []).append(block)

    for shape, shape_blocks in sorted(blocks_by_shape.items()):
      location = SHAPE_TO_EDGE_CENTER.get(shape)
      if location is None or location not in centers:
        continue
      shape_blocks = sorted(shape_blocks)
      center = centers[location]
      positions = _cluster_positions(
          center,
          len(shape_blocks),
          self._cluster_spacing,
          self._rng,
          self._target_jitter,
          workspace_inset=self._target_workspace_inset,
      )
      for block, pos in zip(shape_blocks, positions):
        tasks.append({
            "block": block,
            "location": location,
            "target_translation": pos,
            "color": _block_color(block),
            "shape": shape,
        })
    return tasks

  def _build_move_color_plan(self, blocks_on_table):
    centers = _location_centers(workspace_inset=self._target_workspace_inset)
    available_colors = sorted({_block_color(b) for b in blocks_on_table})
    if not available_colors:
      return []
    color = self._target_color or self._rng.choice(available_colors)
    location = self._target_corner
    if location not in centers:
      raise ValueError(f"Unknown location: {location}")

    self._instruction = (
        f"move all {color} blocks to the {_location_text(location)}"
    )

    blocks = [b for b in blocks_on_table if _block_color(b) == color]
    center = centers[location]
    positions = _cluster_positions(
        center,
        len(blocks),
        self._cluster_spacing,
        self._rng,
        self._target_jitter,
        workspace_inset=self._target_workspace_inset,
    )
    tasks = []
    for block, pos in zip(blocks, positions):
      tasks.append({
          "block": block,
          "location": location,
          "target_translation": pos,
          "color": color,
      })
    return tasks

  def _build_horizontal_line_plan(self, blocks_on_table):
    layout_mode = (
        _LINE_LAYOUT_SINGLE_FILE
        if self._task_mode == "arrange_single_file"
        else self._line_layout_mode
    )
    self._instruction = _line_instruction(layout_mode)
    blocks = _ordered_line_blocks(
        blocks_on_table,
        assignment_mode=self._line_assignment_mode,
        rng=self._rng,
    )
    positions = _line_positions(
        len(blocks),
        self._cluster_spacing,
        self._rng,
        self._target_jitter,
        layout_mode=layout_mode,
        workspace_inset=self._target_workspace_inset,
    )
    location = _line_location_name(layout_mode)
    tasks = []
    for slot, (block, pos) in enumerate(zip(blocks, positions)):
      tasks.append({
          "block": block,
          "location": location,
          "target_translation": pos,
          "color": _block_color(block),
          "line_slot": slot,
      })
    return tasks

  def _build_task_plan(self, blocks_on_table):
    if self._task_mode == "sort_by_color":
      return self._build_sort_by_color_plan(blocks_on_table)
    if self._task_mode == "sort_by_shape_edge_centers":
      return self._build_sort_by_shape_edge_centers_plan(blocks_on_table)
    if self._task_mode == "move_color_to_corner":
      return self._build_move_color_plan(blocks_on_table)
    if self._task_mode in ("arrange_horizontal_line", "arrange_single_file"):
      return self._build_horizontal_line_plan(blocks_on_table)
    raise ValueError(f"Unknown task_mode: {self._task_mode}")

  def reset(self, state, blocks_on_table):
    self._task_plan = self._build_task_plan(blocks_on_table)
    if not self._task_plan:
      return task_info.FAILURE

    if self._shuffle_tasks:
      self._rng.shuffle(self._task_plan)

    if self._max_subtasks is not None:
      self._task_plan = self._task_plan[:self._max_subtasks]

    self._filter_satisfied_tasks(state)
    if not self._task_plan:
      return task_info.FAILURE

    self._cluster_regions_by_location = _cluster_regions_from_tasks(
        self._task_plan, self._cluster_region_padding
    )

    self._current_task_index = 0
    self._in_reward_zone_steps = 0
    return self.get_current_task_info(state)

  def get_current_task_info(self, state):
    if not self._task_plan:
      raise ValueError("Task plan is empty; call reset first.")

    # Advance through any already-satisfied subtasks so the env/oracle always
    # sees the first *unsatisfied* subtask in the plan.
    current_task = self._task_plan[self._current_task_index]
    while (
        self._current_task_index < len(self._task_plan) - 1
        and self._task_satisfied(state, current_task)
    ):
      self._current_task_index += 1
      self._in_reward_zone_steps = 0
      current_task = self._task_plan[self._current_task_index]

    return task_info.Block2LocationTaskInfo(
        instruction=self._build_instruction(current_task),
        block=current_task["block"],
        location=current_task["location"],
        target_translation=current_task["target_translation"],
    )

  def reward(self, state):
    if not self._task_plan:
      return 0.0, False

    current_task = self._task_plan[self._current_task_index]
    is_last = self._current_task_index == len(self._task_plan) - 1
    if self._success_mode in (_SUCCESS_MODE_REGION, _SUCCESS_MODE_CORNER_RADIUS):
      all_satisfied = all(self._success_satisfied(state, task) for task in self._task_plan)
      if all_satisfied:
        if self._in_reward_zone_steps >= self._delay_reward_steps:
          return self._goal_reward, True
        self._in_reward_zone_steps += 1
      else:
        self._in_reward_zone_steps = 0
      return 0.0, False
    if not is_last:
      return 0.0, False

    if self._in_goal_region(state, current_task["block"], current_task["target_translation"]):
      if self._in_reward_zone_steps >= self._delay_reward_steps:
        return self._goal_reward, True
      self._in_reward_zone_steps += 1
    return 0.0, False

  def is_success(self, state) -> bool:
    if not self._task_plan:
      return False
    for task in self._task_plan:
      if not self._success_satisfied(state, task):
        return False
    return True

  def progress_counts(self, state) -> Tuple[int, int]:
    """Return (# blocks satisfying success criterion, # total target blocks)."""
    if not self._task_plan:
      return 0, 0
    total = len(self._task_plan)
    satisfied = 0
    for task in self._task_plan:
      if self._success_satisfied(state, task):
        satisfied += 1
    return satisfied, total

  def progress(self, state) -> float:
    """Fraction of target blocks currently satisfying the success criterion."""
    satisfied, total = self.progress_counts(state)
    if total <= 0:
      return 0.0
    return float(satisfied) / float(total)


def _center_xy() -> np.ndarray:
  return np.array([constants.CENTER_X, constants.CENTER_Y], dtype=np.float32)


class StrategicChainedBlock2LocationReward(base_reward.LanguageTableReward):
  """Strategic long-horizon planner for true color sorting.

  Compared to :class:`ChainedBlock2LocationReward`, this planner:

  * emits explicit subtask labels (`place`, `clear`, `recover`)
  * schedules `clear` subtasks when a target slot is occupied by a wrong-color
    block
  * orders placement subtasks by distance-to-goal (farthest-first)
  * periodically checks whether already-placed blocks got disturbed and, if so,
    inserts `recover` subtasks.

  The returned TaskInfo is a :class:`LongHorizonSubtaskInfo`, which is a
  :class:`task_info.Block2LocationTaskInfo` augmented with `subtask_type` and
  `subtask_id`.
  """

  def __init__(
      self,
      goal_reward,
      rng,
      delay_reward_steps,
      block_mode,
      task_mode="sort_by_color",
      target_color: Optional[str] = None,
      target_corner="top_left",
      instruction_style="high_level",
      shuffle_tasks=False,
      max_subtasks: Optional[int] = None,
      cluster_spacing=0.03,
      target_jitter=0.0,
      target_distance=0.1,
      cluster_region_padding: Optional[float] = None,
      occupied_radius: float = 0.06,
      keepout_radius: float = 0.12,
      staging_spacing: float = 0.08,
      replanning_interval_steps: int = 5,
      max_dynamic_inserts_per_call: int = 2,
      enable_clearing: bool = False,
      clear_target_mode: str = _CLEAR_TARGET_STAGING,
      clear_require_region: bool = False,
      color_to_corner_map: Optional[Dict[str, str]] = None,
      success_mode: str = _SUCCESS_MODE_POINT,
      region_margin: float = 0.0,
      goal_visualization: str = _GOAL_VIS_POINT,
      target_workspace_inset: Optional[float] = None,
      corner_success_radius: Optional[float] = None,
      corner_anchor_offset_x: float = 0.0,
      corner_anchor_offset_y: float = 0.0,
      second_same_color_into_block: bool = False,
      into_block_distance: float = reward_constants.TARGET_BLOCK_DISTANCE,
      subtask_completion_distance: float = -1.0,
      line_assignment_mode: str = _LINE_ASSIGNMENT_SORTED,
      line_layout_mode: str = _LINE_LAYOUT_HORIZONTAL,
  ):
    super(StrategicChainedBlock2LocationReward, self).__init__(
        goal_reward=goal_reward,
        rng=rng,
        delay_reward_steps=delay_reward_steps,
        block_mode=block_mode,
    )
    self._task_mode = task_mode
    self._target_color = target_color
    self._target_corner = target_corner
    self._instruction_style = instruction_style
    self._shuffle_tasks = shuffle_tasks
    self._max_subtasks = max_subtasks
    self._cluster_spacing = cluster_spacing
    self._target_jitter = target_jitter
    self._target_distance = target_distance
    self._cluster_region_padding = _normalize_cluster_region_padding(
        cluster_region_padding, target_distance
    )
    self._target_workspace_inset = _normalize_workspace_inset(target_workspace_inset)
    self._color_to_corner_map = color_to_corner_map or COLOR_TO_CORNER
    self._success_mode = _normalize_success_mode(success_mode)
    self._region_margin = max(0.0, float(region_margin))
    self._corner_success_radius = _normalize_corner_success_radius(
        corner_success_radius, target_distance
    )
    self._corner_anchor_offset_x = float(corner_anchor_offset_x)
    self._corner_anchor_offset_y = float(corner_anchor_offset_y)
    self._second_same_color_into_block = bool(second_same_color_into_block)
    self._into_block_distance = float(into_block_distance)
    self._subtask_completion_distance = float(subtask_completion_distance)
    self._line_assignment_mode = _normalize_line_assignment_mode(line_assignment_mode)
    self._line_layout_mode = _normalize_line_layout_mode(line_layout_mode)
    # Corner-radius success should be anchored to the physical board corners,
    # not the (possibly inset) target placement corners.
    self._corner_centers = _corner_centers(
        workspace_inset=0.0,
        offset_x=self._corner_anchor_offset_x,
        offset_y=self._corner_anchor_offset_y,
    )
    self._goal_visualization = _normalize_goal_vis(goal_visualization)

    # Planner-specific radii.
    self._occupied_radius = float(occupied_radius)
    self._keepout_radius = float(keepout_radius)
    self._staging_spacing = float(staging_spacing)
    self._enable_clearing = bool(enable_clearing)
    self._clear_target_mode = _normalize_clear_target_mode(clear_target_mode)
    self._clear_require_region = bool(clear_require_region)

    # Replanning knobs.
    self._replanning_interval_steps = int(replanning_interval_steps)
    self._max_dynamic_inserts_per_call = int(max_dynamic_inserts_per_call)

    # Episode state.
    self._instruction: Optional[str] = None
    self._blocks_on_table: List[str] = []

    # Task plan is a list of dicts. We mutate it by inserting tasks, but we do
    # not remove tasks; tasks are "completed" by advancing _current_task_index.
    self._task_plan: List[Dict[str, Any]] = []
    self._current_task_index = 0

    # Mapping for canonical placement targets.
    self._place_targets_by_block: Dict[str, np.ndarray] = {}
    self._place_location_by_block: Dict[str, str] = {}
    self._place_color_by_block: Dict[str, str] = {}
    self._cluster_regions_by_location: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # Staging targets used for `clear` tasks.
    self._staging_targets: List[np.ndarray] = []

    # Bookkeeping for replanning.
    self._completed_place_blocks: set[str] = set()
    self._step_counter = 0
    self._last_replan_step = -10**9

    # Current subtask metadata (for data generation).
    self._subtask_id_counter = 0
    self._current_subtask_id: int = -1
    self._current_subtask_type: str = ""
    self._current_subtask_block: str = ""

  # ---------------------------------------------------------------------------
  # Public properties / helpers
  # ---------------------------------------------------------------------------

  @property
  def task_mode(self) -> str:
    return self._task_mode

  @property
  def instruction(self) -> str:
    return self._instruction or ""

  @property
  def current_task_index(self) -> int:
    return self._current_task_index

  @property
  def num_tasks(self) -> int:
    return len(self._task_plan)

  @property
  def current_subtask_id(self) -> int:
    return self._current_subtask_id

  @property
  def current_subtask_type(self) -> str:
    return self._current_subtask_type

  @property
  def current_subtask_block(self) -> str:
    return self._current_subtask_block

  def get_task_plan(self) -> List[Dict[str, object]]:
    plan: List[Dict[str, object]] = []
    for task in self._task_plan:
      entry: Dict[str, object] = {
          "subtask_id": int(task.get("subtask_id", -1)),
          "subtask_type": task.get("subtask_type", ""),
          "block": task["block"],
          "location": task.get("location", ""),
          "target_translation": task["target_translation"].tolist(),
          "color": task.get("color"),
      }
      if "shape" in task:
        entry["shape"] = task.get("shape")
      # Optional debug/context fields.
      if "blocked_location" in task:
        entry["blocked_location"] = task["blocked_location"]
      if "blocked_translation" in task:
        entry["blocked_translation"] = np.asarray(task["blocked_translation"]).tolist()
      if "into_block" in task:
        entry["into_block"] = str(task["into_block"])
      if "into_block_distance" in task:
        entry["into_block_distance"] = float(task["into_block_distance"])
      plan.append(entry)
    return plan

  # ---------------------------------------------------------------------------
  # Planning utilities
  # ---------------------------------------------------------------------------

  def _next_subtask_id(self) -> int:
    subtask_id = self._subtask_id_counter
    self._subtask_id_counter += 1
    return subtask_id

  def _make_task(
      self,
      *,
      subtask_type: str,
      block: str,
      location: str,
      target_translation: np.ndarray,
      color: Optional[str] = None,
      **kwargs,
  ) -> Dict[str, Any]:
    task: Dict[str, Any] = {
        "subtask_id": self._next_subtask_id(),
        "subtask_type": subtask_type,
        "block": block,
        "location": location,
        "target_translation": np.asarray(target_translation, dtype=np.float32),
    }
    if color is not None:
      task["color"] = color
    task.update(kwargs)
    return task

  def _block_xy(self, state, block: str) -> np.ndarray:
    xy, _ = self._get_pose_for_block(block, state)
    return np.asarray(xy, dtype=np.float32)

  def _dist(self, state, block: str, target_xy: np.ndarray) -> float:
    return float(np.linalg.norm(self._block_xy(state, block) - np.asarray(target_xy)))

  def _block_dist(self, state, block_a: str, block_b: str) -> float:
    return float(np.linalg.norm(self._block_xy(state, block_a) - self._block_xy(state, block_b)))

  def _in_goal_region(self, state, block: str, target_translation: np.ndarray) -> bool:
    return self._dist(state, block, target_translation) < self._target_distance

  def _success_for_block(
      self,
      state,
      block: str,
      target_translation: np.ndarray,
      location: str,
  ) -> bool:
    block_xy = self._block_xy(state, block)
    if self._success_mode == _SUCCESS_MODE_CORNER_RADIUS and location:
      if str(location) in self._corner_centers:
        return _block_in_corner_radius(
            block_xy,
            str(location),
            self._corner_success_radius,
            corner_centers=self._corner_centers,
        )
      return _block_in_location_region(
          block_xy,
          str(location),
          self._region_margin,
      )
    if self._success_mode == _SUCCESS_MODE_REGION and location:
      return _block_in_location_region(
          block_xy,
          location,
          self._region_margin,
      )
    if self._success_mode == _SUCCESS_MODE_CLUSTER_REGION and location:
      region = self._cluster_regions_by_location.get(str(location))
      if region is None:
        return False
      return _block_in_cluster_region(block_xy, region)
    return self._in_goal_region(state, block, target_translation)

  def _clear_satisfied(self, state, task: Dict[str, Any]) -> bool:
    # A clear task is satisfied when the blocker is no longer within
    # occupied_radius of the blocked slot, OR when it reaches its staging
    # target. This makes clearing more robust than requiring exact staging.
    block = task["block"]
    if self._clear_require_region:
      clear_region = task.get("clear_region", "")
      if clear_region:
        return _block_in_location_region(
            self._block_xy(state, block),
            str(clear_region),
            self._region_margin,
        )
    if "blocked_translation" in task:
      blocked_xy = np.asarray(task["blocked_translation"], dtype=np.float32)
      if self._dist(state, block, blocked_xy) >= self._occupied_radius:
        return True
    return self._in_goal_region(state, block, task["target_translation"])

  def _task_satisfied(self, state, task: Dict[str, Any]) -> bool:
    ttype = task.get("subtask_type", "")
    if ttype == "clear":
      return self._clear_satisfied(state, task)
    into_block = str(task.get("into_block", "") or "")
    if into_block:
      into_block_distance = float(task.get("into_block_distance", self._into_block_distance))
      return self._block_dist(state, task["block"], into_block) <= into_block_distance
    # Optional strict progression criterion decoupled from eval success.
    # This prevents corner-radius slack from advancing subtasks too early.
    if self._subtask_completion_distance > 0:
      return self._dist(state, task["block"], task["target_translation"]) <= self._subtask_completion_distance
    # place / recover
    return self._success_for_block(
        state,
        task["block"],
        task["target_translation"],
        task.get("location", ""),
    )

  def _build_instruction(self, task: Dict[str, Any]) -> str:
    if self._instruction_style != "high_level_plus_subtask":
      return self._instruction or ""

    prefix = (self._instruction or "").rstrip()
    if prefix and prefix[-1] not in ".!?":
      prefix = prefix + "."

    block_text = _pretty_block(task["block"])
    ttype = task.get("subtask_type", "")
    location = task.get("location", "")
    location_text = _location_text(location) if location else ""

    if ttype == "clear":
      blocked_loc = task.get("blocked_location", "")
      blocked_text = _location_text(blocked_loc) if blocked_loc else ""
      if blocked_loc:
        subtask = (
            f"First clear the {block_text} away from the {blocked_text}."
        )
      else:
        subtask = f"First move the {block_text} to the staging area."
    elif ttype == "recover":
      if location:
        subtask = f"Re-place the {block_text} to the {location_text}."
      else:
        subtask = f"Re-place the {block_text} to its target position."
    else:  # place
      if location:
        subtask = f"Then move the {block_text} to the {location_text}."
      else:
        subtask = f"Then move the {block_text} to its target position."

    return (prefix + " " + subtask).strip() if prefix else subtask

  def _build_place_tasks(self, blocks_on_table: Sequence[str]) -> List[Dict[str, Any]]:
    if self._task_mode == "sort_by_color":
      self._instruction = DEFAULT_INSTRUCTION
      centers = _location_centers(workspace_inset=self._target_workspace_inset)
      blocks_by_color: Dict[str, List[str]] = {}
      for block in blocks_on_table:
        blocks_by_color.setdefault(_block_color(block), []).append(block)

      tasks: List[Dict[str, Any]] = []
      for color, blocks_for_color in blocks_by_color.items():
        location = self._color_to_corner_map.get(color)
        if location is None:
          continue
        # Deterministic ordering for reproducibility.
        blocks_for_color = sorted(blocks_for_color)
        positions = _cluster_positions(
            centers[location],
            len(blocks_for_color),
            self._cluster_spacing,
            self._rng,
            self._target_jitter,
            workspace_inset=self._target_workspace_inset,
        )
        if not blocks_for_color:
          continue
        anchor_block = blocks_for_color[0]
        anchor_pos = positions[0]
        tasks.append(
            self._make_task(
                subtask_type="place",
                block=anchor_block,
                location=location,
                target_translation=anchor_pos,
                color=color,
            )
        )
        for block, pos in zip(blocks_for_color[1:], positions[1:]):
          if self._second_same_color_into_block:
            tasks.append(
                self._make_task(
                    subtask_type="place",
                    block=block,
                    location=location,
                    target_translation=anchor_pos,
                    color=color,
                    into_block=anchor_block,
                    into_block_distance=self._into_block_distance,
                )
            )
          else:
            tasks.append(
                self._make_task(
                    subtask_type="place",
                    block=block,
                    location=location,
                    target_translation=pos,
                    color=color,
                )
            )
      return tasks

    if self._task_mode == "sort_by_shape_edge_centers":
      self._instruction = DEFAULT_SHAPE_EDGE_INSTRUCTION
      centers = _location_centers(workspace_inset=self._target_workspace_inset)
      blocks_by_shape: Dict[str, List[str]] = {}
      for block in blocks_on_table:
        blocks_by_shape.setdefault(_block_shape(block), []).append(block)

      tasks: List[Dict[str, Any]] = []
      for shape, blocks_for_shape in sorted(blocks_by_shape.items()):
        location = SHAPE_TO_EDGE_CENTER.get(shape)
        if location is None or location not in centers:
          continue
        blocks_for_shape = sorted(blocks_for_shape)
        positions = _cluster_positions(
            centers[location],
            len(blocks_for_shape),
            self._cluster_spacing,
            self._rng,
            self._target_jitter,
            workspace_inset=self._target_workspace_inset,
        )
        if not blocks_for_shape:
          continue
        anchor_block = blocks_for_shape[0]
        anchor_pos = positions[0]
        tasks.append(
            self._make_task(
                subtask_type="place",
                block=anchor_block,
                location=location,
                target_translation=anchor_pos,
                color=_block_color(anchor_block),
                shape=shape,
            )
        )
        for block, pos in zip(blocks_for_shape[1:], positions[1:]):
          if self._second_same_color_into_block:
            tasks.append(
                self._make_task(
                    subtask_type="place",
                    block=block,
                    location=location,
                    target_translation=anchor_pos,
                    color=_block_color(block),
                    shape=shape,
                    into_block=anchor_block,
                    into_block_distance=self._into_block_distance,
                )
            )
          else:
            tasks.append(
                self._make_task(
                    subtask_type="place",
                    block=block,
                    location=location,
                    target_translation=pos,
                    color=_block_color(block),
                    shape=shape,
                )
            )
      return tasks

    if self._task_mode == "move_color_to_corner":
      centers = _corner_centers(workspace_inset=self._target_workspace_inset)
      available_colors = sorted({_block_color(b) for b in blocks_on_table})
      if not available_colors:
        return []
      color = self._target_color or self._rng.choice(available_colors)
      corner = self._target_corner
      if corner not in centers:
        raise ValueError(f"Unknown corner: {corner}")

      self._instruction = (
          f"move all {color} blocks to the {corner.replace('_', ' ')} corner of the table"
      )
      blocks_for_color = sorted([b for b in blocks_on_table if _block_color(b) == color])
      positions = _cluster_positions(
          centers[corner],
          len(blocks_for_color),
          self._cluster_spacing,
          self._rng,
          self._target_jitter,
          workspace_inset=self._target_workspace_inset,
      )
      return [
          self._make_task(
              subtask_type="place",
              block=block,
              location=corner,
              target_translation=pos,
              color=color,
          )
          for block, pos in zip(blocks_for_color, positions)
      ]

    if self._task_mode in ("arrange_horizontal_line", "arrange_single_file"):
      layout_mode = (
          _LINE_LAYOUT_SINGLE_FILE
          if self._task_mode == "arrange_single_file"
          else self._line_layout_mode
      )
      self._instruction = _line_instruction(layout_mode)
      blocks = _ordered_line_blocks(
          blocks_on_table,
          assignment_mode=self._line_assignment_mode,
          rng=self._rng,
      )
      positions = _line_positions(
          len(blocks),
          self._cluster_spacing,
          self._rng,
          self._target_jitter,
          layout_mode=layout_mode,
          workspace_inset=self._target_workspace_inset,
      )
      location = _line_location_name(layout_mode)
      return [
          self._make_task(
              subtask_type="place",
              block=block,
              location=location,
              target_translation=pos,
              color=_block_color(block),
              line_slot=slot,
          )
          for slot, (block, pos) in enumerate(zip(blocks, positions))
      ]

    raise ValueError(f"Unknown task_mode: {self._task_mode}")

  def _build_staging_targets(self, blocks_on_table: Sequence[str]) -> List[np.ndarray]:
    # Provide enough staging slots for the number of blocks (deterministic).
    return _cluster_positions(
        _center_xy(),
        len(blocks_on_table),
        self._staging_spacing,
        self._rng,
        jitter=0.0,
        workspace_inset=self._target_workspace_inset,
    )

  def _staging_target_for_block(self, block: str) -> np.ndarray:
    if not self._staging_targets:
      return _center_xy()
    # Stable mapping: sorted index in blocks_on_table.
    if block not in self._blocks_on_table:
      return self._staging_targets[0]
    idx = sorted(self._blocks_on_table).index(block)
    return self._staging_targets[idx % len(self._staging_targets)]

  def _clear_target_for_block(self, block: str) -> np.ndarray:
    if self._clear_target_mode == _CLEAR_TARGET_OWN_GOAL:
      target = self._place_targets_by_block.get(block)
      if target is not None:
        return np.asarray(target, dtype=np.float32)
    return self._staging_target_for_block(block)

  def _clear_region_for_block(self, block: str) -> str:
    return str(self._place_location_by_block.get(block, ""))

  def _find_wrong_block_near_target(
      self,
      state,
      *,
      target_xy: np.ndarray,
      desired_color: str,
      ignore_block: Optional[str] = None,
  ) -> Optional[str]:
    """Return the closest wrong-color block within occupied_radius of target."""
    best_block = None
    best_dist = float("inf")
    for block in self._blocks_on_table:
      if ignore_block is not None and block == ignore_block:
        continue
      if _block_color(block) == desired_color:
        continue
      d = self._dist(state, block, target_xy)
      if d < self._occupied_radius and d < best_dist:
        best_dist = d
        best_block = block
    return best_block

  def _insert_task_at_current(self, task: Dict[str, Any]):
    self._task_plan.insert(self._current_task_index, task)

  def _has_pending_task_for_block(self, block: str) -> bool:
    for task in self._task_plan[self._current_task_index :]:
      if task.get("block") == block and task.get("subtask_type") in ("place", "recover"):
        return True
    return False

  def _advance_past_satisfied(self, state) -> bool:
    """Advance current index while tasks are satisfied.

    Returns True if we advanced at least once.
    """
    advanced = False
    # Match the baseline behavior: never advance past the last task to avoid
    # empty plans / out-of-range indices during env.compute_state().
    while (
        self._task_plan
        and self._current_task_index < len(self._task_plan) - 1
        and self._task_satisfied(state, self._task_plan[self._current_task_index])
    ):
      done_task = self._task_plan[self._current_task_index]
      if done_task.get("subtask_type") in ("place", "recover"):
        self._completed_place_blocks.add(done_task["block"])
      self._current_task_index += 1
      self._in_reward_zone_steps = 0
      advanced = True
    return advanced

  def _maybe_insert_clear_for_current(self, state) -> bool:
    if not self._enable_clearing:
      return False
    if not self._task_plan:
      return False
    task = self._task_plan[self._current_task_index]
    if task.get("subtask_type") not in ("place", "recover"):
      return False

    desired_color = task.get("color") or _block_color(task["block"])
    target_xy = np.asarray(task["target_translation"], dtype=np.float32)
    blocker = self._find_wrong_block_near_target(
        state,
        target_xy=target_xy,
        desired_color=str(desired_color),
        ignore_block=task["block"],
    )
    if blocker is None:
      return False

    clear_task = self._make_task(
        subtask_type="clear",
        block=blocker,
        location="staging",
        target_translation=self._clear_target_for_block(blocker),
        color=_block_color(blocker),
        blocked_location=task.get("location"),
        blocked_translation=target_xy,
        clear_region=self._clear_region_for_block(blocker),
    )
    self._insert_task_at_current(clear_task)
    return True

  def _maybe_insert_recover(self, state) -> bool:
    # Limit how often we do this scan to avoid thrash.
    if self._step_counter - self._last_replan_step < self._replanning_interval_steps:
      return False

    candidates: List[Tuple[str, float]] = []
    for block in sorted(self._completed_place_blocks):
      if self._has_pending_task_for_block(block):
        continue
      target = self._place_targets_by_block.get(block)
      if target is None:
        continue
      d = self._dist(state, block, target)
      if d > self._keepout_radius:
        candidates.append((block, d))

    if not candidates:
      return False

    # Recover the most displaced block first.
    block, _ = max(candidates, key=lambda x: x[1])
    recover_task = self._make_task(
        subtask_type="recover",
        block=block,
        location=self._place_location_by_block.get(block, ""),
        target_translation=self._place_targets_by_block[block],
        color=self._place_color_by_block.get(block, _block_color(block)),
    )
    self._insert_task_at_current(recover_task)
    self._last_replan_step = self._step_counter
    return True

  # ---------------------------------------------------------------------------
  # Gym / env interface
  # ---------------------------------------------------------------------------

  def reset(self, state, blocks_on_table):
    self._blocks_on_table = list(blocks_on_table)
    self._task_plan = []
    self._current_task_index = 0
    self._completed_place_blocks = set()
    self._step_counter = 0
    self._last_replan_step = -10**9
    self._subtask_id_counter = 0

    place_tasks = self._build_place_tasks(blocks_on_table)
    if not place_tasks:
      return task_info.FAILURE

    # Record canonical placement targets for success + recover logic.
    self._place_targets_by_block = {
        t["block"]: np.asarray(t["target_translation"], dtype=np.float32) for t in place_tasks
    }
    self._place_location_by_block = {t["block"]: t.get("location", "") for t in place_tasks}
    self._place_color_by_block = {t["block"]: t.get("color", _block_color(t["block"])) for t in place_tasks}
    self._cluster_regions_by_location = _cluster_regions_from_block_targets(
        self._place_targets_by_block,
        self._place_location_by_block,
        self._cluster_region_padding,
    )

    # Farthest-first ordering for safer placement.
    place_tasks = sorted(
        place_tasks,
        key=lambda t: self._dist(state, t["block"], t["target_translation"]),
        reverse=True,
    )
    if self._shuffle_tasks:
      # If enabled, shuffle *within* distance buckets by small random jitter.
      # This keeps the overall farthest-first bias.
      jitter = self._rng.uniform(0.0, 1e-3, size=(len(place_tasks),))
      place_tasks = [
          t
          for _, t in sorted(
              zip(
                  [
                      self._dist(state, t["block"], t["target_translation"]) + j
                      for t, j in zip(place_tasks, jitter)
                  ],
                  place_tasks,
              ),
              key=lambda x: x[0],
              reverse=True,
          )
      ]

    if self._max_subtasks is not None:
      place_tasks = place_tasks[: self._max_subtasks]

    # Precompute staging targets.
    self._staging_targets = self._build_staging_targets(blocks_on_table)

    # Clear phase: schedule clears for any initial wrong-color blockers.
    clear_tasks: List[Dict[str, Any]] = []
    seen_blockers: set[str] = set()
    if self._enable_clearing:
      for t in place_tasks:
        desired_color = t.get("color") or _block_color(t["block"])
        blocker = self._find_wrong_block_near_target(
            state,
            target_xy=np.asarray(t["target_translation"], dtype=np.float32),
            desired_color=str(desired_color),
            ignore_block=t["block"],
        )
        if blocker is None or blocker in seen_blockers:
          continue
        seen_blockers.add(blocker)
        clear_tasks.append(
            self._make_task(
                subtask_type="clear",
                block=blocker,
                location="staging",
                target_translation=self._clear_target_for_block(blocker),
                color=_block_color(blocker),
                blocked_location=t.get("location"),
                blocked_translation=np.asarray(t["target_translation"], dtype=np.float32),
                clear_region=self._clear_region_for_block(blocker),
            )
        )

    self._task_plan = clear_tasks + place_tasks
    if not self._task_plan:
      return task_info.FAILURE

    # Skip over any already-satisfied tasks at reset (and mark placed blocks).
    # We do one pass of advancement; additional advancement happens in
    # get_current_task_info.
    self._advance_past_satisfied(state)
    self._in_reward_zone_steps = 0
    return self.get_current_task_info(state)

  def get_goal_region(self):
    # For debugging visuals: show the current task target.
    if self._goal_visualization == _GOAL_VIS_NONE:
      return None, None
    if not self._task_plan:
      return None, None
    if self._success_mode == _SUCCESS_MODE_CORNER_RADIUS:
      location = str(self._task_plan[self._current_task_index].get("location", ""))
      center = self._corner_centers.get(location)
      if center is not None:
        return center, float(self._corner_success_radius)
    target = self._task_plan[self._current_task_index]["target_translation"]
    return np.asarray(target, dtype=np.float32), float(self._target_distance)

  def get_current_task_info(self, state):
    if not self._task_plan:
      raise ValueError("Task plan is empty; call reset first.")

    self._step_counter += 1

    # Iterate a small number of times in case inserting a task yields another
    # immediately-satisfied task.
    inserts_remaining = self._max_dynamic_inserts_per_call
    for _ in range(10):
      self._advance_past_satisfied(state)

      inserted = False
      if inserts_remaining > 0 and self._maybe_insert_clear_for_current(state):
        inserted = True
      elif inserts_remaining > 0 and self._maybe_insert_recover(state):
        inserted = True

      if inserted:
        inserts_remaining -= 1
        continue
      break

    current_task = self._task_plan[self._current_task_index]
    self._current_subtask_id = int(current_task.get("subtask_id", -1))
    self._current_subtask_type = str(current_task.get("subtask_type", ""))
    self._current_subtask_block = str(current_task.get("block", ""))

    target_translation = np.asarray(current_task["target_translation"], dtype=np.float32)
    into_block = str(current_task.get("into_block", "") or "")
    if into_block:
      target_translation = self._block_xy(state, into_block)

    return LongHorizonSubtaskInfo(
        instruction=self._build_instruction(current_task),
        block=current_task["block"],
        location=current_task.get("location", ""),
        target_translation=target_translation,
        subtask_type=self._current_subtask_type,
        subtask_id=self._current_subtask_id,
    )

  def reward(self, state):
    if not self._place_targets_by_block:
      return 0.0, False

    all_placed = True
    for block, target in self._place_targets_by_block.items():
      location = self._place_location_by_block.get(block, "")
      if not self._success_for_block(state, block, target, location):
        all_placed = False
        break

    if all_placed:
      if self._in_reward_zone_steps >= self._delay_reward_steps:
        return self._goal_reward, True
      self._in_reward_zone_steps += 1
    else:
      self._in_reward_zone_steps = 0
    return 0.0, False

  def is_success(self, state) -> bool:
    if not self._place_targets_by_block:
      return False
    for block, target in self._place_targets_by_block.items():
      location = self._place_location_by_block.get(block, "")
      if not self._success_for_block(state, block, target, location):
        return False
    return True

  def progress_counts(self, state) -> Tuple[int, int]:
    """Return (# blocks satisfying success criterion, # total target blocks)."""
    if not self._place_targets_by_block:
      return 0, 0
    total = len(self._place_targets_by_block)
    satisfied = 0
    for block, target in self._place_targets_by_block.items():
      location = self._place_location_by_block.get(block, "")
      if self._success_for_block(state, block, target, location):
        satisfied += 1
    return satisfied, total

  def progress(self, state) -> float:
    """Fraction of target blocks currently satisfying the success criterion."""
    satisfied, total = self.progress_counts(state)
    if total <= 0:
      return 0.0
    return float(satisfied) / float(total)
