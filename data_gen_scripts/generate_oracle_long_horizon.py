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

"""Generate long-horizon oracle rollouts (npz) for Language Table."""

import json
import os
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

from absl import app
from absl import flags
from absl import logging
import numpy as np

from language_table.environments import blocks
from language_table.environments import language_table
from language_table.environments.oracles import push_oracle_rrt_slowdown
from language_table.environments.rewards import long_horizon
from language_table.environments.rewards import constants as reward_constants
from language_table.environments import constants
from language_table.environments.utils.utils_pybullet import add_visual_sphere
from tf_agents.environments import gym_wrapper
from tf_agents.trajectories import time_step as ts


SHAPE_NAME_MAP = {
    "moon": "crescent",
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


def _legacy_label_from_task(task: dict) -> str:
  block_name = str(task.get("block", ""))
  color = task.get("color") or _block_color(block_name)
  shape = _format_shape(_block_shape(block_name))
  block_desc = f"{color} {shape}".strip()
  location = _format_corner(task.get("location", ""))
  subtask_type = task.get("subtask_type", "")

  if subtask_type == "clear":
    blocked = _format_corner(task.get("blocked_location", "") or task.get("location", ""))
    if blocked:
      return f"move {block_desc} away from {blocked}"
    return f"move {block_desc} away"

  if location:
    return f"move {block_desc} to {location}"
  if block_desc:
    return f"move {block_desc}"
  return ""


_SUBTASK_LABEL_MODE_MOVE = "move"
_SUBTASK_LABEL_MODE_COLOR_REGION_THEN_INTO = "color_region_then_into"
_SUBTASK_LABEL_MODE_SHAPE_REGION_THEN_INTO_STRICT = "shape_region_then_into_strict"
_SUBTASK_LABEL_MODES = (
    _SUBTASK_LABEL_MODE_MOVE,
    _SUBTASK_LABEL_MODE_COLOR_REGION_THEN_INTO,
    _SUBTASK_LABEL_MODE_SHAPE_REGION_THEN_INTO_STRICT,
)


def _normalize_subtask_label_mode(mode: str) -> str:
  mode = str(mode or "").lower().strip()
  if mode in _SUBTASK_LABEL_MODES:
    return mode
  raise ValueError(
      f"Unknown subtask_label_mode: {mode}; expected one of {_SUBTASK_LABEL_MODES}"
  )


def _location_to_region_text(location: str) -> str:
  formatted = _format_corner(location)
  if not formatted:
    return ""
  words = formatted.split()
  if (
      len(words) == 2
      and words[0] in ("top", "bottom")
      and words[1] in ("left", "right")
  ):
    return f"{words[0]} {words[1]}"
  return formatted


def _first_label_destination_text(location: str) -> str:
  """Natural language destination phrase for first-of-color labels."""
  location_text = _location_to_region_text(location)
  if not location_text:
    return ""
  words = location_text.split()
  if (
      len(words) == 2
      and words[0] in ("top", "bottom")
      and words[1] in ("left", "right")
  ):
    return f"{location_text} corner region of the board"
  return f"{location_text} region of the board"


def _strict_region_destination_text(location: str) -> str:
  """Destination phrasing compatible with strict region-token parsing."""
  location = str(location or "").strip().lower()
  if location == "left":
    return "left side center region of the board"
  if location == "right":
    return "right side center region of the board"
  if location == "top":
    return "top center region of the board"
  if location == "bottom":
    return "bottom center region of the board"
  return _first_label_destination_text(location)


class _SubtaskLabelFormatter:
  """Builds per-subtask text labels with optional color-history strategy."""

  def __init__(self, mode: str, task_plan: Optional[List[Dict[str, Any]]] = None):
    self._mode = _normalize_subtask_label_mode(mode)
    self._labels_by_subtask_id: Dict[int, str] = {}
    self._labels_by_task_key: Dict[Tuple[str, str, str], str] = {}
    self._seen_color_counts: Dict[str, int] = {}
    self._color_to_shapes: Dict[str, set[str]] = {}
    self._seen_shape_counts: Dict[str, int] = {}
    self._shape_to_colors: Dict[str, set[str]] = {}
    if task_plan:
      self.observe_task_plan(task_plan)

  def observe_task_plan(self, task_plan: List[Dict[str, Any]]) -> None:
    for task in task_plan:
      subtask_type = str(task.get("subtask_type", "place") or "place")
      if subtask_type not in ("place", "recover"):
        continue
      block_name = str(task.get("block", ""))
      color = str(task.get("color") or _block_color(block_name))
      shape = _block_shape(block_name)
      if not color or not shape:
        continue
      self._color_to_shapes.setdefault(color, set()).add(shape)
      self._shape_to_colors.setdefault(shape, set()).add(color)

  def _task_key(self, task: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(task.get("subtask_type", "")),
        str(task.get("block", "")),
        str(task.get("location", "")),
    )

  def _subtask_id(self, task: Dict[str, Any]) -> int:
    raw = task.get("subtask_id", -1)
    try:
      return int(raw)
    except (TypeError, ValueError):
      return -1

  def _lookup_cached(self, task: Dict[str, Any]) -> Optional[str]:
    subtask_id = self._subtask_id(task)
    if subtask_id >= 0 and subtask_id in self._labels_by_subtask_id:
      return self._labels_by_subtask_id[subtask_id]
    return self._labels_by_task_key.get(self._task_key(task))

  def _cache(self, task: Dict[str, Any], label: str) -> None:
    subtask_id = self._subtask_id(task)
    if subtask_id >= 0:
      self._labels_by_subtask_id[subtask_id] = label
    self._labels_by_task_key[self._task_key(task)] = label

  def _other_shape(self, color: str, shape: str) -> str:
    all_shapes = sorted(self._color_to_shapes.get(color, set()))
    for candidate in all_shapes:
      if candidate != shape:
        return _format_shape(candidate)
    return _format_shape(shape)

  def _other_color(self, shape: str, color: str) -> str:
    all_colors = sorted(self._shape_to_colors.get(shape, set()))
    for candidate in all_colors:
      if candidate != color:
        return candidate
    return color

  def _color_region_then_into_label(self, task: Dict[str, Any]) -> str:
    block_name = str(task.get("block", ""))
    color = str(task.get("color") or _block_color(block_name))
    shape_raw = _block_shape(block_name)
    shape = _format_shape(shape_raw)
    if not color or not shape:
      return _legacy_label_from_task(task)

    subtask_type = str(task.get("subtask_type", "place") or "place")
    dest_text = _first_label_destination_text(str(task.get("location", "")))

    # Non-placement subtasks can appear in other setups; keep text neutral.
    if subtask_type not in ("place", "recover"):
      if dest_text:
        return f"move the {color} {shape} to the {dest_text}"
      return f"move the {color} {shape}"

    seen = self._seen_color_counts.get(color, 0)
    self._seen_color_counts[color] = seen + 1
    if seen == 0:
      if dest_text:
        return f"move the {color} {shape} to the {dest_text}"
      return f"move the {color} {shape}"

    other_shape = self._other_shape(color, shape_raw)
    return f"move the {color} {shape} to the {color} {other_shape}"

  def _shape_region_then_into_strict_label(self, task: Dict[str, Any]) -> str:
    block_name = str(task.get("block", ""))
    color = str(task.get("color") or _block_color(block_name))
    shape_raw = _block_shape(block_name)
    shape = _format_shape(shape_raw)
    if not color or not shape:
      return _legacy_label_from_task(task)

    subtask_type = str(task.get("subtask_type", "place") or "place")
    dest_text = _strict_region_destination_text(str(task.get("location", "")))

    if subtask_type not in ("place", "recover"):
      if dest_text:
        return f"move the {color} {shape} to the {dest_text}"
      return f"move the {color} {shape}"

    seen = self._seen_shape_counts.get(shape_raw, 0)
    self._seen_shape_counts[shape_raw] = seen + 1
    if seen == 0:
      if dest_text:
        return f"move the {color} {shape} to the {dest_text}"
      return f"move the {color} {shape}"

    other_color = self._other_color(shape_raw, color)
    return f"move the {color} {shape} to the {other_color} {shape}"

  def format(self, task: Dict[str, Any]) -> str:
    cached = self._lookup_cached(task)
    if cached is not None:
      return cached
    self.observe_task_plan([task])
    if self._mode == _SUBTASK_LABEL_MODE_MOVE:
      label = _legacy_label_from_task(task)
    elif self._mode == _SUBTASK_LABEL_MODE_COLOR_REGION_THEN_INTO:
      label = self._color_region_then_into_label(task)
    else:
      label = self._shape_region_then_into_strict_label(task)
    self._cache(task, label)
    return label

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "./oracle_long_horizon_rollouts",
    "Output directory for episodes.")
_NUM_EPISODES = flags.DEFINE_integer(
    "num_episodes",
    10,
    "Number of episodes to generate.")
_MAX_STEPS = flags.DEFINE_integer(
    "max_steps",
    300,
    "Max steps per episode.")
_BLOCK_MODE = flags.DEFINE_string(
    "block_mode",
    "BLOCK_8",
    "Block mode (e.g., BLOCK_4, BLOCK_8, BLOCK_8_PAIRINGS, N_CHOOSE_K).")
_TASK_MODE = flags.DEFINE_string(
    "task_mode",
    "sort_by_color",
    "Task mode: sort_by_color, sort_by_shape_edge_centers, move_color_to_corner, "
    "arrange_horizontal_line, or arrange_single_file.")

_PLANNER = flags.DEFINE_string(
    "planner",
    "strategic",
    "Planner to use: baseline or strategic. Strategic adds explicit clear/place/recover subtasks.")

_OCCUPIED_RADIUS = flags.DEFINE_float(
    "occupied_radius",
    0.06,
    "(Strategic planner) Radius around a slot considered occupied by a wrong-color blocker.")
_KEEPOUT_RADIUS = flags.DEFINE_float(
    "keepout_radius",
    0.12,
    "(Strategic planner) If a placed block drifts farther than this from its slot, insert a recover subtask.")
_STAGING_SPACING = flags.DEFINE_float(
    "staging_spacing",
    0.08,
    "(Strategic planner) Spacing between staging targets near the table center.")
_ENABLE_CLEARING = flags.DEFINE_bool(
    "enable_clearing",
    False,
    "(Strategic planner) Enable clear subtasks when target slots are blocked.")
_CLEAR_TARGET_MODE = flags.DEFINE_string(
    "clear_target_mode",
    "staging",
    "(Strategic planner) Clear target: staging or own_goal.")
_CLEAR_REQUIRE_REGION = flags.DEFINE_bool(
    "clear_require_region",
    False,
    "(Strategic planner) If set, clear subtasks complete only once the blocker reaches its own region.")
_REPLAN_INTERVAL_STEPS = flags.DEFINE_integer(
    "replan_interval_steps",
    5,
    "(Strategic planner) Only scan for disturbed blocks every N task-info updates.")
_MAX_DYNAMIC_INSERTS = flags.DEFINE_integer(
    "max_dynamic_inserts",
    2,
    "(Strategic planner) Max number of tasks to insert per get_current_task_info() call.")
_TARGET_COLOR = flags.DEFINE_string(
    "target_color",
    None,
    "Target color for move_color_to_corner (e.g., red).")
_TARGET_CORNER = flags.DEFINE_string(
    "target_corner",
    "top_left",
    "Corner for move_color_to_corner (top_left, top_right, bottom_left, bottom_right).")
_INSTRUCTION_STYLE = flags.DEFINE_string(
    "instruction_style",
    "high_level",
    "Instruction style: high_level or high_level_plus_subtask.")
_SUBTASK_LABEL_MODE = flags.DEFINE_string(
    "subtask_label_mode",
    _SUBTASK_LABEL_MODE_MOVE,
    "Subtask label style: 'move', 'color_region_then_into', or "
    "'shape_region_then_into_strict'.")
_COMBO_INDEX = flags.DEFINE_integer(
    "combo_index",
    -1,
    "Optional fixed block-combination index for the chosen block_mode "
    "(>=0 to force a deterministic combo each reset).")
_SECOND_SAME_COLOR_INTO_BLOCK = flags.DEFINE_bool(
    "second_same_color_into_block",
    False,
    "If set, the second same-color placement targets the first same-color block "
    "dynamically, and completion uses block-to-block distance.")
_INTO_BLOCK_DISTANCE = flags.DEFINE_float(
    "into_block_distance",
    reward_constants.TARGET_BLOCK_DISTANCE,
    "Distance threshold (meters) for second_same_color_into_block completion.")
_SUBTASK_COMPLETION_DISTANCE = flags.DEFINE_float(
    "subtask_completion_distance",
    -1.0,
    "Optional strict distance (meters) for subtask progression of place/recover. "
    "Set <=0 to use success_mode-based completion.")
_SHUFFLE_TASKS = flags.DEFINE_bool(
    "shuffle_tasks",
    True,
    "Shuffle subtasks within an episode.")
_MAX_SUBTASKS = flags.DEFINE_integer(
    "max_subtasks",
    None,
    "Optional cap on number of subtasks per episode.")
_CLUSTER_SPACING = flags.DEFINE_float(
    "cluster_spacing",
    0.03,
    "Spacing between targets within a cluster.")
_LINE_ASSIGNMENT_MODE = flags.DEFINE_string(
    "line_assignment_mode",
    "sorted",
    "Block-to-slot assignment for line tasks: sorted, random, or fixed_mixed.")
_LINE_LAYOUT_MODE = flags.DEFINE_string(
    "line_layout_mode",
    "horizontal",
    "Line geometry for line tasks: horizontal or single_file.")
_TARGET_JITTER = flags.DEFINE_float(
    "target_jitter",
    0.0,
    "Uniform jitter added to targets.")
_TARGET_DISTANCE = flags.DEFINE_float(
    "target_distance",
    0.1,
    "Distance threshold for subtask completion.")
_TARGET_WORKSPACE_INSET = flags.DEFINE_float(
    "target_workspace_inset",
    -1.0,
    "Inset (meters) from workspace boundary for target placement. "
    "Set <0 to use the default LanguageTable inset.")
_SUCCESS_MODE = flags.DEFINE_string(
    "success_mode",
    "point",
    "Success criterion: 'point', 'region', 'cluster_region', or 'corner_radius'.")
_CORNER_SUCCESS_RADIUS = flags.DEFINE_float(
    "corner_success_radius",
    -1.0,
    "Radius (meters) for corner_radius success (<=0 uses target_distance).")
_CORNER_ANCHOR_OFFSET_X = flags.DEFINE_float(
    "corner_anchor_offset_x",
    0.0,
    "X offset (meters) applied to all corner-radius anchors (success + rendering).")
_CORNER_ANCHOR_OFFSET_Y = flags.DEFINE_float(
    "corner_anchor_offset_y",
    0.0,
    "Y offset (meters) applied to all corner-radius anchors (success + rendering).")
_REGION_MARGIN = flags.DEFINE_float(
    "region_margin",
    0.0,
    "Margin (meters) from the center split line for region-based success.")
_CLUSTER_REGION_PADDING = flags.DEFINE_float(
    "cluster_region_padding",
    -1.0,
    "Padding (meters) around the per-location target cluster for cluster_region success. "
    "Set <0 to use target_distance.")
_COLOR_CORNER_MAP = flags.DEFINE_string(
    "color_corner_map",
    "default",
    "Corner mapping for sort_by_color: default or view.")
_RANDOMIZE_COLOR_CORNER_MAP = flags.DEFINE_bool(
    "randomize_color_corner_map",
    False,
    "If set, sample a new color->corner assignment per episode for sort_by_color.")
_SEED = flags.DEFINE_integer(
    "seed",
    0,
    "Environment RNG seed.")
_REQUIRE_PLAN = flags.DEFINE_bool(
    "require_plan",
    True,
    "Require oracle to find an initial plan before rollout.")
_MAX_INIT_TRIES = flags.DEFINE_integer(
    "max_init_tries",
    20,
    "Max reset attempts to find a valid plan.")
_USE_EE_PLANNER = flags.DEFINE_bool(
    "use_ee_planner",
    True,
    "Use end-effector RRT planner in oracle.")
_ACTION_NOISE_STD = flags.DEFINE_float(
    "action_noise_std",
    0.0,
    "Action noise std for oracle.")
_SLOWDOWN_FREESPACE = flags.DEFINE_bool(
    "slowdown_freespace",
    False,
    "Apply slowdown when moving in freespace.")
_SLOWDOWN_EXPONENT = flags.DEFINE_float(
    "slowdown_exponent",
    1.0,
    "Exponent applied to slowdown factors near target. "
    "0 disables slowdown, 1 is baseline, >1 is more slowdown.")
_RENDER_TEXT = flags.DEFINE_bool(
    "render_text",
    False,
    "Render instruction text into images.")
_IMAGE_WIDTH = flags.DEFINE_integer(
    "image_width",
    constants.IMAGE_WIDTH,
    "Override camera image width in pixels.")
_IMAGE_HEIGHT = flags.DEFINE_integer(
    "image_height",
    constants.IMAGE_HEIGHT,
    "Override camera image height in pixels.")
_SAVE_VIDEO = flags.DEFINE_bool(
    "save_video",
    False,
    "Save rollout videos (mp4).")
_REQUIRE_SUCCESS = flags.DEFINE_bool(
    "require_success",
    False,
    "Only keep episodes that satisfy the success criterion.")
_MAX_EPISODE_ATTEMPTS = flags.DEFINE_integer(
    "max_episode_attempts",
    0,
    "Max attempts to collect episodes (0 = unlimited).")
_FREEZE_TASK_ORDER_ON_RETRY = flags.DEFINE_bool(
    "freeze_task_order_on_retry",
    False,
    "If set, keep the initial subtask order across retries within an episode.")
_RANDOMIZE_RRT_ON_RETRY = flags.DEFINE_bool(
    "randomize_rrt_on_retry",
    False,
    "If set, reseed RRT/pathfinding randomness on each attempt.")
_AVOID_GOAL_REGIONS_ON_RESET = flags.DEFINE_bool(
    "avoid_goal_regions_on_reset",
    False,
    "If set, reject initial block placements inside their goal regions.")
_AVOID_GOAL_REGION_MARGIN = flags.DEFINE_float(
    "avoid_goal_region_margin",
    -1.0,
    "Margin (meters) for avoid_goal_regions_on_reset (<0 uses reward region_margin).")
_EARLY_STOP_ON_SUCCESS = flags.DEFINE_bool(
    "early_stop_on_success",
    True,
    "Stop rollout early when the reward signals success.")
_VIDEO_FPS = flags.DEFINE_integer(
    "video_fps",
    10,
    "FPS for saved videos.")
_VIDEO_DIR = flags.DEFINE_string(
    "video_dir",
    None,
    "Optional directory for videos (defaults to output_dir/videos).")
_DEBUG_TARGET_MARKERS = flags.DEFINE_bool(
    "debug_target_markers",
    False,
    "Render target markers in the sim (debug only).")
_DEBUG_CURRENT_TARGET_ONLY = flags.DEFINE_bool(
    "debug_current_target_only",
    True,
    "Only render the current subtask target when debugging.")
_DEBUG_MARKER_RADIUS = flags.DEFINE_float(
    "debug_marker_radius",
    0.012,
    "Radius for debug target markers.")
_RENDER_CORNER_REGIONS = flags.DEFINE_bool(
    "render_corner_regions",
    False,
    "Render faint color regions for each color corner.")
_CORNER_REGION_SIZE = flags.DEFINE_float(
    "corner_region_size",
    0.10,
    "Side length (meters) for corner color regions.")
_CORNER_REGION_ALPHA = flags.DEFINE_float(
    "corner_region_alpha",
    0.18,
    "Alpha for corner color regions.")
_RENDER_CORNER_GOAL_REGIONS = flags.DEFINE_bool(
    "render_corner_goal_regions",
    False,
    "Render margin-based corner goal regions with colored boundaries.")
_CORNER_GOAL_REGION_MARGIN = flags.DEFINE_float(
    "corner_goal_region_margin",
    -1.0,
    "Margin (meters) used for rendering corner goal regions (<=0 uses region_margin).")
_CORNER_GOAL_RADIUS = flags.DEFINE_float(
    "corner_goal_radius",
    -1.0,
    "Radius (meters) for quarter-circle corner-goal rendering "
    "(<=0 uses corner_success_radius, then target_distance).")
_CORNER_GOAL_REGION_ALPHA = flags.DEFINE_float(
    "corner_goal_region_alpha",
    0.08,
    "Alpha for filled corner goal regions.")
_CORNER_GOAL_BOUNDARY_ALPHA = flags.DEFINE_float(
    "corner_goal_boundary_alpha",
    0.75,
    "Alpha for corner goal region boundaries.")
_CORNER_GOAL_BOUNDARY_THICKNESS = flags.DEFINE_float(
    "corner_goal_boundary_thickness",
    0.004,
    "Boundary thickness (meters) for corner goal region boxes.")
_RENDER_GOAL_RINGS = flags.DEFINE_bool(
    "render_goal_rings",
    False,
    "Render outline rings for goal target positions (visualization only).")
_GOAL_RING_RADIUS = flags.DEFINE_float(
    "goal_ring_radius",
    -1.0,
    "Ring radius for goal outlines (<=0 uses target_distance).")
_GOAL_RING_SEGMENTS = flags.DEFINE_integer(
    "goal_ring_segments",
    36,
    "Number of segments used to draw a goal ring.")
_GOAL_RING_LINE_WIDTH = flags.DEFINE_integer(
    "goal_ring_line_width",
    2,
    "Line width for goal rings.")
_GOAL_RING_THICKNESS = flags.DEFINE_float(
    "goal_ring_thickness",
    -1.0,
    "Thickness (meters) for goal ring outlines (<=0 uses split_line_width).")
_RENDER_REGION_RINGS = flags.DEFINE_bool(
    "render_region_rings",
    False,
    "Render outline rings at region centers (visualization only).")
_REGION_RING_RADIUS = flags.DEFINE_float(
    "region_ring_radius",
    -1.0,
    "Ring radius for region outlines (<=0 uses target_distance).")
_REGION_RING_SEGMENTS = flags.DEFINE_integer(
    "region_ring_segments",
    36,
    "Number of segments used to draw a region ring.")
_REGION_RING_THICKNESS = flags.DEFINE_float(
    "region_ring_thickness",
    -1.0,
    "Thickness (meters) for region ring outlines (<=0 uses split_line_width).")
_RENDER_SPLIT_LINE = flags.DEFINE_bool(
    "render_split_line",
    False,
    "Render a faint split line between left/right regions.")
_SPLIT_LINE_WIDTH = flags.DEFINE_float(
    "split_line_width",
    0.006,
    "Width (meters) of the split line.")
_SPLIT_LINE_ALPHA = flags.DEFINE_float(
    "split_line_alpha",
    0.35,
    "Alpha for the split line.")


# --- Optional motion post-processing for more human-like oracle rollouts. ---
_SMOOTH_ACTIONS = flags.DEFINE_bool(
    "smooth_actions",
    False,
    "Apply EMA smoothing to oracle actions before stepping the environment.")
_EMA_ALPHA = flags.DEFINE_float(
    "ema_alpha",
    0.2,
    "EMA alpha for action smoothing (higher = less smoothing). Only used with --smooth_actions.")
_SMOOTH_DIRECTION_ONLY = flags.DEFINE_bool(
    "smooth_direction_only",
    False,
    "If set, smooth only the action direction and preserve magnitude.")
_MAX_ACTION_DELTA = flags.DEFINE_float(
    "max_action_delta",
    -1.0,
    "Max L2-norm per-step change in action. Set <=0 to disable.")
_ACTION_SCALE = flags.DEFINE_float(
    "action_scale",
    1.0,
    "Global scale applied to oracle actions before smoothing/limiting.")
_MIN_ACTION_NORM = flags.DEFINE_float(
    "min_action_norm",
    0.0,
    "Minimum L2-norm for non-zero actions after smoothing/scaling.")


def _get_block_mode(name: str):
  try:
    return blocks.LanguageTableBlockVariants[name]
  except KeyError as exc:
    raise ValueError(f"Unknown block_mode: {name}") from exc


def _effective_corner_success_radius() -> float:
  if _CORNER_SUCCESS_RADIUS.value > 0:
    return float(_CORNER_SUCCESS_RADIUS.value)
  return float(_TARGET_DISTANCE.value)


def _effective_corner_goal_radius() -> float:
  if _CORNER_GOAL_RADIUS.value > 0:
    return float(_CORNER_GOAL_RADIUS.value)
  return _effective_corner_success_radius()


def _build_env():
  block_mode = _get_block_mode(_BLOCK_MODE.value)
  color_to_corner_map = long_horizon.get_color_to_corner_map(_COLOR_CORNER_MAP.value)
  target_workspace_inset = None
  if _TARGET_WORKSPACE_INSET.value >= 0:
    target_workspace_inset = float(_TARGET_WORKSPACE_INSET.value)
  corner_success_radius = None
  if _CORNER_SUCCESS_RADIUS.value > 0:
    corner_success_radius = float(_CORNER_SUCCESS_RADIUS.value)
  corner_anchor_offset_x = float(_CORNER_ANCHOR_OFFSET_X.value)
  corner_anchor_offset_y = float(_CORNER_ANCHOR_OFFSET_Y.value)
  if _PLANNER.value.lower() == "strategic":
    reward_factory = partial(
        long_horizon.StrategicChainedBlock2LocationReward,
        task_mode=_TASK_MODE.value,
        target_color=_TARGET_COLOR.value,
        target_corner=_TARGET_CORNER.value,
        instruction_style=_INSTRUCTION_STYLE.value,
        shuffle_tasks=_SHUFFLE_TASKS.value,
        max_subtasks=_MAX_SUBTASKS.value,
        cluster_spacing=_CLUSTER_SPACING.value,
        line_assignment_mode=_LINE_ASSIGNMENT_MODE.value,
        line_layout_mode=_LINE_LAYOUT_MODE.value,
        target_jitter=_TARGET_JITTER.value,
        target_distance=_TARGET_DISTANCE.value,
        cluster_region_padding=_CLUSTER_REGION_PADDING.value,
        color_to_corner_map=color_to_corner_map,
        success_mode=_SUCCESS_MODE.value,
        region_margin=_REGION_MARGIN.value,
        occupied_radius=_OCCUPIED_RADIUS.value,
        keepout_radius=_KEEPOUT_RADIUS.value,
        staging_spacing=_STAGING_SPACING.value,
        replanning_interval_steps=_REPLAN_INTERVAL_STEPS.value,
        max_dynamic_inserts_per_call=_MAX_DYNAMIC_INSERTS.value,
        enable_clearing=_ENABLE_CLEARING.value,
        clear_target_mode=_CLEAR_TARGET_MODE.value,
        clear_require_region=_CLEAR_REQUIRE_REGION.value,
        target_workspace_inset=target_workspace_inset,
        corner_success_radius=corner_success_radius,
        corner_anchor_offset_x=corner_anchor_offset_x,
        corner_anchor_offset_y=corner_anchor_offset_y,
        second_same_color_into_block=_SECOND_SAME_COLOR_INTO_BLOCK.value,
        into_block_distance=_INTO_BLOCK_DISTANCE.value,
        subtask_completion_distance=_SUBTASK_COMPLETION_DISTANCE.value,
    )
  elif _PLANNER.value.lower() == "baseline":
    reward_factory = partial(
        long_horizon.ChainedBlock2LocationReward,
        task_mode=_TASK_MODE.value,
        target_color=_TARGET_COLOR.value,
        target_corner=_TARGET_CORNER.value,
        instruction_style=_INSTRUCTION_STYLE.value,
        shuffle_tasks=_SHUFFLE_TASKS.value,
        max_subtasks=_MAX_SUBTASKS.value,
        cluster_spacing=_CLUSTER_SPACING.value,
        line_assignment_mode=_LINE_ASSIGNMENT_MODE.value,
        line_layout_mode=_LINE_LAYOUT_MODE.value,
        target_jitter=_TARGET_JITTER.value,
        target_distance=_TARGET_DISTANCE.value,
        cluster_region_padding=_CLUSTER_REGION_PADDING.value,
        color_to_corner_map=color_to_corner_map,
        success_mode=_SUCCESS_MODE.value,
        region_margin=_REGION_MARGIN.value,
        target_workspace_inset=target_workspace_inset,
        corner_success_radius=corner_success_radius,
        corner_anchor_offset_x=corner_anchor_offset_x,
        corner_anchor_offset_y=corner_anchor_offset_y,
    )
  else:
    raise ValueError(f"Unknown planner: {_PLANNER.value}; expected 'baseline' or 'strategic'.")
  base_env = language_table.LanguageTable(
      block_mode=block_mode,
      reward_factory=reward_factory,
      seed=_SEED.value,
      render_text_in_image=_RENDER_TEXT.value,
      image_width=_IMAGE_WIDTH.value,
      image_height=_IMAGE_HEIGHT.value,
  )
  if _COMBO_INDEX.value >= 0:
    if hasattr(base_env, "set_combo_index_override"):
      base_env.set_combo_index_override(int(_COMBO_INDEX.value))
    else:
      base_env._combo_idx_override = int(_COMBO_INDEX.value)  # pylint: disable=protected-access
  if _AVOID_GOAL_REGIONS_ON_RESET.value:
    base_env._avoid_goal_regions = True
    if _AVOID_GOAL_REGION_MARGIN.value >= 0:
      base_env._avoid_goal_region_margin = float(_AVOID_GOAL_REGION_MARGIN.value)
  # Wrap with TF-Agents gym wrapper so the oracle can access specs.
  return gym_wrapper.GymWrapper(base_env)


def _sample_color_corner_map_for_episode(rng: np.random.RandomState) -> Dict[str, str]:
  base_map = dict(long_horizon.get_color_to_corner_map(_COLOR_CORNER_MAP.value))
  if not _RANDOMIZE_COLOR_CORNER_MAP.value or _TASK_MODE.value != "sort_by_color":
    return base_map

  corner_names = {"top_left", "top_right", "bottom_left", "bottom_right"}
  locations = [base_map[color] for color in sorted(base_map.keys())]
  if not all(loc in corner_names for loc in locations):
    raise ValueError(
        "--randomize_color_corner_map requires a 4-corner mapping "
        f"(got {_COLOR_CORNER_MAP.value!r} -> {base_map})."
    )

  shuffled = list(locations)
  rng.shuffle(shuffled)
  colors = sorted(base_map.keys())
  return {color: shuffled[i] for i, color in enumerate(colors)}


def _set_color_corner_map_on_reward(env, color_to_corner_map: Dict[str, str]) -> None:
  base_env = _get_base_env(env)
  reward_calc = getattr(base_env, "_reward_calculator", None)
  if reward_calc is None:
    return
  if hasattr(reward_calc, "_color_to_corner_map"):
    reward_calc._color_to_corner_map = dict(color_to_corner_map)


def _reset_with_plan(env, oracle):
  for attempt in range(_MAX_INIT_TRIES.value):
    env.reset()
    oracle.reset()
    if not _REQUIRE_PLAN.value:
      return True
    raw_state = env.compute_state()
    plan_success = oracle.get_plan(raw_state)
    if plan_success:
      return True
    logging.info("Resetting env; oracle plan failed on attempt %d", attempt + 1)
  return False


def _save_episode(episode_dir, episode_idx, payload, metadata):
  npz_path = os.path.join(episode_dir, f"episode_{episode_idx:06d}.npz")
  np.savez_compressed(npz_path, **payload)
  meta_path = os.path.join(episode_dir, f"episode_{episode_idx:06d}.json")
  with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)


def _unwrap_step(step_result):
  if isinstance(step_result, ts.TimeStep):
    reward = float(step_result.reward) if np.size(step_result.reward) == 1 else step_result.reward
    done = bool(step_result.is_last())
    observation = step_result.observation
    return observation, reward, done
  observation, reward, done, _ = step_result
  return observation, reward, done


def _get_base_env(env):
  return env.gym if hasattr(env, "gym") else env


def _clear_markers(pybullet_client, marker_ids):
  for marker_id in marker_ids:
    try:
      pybullet_client.removeBody(marker_id)
    except Exception:
      pass


def _clear_debug_items(pybullet_client, item_ids):
  for item_id in item_ids:
    try:
      pybullet_client.removeUserDebugItem(item_id)
    except Exception:
      pass


def _iter_success_blocks(reward_calc):
  if reward_calc is None:
    return []
  if hasattr(reward_calc, "_place_location_by_block"):
    mapping = getattr(reward_calc, "_place_location_by_block", {}) or {}
    return list(mapping.items())
  if hasattr(reward_calc, "get_task_plan"):
    blocks = []
    for task in reward_calc.get_task_plan() or []:
      location = task.get("location", "")
      if not location or location == "staging":
        continue
      blocks.append((task.get("block", ""), location))
    return blocks
  return []


def _task_key(task: Dict[str, Any]) -> tuple:
  return (
      task.get("subtask_type", "") or "place",
      task.get("block", ""),
      task.get("location", ""),
  )


def _extract_task_order(reward_calc) -> list:
  plan = reward_calc.get_task_plan() if hasattr(reward_calc, "get_task_plan") else []
  if not plan:
    return []
  return [_task_key(t) for t in plan]


def _apply_task_order(reward_calc, cached_order: list) -> None:
  if not cached_order or not hasattr(reward_calc, "_task_plan"):
    return
  plan = list(getattr(reward_calc, "_task_plan", []))
  if not plan:
    return
  buckets: Dict[tuple, List[Dict[str, Any]]] = {}
  for task in plan:
    buckets.setdefault(_task_key(task), []).append(task)
  new_plan: List[Dict[str, Any]] = []
  for key in cached_order:
    tasks = buckets.get(key)
    if tasks:
      new_plan.append(tasks.pop(0))
  new_ids = {id(t) for t in new_plan}
  for task in plan:
    if id(task) not in new_ids:
      new_plan.append(task)
  reward_calc._task_plan = new_plan
  if hasattr(reward_calc, "_current_task_index"):
    reward_calc._current_task_index = 0
  if hasattr(reward_calc, "_in_reward_zone_steps"):
    reward_calc._in_reward_zone_steps = 0


def _seed_rrt_for_attempt(attempt_idx: int, oracle) -> None:
  if not _RANDOMIZE_RRT_ON_RETRY.value:
    return
  base_seed = _SEED.value if _SEED.value is not None else 0
  seed = int(base_seed + attempt_idx * 100003) % (2**32 - 1)
  np.random.seed(seed)
  if hasattr(oracle, "_np_random_state"):
    oracle._np_random_state = np.random.RandomState(seed + 17)


def _success_by_region(state, reward_calc, margin):
  if state is None:
    return False
  for block, location in _iter_success_blocks(reward_calc):
    if not block or not location:
      continue
    key = f"block_{block}_translation"
    if key not in state:
      return False
    block_xy = np.asarray(state[key], dtype=np.float32)
    if not long_horizon._block_in_location_region(block_xy, location, margin):
      return False
  return True


def _compute_success(state, reward_calc):
  if reward_calc is None:
    return False
  mode = str(_SUCCESS_MODE.value).lower()
  if mode == "region":
    return _success_by_region(state, reward_calc, _REGION_MARGIN.value)
  if mode == "cluster_region":
    if hasattr(reward_calc, "is_success"):
      return bool(reward_calc.is_success(state))
    padding = _CLUSTER_REGION_PADDING.value
    if padding < 0:
      padding = _TARGET_DISTANCE.value
    if hasattr(reward_calc, "get_task_plan"):
      return long_horizon._success_by_cluster_region(
          state, reward_calc.get_task_plan() or [], padding
      )
    return False
  if hasattr(reward_calc, "is_success"):
    return bool(reward_calc.is_success(state))
  return False


def _add_target_markers(env, reward_calc, current_only):
  base_env = _get_base_env(env)
  pyb = base_env.pybullet_client
  marker_ids = []
  if not hasattr(reward_calc, "get_task_plan"):
    return marker_ids
  task_plan = reward_calc.get_task_plan()
  if not task_plan:
    return marker_ids
  if current_only:
    idx = getattr(reward_calc, "current_task_index", 0)
    task_plan = [task_plan[idx]]
  for task in task_plan:
    target_xy = task["target_translation"]
    marker_id = add_visual_sphere(
        pyb,
        center=(target_xy[0], target_xy[1], 0.001),
        radius=_DEBUG_MARKER_RADIUS.value,
        rgba=(0.1, 0.9, 0.1, 0.6))
    marker_ids.append(marker_id)
  return marker_ids


def _add_corner_regions(env, color_to_corner_map):
  base_env = _get_base_env(env)
  pyb = base_env.pybullet_client
  centers = long_horizon._location_centers()
  size = float(_CORNER_REGION_SIZE.value)
  alpha = float(_CORNER_REGION_ALPHA.value)
  half = max(0.0, size / 2.0)
  z = 0.001
  color_rgba = {
      "red": (1.0, 0.0, 0.0, alpha),
      "blue": (0.0, 0.3, 1.0, alpha),
      "green": (0.0, 0.8, 0.0, alpha),
      "yellow": (1.0, 0.9, 0.0, alpha),
  }
  ids = []
  for color, corner in color_to_corner_map.items():
    center = centers.get(corner)
    if center is None:
      continue
    rgba = color_rgba.get(color, (0.7, 0.7, 0.7, alpha))
    vis_id = pyb.createVisualShape(
        pyb.GEOM_BOX,
        halfExtents=(half, half, 0.0005),
        rgbaColor=rgba,
    )
    body_id = pyb.createMultiBody(
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vis_id,
        basePosition=(float(center[0]), float(center[1]), z),
    )
    ids.append(body_id)
  return ids


def _corner_goal_bounds(location: str, margin: float) -> Optional[Tuple[float, float, float, float]]:
  x_min = constants.X_MIN + constants.WORKSPACE_BOUNDS_BUFFER
  x_max = constants.X_MAX - constants.WORKSPACE_BOUNDS_BUFFER
  y_min = constants.Y_MIN + constants.WORKSPACE_BOUNDS_BUFFER
  y_max = constants.Y_MAX - constants.WORKSPACE_BOUNDS_BUFFER
  cx = float(constants.CENTER_X)
  cy = float(constants.CENTER_Y)
  m = max(0.0, float(margin))
  if location == "top_left":
    return x_min, cx - m, y_min, cy - m
  if location == "top_right":
    return cx + m, x_max, y_min, cy - m
  if location == "bottom_left":
    return x_min, cx - m, cy + m, y_max
  if location == "bottom_right":
    return cx + m, x_max, cy + m, y_max
  return None


def _corner_quadrant_signs(location: str) -> Optional[Tuple[float, float]]:
  if location == "top_left":
    return 1.0, 1.0
  if location == "top_right":
    return -1.0, 1.0
  if location == "bottom_left":
    return 1.0, -1.0
  if location == "bottom_right":
    return -1.0, -1.0
  return None


def _add_corner_goal_regions(env, color_to_corner_map, margin: float):
  base_env = _get_base_env(env)
  pyb = base_env.pybullet_client
  z = 0.001
  fill_alpha = float(_CORNER_GOAL_REGION_ALPHA.value)
  boundary_alpha = float(_CORNER_GOAL_BOUNDARY_ALPHA.value)
  boundary_thickness = max(0.001, float(_CORNER_GOAL_BOUNDARY_THICKNESS.value))
  color_rgba = {
      "red": (1.0, 0.0, 0.0, fill_alpha),
      "blue": (0.0, 0.3, 1.0, fill_alpha),
      "green": (0.0, 0.8, 0.0, fill_alpha),
      "yellow": (1.0, 0.9, 0.0, fill_alpha),
  }
  ids = []
  for color, location in color_to_corner_map.items():
    bounds = _corner_goal_bounds(str(location), margin)
    if bounds is None:
      continue
    x_lo, x_hi, y_lo, y_hi = bounds
    if x_hi <= x_lo or y_hi <= y_lo:
      continue
    rgba_fill = color_rgba.get(color, (0.7, 0.7, 0.7, fill_alpha))
    rgba_boundary = (rgba_fill[0], rgba_fill[1], rgba_fill[2], boundary_alpha)

    # Filled region.
    fill_half_x = (x_hi - x_lo) / 2.0
    fill_half_y = (y_hi - y_lo) / 2.0
    fill_vis_id = pyb.createVisualShape(
        pyb.GEOM_BOX,
        halfExtents=(fill_half_x, fill_half_y, 0.0003),
        rgbaColor=rgba_fill,
    )
    fill_body_id = pyb.createMultiBody(
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=fill_vis_id,
        basePosition=(float((x_lo + x_hi) / 2.0), float((y_lo + y_hi) / 2.0), z),
    )
    ids.append(fill_body_id)

    # Boundary edges.
    horiz_half_x = fill_half_x
    horiz_half_y = boundary_thickness / 2.0
    vert_half_x = boundary_thickness / 2.0
    vert_half_y = fill_half_y
    edge_specs = [
        ((x_lo + x_hi) / 2.0, y_lo, horiz_half_x, horiz_half_y),  # top/bottom edge
        ((x_lo + x_hi) / 2.0, y_hi, horiz_half_x, horiz_half_y),
        (x_lo, (y_lo + y_hi) / 2.0, vert_half_x, vert_half_y),  # left/right edge
        (x_hi, (y_lo + y_hi) / 2.0, vert_half_x, vert_half_y),
    ]
    for edge in edge_specs:
      cx_edge, cy_edge, half_x, half_y = edge
      edge_vis_id = pyb.createVisualShape(
          pyb.GEOM_BOX,
          halfExtents=(float(half_x), float(half_y), 0.0004),
          rgbaColor=rgba_boundary,
      )
      edge_body_id = pyb.createMultiBody(
          baseCollisionShapeIndex=-1,
          baseVisualShapeIndex=edge_vis_id,
          basePosition=(float(cx_edge), float(cy_edge), z),
      )
      ids.append(edge_body_id)
  return ids


def _add_corner_goal_quarter_circles(env, color_to_corner_map, radius: float):
  base_env = _get_base_env(env)
  pyb = base_env.pybullet_client
  z = 0.001
  fill_alpha = float(_CORNER_GOAL_REGION_ALPHA.value)
  boundary_alpha = float(_CORNER_GOAL_BOUNDARY_ALPHA.value)
  boundary_thickness = max(0.001, float(_CORNER_GOAL_BOUNDARY_THICKNESS.value))
  radius = max(0.0, float(radius))
  if radius <= 0:
    return []

  # Keep quarter-circle goals aligned with physical board corners.
  corner_centers = long_horizon._corner_centers(
      workspace_inset=0.0,
      offset_x=float(_CORNER_ANCHOR_OFFSET_X.value),
      offset_y=float(_CORNER_ANCHOR_OFFSET_Y.value),
  )

  color_rgba = {
      "red": (1.0, 0.0, 0.0, fill_alpha),
      "blue": (0.0, 0.3, 1.0, fill_alpha),
      "green": (0.0, 0.8, 0.0, fill_alpha),
      "yellow": (1.0, 0.9, 0.0, fill_alpha),
  }
  ids = []

  # Coarse tile fill inside each quarter-circle.
  cell_size = max(boundary_thickness * 2.0, min(0.02, radius / 8.0))
  cell_size = max(0.006, cell_size)
  half_cell = cell_size / 2.0
  num_cells = max(1, int(np.ceil(radius / cell_size)))
  half_height_fill = 0.00025
  half_height_line = 0.0004

  for color, location in color_to_corner_map.items():
    signs = _corner_quadrant_signs(str(location))
    center = corner_centers.get(str(location))
    if signs is None or center is None:
      continue
    sx, sy = signs
    cx = float(center[0])
    cy = float(center[1])

    rgba_fill = color_rgba.get(color, (0.7, 0.7, 0.7, fill_alpha))
    rgba_boundary = (rgba_fill[0], rgba_fill[1], rgba_fill[2], boundary_alpha)

    fill_vis_id = pyb.createVisualShape(
        pyb.GEOM_BOX,
        halfExtents=(half_cell, half_cell, half_height_fill),
        rgbaColor=rgba_fill,
    )
    for i in range(num_cells):
      dx = (i + 0.5) * cell_size
      for j in range(num_cells):
        dy = (j + 0.5) * cell_size
        if dx * dx + dy * dy > radius * radius:
          continue
        body_id = pyb.createMultiBody(
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=fill_vis_id,
            basePosition=(cx + sx * dx, cy + sy * dy, z),
        )
        ids.append(body_id)

    # Radial edges from the corner to the arc.
    edge_half_len = max(0.001, radius / 2.0)
    edge_half_thickness = boundary_thickness / 2.0
    edge_vis_id = pyb.createVisualShape(
        pyb.GEOM_BOX,
        halfExtents=(edge_half_len, edge_half_thickness, half_height_line),
        rgbaColor=rgba_boundary,
    )
    edge_specs = [
      (cx + sx * edge_half_len, cy, 0.0),
      (cx, cy + sy * edge_half_len, np.pi / 2.0),
    ]
    for edge_cx, edge_cy, yaw in edge_specs:
      edge_body_id = pyb.createMultiBody(
          baseCollisionShapeIndex=-1,
          baseVisualShapeIndex=edge_vis_id,
          basePosition=(float(edge_cx), float(edge_cy), z),
          baseOrientation=pyb.getQuaternionFromEuler((0.0, 0.0, float(yaw))),
      )
      ids.append(edge_body_id)

    # Curved arc boundary (polyline approximation).
    arc_segments = max(12, int(np.ceil(radius / 0.004)))
    dphi = (np.pi / 2.0) / float(arc_segments)
    arc_seg_length = max(boundary_thickness, radius * dphi)
    arc_vis_id = pyb.createVisualShape(
        pyb.GEOM_BOX,
        halfExtents=(arc_seg_length / 2.0, boundary_thickness / 2.0, half_height_line),
        rgbaColor=rgba_boundary,
    )
    for idx in range(arc_segments):
      phi = (idx + 0.5) * dphi
      px = cx + sx * radius * np.cos(phi)
      py = cy + sy * radius * np.sin(phi)
      tx = -sx * np.sin(phi)
      ty = sy * np.cos(phi)
      yaw = float(np.arctan2(ty, tx))
      arc_body_id = pyb.createMultiBody(
          baseCollisionShapeIndex=-1,
          baseVisualShapeIndex=arc_vis_id,
          basePosition=(float(px), float(py), z),
          baseOrientation=pyb.getQuaternionFromEuler((0.0, 0.0, yaw)),
      )
      ids.append(arc_body_id)
  return ids


def _add_goal_rings(env, reward_calc, radius, segments, line_width, thickness):
  base_env = _get_base_env(env)
  pyb = base_env.pybullet_client
  if not hasattr(reward_calc, "get_task_plan"):
    return []
  task_plan = reward_calc.get_task_plan()
  if not task_plan:
    return []
  if radius <= 0:
    radius = float(_TARGET_DISTANCE.value)
  segments = max(8, int(segments))
  thickness = float(thickness)
  if thickness <= 0:
    thickness = float(line_width) * 0.001
  if thickness <= 0:
    thickness = float(_SPLIT_LINE_WIDTH.value)
  thickness = max(0.001, thickness)
  z = 0.001
  color_map = {
      "red": (1.0, 0.2, 0.2, 0.9),
      "blue": (0.2, 0.4, 1.0, 0.9),
      "green": (0.2, 1.0, 0.4, 0.9),
      "yellow": (1.0, 0.9, 0.2, 0.9),
  }
  seen = set()
  ids = []
  seg_angle = (2.0 * np.pi) / segments
  seg_length = 2.0 * radius * np.sin(seg_angle / 2.0)
  half_thickness = thickness / 2.0
  half_height = 0.0005
  for task in task_plan:
    location = task.get("location", "")
    if location == "staging":
      continue
    target = task.get("target_translation")
    if target is None:
      continue
    key = (float(target[0]), float(target[1]))
    if key in seen:
      continue
    seen.add(key)
    color = task.get("color", "")
    line_color = color_map.get(color, (0.9, 0.9, 0.9, 0.9))
    for i in range(segments):
      angle = (i + 0.5) * seg_angle
      cx = key[0] + radius * np.cos(angle)
      cy = key[1] + radius * np.sin(angle)
      quat = pyb.getQuaternionFromEuler((0.0, 0.0, float(angle + np.pi / 2.0)))
      vis_id = pyb.createVisualShape(
          pyb.GEOM_BOX,
          halfExtents=(seg_length / 2.0, half_thickness, half_height),
          rgbaColor=line_color,
      )
      body_id = pyb.createMultiBody(
          baseCollisionShapeIndex=-1,
          baseVisualShapeIndex=vis_id,
          basePosition=(float(cx), float(cy), z),
          baseOrientation=quat,
      )
      ids.append(body_id)
  return ids


def _region_center(location: str) -> Optional[Tuple[float, float]]:
  x_min = constants.X_MIN + constants.WORKSPACE_BOUNDS_BUFFER
  x_max = constants.X_MAX - constants.WORKSPACE_BOUNDS_BUFFER
  y_min = constants.Y_MIN + constants.WORKSPACE_BOUNDS_BUFFER
  y_max = constants.Y_MAX - constants.WORKSPACE_BOUNDS_BUFFER
  cx = float(constants.CENTER_X)
  cy = float(constants.CENTER_Y)
  if location == "left":
    return ((x_min + cx) / 2.0, cy)
  if location == "right":
    return ((x_max + cx) / 2.0, cy)
  if location == "left_view":
    return (cx, (y_min + cy) / 2.0)
  if location == "right_view":
    return (cx, (y_max + cy) / 2.0)
  if location == "top_left":
    return ((x_min + cx) / 2.0, (y_min + cy) / 2.0)
  if location == "top_right":
    return ((x_max + cx) / 2.0, (y_min + cy) / 2.0)
  if location == "bottom_left":
    return ((x_min + cx) / 2.0, (y_max + cy) / 2.0)
  if location == "bottom_right":
    return ((x_max + cx) / 2.0, (y_max + cy) / 2.0)
  return None


def _add_region_rings(env, color_to_corner_map, radius, segments, thickness):
  base_env = _get_base_env(env)
  pyb = base_env.pybullet_client
  if radius <= 0:
    radius = float(_TARGET_DISTANCE.value)
  segments = max(8, int(segments))
  thickness = float(thickness)
  if thickness <= 0:
    thickness = float(_SPLIT_LINE_WIDTH.value)
  thickness = max(0.001, thickness)
  z = 0.001
  color_map = {
      "red": (1.0, 0.2, 0.2, 0.9),
      "blue": (0.2, 0.4, 1.0, 0.9),
      "green": (0.2, 1.0, 0.4, 0.9),
      "yellow": (1.0, 0.9, 0.2, 0.9),
  }
  seg_angle = (2.0 * np.pi) / segments
  seg_length = 2.0 * radius * np.sin(seg_angle / 2.0)
  half_thickness = thickness / 2.0
  half_height = 0.0005
  ids = []
  for color, location in color_to_corner_map.items():
    center = _region_center(location)
    if center is None:
      continue
    line_color = color_map.get(color, (0.9, 0.9, 0.9, 0.9))
    for i in range(segments):
      angle = (i + 0.5) * seg_angle
      cx = center[0] + radius * np.cos(angle)
      cy = center[1] + radius * np.sin(angle)
      quat = pyb.getQuaternionFromEuler((0.0, 0.0, float(angle + np.pi / 2.0)))
      vis_id = pyb.createVisualShape(
          pyb.GEOM_BOX,
          halfExtents=(seg_length / 2.0, half_thickness, half_height),
          rgbaColor=line_color,
      )
      body_id = pyb.createMultiBody(
          baseCollisionShapeIndex=-1,
          baseVisualShapeIndex=vis_id,
          basePosition=(float(cx), float(cy), z),
          baseOrientation=quat,
      )
      ids.append(body_id)
  return ids


def _add_split_line(env, orientation: str):
  base_env = _get_base_env(env)
  pyb = base_env.pybullet_client
  x_min = constants.X_MIN + constants.WORKSPACE_BOUNDS_BUFFER
  x_max = constants.X_MAX - constants.WORKSPACE_BOUNDS_BUFFER
  y_min = constants.Y_MIN + constants.WORKSPACE_BOUNDS_BUFFER
  y_max = constants.Y_MAX - constants.WORKSPACE_BOUNDS_BUFFER
  z = 0.001
  width = max(0.001, float(_SPLIT_LINE_WIDTH.value))
  alpha = float(_SPLIT_LINE_ALPHA.value)
  color = (1.0, 1.0, 1.0, alpha)
  if orientation == "horizontal":
    half_x = max(0.001, (x_max - x_min) / 2.0)
    half_y = width / 2.0
    center = (constants.CENTER_X, constants.CENTER_Y, z)
    half_extents = (half_x, half_y, 0.0005)
  else:
    half_x = width / 2.0
    half_y = max(0.001, (y_max - y_min) / 2.0)
    center = (constants.CENTER_X, constants.CENTER_Y, z)
    half_extents = (half_x, half_y, 0.0005)
  vis_id = pyb.createVisualShape(
      pyb.GEOM_BOX,
      halfExtents=half_extents,
      rgbaColor=color,
  )
  body_id = pyb.createMultiBody(
      baseCollisionShapeIndex=-1,
      baseVisualShapeIndex=vis_id,
      basePosition=center,
  )
  return [body_id]


def _config_snapshot():
  return {
      "output_dir": _OUTPUT_DIR.value,
      "num_episodes": _NUM_EPISODES.value,
      "max_steps": _MAX_STEPS.value,
      "block_mode": _BLOCK_MODE.value,
      "combo_index": _COMBO_INDEX.value,
      "task_mode": _TASK_MODE.value,
      "planner": _PLANNER.value,
      "occupied_radius": _OCCUPIED_RADIUS.value,
      "keepout_radius": _KEEPOUT_RADIUS.value,
      "staging_spacing": _STAGING_SPACING.value,
      "enable_clearing": _ENABLE_CLEARING.value,
      "clear_target_mode": _CLEAR_TARGET_MODE.value,
      "clear_require_region": _CLEAR_REQUIRE_REGION.value,
      "replan_interval_steps": _REPLAN_INTERVAL_STEPS.value,
      "max_dynamic_inserts": _MAX_DYNAMIC_INSERTS.value,
      "target_color": _TARGET_COLOR.value,
      "target_corner": _TARGET_CORNER.value,
      "instruction_style": _INSTRUCTION_STYLE.value,
      "subtask_label_mode": _SUBTASK_LABEL_MODE.value,
      "second_same_color_into_block": _SECOND_SAME_COLOR_INTO_BLOCK.value,
      "into_block_distance": _INTO_BLOCK_DISTANCE.value,
      "subtask_completion_distance": _SUBTASK_COMPLETION_DISTANCE.value,
      "shuffle_tasks": _SHUFFLE_TASKS.value,
      "max_subtasks": _MAX_SUBTASKS.value,
      "cluster_spacing": _CLUSTER_SPACING.value,
      "line_assignment_mode": _LINE_ASSIGNMENT_MODE.value,
      "line_layout_mode": _LINE_LAYOUT_MODE.value,
      "target_jitter": _TARGET_JITTER.value,
      "target_distance": _TARGET_DISTANCE.value,
      "target_workspace_inset": _TARGET_WORKSPACE_INSET.value,
      "success_mode": _SUCCESS_MODE.value,
      "corner_success_radius": _CORNER_SUCCESS_RADIUS.value,
      "corner_anchor_offset_x": _CORNER_ANCHOR_OFFSET_X.value,
      "corner_anchor_offset_y": _CORNER_ANCHOR_OFFSET_Y.value,
      "region_margin": _REGION_MARGIN.value,
      "cluster_region_padding": _CLUSTER_REGION_PADDING.value,
      "color_corner_map": _COLOR_CORNER_MAP.value,
      "randomize_color_corner_map": _RANDOMIZE_COLOR_CORNER_MAP.value,
      "render_corner_regions": _RENDER_CORNER_REGIONS.value,
      "corner_region_size": _CORNER_REGION_SIZE.value,
      "corner_region_alpha": _CORNER_REGION_ALPHA.value,
      "render_corner_goal_regions": _RENDER_CORNER_GOAL_REGIONS.value,
      "corner_goal_region_margin": _CORNER_GOAL_REGION_MARGIN.value,
      "corner_goal_radius": _CORNER_GOAL_RADIUS.value,
      "corner_goal_region_alpha": _CORNER_GOAL_REGION_ALPHA.value,
      "corner_goal_boundary_alpha": _CORNER_GOAL_BOUNDARY_ALPHA.value,
      "corner_goal_boundary_thickness": _CORNER_GOAL_BOUNDARY_THICKNESS.value,
      "render_goal_rings": _RENDER_GOAL_RINGS.value,
      "goal_ring_radius": _GOAL_RING_RADIUS.value,
      "goal_ring_segments": _GOAL_RING_SEGMENTS.value,
      "goal_ring_line_width": _GOAL_RING_LINE_WIDTH.value,
      "goal_ring_thickness": _GOAL_RING_THICKNESS.value,
      "render_region_rings": _RENDER_REGION_RINGS.value,
      "region_ring_radius": _REGION_RING_RADIUS.value,
      "region_ring_segments": _REGION_RING_SEGMENTS.value,
      "region_ring_thickness": _REGION_RING_THICKNESS.value,
      "render_split_line": _RENDER_SPLIT_LINE.value,
      "split_line_width": _SPLIT_LINE_WIDTH.value,
      "split_line_alpha": _SPLIT_LINE_ALPHA.value,
      "seed": _SEED.value,
      "require_plan": _REQUIRE_PLAN.value,
      "max_init_tries": _MAX_INIT_TRIES.value,
      "use_ee_planner": _USE_EE_PLANNER.value,
      "action_noise_std": _ACTION_NOISE_STD.value,
      "slowdown_freespace": _SLOWDOWN_FREESPACE.value,
      "slowdown_exponent": _SLOWDOWN_EXPONENT.value,
      "render_text": _RENDER_TEXT.value,
      "image_width": _IMAGE_WIDTH.value,
      "image_height": _IMAGE_HEIGHT.value,
      "save_video": _SAVE_VIDEO.value,
      "require_success": _REQUIRE_SUCCESS.value,
      "max_episode_attempts": _MAX_EPISODE_ATTEMPTS.value,
      "freeze_task_order_on_retry": _FREEZE_TASK_ORDER_ON_RETRY.value,
      "randomize_rrt_on_retry": _RANDOMIZE_RRT_ON_RETRY.value,
      "avoid_goal_regions_on_reset": _AVOID_GOAL_REGIONS_ON_RESET.value,
      "avoid_goal_region_margin": _AVOID_GOAL_REGION_MARGIN.value,
      "early_stop_on_success": _EARLY_STOP_ON_SUCCESS.value,
      "video_fps": _VIDEO_FPS.value,
      "video_dir": _VIDEO_DIR.value,
      "debug_target_markers": _DEBUG_TARGET_MARKERS.value,
      "debug_current_target_only": _DEBUG_CURRENT_TARGET_ONLY.value,
      "debug_marker_radius": _DEBUG_MARKER_RADIUS.value,
      "smooth_actions": _SMOOTH_ACTIONS.value,
      "ema_alpha": _EMA_ALPHA.value,
      "smooth_direction_only": _SMOOTH_DIRECTION_ONLY.value,
      "max_action_delta": _MAX_ACTION_DELTA.value,
      "action_scale": _ACTION_SCALE.value,
      "min_action_norm": _MIN_ACTION_NORM.value,
  }


class _ActionPostProcessor:
  """Optional post-processing for oracle actions (scale / EMA / rate-limit).

  This is intentionally lightweight and *off by default* to preserve baseline
  behavior when flags are not set.
  """

  def __init__(self, *, enabled: bool, smooth: bool, ema_alpha: float,
               smooth_direction_only: bool, max_delta: float, action_scale: float,
               min_action_norm: float):
    self._enabled = bool(enabled)
    self._smooth = bool(smooth)
    self._ema_alpha = float(ema_alpha)
    self._smooth_direction_only = bool(smooth_direction_only)
    self._max_delta = float(max_delta)
    self._action_scale = float(action_scale)
    self._min_action_norm = float(min_action_norm)
    self._prev_action = None
    self._prev_dir = None

  @property
  def enabled(self) -> bool:
    return self._enabled

  def reset(self):
    self._prev_action = None
    self._prev_dir = None

  def apply(self, raw_action):
    """Returns the post-processed action as np.float32 array.

    Args:
      raw_action: Oracle-proposed action (array-like).
    """
    action = np.asarray(raw_action, dtype=np.float32)

    if not self._enabled:
      # Preserve baseline stepping by the caller; still return float32 copy for
      # consistent logging/saving.
      self._prev_action = action
      norm = float(np.linalg.norm(action))
      self._prev_dir = action / norm if norm > 1e-8 else None
      return action

    # 1) EMA smoothing (optional).
    if self._smooth and self._smooth_direction_only:
      norm = float(np.linalg.norm(action))
      if norm > 1e-8:
        direction = action / norm
      else:
        direction = action
      if self._prev_dir is not None:
        alpha = np.clip(self._ema_alpha, 0.0, 1.0)
        direction = (1.0 - alpha) * self._prev_dir + alpha * direction
        dnorm = float(np.linalg.norm(direction))
        if dnorm > 1e-8:
          direction = direction / dnorm
      action = direction * norm
      self._prev_dir = direction if norm > 1e-8 else self._prev_dir
    elif self._smooth and self._prev_action is not None:
      alpha = np.clip(self._ema_alpha, 0.0, 1.0)
      action = (1.0 - alpha) * self._prev_action + alpha * action

    # 2) Global scale.
    if self._action_scale != 1.0:
      action = action * self._action_scale

    # 3) Enforce minimum action norm if requested.
    if self._min_action_norm > 0:
      norm = float(np.linalg.norm(action))
      if norm > 1e-8 and norm < self._min_action_norm:
        action = (action / norm) * self._min_action_norm

    # 4) Per-step delta cap (rate limiting).
    if self._max_delta > 0 and self._prev_action is not None:
      delta = action - self._prev_action
      norm = float(np.linalg.norm(delta))
      if norm > self._max_delta:
        # Clip in L2 to preserve direction.
        delta = delta / (norm + 1e-8) * self._max_delta
        action = self._prev_action + delta

    self._prev_action = action
    if self._prev_dir is None:
      norm = float(np.linalg.norm(action))
      self._prev_dir = action / norm if norm > 1e-8 else None
    return action


def _compute_motion_metrics(action_array: np.ndarray) -> dict:
  """Compute simple per-episode motion metrics from executed actions.

  Args:
    action_array: np.ndarray [T, A]

  Returns:
    Dict with mean_action_norm and mean_action_delta_norm.
  """
  if action_array.size == 0:
    return {
        "mean_action_norm": 0.0,
        "mean_action_delta_norm": 0.0,
    }
  norms = np.linalg.norm(action_array, axis=-1)
  mean_action_norm = float(np.mean(norms))
  if action_array.shape[0] < 2:
    mean_action_delta_norm = 0.0
  else:
    deltas = np.diff(action_array, axis=0)
    delta_norms = np.linalg.norm(deltas, axis=-1)
    mean_action_delta_norm = float(np.mean(delta_norms))
  return {
      "mean_action_norm": mean_action_norm,
      "mean_action_delta_norm": mean_action_delta_norm,
  }


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  subtask_label_mode = _normalize_subtask_label_mode(_SUBTASK_LABEL_MODE.value)

  output_dir = _OUTPUT_DIR.value
  os.makedirs(output_dir, exist_ok=True)
  episode_dir = os.path.join(output_dir, "episodes")
  os.makedirs(episode_dir, exist_ok=True)
  video_dir = _VIDEO_DIR.value
  if _SAVE_VIDEO.value:
    if video_dir is None:
      video_dir = os.path.join(output_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    mediapy_lib = None
    try:
      import mediapy as mediapy_lib  # pylint: disable=g-import-not-at-top
    except Exception:  # fallback to imageio if mediapy isn't available
      mediapy_lib = None
  else:
    mediapy_lib = None

  with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
    json.dump(_config_snapshot(), f, indent=2)

  env = _build_env()
  oracle = push_oracle_rrt_slowdown.ObstacleOrientedPushOracleBoard2dRRT(
      env,
      use_ee_planner=_USE_EE_PLANNER.value,
      action_noise_std=_ACTION_NOISE_STD.value,
      slowdown_freespace=_SLOWDOWN_FREESPACE.value,
      slowdown_exponent=_SLOWDOWN_EXPONENT.value,
  )

  successes = 0
  saved_episodes = 0
  attempts = 0
  max_attempts = _MAX_EPISODE_ATTEMPTS.value if _MAX_EPISODE_ATTEMPTS.value > 0 else None
  cached_task_order = None
  while saved_episodes < _NUM_EPISODES.value:
    if max_attempts is not None and attempts >= max_attempts:
      logging.warning(
          "Stopping early after %d attempts; saved %d / %d episodes.",
          attempts,
          saved_episodes,
          _NUM_EPISODES.value,
      )
      break
    attempts += 1
    _seed_rrt_for_attempt(attempts, oracle)
    episode_rng = np.random.RandomState(int(_SEED.value) + attempts * 1009 + saved_episodes * 9173)
    episode_color_corner_map = _sample_color_corner_map_for_episode(episode_rng)
    _set_color_corner_map_on_reward(env, episode_color_corner_map)
    if not _reset_with_plan(env, oracle):
      logging.warning(
          "Skipping attempt %d; could not find valid plan.",
          attempts,
      )
      continue

    base_env = _get_base_env(env)
    reward_calc = getattr(base_env, "_reward_calculator", None)
    initial_task_plan = reward_calc.get_task_plan() if hasattr(reward_calc, "get_task_plan") else []
    label_formatter = _SubtaskLabelFormatter(
        mode=subtask_label_mode,
        task_plan=initial_task_plan,
    )
    if _FREEZE_TASK_ORDER_ON_RETRY.value and reward_calc is not None:
      if cached_task_order is None:
        cached_task_order = _extract_task_order(reward_calc)
      else:
        _apply_task_order(reward_calc, cached_task_order)
    prev_task_idx = getattr(reward_calc, "current_task_index", None)
    prev_subtask_id = getattr(reward_calc, "current_subtask_id", None)
    prev_signature = (prev_subtask_id if prev_subtask_id is not None else prev_task_idx)
    marker_ids = []
    corner_region_ids = []
    corner_goal_region_ids = []
    split_line_ids = []
    goal_ring_ids = []
    region_ring_ids = []
    if _DEBUG_TARGET_MARKERS.value:
      marker_ids = _add_target_markers(
          env, reward_calc, _DEBUG_CURRENT_TARGET_ONLY.value)
    if _RENDER_CORNER_REGIONS.value:
      corner_region_ids = _add_corner_regions(env, episode_color_corner_map)
    if _RENDER_CORNER_GOAL_REGIONS.value:
      if str(_SUCCESS_MODE.value).lower() == "corner_radius":
        corner_goal_region_ids = _add_corner_goal_quarter_circles(
            env,
            episode_color_corner_map,
            radius=_effective_corner_goal_radius(),
        )
      else:
        corner_goal_margin = float(_CORNER_GOAL_REGION_MARGIN.value)
        if corner_goal_margin <= 0:
          corner_goal_margin = float(_REGION_MARGIN.value)
        corner_goal_region_ids = _add_corner_goal_regions(
            env,
            episode_color_corner_map,
            margin=corner_goal_margin,
        )
    if _RENDER_SPLIT_LINE.value:
      color_to_corner_map = episode_color_corner_map
      orientation = "horizontal" if any(
          loc in ("left_view", "right_view") for loc in color_to_corner_map.values()
      ) else "vertical"
      split_line_ids = _add_split_line(env, orientation)
    if _RENDER_GOAL_RINGS.value:
      goal_ring_ids = _add_goal_rings(
          env,
          reward_calc,
          _GOAL_RING_RADIUS.value,
          _GOAL_RING_SEGMENTS.value,
          _GOAL_RING_LINE_WIDTH.value,
          _GOAL_RING_THICKNESS.value,
      )
    if _RENDER_REGION_RINGS.value:
      region_ring_ids = _add_region_rings(
          env,
          episode_color_corner_map,
          _REGION_RING_RADIUS.value,
          _REGION_RING_SEGMENTS.value,
          _REGION_RING_THICKNESS.value,
      )

    states = []
    oracle_actions = []
    actions = []
    rewards = []
    dones = []
    subtask_indices = []
    subtask_ids = []
    subtask_types = []
    subtask_blocks = []
    subtask_labels = []
    last_subtask_label = ""
    frames = [] if _SAVE_VIDEO.value else None

    motion_enabled = (
        _SMOOTH_ACTIONS.value or
        _MAX_ACTION_DELTA.value > 0 or
        abs(_ACTION_SCALE.value - 1.0) > 1e-8
    )
    action_post = _ActionPostProcessor(
        enabled=motion_enabled,
        smooth=_SMOOTH_ACTIONS.value,
        ema_alpha=_EMA_ALPHA.value,
        smooth_direction_only=_SMOOTH_DIRECTION_ONLY.value,
        max_delta=_MAX_ACTION_DELTA.value,
        action_scale=_ACTION_SCALE.value,
        min_action_norm=_MIN_ACTION_NORM.value,
    )

    done = False
    step = 0
    if _SAVE_VIDEO.value:
      frames.append(env.render())
    while step < _MAX_STEPS.value and not done:
      state = env.compute_state()
      curr_task_idx = getattr(reward_calc, "current_task_index", None)
      curr_subtask_id = getattr(reward_calc, "current_subtask_id", None)
      curr_signature = (curr_subtask_id if curr_subtask_id is not None else curr_task_idx)
      if curr_signature is not None and curr_signature != prev_signature:
        oracle.reset()
        # Avoid carrying smoothing state across subtasks.
        action_post.reset()
        prev_signature = curr_signature
        prev_task_idx = curr_task_idx
        if _DEBUG_TARGET_MARKERS.value and _DEBUG_CURRENT_TARGET_ONLY.value:
          _clear_markers(base_env.pybullet_client, marker_ids)
          marker_ids = _add_target_markers(env, reward_calc, True)
      raw_action = oracle._get_action_for_block_target(state)
      applied_action = action_post.apply(raw_action)

      # Preserve exact baseline stepping when motion post-processing is disabled.
      step_action = applied_action if action_post.enabled else raw_action
      _, reward, done = _unwrap_step(env.step(step_action))
      if _EARLY_STOP_ON_SUCCESS.value and not done and reward_calc is not None:
        if hasattr(reward_calc, "is_success"):
          post_state = env.compute_state(request_task_update=False)
          if reward_calc.is_success(post_state):
            done = True

      states.append(state)
      oracle_actions.append(np.asarray(raw_action, dtype=np.float32))
      actions.append(np.asarray(applied_action, dtype=np.float32))
      rewards.append(reward)
      dones.append(done)
      if curr_task_idx is None:
        subtask_indices.append(-1)
      else:
        subtask_indices.append(curr_task_idx)

      # Record explicit subtask labels when available.
      subtask_ids.append(int(curr_subtask_id) if curr_subtask_id is not None else -1)
      subtask_types.append(getattr(reward_calc, "current_subtask_type", ""))
      subtask_blocks.append(getattr(reward_calc, "current_subtask_block", ""))
      if reward_calc is not None and curr_task_idx is not None and curr_task_idx >= 0:
        task_plan = reward_calc.get_task_plan() if hasattr(reward_calc, "get_task_plan") else []
        label_formatter.observe_task_plan(task_plan)
        if curr_task_idx < len(task_plan):
          label = label_formatter.format(task_plan[curr_task_idx])
          if label:
            last_subtask_label = label
      subtask_labels.append(last_subtask_label)
      if _SAVE_VIDEO.value:
        frames.append(env.render())

      step += 1

    post_state = env.compute_state(request_task_update=False)
    success = _compute_success(post_state, reward_calc)

    if _REQUIRE_SUCCESS.value and not success:
      if _DEBUG_TARGET_MARKERS.value and marker_ids:
        _clear_markers(base_env.pybullet_client, marker_ids)
      if corner_region_ids:
        _clear_markers(base_env.pybullet_client, corner_region_ids)
      if corner_goal_region_ids:
        _clear_markers(base_env.pybullet_client, corner_goal_region_ids)
      if split_line_ids:
        _clear_markers(base_env.pybullet_client, split_line_ids)
      if goal_ring_ids:
        _clear_markers(base_env.pybullet_client, goal_ring_ids)
      if region_ring_ids:
        _clear_markers(base_env.pybullet_client, region_ring_ids)
      logging.info(
          "Skipping attempt %d (not successful).",
          attempts,
      )
      continue

    if success:
      successes += 1

    payload = {
        "states": np.array(states, dtype=object),
        "oracle_actions": (
            np.stack(oracle_actions) if oracle_actions else
            np.zeros((0, 2), dtype=np.float32)
        ),
        "actions": np.stack(actions) if actions else np.zeros((0, 2), dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "dones": np.array(dones, dtype=np.bool_),
        "subtask_indices": np.array(subtask_indices, dtype=np.int32),
        "subtask_ids": np.array(subtask_ids, dtype=np.int32),
        "subtask_types": np.array(subtask_types, dtype=object),
        "subtask_blocks": np.array(subtask_blocks, dtype=object),
        "subtask_labels": np.array(subtask_labels, dtype=object),
    }

    metadata = {
        "episode_index": saved_episodes,
        "attempt_index": attempts - 1,
        "num_steps": len(actions),
        **_compute_motion_metrics(np.stack(actions) if actions else np.zeros((0, 2), dtype=np.float32)),
        "success": success,
        "instruction": getattr(reward_calc, "instruction", ""),
        "task_mode": getattr(reward_calc, "task_mode", ""),
        "color_corner_map_assignment": dict(episode_color_corner_map),
        "planner": _PLANNER.value,
        "task_plan": reward_calc.get_task_plan() if hasattr(reward_calc, "get_task_plan") else [],
    }

    _save_episode(episode_dir, saved_episodes, payload, metadata)
    if _SAVE_VIDEO.value and frames:
      video_path = os.path.join(video_dir, f"episode_{saved_episodes:06d}.mp4")
      if mediapy_lib is not None:
        mediapy_lib.write_video(video_path, frames, fps=_VIDEO_FPS.value)
      else:
        import imageio.v2 as imageio  # pylint: disable=g-import-not-at-top
        with imageio.get_writer(video_path, fps=_VIDEO_FPS.value) as writer:
          for frame in frames:
            writer.append_data(frame)
    if _DEBUG_TARGET_MARKERS.value and marker_ids:
      base_env = _get_base_env(env)
      _clear_markers(base_env.pybullet_client, marker_ids)
    if corner_region_ids:
      base_env = _get_base_env(env)
      _clear_markers(base_env.pybullet_client, corner_region_ids)
    if corner_goal_region_ids:
      base_env = _get_base_env(env)
      _clear_markers(base_env.pybullet_client, corner_goal_region_ids)
    if split_line_ids:
      base_env = _get_base_env(env)
      _clear_markers(base_env.pybullet_client, split_line_ids)
    if goal_ring_ids:
      base_env = _get_base_env(env)
      _clear_markers(base_env.pybullet_client, goal_ring_ids)
    if region_ring_ids:
      base_env = _get_base_env(env)
      _clear_markers(base_env.pybullet_client, region_ring_ids)
    logging.info(
        "Episode %d (attempt %d) finished with %d steps.",
        saved_episodes,
        attempts,
        len(actions),
    )
    saved_episodes += 1
    cached_task_order = None

  logging.info(
      "Finished. Successes: %d / %d (attempts: %d).",
      successes,
      saved_episodes,
      attempts,
  )


if __name__ == "__main__":
  app.run(main)
