"""Minimal TimeStep shim used by the scripted-data bundle."""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass
class TimeStep:
  step_type: Any = None
  reward: Any = None
  discount: Any = None
  observation: Any = None

  def is_first(self):
    return self.step_type == 0

  def is_last(self):
    return self.step_type == 2
