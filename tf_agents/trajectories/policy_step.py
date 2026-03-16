"""Minimal PolicyStep shim used by the scripted-data bundle."""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass
class PolicyStep:
  action: Any = None
  state: Any = ()
  info: Any = ()
