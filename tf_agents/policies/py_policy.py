"""Minimal PyPolicy shim used by the scripted-data bundle."""


class PyPolicy:
  """No-op base class matching the constructor shape used by the oracle."""

  def __init__(self, time_step_spec=None, action_spec=None):
    self._time_step_spec = time_step_spec
    self._action_spec = action_spec
