"""Minimal GymWrapper shim used by the scripted-data bundle."""

from __future__ import annotations


class GymWrapper:
  """Thin wrapper around a gym.Env with the methods used by the bundle."""

  def __init__(self, gym_env, auto_reset=True):
    self.gym = gym_env
    self._auto_reset = auto_reset

  def reset(self):
    return self.gym.reset()

  def step(self, action):
    return self.gym.step(action)

  def render(self, *args, **kwargs):
    return self.gym.render(*args, **kwargs)

  def compute_state(self, *args, **kwargs):
    return self.gym.compute_state(*args, **kwargs)

  def get_control_frequency(self):
    return self.gym.get_control_frequency()

  def action_spec(self):
    return self.gym.action_space

  def time_step_spec(self):
    return None

  def __getattr__(self, name):
    return getattr(self.gym, name)
