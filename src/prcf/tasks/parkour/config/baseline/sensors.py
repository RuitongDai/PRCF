"""Task-local period-five camera sensor for true 10 Hz depth capture."""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.sensor import CameraSensor, CameraSensorCfg


class PeriodicCameraSensor(CameraSensor):
  """Camera whose render cadence is controlled by the task environment."""

  def __init__(self, cfg: "PeriodicCameraSensorCfg") -> None:
    super().__init__(cfg)
    if cfg.update_period_steps < 1:
      raise ValueError("update_period_steps must be a positive integer")
    self.update_period_steps = int(cfg.update_period_steps)
    self.sample_id = 0
    self._sense_call_count = 0

  def should_render(self) -> bool:
    return self._sense_call_count % self.update_period_steps == 0

  def finish_sense(self, rendered: bool) -> None:
    if rendered:
      self.sample_id += 1
    self._sense_call_count += 1


@dataclass
class PeriodicCameraSensorCfg(CameraSensorCfg):
  """Camera config with an environment-step render period."""

  update_period_steps: int = 5

  def build(self) -> PeriodicCameraSensor:
    return PeriodicCameraSensor(self)


__all__ = ["PeriodicCameraSensor", "PeriodicCameraSensorCfg"]
