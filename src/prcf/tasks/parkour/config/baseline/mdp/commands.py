"""Perlin omnidirectional commands plus target navigation elsewhere."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import wrap_to_pi


class ParkourTargetVelocityCommand(CommandTerm):
  """Sample full planar motion on Perlin and navigate targets elsewhere."""

  cfg: "ParkourTargetVelocityCommandCfg"

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.terrain = env.scene.terrain
    generator = env.cfg.scene.terrain.terrain_generator
    if generator is None:
      raise RuntimeError("ParkourTargetVelocityCommand requires generated terrain.")
    self._terrain_names = tuple(generator.sub_terrains)
    self._rough_id = self._terrain_names.index(cfg.rough_terrain_name) if cfg.rough_terrain_name in self._terrain_names else -1

    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.target_pos_b = torch.zeros_like(self.target_pos_w)
    self.max_command_b = torch.zeros_like(self.vel_command_b)
    self.omni_command_b = torch.zeros_like(self.vel_command_b)
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.is_standing_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.initial_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.previous_pos_w = torch.zeros_like(self.initial_pos_w)
    self.cumulative_travel = torch.zeros(self.num_envs, device=self.device)
    self._reference_pending = torch.ones(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    for name in (
      "error_vel_xy", "error_vel_yaw", "tracking_exp_vel_xy",
      "tracking_exp_vel_yaw", "moving_tracking_linear_sum",
      "moving_tracking_yaw_sum", "moving_steps", "cumulative_travel",
      "distance", "target_progress",
    ):
      self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self):
    return self.vel_command_b

  @property
  def rough_terrain(self):
    return self.terrain.terrain_types == self._rough_id

  @property
  def complex_terrain(self):
    return ~self.rough_terrain

  @property
  def route(self):
    return self.complex_terrain.float()

  def reset(self, env_ids):
    extras = super().reset(env_ids)
    self.cumulative_travel[env_ids] = 0.0
    self._reference_pending[env_ids] = True
    return extras

  def _resample_command(self, env_ids):
    if len(env_ids) == 0:
      return
    if self.cfg.fixed_command is not None:
      self.vel_command_b[env_ids] = torch.tensor(
        self.cfg.fixed_command, device=self.device, dtype=torch.float
      )
      self.is_standing_env[env_ids] = False
      return
    rough = self.rough_terrain[env_ids]
    rough_ids = env_ids[rough]
    complex_ids = env_ids[~rough]

    # Keep one body-frame omnidirectional command for the full Perlin episode.
    # The heading controller converts a sampled world heading into command wz.
    if len(rough_ids):
      # Advanced indexing returns a copy in PyTorch, so calling ``uniform_``
      # directly on ``tensor[rough_ids, column]`` would leave the stored
      # command at zero.  Sample into temporary tensors and assign them back.
      self.omni_command_b[rough_ids, 0] = torch.empty(
        len(rough_ids), device=self.device
      ).uniform_(*self.cfg.omni_lin_vel_x)
      self.omni_command_b[rough_ids, 1] = torch.empty(
        len(rough_ids), device=self.device
      ).uniform_(*self.cfg.omni_lin_vel_y)
      self.omni_command_b[rough_ids, 2] = 0.0
      self.heading_target[rough_ids] = torch.empty(
        len(rough_ids), device=self.device
      ).uniform_(*self.cfg.omni_heading)
      self.time_left[rough_ids] = self.cfg.episode_length_s

    # Obstacles and stairs have a forward route ending near the far platform.
    if len(complex_ids):
      local = torch.zeros(len(complex_ids), 3, device=self.device)
      local[:, 0] = self.cfg.complex_target_x
      self.target_pos_w[complex_ids] = self._env.scene.env_origins[complex_ids] + local
      self.max_command_b[complex_ids, 0] = torch.empty(
        len(complex_ids), device=self.device
      ).uniform_(*self.cfg.complex_lin_vel_x)
      self.max_command_b[complex_ids, 1:] = 0.0
    # Perlin is the all-direction locomotion comparison and must accumulate
    # travel for its curriculum gate.  Preserve the Camera baseline's random
    # standing episodes only on target-navigation terrains.
    self.is_standing_env[env_ids] = False
    if len(complex_ids):
      self.is_standing_env[complex_ids] = (
        torch.rand(len(complex_ids), device=self.device)
        < self.cfg.rel_standing_envs
      )

  def _update_command(self, env_ids=None):
    if self.cfg.fixed_command is not None:
      self.vel_command_b[:] = torch.tensor(
        self.cfg.fixed_command, device=self.device, dtype=torch.float
      )
      return
    rough = self.rough_terrain
    complex_mask = ~rough
    heading = self.robot.data.heading_w

    self.vel_command_b[rough, :2] = self.omni_command_b[rough, :2]
    omni_heading_error = wrap_to_pi(self.heading_target - heading)
    self.vel_command_b[rough, 2] = torch.clamp(
      omni_heading_error[rough] * self.cfg.omni_heading_control_stiffness,
      min=self.cfg.omni_ang_vel_z[0], max=self.cfg.omni_ang_vel_z[1],
    )

    delta_w = self.target_pos_w - self.robot.data.root_link_pos_w
    distance = torch.linalg.vector_norm(delta_w[:, :2], dim=1)
    c, s = torch.cos(heading), torch.sin(heading)
    self.target_pos_b[:, 0] = c * delta_w[:, 0] + s * delta_w[:, 1]
    self.target_pos_b[:, 1] = -s * delta_w[:, 0] + c * delta_w[:, 1]
    self.target_pos_b[:, 2] = delta_w[:, 2]

    desired_x = self.target_pos_b[:, 0] * self.cfg.velocity_control_stiffness
    self.vel_command_b[complex_mask, 0] = torch.minimum(
      torch.clamp_min(desired_x[complex_mask], 0.0),
      self.max_command_b[complex_mask, 0],
    )
    self.vel_command_b[complex_mask, 1] = 0.0
    desired_heading = torch.atan2(delta_w[:, 1], delta_w[:, 0])
    heading_error = wrap_to_pi(desired_heading - heading)
    self.vel_command_b[complex_mask, 2] = torch.clamp(
      heading_error[complex_mask] * self.cfg.heading_control_stiffness,
      min=self.cfg.ang_vel_z[0], max=self.cfg.ang_vel_z[1],
    )
    active = distance > self.cfg.target_distance_threshold
    self.vel_command_b[complex_mask & ~active] = 0.0
    self.vel_command_b[self.is_standing_env] = 0.0

  def _update_metrics(self):
    steps = max(float(self._env.max_episode_length), 1.0)
    linear_error = (
      self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2]
    )
    yaw_error = self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2]
    self.metrics["error_vel_xy"] += (
      torch.linalg.vector_norm(linear_error, dim=1) / steps
    )
    self.metrics["error_vel_yaw"] += torch.abs(yaw_error) / steps
    linear = torch.exp(-torch.sum(linear_error.square(), dim=1) / 0.5**2)
    yaw = torch.exp(-yaw_error.square() / 0.5**2)
    self.metrics["tracking_exp_vel_xy"] += linear / steps
    self.metrics["tracking_exp_vel_yaw"] += yaw / steps
    moving = torch.linalg.vector_norm(self.vel_command_b, dim=1) > 0.1
    self.metrics["moving_tracking_linear_sum"] += linear * moving
    self.metrics["moving_tracking_yaw_sum"] += yaw * moving
    self.metrics["moving_steps"] += moving.float()
    self.metrics["cumulative_travel"].copy_(self.cumulative_travel)
    displacement = self.robot.data.root_link_pos_w - self.initial_pos_w
    # Complex routes always point along the terrain patch's +x axis. Counting
    # only +x progress prevents lateral wandering or a 180-degree reversal from
    # satisfying the promotion guard.
    forward_progress = displacement[:, 0].clamp_min(0.0)
    distance = torch.where(
      self.complex_terrain, forward_progress, self.cumulative_travel
    )
    self.metrics["distance"].copy_(distance)
    progress_scale = torch.where(
      self.complex_terrain,
      torch.full_like(distance, self.cfg.complex_target_x),
      torch.full_like(distance, self.cfg.perlin_min_travel),
    )
    self.metrics["target_progress"].copy_((distance / progress_scale).clamp(0.0, 1.5))

  def _debug_vis_impl(self, visualizer):
    for env_idx in visualizer.get_env_indices(self.num_envs):
      start = self.robot.data.root_link_pos_w[env_idx].detach().cpu().numpy().copy()
      start[2] += 0.15
      if bool(self.rough_terrain[env_idx]):
        heading = self.robot.data.heading_w[env_idx]
        command = self.vel_command_b[env_idx, :2]
        c, s = torch.cos(heading), torch.sin(heading)
        delta = torch.stack((
          c * command[0] - s * command[1],
          s * command[0] + c * command[1],
        )).detach().cpu().numpy()
        target = start.copy()
        target[:2] += delta
        visualizer.add_arrow(
          start, target, color=(1.0, 0.65, 0.1, 0.9), width=0.018
        )
        continue
      target = self.target_pos_w[env_idx].detach().cpu().numpy().copy()
      target[2] += 0.08
      distance = torch.linalg.vector_norm(
        self.target_pos_w[env_idx, :2] - self.robot.data.root_link_pos_w[env_idx, :2]
      )
      active = bool(distance > self.cfg.target_distance_threshold)
      color = (0.1, 0.9, 0.2, 1.0) if active else (0.95, 0.2, 0.1, 1.0)
      visualizer.add_sphere(target, radius=0.07, color=color, label="target")
      visualizer.add_arrow(start, target, color=(1.0, 0.65, 0.1, 0.9), width=0.018)


@dataclass(kw_only=True)
class ParkourTargetVelocityCommandCfg(CommandTermCfg):
  entity_name: str
  fixed_command: tuple[float, float, float] | None = None
  rough_terrain_name: str = "perlin_rough"
  episode_length_s: float = 20.0
  omni_lin_vel_x: tuple[float, float] = (-0.6, 0.8)
  omni_lin_vel_y: tuple[float, float] = (-0.6, 0.6)
  omni_heading: tuple[float, float] = (-3.14, 3.14)
  omni_heading_control_stiffness: float = 0.5
  omni_ang_vel_z: tuple[float, float] = (-1.5, 1.5)
  perlin_min_travel: float = 1.5
  complex_target_x: float = 3.7
  complex_lin_vel_x: tuple[float, float] = (0.45, 0.8)
  ang_vel_z: tuple[float, float] = (-1.0, 1.0)
  velocity_control_stiffness: float = 2.0
  heading_control_stiffness: float = 2.0
  rel_standing_envs: float = 0.05
  target_distance_threshold: float = 0.4

  def build(self, env):
    return ParkourTargetVelocityCommand(self, env)


__all__ = ["ParkourTargetVelocityCommandCfg"]
