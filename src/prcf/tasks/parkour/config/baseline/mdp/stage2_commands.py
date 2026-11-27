"""Stage-two navigation with a simple mid-stair zero-command event."""

from dataclasses import dataclass

import torch

from mjlab.utils.lab_api.math import wrap_to_pi

from .box_traversal import (
  BoxTaskGeometry, CLIMB_PELVIS_CLEARANCE, CLIMB_TARGET_OFFSET, CLIMB_TARGET_RADIUS,
)
from .commands import ParkourTargetVelocityCommand, ParkourTargetVelocityCommandCfg


class Stage2TargetVelocityCommand(ParkourTargetVelocityCommand):
  """Keep the stage-two routes and optionally zero commands mid-stair."""

  FORWARD_CENTER = 0
  FORWARD_DIAGONAL = 1
  BACKWARD_DIAGONAL = 2
  cfg: "Stage2TargetVelocityCommandCfg"

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self._gap_id = self._terrain_names.index(cfg.gap_terrain_name) if cfg.gap_terrain_name in self._terrain_names else -1
    self._no_backward_stair_ids = tuple(
      self._terrain_names.index(name) for name in cfg.stair_no_backward_terrain_names
      if name in self._terrain_names
    )
    self.box_course = BoxTaskGeometry(env)
    self.navigation_mode = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    self.target_offset_b = torch.zeros_like(self.target_pos_w)
    self.route_start_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.route_direction_w = torch.zeros_like(self.route_start_w)
    self.route_length = torch.ones(self.num_envs, device=self.device)
    self.heading_error = torch.zeros(self.num_envs, device=self.device)
    self.zero_stop_enabled = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    self.zero_stop_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    self.zero_stop_trigger_progress = torch.zeros(self.num_envs, device=self.device)
    for key in ("zero_stop_triggered", "zero_stop_seconds"):
      self.metrics[key] = torch.zeros(self.num_envs, device=self.device)

  @property
  def square_gaps(self):
    return self.terrain.terrain_types == self._gap_id

  @property
  def stairs(self):
    return self.complex_terrain & ~self.square_gaps & ~self.boxes

  @property
  def boxes(self):
    return self.box_course.up | self.box_course.down

  @property
  def backward_diagonal(self):
    return self.stairs & (self.navigation_mode == self.BACKWARD_DIAGONAL)

  def reset(self, env_ids):
    extras = super().reset(env_ids)
    self.zero_stop_active[env_ids] = False
    return extras

  def compute(self, dt, env_ids=None) -> None:
    # Anchor each route to the actual pose *after* reset events, not the tile
    # origin. This makes x=-4 a true backwards goal even with spawn offsets.
    pending = self._reference_pending.nonzero().flatten()
    if len(pending):
      position = self.robot.data.root_link_pos_w[pending]
      self.initial_pos_w[pending] = position
      self.previous_pos_w[pending] = position
      self.route_start_w[pending] = position[:, :2]
      nav_ids = pending[(self.complex_terrain & ~self.boxes)[pending]]
      if len(nav_ids):
        heading = self.robot.data.heading_w[nav_ids]
        offset = self.target_offset_b[nav_ids]
        self.target_pos_w[nav_ids] = self.robot.data.root_link_pos_w[nav_ids]
        self.target_pos_w[nav_ids, 0] += torch.cos(heading) * offset[:, 0] - torch.sin(heading) * offset[:, 1]
        self.target_pos_w[nav_ids, 1] += torch.sin(heading) * offset[:, 0] + torch.cos(heading) * offset[:, 1]
      route = self.target_pos_w[pending, :2] - position[:, :2]
      self.route_length[pending] = torch.linalg.vector_norm(route, dim=1).clamp_min(1.0e-6)
      self.route_direction_w[pending] = route / self.route_length[pending, None]
      self._reference_pending[pending] = False
    current = self.robot.data.root_link_pos_w
    ids = slice(None) if env_ids is None else env_ids
    self.cumulative_travel[ids] += torch.linalg.vector_norm(current[ids, :2] - self.previous_pos_w[ids, :2], dim=1)
    self.previous_pos_w[ids] = current[ids]
    self._update_metrics()
    self._update_zero_stop(dt, env_ids)
    self.time_left[ids] -= dt
    resample = (self.time_left <= 0).nonzero().flatten()
    if len(resample):
      self._resample(resample)
    self._update_command()

  def _resample_command(self, env_ids):
    if len(env_ids) == 0:
      return
    self.zero_stop_enabled[env_ids] = False
    self.zero_stop_active[env_ids] = False
    self.zero_stop_trigger_progress[env_ids] = 0.0
    self.navigation_mode[env_ids] = self.FORWARD_CENTER
    super()._resample_command(env_ids)
    box_ids = env_ids[self.boxes[env_ids]]
    self.is_standing_env[box_ids] = False
    front, rear, _ = self.box_course.bounds(box_ids).unbind(dim=1)
    up = self.box_course.up[box_ids]
    self.target_pos_w[box_ids] = self._env.scene.env_origins[box_ids]
    self.target_pos_w[box_ids, 0] += torch.where(up, front + CLIMB_TARGET_OFFSET, rear + 0.50)
    self.target_pos_w[box_ids, 2] += torch.where(up, self.box_course.height(box_ids), 0.0)
    if self.cfg.fixed_command is not None:
      return
    nav_ids = env_ids[self.complex_terrain[env_ids]]
    if not len(nav_ids):
      return
    offsets = torch.zeros(len(nav_ids), 3, device=self.device)
    offsets[:, 0] = self.cfg.complex_target_x
    gaps = self.square_gaps[nav_ids]
    gap_ids = nav_ids[gaps]
    if len(gap_ids):
      diagonal = torch.rand(len(gap_ids), device=self.device) < self.cfg.square_gap_diagonal_fraction
      self.navigation_mode[gap_ids[diagonal]] = self.FORWARD_DIAGONAL
      rows = gaps.nonzero().flatten()[diagonal]
      if len(rows):
        offsets[rows, 1] = torch.empty(len(rows), device=self.device).uniform_(
          -self.cfg.square_gap_lateral_target_range, self.cfg.square_gap_lateral_target_range
        )
    stair_mask = self.stairs[nav_ids]
    stair_ids = nav_ids[stair_mask]
    if len(stair_ids):
      sample = torch.rand(len(stair_ids), device=self.device)
      diagonal = ((sample >= self.cfg.stair_forward_center_fraction)
                  & (sample < self.cfg.stair_forward_center_fraction + self.cfg.stair_forward_diagonal_fraction))
      backward = sample >= (self.cfg.stair_forward_center_fraction + self.cfg.stair_forward_diagonal_fraction)
      # Reassign only these terrains' backward draws to the initialized
      # FORWARD_CENTER mode. Preserve the existing 30% diagonal draws.
      for terrain_id in self._no_backward_stair_ids:
        backward &= self.terrain.terrain_types[stair_ids] != terrain_id
      self.navigation_mode[stair_ids[diagonal]] = self.FORWARD_DIAGONAL
      self.navigation_mode[stair_ids[backward]] = self.BACKWARD_DIAGONAL
      rows = stair_mask.nonzero().flatten()
      # Both forward-diagonal and backward routes need the lateral target
      # offset; otherwise the 20% backward mode silently becomes straight back.
      diagonal_rows = rows[diagonal | backward]
      if len(diagonal_rows):
        offsets[diagonal_rows, 1] = torch.empty(len(diagonal_rows), device=self.device).uniform_(
          -self.cfg.stair_lateral_target_range, self.cfg.stair_lateral_target_range
        )
      offsets[rows[backward], 0] = -self.cfg.complex_target_x
      if self.cfg.zero_stop_probability > 0:
        selected = stair_ids[torch.rand(len(stair_ids), device=self.device) < self.cfg.zero_stop_probability]
        self.zero_stop_enabled[selected] = True
        if len(selected):
          self.zero_stop_trigger_progress[selected] = torch.empty(len(selected), device=self.device).uniform_(
            *self.cfg.zero_stop_progress_range
          )
    self.target_offset_b[nav_ids] = offsets
    # Only reset uses command resampling; route metrics will be anchored on the
    # next compute call once the robot pose is final.
    self._reference_pending[nav_ids] = True

  def _update_command(self, env_ids=None):
    super()._update_command(env_ids)
    if self.cfg.fixed_command is not None:
      return
    position = self.robot.data.root_link_pos_w
    delta = self.target_pos_w[:, :2] - position[:, :2]
    heading = self.robot.data.heading_w
    desired = torch.atan2(delta[:, 1], delta[:, 0])
    desired = torch.where(self.backward_diagonal, wrap_to_pi(desired + torch.pi), desired)
    up = self.box_course.up
    climbing = up & (position[:, 2] < self.target_pos_w[:, 2] + CLIMB_PELVIS_CLEARANCE)
    # Aim toward the box centerline while approaching and climbing.
    desired = torch.where(climbing, torch.atan2(delta[:, 1], delta[:, 0].clamp_min(0.35)), desired)
    self.heading_error.copy_(wrap_to_pi(desired - heading))
    distance = torch.linalg.vector_norm(delta, dim=1)
    backward = self.backward_diagonal
    if backward.any():
      desired_x = (self.target_pos_b[:, 0] * self.cfg.velocity_control_stiffness).clamp(max=0.0)
      self.vel_command_b[backward, 0] = torch.maximum(desired_x[backward], -self.max_command_b[backward, 0])
      self.vel_command_b[backward, 2] = torch.clamp(
        self.heading_error[backward] * self.cfg.heading_control_stiffness,
        min=self.cfg.ang_vel_z[0], max=self.cfg.ang_vel_z[1],
      )
      self.vel_command_b[backward & (distance <= self.cfg.target_distance_threshold)] = 0.0
    desired_x = self.target_pos_b[:, 0] * self.cfg.velocity_control_stiffness
    aligned_climb = climbing & (self.heading_error.abs() <= torch.pi / 3)
    desired_x = torch.where(aligned_climb, desired_x.clamp_min(0.20), desired_x.clamp_min(0.0))
    self.vel_command_b[up, 0] = torch.minimum(desired_x[up], self.max_command_b[up, 0])
    self.vel_command_b[up, 2] = (self.heading_error[up] * self.cfg.heading_control_stiffness).clamp(*self.cfg.ang_vel_z)
    self.vel_command_b[up & ~climbing & (distance <= CLIMB_TARGET_RADIUS)] = 0.0
    self.vel_command_b[self.is_standing_env | self.zero_stop_active] = 0.0

  def _update_metrics(self):
    super()._update_metrics()
    displacement = self.robot.data.root_link_pos_w[:, :2] - self.route_start_w
    directed = torch.sum(displacement * self.route_direction_w, dim=1).clamp_min(0.0)
    progress = (directed / self.route_length).clamp(0.0, 1.5)
    self.metrics["target_progress"].copy_(torch.where(self.complex_terrain, progress, self.metrics["target_progress"]))
    self.metrics["distance"].copy_(torch.where(self.complex_terrain, directed, self.metrics["distance"]))

  def _update_zero_stop(self, dt: float | torch.Tensor, env_ids=None) -> None:
    if self.cfg.zero_stop_probability <= 0 or self.cfg.fixed_command is not None:
      return
    elapsed = torch.zeros_like(self.time_left)
    elapsed[slice(None) if env_ids is None else env_ids] = dt
    trigger = (
      self.zero_stop_enabled
      & self.stairs
      & ~self.zero_stop_active
      & (self.metrics["target_progress"] >= self.zero_stop_trigger_progress)
      & (elapsed > 0)
    )
    trigger_ids = trigger.nonzero().flatten()
    if len(trigger_ids):
      self.zero_stop_active[trigger_ids] = True
      self.zero_stop_enabled[trigger_ids] = False
      self.metrics["zero_stop_triggered"][trigger_ids] += 1.0
    self.metrics["zero_stop_seconds"] += elapsed * self.zero_stop_active


@dataclass(kw_only=True)
class Stage2TargetVelocityCommandCfg(ParkourTargetVelocityCommandCfg):
  gap_terrain_name: str = "square_gaps"
  complex_target_x: float = 4.0
  target_distance_threshold: float = 0.35
  resampling_time_range: tuple[float, float] = (20.0, 20.0)
  stair_forward_center_fraction: float = 0.50
  stair_forward_diagonal_fraction: float = 0.30
  stair_backward_diagonal_fraction: float = 0.20
  stair_no_backward_terrain_names: tuple[str, ...] = ("stairs_up", "stairs_up_high")
  stair_lateral_target_range: float = 1.50
  square_gap_diagonal_fraction: float = 0.30
  square_gap_lateral_target_range: float = 1.00
  zero_stop_probability: float = 0.25
  zero_stop_progress_range: tuple[float, float] = (0.20, 0.80)

  def build(self, env):
    total = (self.stair_forward_center_fraction + self.stair_forward_diagonal_fraction
             + self.stair_backward_diagonal_fraction)
    if abs(total - 1.0) > 1e-6:
      raise ValueError(f"Stair route probabilities must sum to one: {total}")
    return Stage2TargetVelocityCommand(self, env)


__all__ = ["Stage2TargetVelocityCommandCfg"]
