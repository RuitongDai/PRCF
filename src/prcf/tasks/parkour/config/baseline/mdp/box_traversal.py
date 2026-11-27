"""Reset and score the two box tasks using their course geometry and contacts."""

import math

import torch

from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.manager_base import ManagerTermBase

from ..terrains import SingleBoxTerrainCfg, terrain_difficulties

CLIMB_TARGET_OFFSET = 0.45
CLIMB_TARGET_RADIUS = 0.25
CLIMB_PELVIS_CLEARANCE = 0.45


class BoxTaskGeometry:
  def __init__(self, env):
    self.terrain = env.scene.terrain
    generator = self.terrain.cfg.terrain_generator
    names = tuple(generator.sub_terrains)
    self._up_column = names.index("long_high_box") if "long_high_box" in names else -1
    self._down_column = names.index("high_box") if "high_box" in names else -1
    tiles = tuple(generator.sub_terrains.values())
    self._bounds = torch.tensor([
      (tile.front_x, tile.end_x, tile.box_width / 2)
      if isinstance(tile, SingleBoxTerrainCfg) else (0.0, 0.0, 0.0)
      for tile in tiles
    ], device=env.device)
    self._heights = torch.tensor([
      [tile.height_at(row) if isinstance(tile, SingleBoxTerrainCfg) else 0.0 for tile in tiles]
      for row in terrain_difficulties(generator)
    ], device=env.device, dtype=torch.float32)

  @property
  def up(self):
    return self.terrain.terrain_types == self._up_column

  @property
  def down(self):
    return self.terrain.terrain_types == self._down_column

  def bounds(self, env_ids=slice(None)):
    return self._bounds[self.terrain.terrain_types[env_ids]]

  def height(self, env_ids=slice(None)):
    return self._heights[self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids]]


def reset_box_pose(env, env_ids, command_name="base_velocity"):
  """Place E1 on the ground for climbing and on the top for descending."""
  command = env.command_manager.get_term(command_name)
  course = command.box_course
  ids = resolve_env_ids(env, env_ids)
  ids = ids[(course.up | course.down)[ids]]
  if not ids.numel():
    return
  front, rear, _ = course.bounds(ids).unbind(dim=1)
  down = course.down[ids]
  robot = command.robot
  root = robot.data.default_root_state[ids].clone()
  root[:, :3] += env.scene.env_origins[ids]
  root[:, 0] += torch.where(down, rear - 0.40, front - 0.90)
  root[:, 2] += torch.where(down, course.height(ids), 0.0)
  root[:, 7:] = 0.0
  robot.write_root_state_to_sim(root, env_ids=ids)
  joints = robot.data.default_joint_pos[ids]
  robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=ids)


class BoxTraversal(ManagerTermBase):
  def __init__(self, cfg, env):
    super().__init__(env)
    self.course = env.command_manager.get_term(cfg.params["command_name"]).box_course
    self.mounted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.passed = torch.zeros_like(self.mounted)
    self.mount_bonus = torch.zeros_like(self.mounted)
    self.exit_bonus = torch.zeros_like(self.mounted)
    self.lift = torch.zeros(self.num_envs, device=self.device)
    self._previous_z = torch.zeros_like(self.lift)
    self._was_approaching = torch.zeros_like(self.mounted)
    self._top_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    self._ground_steps = torch.zeros_like(self._top_steps)
    self._outside_steps = torch.zeros_like(self._ground_steps)

  @property
  def on_box(self):
    return self.course.up | self.course.down

  def reset(self, env_ids=None):
    if env_ids is None:
      env_ids = slice(None)
    for buffer in (
      self.mounted, self.passed, self.mount_bonus, self.exit_bonus, self.lift,
      self._previous_z, self._was_approaching, self._top_steps,
      self._ground_steps, self._outside_steps,
    ):
      buffer[env_ids] = 0

  def __call__(
    self, env, *, command_name, sensor_name, contact_force_threshold,
    mount_seconds, landing_seconds, bypass_seconds,
  ):
    front, rear, half_width = self.course.bounds().unbind(dim=1)
    top = self.course.height()
    corridor = half_width + 0.15
    command = env.command_manager.get_term(command_name)
    robot = command.robot.data
    # The box configuration measures its edges from each tile's spawn origin.
    x, y, z = (robot.root_link_pos_w - env.scene.env_origins).unbind(dim=1)
    in_corridor = y.abs() <= corridor
    active = ~command.is_standing_env & ~env.termination_manager.terminated
    climbing = active & self.course.up & ~self.mounted
    descending = active & self.course.down & ~self.passed
    attempting = climbing | descending

    outside = attempting & (x >= front) & (x <= rear) & ~in_corridor
    self._outside_steps.copy_(torch.where(outside, self._outside_steps + 1, 0))
    bypass = attempting & (
      (self._outside_steps >= math.ceil(bypass_seconds / env.step_dt))
      | (climbing & (x >= rear + 0.15))
    )

    contacts = env.scene[sensor_name].data
    assert contacts.pos is not None and contacts.normal is not None
    assert contacts.found is not None and contacts.force is not None
    px, py, pz = (contacts.pos - env.scene.env_origins[:, None, :]).unbind(dim=-1)
    # Each E1 foot distributes its load across several collision contacts.
    force = torch.where(
      (contacts.found > 0) & (contacts.normal[..., 2].abs() >= 0.7),
      contacts.force[..., 0].abs(), 0.0,
    )
    top_region = (
      (px >= front[:, None]) & (px <= rear[:, None])
      & (py.abs() <= half_width[:, None]) & ((pz - top[:, None]).abs() <= 0.04)
    )
    on_top = self._supported_feet(force, top_region, contact_force_threshold)
    stable = (-robot.projected_gravity_b[:, 2] > 0.8) & (robot.root_link_lin_vel_w[:, 2].abs() < 0.5)
    near_top_goal = (x - front - CLIMB_TARGET_OFFSET).square() + y.square() <= CLIMB_TARGET_RADIUS**2
    mounting = (
      climbing & ~bypass & stable & near_top_goal & on_top.all(dim=1)
      & (z >= top + CLIMB_PELVIS_CLEARANCE)
      & (torch.linalg.vector_norm(robot.root_link_lin_vel_w[:, :2], dim=1) < 0.30)
    )
    self._top_steps.copy_(torch.where(mounting, self._top_steps + 1, 0))
    self.mount_bonus.copy_(self._top_steps >= math.ceil(mount_seconds / env.step_dt))
    self.mounted |= self.mount_bonus

    ground_region = (px >= rear[:, None]) & (py.abs() <= corridor[:, None]) & (pz.abs() <= 0.04)
    on_ground = self._supported_feet(force, ground_region, contact_force_threshold).any(dim=1)
    landing = (
      descending & ~bypass & in_corridor
      & (x >= rear + 0.15) & on_ground & stable
    )
    self._ground_steps.copy_(torch.where(landing, self._ground_steps + 1, 0))
    self.exit_bonus.copy_(self._ground_steps >= math.ceil(landing_seconds / env.step_dt))
    self.passed |= self.exit_bonus

    approaching = (
      climbing & in_corridor & ~bypass
      & (x >= front - 0.45) & (x <= front + 0.25)
    )
    lift = ((z - self._previous_z) / (0.4 * env.step_dt)).clamp(-1.0, 1.0)
    self.lift.copy_(torch.where(approaching & self._was_approaching, lift, 0.0))
    self._previous_z.copy_(z)
    self._was_approaching.copy_(approaching)
    return bypass

  def _supported_feet(self, force, region, threshold):
    foot_forces = torch.where(region, force, 0.0).reshape(self.num_envs, 2, -1).sum(dim=-1)
    return foot_forces >= threshold


def box_traversal(env) -> BoxTraversal | None:
  manager = env.termination_manager
  if "box_bypass" in manager.active_terms:
    return manager.get_term_cfg("box_bypass").func
  return None


def box_lift_progress(env):
  return box_traversal(env).lift


def box_mount_bonus(env):
  return box_traversal(env).mount_bonus.float()


def box_exit_bonus(env):
  return box_traversal(env).exit_bonus.float()


def box_task_succeeded(env):
  box = box_traversal(env)
  return box.mounted | box.passed
