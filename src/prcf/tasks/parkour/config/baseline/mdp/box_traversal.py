"""Track box traversal from measured foot contacts once per control step."""

import math

import torch

from mjlab.managers.manager_base import ManagerTermBase

from ..terrains import SingleBoxTerrainCfg, terrain_difficulties


class BoxTraversal(ManagerTermBase):
  def __init__(self, cfg, env):
    super().__init__(env)
    generator = env.scene.terrain.cfg.terrain_generator
    tiles = tuple(generator.sub_terrains.values())
    rows = terrain_difficulties(generator)
    self._box_columns = torch.tensor(
      [isinstance(tile, SingleBoxTerrainCfg) for tile in tiles], device=self.device
    )
    self._bounds = torch.tensor([
      (tile.front_x, tile.end_x, tile.box_width / 2)
      if isinstance(tile, SingleBoxTerrainCfg) else (0.0, 0.0, 0.0)
      for tile in tiles
    ], device=self.device)
    self._heights = torch.tensor([
      [tile.height_at(row) if isinstance(tile, SingleBoxTerrainCfg) else 0.0
       for tile in tiles]
      for row in rows
    ], device=self.device, dtype=torch.float32)

    self.mounted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.passed = torch.zeros_like(self.mounted)
    self.mount_bonus = torch.zeros_like(self.mounted)
    self.exit_bonus = torch.zeros_like(self.mounted)
    self.lift = torch.zeros(self.num_envs, device=self.device)
    self._previous_z = torch.zeros_like(self.lift)
    self._was_approaching = torch.zeros_like(self.mounted)
    self._top_steps = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.long)
    self._ground_steps = torch.zeros_like(self._top_steps[:, 0])
    self._outside_steps = torch.zeros_like(self._ground_steps)

  @property
  def on_box(self):
    return self._box_columns[self._env.scene.terrain.terrain_types]

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
    terrain = env.scene.terrain
    front, rear, half_width = self._bounds[terrain.terrain_types].unbind(dim=1)
    top = self._heights[terrain.terrain_levels, terrain.terrain_types]
    corridor = half_width + 0.15
    robot = env.scene["robot"].data
    # The box configuration measures its edges from each tile's spawn origin.
    x, y, z = (robot.root_link_pos_w - env.scene.env_origins).unbind(dim=1)
    in_corridor = y.abs() <= corridor
    active = self.on_box & ~env.command_manager.get_term(command_name).is_standing_env

    outside = active & ~self.passed & (x >= front) & (x <= rear) & ~in_corridor
    self._outside_steps.copy_(torch.where(outside, self._outside_steps + 1, 0))
    bypass = active & ~self.passed & (
      (self._outside_steps >= math.ceil(bypass_seconds / env.step_dt))
      | ((x >= rear + 0.15) & ~self.mounted)
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
    mounting = (active & in_corridor & ~self.mounted & ~bypass)[:, None] & on_top
    self._top_steps.copy_(torch.where(mounting, self._top_steps + 1, 0))
    self.mount_bonus.copy_((self._top_steps >= math.ceil(mount_seconds / env.step_dt)).any(dim=1))
    self.mounted |= self.mount_bonus

    ground_region = (px >= rear[:, None]) & (py.abs() <= corridor[:, None]) & (pz.abs() <= 0.04)
    on_ground = self._supported_feet(force, ground_region, contact_force_threshold).any(dim=1)
    stable = (-robot.projected_gravity_b[:, 2] > 0.8) & (robot.root_link_lin_vel_w[:, 2].abs() < 0.5)
    landing = (
      active & self.mounted & ~self.passed & ~bypass & in_corridor
      & (x >= rear + 0.15) & on_ground & stable
    )
    self._ground_steps.copy_(torch.where(landing, self._ground_steps + 1, 0))
    self.exit_bonus.copy_(self._ground_steps >= math.ceil(landing_seconds / env.step_dt))
    self.passed |= self.exit_bonus

    approaching = (
      active & in_corridor & (~self.mounted | self.mount_bonus) & ~bypass
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
