"""Rewards ported from AMP_lab's E1 25-DoF Walk task."""


import inspect


import torch


from mjlab.managers.manager_base import ManagerTermBase


from mjlab.entity import Entity


from mjlab.managers.scene_entity_config import SceneEntityCfg


from mjlab.sensor import ContactSensor, TerrainHeightSensor


from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse


from .observations import isaacgym_gait_phase


from ..terrains import HEADS, amp_head


_ROBOT = SceneEntityCfg("robot")


def _terrain_name_mask(env, terrain_name: str) -> torch.Tensor:
  """Return the environment mask for a named generated terrain, if present."""
  generator = env.cfg.scene.terrain.terrain_generator
  if generator is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  terrain_names = tuple(generator.sub_terrains)
  if terrain_name not in terrain_names:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return env.scene.terrain.terrain_types == terrain_names.index(terrain_name)


def lin_vel_z_l2(env, asset_cfg=_ROBOT):
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def ang_vel_xy_l2(env, asset_cfg=_ROBOT):
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def track_lin_vel_xy(env, command_name: str, std: float, asset_cfg=_ROBOT):
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  error = torch.sum(
    torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]), dim=1
  )
  return torch.exp(-error / std**2)


def track_ang_vel_z(env, command_name: str, std: float, asset_cfg=_ROBOT):
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
  return torch.exp(-error / std**2)


def navigation_heading_error(env, command_name: str, weight_mask: bool = True):
  """Penalize target-heading error while a Parkour route is active.

  The Hybrid Nav implementation exposes ``heading_error`` and
  ``active_navigation`` directly.  This task intentionally keeps the Parkour
  command term, so derive the same quantities from its target point instead.
  """
  term = env.command_manager.get_term(command_name)
  if hasattr(term, "heading_error"):
    error = term.heading_error
  else:
    delta = term.target_pos_w[:, :2] - term.robot.data.root_link_pos_w[:, :2]
    desired = torch.atan2(delta[:, 1], delta[:, 0])
    heading = term.robot.data.heading_w
    error = torch.atan2(torch.sin(desired - heading), torch.cos(desired - heading))
  distance = torch.linalg.vector_norm(
    term.target_pos_w[:, :2] - term.robot.data.root_link_pos_w[:, :2], dim=1
  )
  active = term.complex_terrain & (distance > term.cfg.target_distance_threshold)
  active &= ~term.is_standing_env
  active &= ~term.zero_stop_active
  value = 1.0 - torch.cos(error)
  return value * active.float() if weight_mask else value


def dont_wait(env, command_name: str, asset_cfg=_ROBOT):
  """Waiting penalty measured along either commanded traversal direction."""
  asset: Entity = env.scene[asset_cfg.name]
  command_x = env.command_manager.get_command(command_name)[:, 0]
  velocity_x = asset.data.root_link_lin_vel_b[:, 0]
  directed_velocity = velocity_x * torch.sign(command_x)
  return (command_x.abs() > 0.3).float() * (
    (directed_velocity < 0.15).float()
    + (directed_velocity < 0.0).float()
    + (directed_velocity < -0.15).float()
  )


def energy(env, asset_cfg=_ROBOT):
  asset: Entity = env.scene[asset_cfg.name]
  power = torch.abs(
    asset.data.qfrc_actuator[:, asset_cfg.joint_ids]
    * asset.data.joint_vel[:, asset_cfg.joint_ids]
  )
  return torch.linalg.vector_norm(power, dim=1)


def joint_deviation_exp(
  env,
  asset_cfg: SceneEntityCfg,
  tolerance: float = 0.1,
  scale: float = 3.0,
  max_error: float = 50.0,
):
  asset: Entity = env.scene[asset_cfg.name]
  error = torch.linalg.vector_norm(
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids],
    dim=1,
  )
  error = torch.clamp(error - tolerance, min=0.0, max=max_error)
  return torch.exp(-scale * error)


def feet_slide(
  env,
  sensor_name: str,
  feet_cfg: SceneEntityCfg,
  force_threshold: float = 1.0,
):
  sensor: ContactSensor = env.scene[sensor_name]
  asset: Entity = env.scene[feet_cfg.name]
  if sensor.data.force_history is not None:
    force = torch.linalg.vector_norm(sensor.data.force_history, dim=-1)
    contact = force.amax(dim=2) > force_threshold
  else:
    assert sensor.data.found is not None
    contact = sensor.data.found
  velocity = torch.linalg.vector_norm(
    asset.data.body_link_lin_vel_w[:, feet_cfg.body_ids, :2], dim=2
  )
  return torch.sum(velocity * contact.float(), dim=1)


def feet_stumble(env, sensor_name: str, force_ratio: float = 5.0):
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  lateral = torch.linalg.vector_norm(sensor.data.force[..., :2], dim=-1)
  vertical = torch.abs(sensor.data.force[..., 2])
  return torch.any(lateral > force_ratio * vertical, dim=1).float()


def foot_impact_velocity(env, feet_cfg: SceneEntityCfg, threshold: float = 0.4):
  """Penalize only downward/vertical foot speed above a safe landing speed."""
  asset: Entity = env.scene[feet_cfg.name]
  speed_z = torch.abs(asset.data.body_link_lin_vel_w[:, feet_cfg.body_ids, 2])
  return torch.sum(torch.relu(speed_z - threshold), dim=1)


def feet_lateral_distance_limit(
  env,
  feet_cfg: SceneEntityCfg,
  minimum: float = 0.20,
  maximum: float = 0.28,
  scale: float = 10.0,
):
  """Penalize lateral foot spacing outside a permitted base-frame band."""
  if len(feet_cfg.body_ids) != 2:
    raise ValueError("feet_lateral_distance_limit requires exactly two feet")
  asset: Entity = env.scene[feet_cfg.name]
  feet_w = asset.data.body_link_pos_w[:, feet_cfg.body_ids, :]
  root_w = asset.data.root_link_pos_w.unsqueeze(1)
  root_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
  feet_b = quat_apply_inverse(root_quat, feet_w - root_w)
  distance = torch.abs(feet_b[:, 0, 1] - feet_b[:, 1, 1])
  below = torch.relu(float(minimum) - distance)
  above = torch.relu(distance - float(maximum))
  return float(scale) * (below.square() + above.square())


def feet_orientation_contact(
  env,
  sensor_name: str,
  feet_cfg: SceneEntityCfg,
  contact_force_threshold: float = 1.0,
):
  sensor: ContactSensor = env.scene[sensor_name]
  asset: Entity = env.scene[feet_cfg.name]
  assert sensor.data.force_history is not None
  force = torch.linalg.vector_norm(sensor.data.force_history, dim=-1).amax(dim=2)
  contact = force > contact_force_threshold
  quat_w = asset.data.body_link_quat_w[:, feet_cfg.body_ids]
  gravity_w = asset.data.gravity_vec_w.unsqueeze(1).expand_as(quat_w[..., :3])
  gravity_b = quat_apply_inverse(quat_w, gravity_w)
  tilt = torch.linalg.vector_norm(gravity_b[..., :2], dim=-1)
  return torch.sum(tilt * contact.float(), dim=1)


def feet_roll_downstairs(
  env,
  feet_cfg: SceneEntityCfg,
  command_name: str = "base_velocity",
  contact_sensor_name: str = "feet_ground_contact",
  contact_force_threshold: float = 1.0,
  downward_speed_threshold: float = 0.05,
  roll_tolerance: float = 0.08,
  command_x_threshold: float = 0.2,
  terrain_names: tuple[str, ...] = ("stairs_down", "stairs_down_high"),
):
  """Penalize excessive foot roll while descending and at first touchdown.

  Use measured downward foot velocity and the contact sensor's first-touchdown
  event instead of a nominal gait phase. This catches the actual landing even
  when stair cadence differs from the reference phase.
  """
  generator = env.cfg.scene.terrain.terrain_generator
  if generator is None:
    return torch.zeros(env.num_envs, device=env.device)
  all_terrain_names = tuple(generator.sub_terrains)
  missing = [name for name in terrain_names if name not in all_terrain_names]
  if missing:
    raise ValueError(f"Unknown downhill terrain names: {missing}")
  terrain_types = env.scene.terrain.terrain_types
  downhill = torch.zeros_like(terrain_types, dtype=torch.bool)
  for name in terrain_names:
    downhill |= terrain_types == all_terrain_names.index(name)

  command_x = env.command_manager.get_command(command_name)[:, 0]
  active = downhill & (command_x.abs() > float(command_x_threshold))

  asset: Entity = env.scene[feet_cfg.name]
  sensor: ContactSensor = env.scene[contact_sensor_name]
  assert sensor.data.force is not None
  contact = torch.abs(sensor.data.force[..., 2]) > float(contact_force_threshold)
  first_contact = sensor.compute_first_contact(dt=env.step_dt)
  foot_velocity_z = asset.data.body_link_lin_vel_w[:, feet_cfg.body_ids, 2]
  descending_swing = (~contact) & (
    foot_velocity_z < -float(downward_speed_threshold)
  )
  landing_window = descending_swing | first_contact

  quat_w = asset.data.body_link_quat_w[:, feet_cfg.body_ids]
  gravity_w = asset.data.gravity_vec_w.unsqueeze(1).expand_as(quat_w[..., :3])
  gravity_b = quat_apply_inverse(quat_w, gravity_w)
  # E1 foot frames use x-forward/y-lateral/z-up.  The y gravity component
  # therefore measures roll; atan2 keeps the quantity well-conditioned.
  roll = torch.atan2(
    torch.abs(gravity_b[..., 1]),
    torch.clamp(-gravity_b[..., 2], min=1.0e-6),
  )
  # Linear excess gives a useful gradient for modest roll errors; a squared
  # excess made the previous signal vanish for errors just above tolerance.
  excess = torch.relu(roll - float(roll_tolerance))
  return (excess * landing_window.float()).sum(dim=1) * active.float()


def feet_both_air_downstairs(
  env,
  sensor_name: str = "feet_ground_contact",
  command_name: str = "base_velocity",
  contact_force_threshold: float = 5.0,
  downward_speed_threshold: float = 0.15,
  command_x_threshold: float = 0.2,
  terrain_names: tuple[str, ...] = ("stairs_down", "stairs_down_high"),
):
  """Penalize downhill drops while both feet are unsupported.

  This is active on descending stairs in either travel direction when root
  velocity points down. Ordinary gait swing is not penalized.
  """
  generator = env.cfg.scene.terrain.terrain_generator
  if generator is None:
    return torch.zeros(env.num_envs, device=env.device)
  terrain_types = env.scene.terrain.terrain_types
  terrain_names_all = tuple(generator.sub_terrains)
  missing = [name for name in terrain_names if name not in terrain_names_all]
  if missing:
    raise ValueError(f"Unknown downhill terrain names: {missing}")
  descending = torch.zeros_like(terrain_types, dtype=torch.bool)
  for name in terrain_names:
    descending |= terrain_types == terrain_names_all.index(name)

  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  vertical_force = torch.abs(sensor.data.force[..., 2])
  contact = vertical_force > float(contact_force_threshold)
  both_air = ~contact.any(dim=1)

  asset: Entity = env.scene["robot"]
  downward = -asset.data.root_link_lin_vel_w[:, 2]
  falling = downward > float(downward_speed_threshold)
  command_x = env.command_manager.get_command(command_name)[:, 0]
  active = descending & both_air & falling & (command_x.abs() > float(command_x_threshold))

  severity = 1.0 + torch.relu(
    downward - float(downward_speed_threshold)
  ) / float(max(downward_speed_threshold, 1.0e-6))
  return active.float() * severity


def feet_gait_contact_isaacgym(
  env,
  sensor_name: str,
  command_name="base_velocity",
  period: float = 1.2,
  phase_offset: float = 0.5,
  stance_ratio: float = 0.65,
  force_threshold: float = 20.0,
):
  """Reward contact matching Isaac Gym's left/right phase masks."""
  phase, standing = isaacgym_gait_phase(env, command_name, period)
  leg_phase = torch.stack((phase, torch.remainder(phase + phase_offset, 1.0)), dim=1)
  stance = leg_phase <= stance_ratio
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  contact = torch.linalg.vector_norm(sensor.data.force, dim=-1) > force_threshold
  reward = (contact == stance).float().sum(dim=1)
  reward[standing] = 2.0
  return reward


def feet_gait_stance_mask_isaacgym(
  env,
  feet_cfg: SceneEntityCfg,
  command_name="base_velocity",
  period: float = 1.2,
  phase_offset: float = 0.5,
  stance_ratio: float = 0.65,
  fast_command_threshold: float = 0.3,
  fast_command_scale: float = 0.2,
):
  """Penalize horizontal foot motion during the scheduled stance phase."""
  phase, standing = isaacgym_gait_phase(env, command_name, period)
  leg_phase = torch.stack((phase, torch.remainder(phase + phase_offset, 1.0)), dim=1)
  stance = (leg_phase <= stance_ratio).float()
  asset: Entity = env.scene[feet_cfg.name]
  foot_speed = torch.linalg.vector_norm(
    asset.data.body_link_lin_vel_w[:, feet_cfg.body_ids, :2], dim=-1
  ) / 5.0
  penalty = ((1.0 - torch.exp(-foot_speed)) * stance).sum(dim=1)
  penalty[standing] = 0.0
  # Keep full phase-gait strength on Perlin terrain. Other terrains retain
  # Isaac Gym's reduction for commanded walking above 0.3 m/s.
  command_x = env.command_manager.get_command(command_name)[:, 0]
  discounted = (torch.abs(command_x) > fast_command_threshold) & ~_terrain_name_mask(
    env, "perlin_rough"
  )
  penalty[discounted] *= fast_command_scale
  return penalty


def feet_gait_swing_mask_isaacgym(
  env,
  sensor_name: str,
  command_name="base_velocity",
  period: float = 1.2,
  phase_offset: float = 0.5,
  stance_ratio: float = 0.65,
  fast_command_threshold: float = 0.3,
  fast_command_scale: float = 0.2,
):
  """Penalize contact force during the scheduled swing phase."""
  phase, standing = isaacgym_gait_phase(env, command_name, period)
  leg_phase = torch.stack((phase, torch.remainder(phase + phase_offset, 1.0)), dim=1)
  swing = (leg_phase > stance_ratio).float()
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  force = torch.linalg.vector_norm(sensor.data.force, dim=-1) / 50.0
  penalty = ((1.0 - torch.exp(-force)) * swing).sum(dim=1)
  penalty[standing] = 0.0
  command_x = env.command_manager.get_command(command_name)[:, 0]
  discounted = (torch.abs(command_x) > fast_command_threshold) & ~_terrain_name_mask(
    env, "perlin_rough"
  )
  penalty[discounted] *= fast_command_scale
  return penalty


class FootholdPlacementExp:
  """Liftoff-frozen flat-candidate touchdown reward."""
  def __init__(self, cfg, env):
    del cfg
    self.prev = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
    self.init = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.candidates = torch.zeros(env.num_envs, 2, 30, 3, device=env.device)
    self.valid = torch.zeros(env.num_envs, 2, 30, dtype=torch.bool, device=env.device)
    self.error = torch.zeros(env.num_envs, 2, device=env.device)
    # The support reference at liftoff is retained only to color elevated
    # candidates in the viewer; it does not enter observations or rewards.
    self.candidate_ground_z = torch.zeros(env.num_envs, 2, device=env.device)
  def reset(self, env_ids=None, ids=None):
    ids = env_ids if env_ids is not None else ids
    if ids is None:
      return
    self.prev[ids] = False; self.init[ids] = False; self.candidates[ids] = 0; self.valid[ids] = False; self.error[ids] = 0
    self.candidate_ground_z[ids] = 0

  def debug_vis(self, visualizer):
    """Draw frozen foothold candidates for the selected environment."""
    env_indices = list(visualizer.get_env_indices(self.candidates.shape[0]))
    if not bool(self.init.any()):
      return
    if not env_indices:
      return
    candidates = self.candidates.detach().cpu().numpy()
    valid = self.valid.detach().cpu().numpy()
    ready = self.init.detach().cpu().numpy()
    candidate_ground_z = self.candidate_ground_z.detach().cpu().numpy()
    radius = max(0.018, 0.16 * visualizer.meansize)
    for env_idx in env_indices:
      if not ready[env_idx]:
        continue
      for foot in range(2):
        prefix = "left" if foot == 0 else "right"
        for candidate_idx in range(candidates.shape[2]):
          point = candidates[env_idx, foot, candidate_idx]
          # Hide padding/fallback slots. They are fixed-shape implementation
          # details rather than real safe footholds.
          if not valid[env_idx, foot, candidate_idx]:
            continue
          raised = point[2] > candidate_ground_z[env_idx, foot] + 0.04
          if raised:
            color = (1.0, 0.42, 0.04, 1.0)
            label = f"foothold_{prefix}_raised_{candidate_idx}"
          elif foot == 0:
            color = (0.05, 0.80, 1.0, 0.95)
            label = f"foothold_left_flat_{candidate_idx}"
          else:
            color = (0.10, 1.0, 0.22, 0.95)
            label = f"foothold_right_flat_{candidate_idx}"
          # Lift the marker slightly above the actual support surface. The
          # reward continues to use the unmodified candidate coordinates.
          draw_point = point.copy()
          draw_point[2] += 0.01
          visualizer.add_sphere(draw_point, radius=radius, color=color, label=label)
  def _build_candidates(self, points, command_x, hit_mask):
    num_envs, num_points, _ = points.shape
    count = 30
    filter_indices = torch.clamp(torch.floor(torch.clamp(command_x.abs() * 0.3, 0.02, 0.60) / 0.01).long(), 0, num_points - 1)
    window_points, stride = min(26, num_points), 3
    max_steps = max((num_points - window_points) // stride + 1, 1)
    starts = filter_indices[:, None] + torch.arange(max_steps, device=points.device)[None] * stride
    sample = starts[..., None] + torch.arange(window_points, device=points.device)[None, None]
    valid_window = sample < num_points
    sample = sample.clamp(0, num_points - 1)
    heights = points[..., 2][:, None].expand(-1, max_steps, -1)
    window_heights = torch.gather(heights, 2, sample)
    roughness = window_heights.amax(-1) - window_heights.amin(-1)
    expanded_hit_mask = hit_mask[:, None].expand(-1, max_steps, -1)
    window_hit_mask = torch.gather(expanded_hit_mask, 2, sample)
    # RayCastSensor collapses a miss to its ray origin. Reject those samples;
    # otherwise a row of misses is incorrectly classified as a flat platform.
    valid_window_all = valid_window.all(-1) & window_hit_mask.all(-1)
    flat = (roughness < 0.08) & valid_window_all
    centers = torch.gather(points, 1, (starts + window_points // 2).clamp(0, num_points - 1)[..., None].expand(-1, -1, 3))
    rank = flat.long().cumsum(1) - 1
    selected = flat & (rank < count)
    candidates = torch.zeros(num_envs, count, 3, device=points.device, dtype=points.dtype)
    valid = torch.zeros(num_envs, count, device=points.device, dtype=torch.bool)
    env_ids, step_ids = selected.nonzero(as_tuple=True)
    if env_ids.numel():
      candidate_ids = rank[env_ids, step_ids]
      candidates[env_ids, candidate_ids] = centers[env_ids, step_ids]
      valid[env_ids, candidate_ids] = True
    valid_count = valid.sum(1)
    last_ids = (valid_count - 1).clamp(min=0)
    last_points = candidates[torch.arange(num_envs, device=points.device), last_ids]
    fallback = points[torch.arange(num_envs, device=points.device), (filter_indices + window_points // 2).clamp(0, num_points - 1)]
    fill = torch.where((valid_count > 0)[:, None], last_points, fallback)
    return torch.where(valid[..., None], candidates, fill[:, None]), valid

  def __call__(self, env, feet_cfg, sensor_name="foot_forward_scan", contact_sensor_name="feet_ground_contact", command_name="base_velocity", foot_offset=0.06, sigma_xz=5.0):
    del sensor_name
    sensor = env.scene[contact_sensor_name]
    force = torch.linalg.vector_norm(sensor.data.force, dim=-1)
    contact = force > 1.0
    first_air = (~contact) & self.prev & self.init[:, None]
    first_contact = contact & (~self.prev) & self.init[:, None]
    forward_sensors = (
      env.scene["left_foot_forward_scan"],
      env.scene["right_foot_forward_scan"],
    )
    points = torch.stack(
      [forward_sensor.data.hit_pos_w for forward_sensor in forward_sensors],
      dim=1,
    )
    hit_mask = torch.stack(
      [forward_sensor.data.distances >= 0.0 for forward_sensor in forward_sensors],
      dim=1,
    )
    command = env.command_manager.get_command(command_name)[:, 0]
    robot = env.scene[feet_cfg.name]
    foot_quat = robot.data.body_link_quat_w[:, feet_cfg.body_ids]
    offset = torch.zeros(env.num_envs, 2, 3, device=env.device); offset[..., 2] = -foot_offset
    foot_pos = robot.data.body_link_pos_w[:, feet_cfg.body_ids] + quat_apply(foot_quat, offset)
    for foot in range(2):
      candidates, valid = self._build_candidates(
        points[:, foot], command, hit_mask[:, foot]
      )
      scan_z = points[:, foot, :, 2]
      has_hit = hit_mask[:, foot].any(1)
      scan_z_min = torch.where(
        has_hit,
        scan_z.masked_fill(~hit_mask[:, foot], float("inf")).amin(1),
        torch.zeros_like(command),
      )
      self.candidates[:, foot] = torch.where(first_air[:, foot, None, None], candidates, self.candidates[:, foot])
      self.valid[:, foot] = torch.where(first_air[:, foot, None], valid, self.valid[:, foot])
      refresh = first_air[:, foot]
      self.candidate_ground_z[:, foot] = torch.where(
        refresh,
        scan_z_min,
        self.candidate_ground_z[:, foot],
      )
    distance = torch.linalg.vector_norm(foot_pos[..., (0, 2)].unsqueeze(2) - self.candidates[..., (0, 2)], dim=-1)
    distance = torch.where(self.valid, distance, torch.full_like(distance, 1e6))
    minimum = distance.amin(-1); minimum = torch.where(self.valid.any(-1), minimum, torch.zeros_like(minimum))
    self.error = torch.where(first_contact, minimum, self.error)
    self.prev.copy_(contact); self.init[:] = True
    forward = (command > 0.2) & (torch.linalg.vector_norm(env.command_manager.get_command(command_name)[:, 1:], dim=1) < 0.2)
    return torch.exp(-sigma_xz * self.error).sum(1) * forward


class FootHoldPenalty:
  """Penalize touchdown when the sampled sole is not on one support plane."""
  def __init__(self, cfg, env):
    del cfg
    self._env = env
    self.prev=torch.zeros(env.num_envs,2,dtype=torch.bool,device=env.device); self.init=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device); self.saved=None
  def reset(self, env_ids=None, ids=None):
    ids = env_ids if env_ids is not None else ids
    if ids is None:
      return
    self.prev[ids]=False; self.init[ids]=False
    if self.saved is not None: self.saved[ids]=0

  def debug_vis(self, visualizer):
    """Hide dense sole samples; play draws only frozen foothold candidates."""
    del visualizer
  def __call__(self, env, sensor_name="foot_hold_scan", contact_sensor_name="feet_ground_contact", threshold=0.05):
    self._env = env
    s=env.scene[contact_sensor_name]; c=torch.linalg.vector_norm(s.data.force,dim=-1)>1.0; h=torch.cat([env.scene[n].data.hit_pos_w[...,2] for n in ("left_foot_hold_scan","right_foot_hold_scan")],1).reshape(env.num_envs,2,-1); span=h.amax(-1,keepdim=True)-h
    if self.saved is None: self.saved=torch.zeros_like(span)
    first=c&~self.prev&self.init[:,None]; self.saved=torch.where(first[...,None],span,self.saved); self.prev.copy_(c); self.init[:]=True
    return (self.saved>threshold).float().mean(-1).gt(0.4).sum(-1).float()


class LocomotionReward(ManagerTermBase):
  """Gate the original reward while preserving class-based term state."""

  def __init__(self, cfg, env):
    super().__init__(env)
    reward_fn = cfg.params["reward_fn"]
    self.reward = reward_fn(cfg, env) if inspect.isclass(reward_fn) else reward_fn

  def reset(self, env_ids=None):
    if hasattr(self.reward, "reset"):
      self.reward.reset(env_ids)

  def __call__(self, env, reward_fn, **params):
    return self.reward(env, **params) * (amp_head(env) != HEADS.index("box"))


__all__ = ['lin_vel_z_l2', 'ang_vel_xy_l2', 'track_lin_vel_xy', 'track_ang_vel_z', 'navigation_heading_error', 'dont_wait', 'energy', 'joint_deviation_exp', 'feet_slide', 'feet_stumble', 'foot_impact_velocity', 'feet_lateral_distance_limit', 'feet_orientation_contact', 'feet_roll_downstairs', 'feet_both_air_downstairs', 'feet_gait_contact_isaacgym', 'feet_gait_stance_mask_isaacgym', 'feet_gait_swing_mask_isaacgym', 'FootholdPlacementExp', 'FootHoldPenalty', 'LocomotionReward']
