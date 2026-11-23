"""E1 visual AMP observation helpers with Isaac Gym VDWL depth noise."""

import torch
import torch.nn.functional as F
from dataclasses import dataclass

from mjlab.entity import Entity
from mjlab.envs.mdp.observations import builtin_sensor
from mjlab.sensor import CameraSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse



@dataclass
class LegacyGridPatternCfg:
  """Explicit x/y ray coordinates matching the Isaac Gym VDWL grids."""
  x_values: tuple[float, ...]
  y_values: tuple[float, ...]
  direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
  z_offset: float = 0.0
  """World-up offset after yaw alignment for the ray origins."""

  def generate_rays(self, mj_model, device):
    del mj_model
    x = torch.tensor(self.x_values, device=device, dtype=torch.float32)
    y = torch.tensor(self.y_values, device=device, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    offsets = torch.zeros((grid_x.numel(), 3), device=device)
    offsets[:, 0] = grid_x.flatten()
    offsets[:, 1] = grid_y.flatten()
    offsets[:, 2] = self.z_offset
    direction = torch.tensor(self.direction, device=device, dtype=torch.float32)
    direction = direction / direction.norm().clamp_min(1.0e-8)
    return offsets, direction.unsqueeze(0).expand(offsets.shape[0], -1).clone()


def torso_imu_projected_gravity(env, sensor_name="robot/orientation"):
  """Gravity in the XML torso IMU frame, using its wxyz framequat sensor."""
  quat = builtin_sensor(env, sensor_name)
  quat = F.normalize(quat, dim=-1)
  gravity_w = env.scene["robot"].data.gravity_vec_w
  return quat_apply_inverse(quat, gravity_w)


def isaacgym_gait_phase(
  env,
  command_name="base_velocity",
  period=1.2,
  stand_command_threshold=0.1,
  stand_speed_threshold=0.5,
):
  """Shared Isaac Gym gait phase with a random 0/0.5 episode offset."""
  episode_step = env.episode_length_buf
  start = getattr(env, "_vdwl_gait_phase_start", None)
  previous_step = getattr(env, "_vdwl_gait_previous_episode_step", None)
  if start is None or start.shape != episode_step.shape:
    start = torch.randint(0, 2, episode_step.shape, device=env.device).float() * 0.5
    previous_step = episode_step.clone()
    env._vdwl_gait_phase_start = start
    env._vdwl_gait_previous_episode_step = previous_step
  else:
    reset = episode_step < previous_step
    if torch.any(reset):
      start[reset] = torch.randint(
        0, 2, (int(reset.sum().item()),), device=env.device
      ).float() * 0.5
    previous_step.copy_(episode_step)

  command = env.command_manager.get_command(command_name)
  robot: Entity = env.scene["robot"]
  low_command = torch.linalg.vector_norm(command[:, :3], dim=1) <= stand_command_threshold
  low_speed = torch.linalg.vector_norm(robot.data.root_link_lin_vel_b[:, :2], dim=1) <= stand_speed_threshold
  standing = low_command & low_speed
  phase = torch.remainder(start + episode_step.float() * env.step_dt / period, 1.0)
  phase = torch.where(standing, torch.zeros_like(phase), phase)
  return phase, standing


def gait_clock(
  env,
  command_name="base_velocity",
  period=1.2,
  stand_command_threshold=0.1,
  stand_speed_threshold=0.5,
):
  """Isaac Gym clock observation: sin/cos phase, zeroed while standing."""
  phase, standing = isaacgym_gait_phase(
    env, command_name, period, stand_command_threshold, stand_speed_threshold
  )
  clock = torch.stack((
    torch.sin(2.0 * torch.pi * phase),
    torch.cos(2.0 * torch.pi * phase),
  ), dim=1)
  clock[standing] = 0.0
  return clock


class DepthHistory:
  """Return the previous/current pair from the true 10 Hz camera stream."""

  def __init__(self, cfg, env):
    del cfg, env
    self.buffer: torch.Tensor | None = None
    self.needs_fill: torch.Tensor | None = None
    self.last_step = -1
    self.last_sensor_sample_id = -1

  def reset(self, env_ids):
    if self.buffer is not None:
      self.buffer[env_ids] = 0.0
    if self.needs_fill is not None:
      self.needs_fill[env_ids] = True

  def __call__(
    self,
    env,
    sensor_name="depth_camera",
    history_length=2,
    history_stride=5,
    output_frames=2,
    delayed_frame_range=(0, 0),
    crop_top=18,
    crop_bottom=0,
    crop_left=16,
    crop_right=16,
    min_distance=0.1,
    max_distance=3.0,
    enable_noise=True,
    depth_std=0.01,
    depth_std_multiplier=0.015,
    dropout_prob=0.02,
    dropout_value=-1.0,
  ):
    sensor: CameraSensor = env.scene[sensor_name]
    out_height = sensor.cfg.height - crop_top - crop_bottom
    out_width = sensor.cfg.width - crop_left - crop_right
    shape = (env.num_envs, history_length, out_height, out_width)
    if self.buffer is None or tuple(self.buffer.shape) != shape:
      self.buffer = torch.zeros(shape, device=env.device)
      self.needs_fill = torch.ones(
        env.num_envs, device=env.device, dtype=torch.bool
      )
      self.last_step = -1
      self.last_sensor_sample_id = -1

    step = int(env.common_step_counter)
    sensor_sample_id = int(getattr(sensor, "sample_id", step))
    has_new_sample = sensor_sample_id != self.last_sensor_sample_id
    if has_new_sample:
      raw = sensor.data.depth
      if raw is None:
        raise RuntimeError("Depth camera has no depth output.")
      depth = raw[..., 0]

      # Match Isaac Gym's `_set_nan_inf_zero()` exactly: renderer-invalid
      # pixels are zero before crop/clip, rather than being treated as far
      # range. The following near clip turns them into 0.1 m (actor 0.1/3),
      # so invalid/self-occluded pixels stay dark instead of becoming white.
      depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

      bottom = sensor.cfg.height - crop_bottom if crop_bottom else sensor.cfg.height
      right = sensor.cfg.width - crop_right if crop_right else sensor.cfg.width
      depth = depth[:, crop_top:bottom, crop_left:right]
      depth = torch.where(depth > max_distance, max_distance, depth)
      depth = depth.clamp(min=min_distance, max=max_distance)

      # Isaac Gym applies metric Gaussian noise and dropout before dividing
      # by far_clip. Thus dropout_value=-1.0 reaches the actor as -1/3.
      if enable_noise:
        std = depth_std + depth * depth_std_multiplier
        depth = depth + torch.randn_like(depth) * std
        depth = depth.masked_fill(torch.rand_like(depth) < dropout_prob, dropout_value)
      depth = depth / max_distance

      if self.last_step >= 0:
        self.buffer[:, 1:].copy_(self.buffer[:, :-1].clone())
      self.buffer[:, 0].copy_(depth)
      assert self.needs_fill is not None
      if torch.any(self.needs_fill):
        fill = self.needs_fill
        self.buffer[fill] = depth[fill].unsqueeze(1).expand(
          -1, history_length, -1, -1
        )
        self.needs_fill[fill] = False
      self.last_step = step
      self.last_sensor_sample_id = sensor_sample_id

    # history_stride is expressed in 50 Hz policy steps, while this ring
    # buffer stores only actual 10 Hz captures.
    capture_period = int(getattr(sensor, "update_period_steps", 1))
    if history_stride % capture_period != 0:
      raise ValueError(
        "history_stride must be divisible by camera update_period_steps: "
        f"{history_stride} % {capture_period} != 0."
      )
    sample_stride = history_stride // capture_period
    maximum_offset = (output_frames - 1) * sample_stride + delayed_frame_range[1]
    if maximum_offset >= history_length:
      raise ValueError(
        f"Depth history length {history_length} must exceed offset {maximum_offset}."
      )
    # Newest is index zero; reverse [0, 1] to return previous then current.
    indices = torch.flip(
      torch.arange(
        0, output_frames * sample_stride, sample_stride, device=env.device
      ),
      dims=(0,),
    )
    indices = indices.unsqueeze(0)
    env_ids = torch.arange(env.num_envs, device=env.device).unsqueeze(1)
    return self.buffer[env_ids, indices]


def foot_contacts(env, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  if sensor.data.force_history is not None:
    force = torch.linalg.vector_norm(sensor.data.force_history, dim=-1)
    return (force.amax(dim=2) > 0.5).float()
  assert sensor.data.found is not None
  return sensor.data.found.float()


def foot_clearance(
  env,
  feet_cfg,
  height_sensor_name: str = "terrain_scan",
  sole_offset: float = 0.06,
) -> torch.Tensor:
  """Terrain-relative sole clearance used by the PIE estimator target.

  The ankle-roll link origin is above the physical sole.  Subtracting the
  calibrated offset keeps this target consistent with the foot collision
  geometry and with the edge/penetration penalties.
  """
  asset: Entity = env.scene[feet_cfg.name]
  feet = asset.data.body_link_pos_w[:, feet_cfg.body_ids]
  scan = env.scene[height_sensor_name].data.hit_pos_w
  delta = feet[:, :, None, :2] - scan[:, None, :, :2]
  nearest = torch.argmin(torch.sum(delta.square(), dim=-1), dim=-1)
  ground_z = torch.gather(scan[..., 2], 1, nearest)
  clearance = feet[..., 2] - float(sole_offset) - ground_z
  return torch.nan_to_num(clearance, nan=0.0, posinf=0.6, neginf=-0.05).clamp(-0.05, 0.6)


