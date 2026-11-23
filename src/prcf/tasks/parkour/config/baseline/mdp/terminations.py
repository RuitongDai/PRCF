"""Contact termination local to the E1 AMP task."""

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from ..terrains import HEADS, amp_head


def invalid_body_contacts(env, sensor_name: str, threshold: float = 1.0) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  if sensor.data.force_history is not None:
    forces = torch.linalg.vector_norm(sensor.data.force_history, dim=-1).amax(dim=2)
  else:
    forces = sensor.data.found.float()
  invalid = forces > threshold
  on_box = amp_head(env) == HEADS.index("box")
  torso = torch.tensor([name.rsplit("/", 1)[-1] in ("pelvis", "torso_link", "waist_yaw_link")
                        for name in sensor.primary_names], device=env.device)
  invalid &= ~on_box[:, None] | torso[None]
  return invalid


def illegal_contact(env, sensor_name: str, threshold: float = 1.0) -> torch.Tensor:
  return invalid_body_contacts(env, sensor_name, threshold).any(dim=1)


def undesired_contact_count(env, sensor_name: str, threshold: float = 1.0) -> torch.Tensor:
  return invalid_body_contacts(env, sensor_name, threshold).sum(dim=1).float()


def traversal_orientation(env, limit_angle):
  robot = env.scene["robot"].data
  angle = (-robot.projected_gravity_b[:, 2]).clamp(-1, 1).acos()
  maximum = torch.where(amp_head(env) == HEADS.index("box"), 1.77, limit_angle)
  return angle > maximum


def terrain_out_of_bounds(
  env,
  distance_buffer: float = 2.0,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Terminate when the robot approaches the outer edge of the terrain map."""
  terrain_cfg = env.cfg.scene.terrain
  if terrain_cfg.terrain_type == "plane":
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  if terrain_cfg.terrain_type not in ("generator", "hacked_generator"):
    raise ValueError(
      "terrain_out_of_bounds supports plane, generator, and hacked_generator "
      f"terrains, got {terrain_cfg.terrain_type!r}."
    )

  generator = terrain_cfg.terrain_generator
  if generator is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  grid_width, grid_length = generator.size
  map_width = generator.num_rows * grid_width + 2.0 * generator.border_width
  map_length = generator.num_cols * grid_length + 2.0 * generator.border_width
  asset: Entity = env.scene[asset_cfg.name]
  root_pos_w = asset.data.root_link_pos_w
  x_limit = max(0.0, 0.5 * map_width - distance_buffer)
  y_limit = max(0.0, 0.5 * map_length - distance_buffer)
  return (torch.abs(root_pos_w[:, 0]) > x_limit) | (
    torch.abs(root_pos_w[:, 1]) > y_limit
  )


def gap_base_height(
  env,
  minimum_height: float,
  gap_terrain_name: str = "square_gaps",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Terminate a fall below the nominal surface only on square-gap terrain."""
  asset: Entity = env.scene[asset_cfg.name]
  terrain = env.scene.terrain
  generator = env.cfg.scene.terrain.terrain_generator
  if generator is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  terrain_names = tuple(generator.sub_terrains)
  if gap_terrain_name not in terrain_names:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  gap_id = terrain_names.index(gap_terrain_name)
  on_gap = terrain.terrain_types == gap_id
  height_from_nominal_surface = (
    asset.data.root_link_pos_w[:, 2] - terrain.env_origins[:, 2]
  )
  return on_gap & (height_from_nominal_surface < minimum_height)


def unstable_velocity(
  env,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  max_root_linear_velocity: float = 10.0,
  max_root_angular_velocity: float = 20.0,
  max_joint_velocity: float = 50.0,
) -> torch.Tensor:
  """Terminate unusually large finite velocities before physics overflows."""
  asset: Entity = env.scene[asset_cfg.name]
  root_linear = torch.linalg.vector_norm(asset.data.root_link_lin_vel_b, dim=1)
  root_angular = torch.linalg.vector_norm(asset.data.root_link_ang_vel_b, dim=1)
  joint = torch.amax(
    torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1
  )
  return (
    (root_linear > max_root_linear_velocity)
    | (root_angular > max_root_angular_velocity)
    | (joint > max_joint_velocity)
  )


def target_reached(
  env,
  command_name: str = "base_velocity",
  contact_sensor_name: str = "feet_ground_contact",
  contact_force_threshold: float = 5.0,
  minimum_upright: float = 0.80,
  maximum_vertical_speed: float = 0.50,
) -> torch.Tensor:
  """End a play route only after reaching the target in a stable state."""
  term = env.command_manager.get_term(command_name)
  if getattr(term.cfg, "fixed_command", None) is not None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  robot: Entity = env.scene[term.cfg.entity_name]
  distance = torch.linalg.vector_norm(
    term.target_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2], dim=1
  )
  sensor: ContactSensor = env.scene[contact_sensor_name]
  assert sensor.data.force_history is not None
  force = torch.linalg.vector_norm(sensor.data.force_history, dim=-1).amax(dim=2)
  supported = (force > contact_force_threshold).any(dim=1)
  upright = -robot.data.projected_gravity_b[:, 2] > minimum_upright
  vertical_stable = (
    torch.abs(robot.data.root_link_lin_vel_w[:, 2]) < maximum_vertical_speed
  )
  reached = (
    term.complex_terrain
    & (distance <= term.cfg.target_distance_threshold)
    & supported
    & upright
    & vertical_stable
  )
  return reached & ~term.is_standing_env & (env.episode_length_buf > 1)


__all__ = [
  "illegal_contact", "traversal_orientation",
  "terrain_out_of_bounds",
  "gap_base_height",
  "target_reached",
  "unstable_velocity",
]
