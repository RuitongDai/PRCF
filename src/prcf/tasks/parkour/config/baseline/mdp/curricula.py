"""Perlin travel curriculum with legacy navigation curriculum elsewhere."""

import torch


def terrain_tracking_progress(
  env,
  env_ids,
  command_name="base_velocity",
  linear_thresholds=(0.30, 0.60),
  yaw_thresholds=(0.30, 0.50),
  complex_progress_fraction=0.70,
  normalize_by_elapsed=False,
  require_target_reached=False,
  perlin_min_travel=1.5,
  perlin_min_linear_tracking=0.55,
  perlin_min_yaw_tracking=0.45,
):
  if isinstance(env_ids, slice):
    env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
  terrain = env.scene.terrain
  term = env.command_manager.get_term(command_name)
  linear = term.metrics["tracking_exp_vel_xy"][env_ids]
  yaw = term.metrics["tracking_exp_vel_yaw"][env_ids]
  # Command metrics are accumulated with max_episode_length as denominator.
  # Renormalize by the actual completed length so early target completion in
  # play receives a genuine episode-average tracking score.
  if normalize_by_elapsed:
    elapsed = env.episode_length_buf[env_ids].float().clamp_min(1.0)
    renormalize = float(env.max_episode_length) / elapsed
    linear = (linear * renormalize).clamp(0.0, 1.0)
    yaw = (yaw * renormalize).clamp(0.0, 1.0)
  progress = term.metrics["target_progress"][env_ids]
  complex_mask = term.complex_terrain[env_ids]
  perlin_mask = ~complex_mask
  standing = term.is_standing_env[env_ids]
  valid = env.episode_length_buf[env_ids] > 0
  moving_steps = term.metrics["moving_steps"][env_ids].clamp_min(1.0)
  moving_linear = term.metrics["moving_tracking_linear_sum"][env_ids] / moving_steps
  moving_yaw = term.metrics["moving_tracking_yaw_sum"][env_ids] / moving_steps
  travel = term.metrics["cumulative_travel"][env_ids]
  time_out = env.termination_manager.get_term("time_out")[env_ids]

  route_ok = progress >= complex_progress_fraction
  if require_target_reached:
    if "target_reached" not in env.termination_manager.active_terms:
      raise RuntimeError(
        "require_target_reached=True requires a target_reached termination term."
      )
    target_success = env.termination_manager.get_term("target_reached")[env_ids]
  else:
    target_success = torch.ones_like(route_ok)
  complex_up = (
    complex_mask
    & (linear > linear_thresholds[1])
    & (yaw > yaw_thresholds[1])
    & ~standing
    & valid
    & route_ok
    & target_success
  )
  perlin_up = (
    perlin_mask
    & time_out
    & ~standing
    & (travel >= perlin_min_travel)
    & (moving_linear >= perlin_min_linear_tracking)
    & (moving_yaw >= perlin_min_yaw_tracking)
  )
  move_up = complex_up | perlin_up
  complex_down = (
    complex_mask
    & (
      (linear < linear_thresholds[0])
      | (yaw < yaw_thresholds[0])
      | (~standing & ~route_ok)
      | (require_target_reached & ~target_success)
    )
    & ~move_up
    & valid
  )
  # Match the hybrid task on Perlin: a reset that does not satisfy the full
  # timeout/travel/tracking gate moves down. Navigation keeps its neutral band.
  perlin_down = perlin_mask & valid & ~perlin_up
  move_down = complex_down | perlin_down
  terrain.update_env_origins(env_ids, move_up, move_down)

  return {
    "mean_level": terrain.terrain_levels.float().mean(),
    "max_level": terrain.terrain_levels.float().max(),
    "move_up_fraction": move_up.float().mean(),
    "move_down_fraction": move_down.float().mean(),
    "tracking_linear": linear.mean(),
    "tracking_yaw": yaw.mean(),
    "perlin_tracking_linear": (
      moving_linear[perlin_mask].mean() if perlin_mask.any()
      else moving_linear.new_zeros(())
    ),
    "perlin_tracking_yaw": (
      moving_yaw[perlin_mask].mean() if perlin_mask.any()
      else moving_yaw.new_zeros(())
    ),
    "perlin_travel": (
      travel[perlin_mask].mean() if perlin_mask.any() else travel.new_zeros(())
    ),
    "perlin_success": perlin_up.float().mean(),
    "complex_progress_ok": (route_ok & complex_mask).float().mean(),
    "target_success": target_success.float().mean(),
  }


__all__ = ["terrain_tracking_progress"]
