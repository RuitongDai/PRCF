"""Velocity tracking and task penalties for E1 traversal."""
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

from .observations import HEIGHT_GRID_X, HEIGHT_GRID_Y, contact_force, scan_height
from ..terrains import SingleBoxTerrainCfg, on_terrain, terrain_configs


CENTER_RAY = HEIGHT_GRID_X.index(0.0) * len(HEIGHT_GRID_Y) + HEIGHT_GRID_Y.index(0.0)


def _course_limits(env):
    """Approach, exit, corridor width and tile bounds relative to each spawn."""
    limits = []
    for tile in terrain_configs(env).values():
        sx, sy = tile.size
        if isinstance(tile, SingleBoxTerrainCfg):
            limits.append((tile.front_x - 0.38, tile.end_x + 0.15, tile.box_width / 2 - 0.02,
                           -tile.spawn_x, sx - tile.spawn_x, sy / 2))
        else:
            limits.append((-sx / 2, sx / 2 - tile.border_width + 0.10, sy / 2 - 0.25,
                           -sx / 2, sx / 2, sy / 2))
    return torch.tensor(limits, device=env.device)


def track_planar_velocity(env, std):
    robot = env.scene["robot"].data
    heading = yaw_quat(robot.root_link_quat_w)
    velocity = quat_apply_inverse(heading, robot.root_link_lin_vel_w)
    command = env.command_manager.get_command("twist")
    return torch.exp(-((velocity[:, :2] - command[:, :2]).square().sum(-1)) / std**2)


def track_yaw_velocity(env, std):
    velocity = env.scene["robot"].data.root_link_ang_vel_w[:, 2]
    command = env.command_manager.get_command("twist")[:, 2]
    return torch.exp(-(velocity - command).square() / std**2)


def penalize_torso_contact(env):
    return (contact_force(env, "torso") > 10).float()


def bad_base_pose(env):
    robot = env.scene["robot"].data
    pos = robot.root_link_pos_w - env.scene.env_origins
    x = pos[:, 0]
    on_box_top = torch.zeros_like(x, dtype=torch.bool)
    types = env.scene.terrain.terrain_types
    for column, tile in enumerate(terrain_configs(env).values()):
        if isinstance(tile, SingleBoxTerrainCfg):
            on_box_top |= (types == column) & (x >= tile.front_x) & (x <= tile.end_x)
    height = scan_height(env)[:, CENTER_RAY]
    # Above the leading lip a legitimate climb can briefly bring the base very close to the top.
    minimal_height = torch.where(on_box_top, 0.06, 0.28)
    below_gap = on_terrain(env, "square_gaps") & (pos[:, 2] < -0.30)
    return below_gap | (height < minimal_height) | (robot.projected_gravity_b[:, 2] > 0.20)


class LeftCourse:
    def __init__(self, cfg, env):
        self.corridors = _course_limits(env)

    def __call__(self, env):
        start, end, width, left, right, tile_half_width = self.corridors[env.scene.terrain.terrain_types].unbind(1)
        pos = env.scene["robot"].data.root_link_pos_w - env.scene.env_origins
        x, y = pos[:, 0], pos[:, 1].abs()
        outside = (x < left) | (x > right) | (y > tile_half_width)
        return outside | ((x >= start) & (x < end) & (y > width))


class CourseSuccess:
    def __init__(self, cfg, env):
        self.corridors = _course_limits(env)
        self.stable_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
        self.required_steps = round(1.0 / env.step_dt)

    def reset(self, env_ids=None):
        self.stable_steps[slice(None) if env_ids is None else env_ids] = 0

    def __call__(self, env):
        robot = env.scene["robot"].data
        pos = robot.root_link_pos_w - env.scene.env_origins
        x = pos[:, 0]
        _, end, _, _, right, tile_half_width = self.corridors[env.scene.terrain.terrain_types].unbind(1)
        upright = robot.projected_gravity_b[:, 2] < -0.65
        stable = upright & (x >= end) & (x <= right) & (pos[:, 1].abs() <= tile_half_width) & ~bad_base_pose(env)
        self.stable_steps.copy_(torch.where(stable, self.stable_steps + 1, 0))
        return self.stable_steps >= self.required_steps


def success_indicator(env):
    return env.termination_manager.get_term("success").float()
