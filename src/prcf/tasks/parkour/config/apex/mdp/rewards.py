"""APEX task rewards and regularization adapted to E1."""

from itertools import product
import math

import torch
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.utils.lab_api.math import quat_apply

from ..poses import STANDING_PELVIS_HEIGHT, prone_joint_pos
from ..terrains import BOX_EDGE_X
from .progress import progress_state


def wait_at_edge(env, *, descent):
    data = env.scene["robot"].data
    speed = data.root_link_lin_vel_w.norm(dim=1)
    if descent:
        on_platform = data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0] > BOX_EDGE_X + 0.10
        return (speed < 0.10) & on_platform
    incomplete = data.body_link_pos_w[..., 2].amin(dim=1) < env.scene.env_origins[:, 2]
    return (speed < 0.15) & incomplete


def prone_pose(env):
    state = progress_state(env)
    q = env.scene["robot"].data.joint_pos
    score = torch.exp(-0.1 * (q - prone_joint_pos(q)).square().sum(dim=1))
    return score * state.at_goal * (env.episode_length_buf > 200)


def elbows_knees_on_top(env):
    state = progress_state(env)
    return state.support_count * state.at_goal * (env.episode_length_buf > 200)


def standing_joint(env):
    q = env.scene["robot"].data.joint_pos
    nominal = env.action_manager.get_term("joint_pos").nominal_pose
    return torch.exp(-0.1 * (q - nominal).square().sum(dim=1)) * progress_state(env).at_goal


def standing_orientation(env):
    gravity = env.scene["robot"].data.projected_gravity_b[:, :2]
    return torch.exp(-5.0 * gravity.square().sum(dim=1)) * progress_state(env).at_goal


def standing_linear_velocity(env, *, asset_cfg):
    vel = env.scene["robot"].data.body_link_lin_vel_w[:, asset_cfg.body_ids[0]]
    return torch.exp(-5.0 * vel.square().sum(dim=1)) * progress_state(env).feet_near_ground


def standing_angular_velocity(env, *, asset_cfg):
    vel = env.scene["robot"].data.body_link_ang_vel_w[:, asset_cfg.body_ids[0]]
    return torch.exp(-2.0 * vel.square().sum(dim=1)) * progress_state(env).feet_near_ground


def standing_height(env, *, asset_cfg):
    data = env.scene["robot"].data
    shoulder_height = data.body_link_pos_w[:, asset_cfg.body_ids, 2].mean(dim=1)
    # Preserve the original 0.27 m scoring interval relative to the E1 stance.
    scale = (shoulder_height - (STANDING_PELVIS_HEIGHT + 0.025)) / 0.27
    return scale.clamp(0.0, 1.0) * progress_state(env).at_goal


class FootPressure(ManagerTermBase):
    """Estimate pressure as net foot force divided by a ground-plane cross section."""

    def __init__(self, cfg, env):
        super().__init__(env)
        signs = list(product((-1, 1), repeat=3))
        # Envelope of E1's seven sole capsules. Dimensions include their radius.
        half_size = torch.tensor((0.107, 0.047, 0.010), device=env.device)
        self.vertices = half_size * torch.tensor(signs, device=env.device)
        self.vertices[:, 2] -= 0.0578
        self.edges = torch.tensor(
            [(i, i ^ bit) for i in range(8) for bit in (1, 2, 4) if i & bit == 0],
            device=env.device,
        )
        self.columns = torch.arange(len(self.edges), device=env.device)
        self.force_ids = [
            env.scene["terrain_forces"].primary_names.index(name)
            for name in ("left_ankle_roll_link", "right_ankle_roll_link")
        ]

    def __call__(self, env, *, descent, asset_cfg):
        data = env.scene["robot"].data
        foot_quat = data.body_link_quat_w[:, asset_cfg.body_ids, None, :]
        foot_pos = data.body_link_pos_w[:, asset_cfg.body_ids, None, :]
        vertices = quat_apply(
            foot_quat.expand(-1, -1, 8, -1), self.vertices.expand(env.num_envs, 2, -1, -1),
        ) + foot_pos
        start, end = vertices[:, :, self.edges, :].unbind(dim=3)
        h0 = start[..., 2] - 0.01
        h1 = end[..., 2] - 0.01
        crosses = (h0 <= 0) != (h1 <= 0)
        denominator = torch.where(crosses, h1 - h0, 1.0)
        fraction = (-h0 / denominator).clamp(0.0, 1.0)
        xy = start[..., :2] + fraction[..., None] * (end[..., :2] - start[..., :2])
        count = crosses.sum(dim=-1, keepdim=True)
        centroid = torch.where(crosses[..., None], xy, 0.0).sum(dim=-2) / count.clamp_min(1)
        offset = xy - centroid[..., None, :]
        angle = torch.atan2(offset[..., 1], offset[..., 0]).masked_fill(~crosses, torch.inf)
        order = angle.argsort(dim=-1)
        xy = offset.gather(-2, order[..., None].expand(-1, -1, -1, 2))
        following = (self.columns + 1) % count.clamp_min(1)
        next_xy = xy.gather(-2, following[..., None].expand(-1, -1, -1, 2))
        cross = xy[..., 0] * next_xy[..., 1] - xy[..., 1] * next_xy[..., 0]
        area = 0.5 * torch.where(self.columns < count, cross, 0).sum(dim=-1).abs()
        net_force = env.scene["terrain_forces"].data.force[:, self.force_ids].norm(dim=-1)
        pressure = torch.where(area > 1e-8, net_force / area.clamp_min(1e-8), 0.0).amax(dim=1)
        threshold, gain = (10000.0, 1e-5) if descent else (100000.0, 1e-4)
        return ((pressure - threshold) * gain).clamp(0.0, 100.0)


def peak_force(env, *, sensor_name, asset_cfg=None):
    force = env.scene[sensor_name].data.force_history
    if asset_cfg is not None:
        force = force[:, asset_cfg.body_ids]
    return force.norm(dim=-1).amax(dim=(1, 2))


def excessive_force(env, *, sensor_name, threshold, gain, asset_cfg=None):
    force = peak_force(env, sensor_name=sensor_name, asset_cfg=asset_cfg)
    return torch.expm1(((force - threshold).clamp_min(0.0) * gain).clamp_max(math.log(201.0)))


def contact_slip(env, *, descent):
    force = env.scene["body_forces"].data.force.norm(dim=-1)
    vel = env.scene["robot"].data.body_link_lin_vel_w
    speed = vel.norm(dim=-1) * force / 100.0 if descent else vel[..., :2].norm(dim=-1)
    return torch.where(force > 1.0, speed, 0.0).sum(dim=1)


def group_air_time(env, *, upper_cfg, lower_cfg, feet_cfg):
    air = env.scene["terrain_forces"].data.current_air_time
    upper = air[:, upper_cfg.body_ids].amin(dim=1)
    lower = air[:, lower_cfg.body_ids].amin(dim=1)
    feet = air[:, feet_cfg.body_ids].amax(dim=1)
    duration = torch.maximum(upper, lower).minimum(feet)
    return torch.expm1((20.0 * duration).clamp_max(math.log(201.0)))


def action_target_rate(env):
    action = env.action_manager.get_term("joint_pos")
    return (action.target - action.previous_target).square().sum(dim=1)


def hip_joint_deviation(env, *, yaw_cfg, roll_cfg):
    q = env.scene["robot"].data.joint_pos.abs()
    return (q[:, yaw_cfg.joint_ids] > 1.5).any(dim=1) | (q[:, roll_cfg.joint_ids] > 1.4).any(dim=1)


def waist_joint_deviation(env, *, asset_cfg):
    q = env.scene["robot"].data.joint_pos[:, asset_cfg.joint_ids]
    return (q.abs() > 1.4).any(dim=1)


def mechanical_power(env):
    robot = env.scene["robot"].data
    return (robot.actuator_force * robot.joint_vel).abs().sum(dim=1)


def joint_speed_exp(env):
    max_speed = env.scene["robot"].data.joint_vel.amax(dim=1)
    return torch.expm1((0.60 * (max_speed - 10.0).clamp_min(0.0)).clamp_max(math.log(201.0)))


def joint_speed_limit(env, *, limits):
    speed = env.scene["robot"].data.joint_vel.abs()
    return (speed - speed.new_tensor(limits)).clamp_min(0).sum(dim=1)


def angular_velocity_xy(env):
    return env.scene["robot"].data.root_link_ang_vel_b[:, :2].square().sum(dim=1)


def torso_acceleration(env):
    lin = env.scene["robot/torso_lin_acc"].data
    ang = env.scene["robot/torso_ang_acc"].data
    return lin.square().sum(dim=1) + 0.02 * ang.square().sum(dim=1)


def body_linear_acceleration(env):
    data = env.scene["robot"].data
    # Convert MuJoCo's spatial acceleration at the subtree CoM to classical
    # acceleration at each link origin, as mjSENS_FRAMELINACC does.
    acc = data.data.cacc[:, data.indexing.body_ids]
    offset = data.body_link_pos_w - data.data.subtree_com[:, data.indexing.root_body_id, None, :]
    vel = data.body_link_vel_w
    linear_acc = acc[..., 3:] - torch.cross(offset, acc[..., :3], dim=-1)
    linear_acc = linear_acc + torch.cross(vel[..., 3:], vel[..., :3], dim=-1)
    return linear_acc.norm(dim=-1).sum(dim=1)
