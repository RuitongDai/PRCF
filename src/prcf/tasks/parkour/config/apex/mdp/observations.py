"""APEX observation order, E1 joint history, and privileged critic terms."""

from copy import deepcopy

import torch
from mjlab.envs import mdp
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from mjlab.utils.noise import UniformNoiseCfg

from .progress import UP_PROGRESS_DEFAULTS, progress_state

HISTORY = 6


def height_scan(env):
    data = env.scene["height_scan"].data
    relative = data.frame_pos_w[:, 0, 2:3] - data.hit_pos_w[..., 2]
    hit = (data.distances >= 0.0) & torch.isfinite(relative)
    return torch.where(hit, relative, 2.0).clamp(-1.0, 2.0)


def previous_target(env):
    return env.action_manager.get_term("joint_pos").target


def box_height(env):
    return env.scene.env_origins[:, 2:3]


def progress_memory(env, *, descent):
    # mjlab probes observation widths before constructing the reward manager.
    if not hasattr(env, "reward_manager"):
        width = 2 if descent else 2 + len(UP_PROGRESS_DEFAULTS)
        return env.scene.env_origins.new_zeros((env.num_envs, width))
    return progress_state(env).memory()


def episode_time(env):
    return env.episode_length_buf[:, None] * env.step_dt


def torso_state(env, *, field, asset_cfg):
    data = env.scene["robot"].data
    index = asset_cfg.body_ids[0]
    if field == "position":
        return data.body_link_pos_w[:, index] - env.scene.env_origins
    if field == "orientation":
        quat = data.body_link_quat_w[:, index]
        return torch.where(quat[:, :1] < 0, -quat, quat)
    if field == "linear_velocity":
        return data.body_link_lin_vel_w[:, index]
    return data.body_link_ang_vel_w[:, index]


def observations_cfg(*, descent, play):
    actor = {
        "ang_vel": ObservationTermCfg(func=mdp.base_ang_vel, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2)),
        "gravity": ObservationTermCfg(func=mdp.projected_gravity, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05)),
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01)),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5)),
        "last_target": ObservationTermCfg(func=previous_target),
    }
    for term in actor.values():
        term.history_length = HISTORY
    actor["height_scan"] = ObservationTermCfg(func=height_scan, noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1))

    torso = SceneEntityCfg("robot", body_names=("torso_link",))
    linear, angular, orientation, position = (
        ObservationTermCfg(func=torso_state, params={"field": field, "asset_cfg": torso}, history_length=HISTORY)
        for field in ("linear_velocity", "angular_velocity", "orientation", "position")
    )
    critic = {
        "torso_lin_vel": linear,
        "torso_ang_vel": angular,
        "torso_quat": orientation,
        "gravity": deepcopy(actor["gravity"]),
        "torso_pos": position,
        "joint_pos": deepcopy(actor["joint_pos"]),
        "joint_vel": deepcopy(actor["joint_vel"]),
        "last_target": deepcopy(actor["last_target"]),
        "box_height": ObservationTermCfg(func=box_height),
        "progress": ObservationTermCfg(func=progress_memory, params={"descent": descent}),
        "time": ObservationTermCfg(func=episode_time),
        "height_scan": deepcopy(actor["height_scan"]),
    }
    return {
        "actor": ObservationGroupCfg(terms=actor, enable_corruption=not play),
        "critic": ObservationGroupCfg(terms=critic, enable_corruption=False),
    }
