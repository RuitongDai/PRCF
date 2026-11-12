"""A separate reference-residual experiment; baseline checkpoints keep their semantics."""
from dataclasses import fields
from copy import deepcopy
from mjlab.managers import RewardTermCfg
from ..e1.env_cfgs import e1_mimic_env_cfg
from .mdp.actions import ReferenceJointPositionActionCfg
from .mdp.rewards import feet_world_position_tracking


def e1_reference_env_cfg(mode='train'):
    cfg = e1_mimic_env_cfg(mode)
    original = cfg.actions['joint_pos']
    cfg.actions['joint_pos'] = ReferenceJointPositionActionCfg(**{
        item.name: deepcopy(getattr(original, item.name)) for item in fields(original)
    })
    cfg.rewards['feet_world_position'] = RewardTermCfg(
        func=feet_world_position_tracking, weight=1.0, params={'std': 0.15})
    return cfg
