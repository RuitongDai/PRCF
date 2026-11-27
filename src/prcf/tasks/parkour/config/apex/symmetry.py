"""Left/right augmentation of the two APEX observation layouts."""

import torch

from prcf.asset.robot.e1.e1_constants import E1_JOINT_NAMES

from ..baseline.symmetry import mirror_joints
from .mdp.observations import HISTORY
from .sensors import ApexHeightGrid

_ANGULAR_SIGNS = (-1, 1, -1)
_POLAR_SIGNS = (1, -1, 1)
_QUAT_SIGNS = (1, -1, 1, -1)
_JOINT_WIDTH = len(E1_JOINT_NAMES)
_GRID = ApexHeightGrid()
_NX, _NY = (round(size / _GRID.resolution) + 1 for size in _GRID.size)
_SCAN_WIDTH = _NX * _NY


def _mirror_observations(values, *, critic):
    result = values.clone()
    start = 0
    transforms = (
        (_POLAR_SIGNS, _ANGULAR_SIGNS, _QUAT_SIGNS, _POLAR_SIGNS, _POLAR_SIGNS)
        if critic else (_ANGULAR_SIGNS, _POLAR_SIGNS)
    )
    for signs in (*transforms, None, None, None):
        width = _JOINT_WIDTH if signs is None else len(signs)
        stop = start + HISTORY * width
        block = values[..., start:stop].unflatten(-1, (HISTORY, width))
        block = mirror_joints(block) if signs is None else block * block.new_tensor(signs)
        result[..., start:stop] = block.flatten(-2)
        start = stop

    if critic and values.shape[-1] - start - _SCAN_WIDTH == 10:
        # Box height, then the eight up-progress fields and episode time.
        result[..., start + 3:start + 7] = values[..., [start + 4, start + 3, start + 6, start + 5]]
    # GridPatternCfg uses XY indexing with Y on the first grid axis.
    result[..., -_SCAN_WIDTH:] = values[..., -_SCAN_WIDTH:].unflatten(-1, (_NY, _NX)).flip(-2).flatten(-2)
    return result


def mirror_apex(obs=None, actions=None, env=None):
    del env
    if obs is not None:
        mirrored = obs.clone()
        mirrored["actor"] = _mirror_observations(obs["actor"], critic=False)
        mirrored["critic"] = _mirror_observations(obs["critic"], critic=True)
        obs = torch.cat((obs, mirrored), dim=0)
    if actions is not None:
        actions = torch.cat((actions, mirror_joints(actions)), dim=0)
    return obs, actions
