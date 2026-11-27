"""E1 joint template and pelvis alignment for the box-top prone pose."""

import math

import torch

from prcf.asset.robot.e1.e1_constants import E1_JOINT_NAMES

PRONE_PELVIS_HEIGHT = 0.337
PRONE_PITCH = math.radians(110)
STANDING_PELVIS_HEIGHT = 0.665

_PRONE_JOINTS = {
    "hip_pitch": -1.50,
    "hip_roll": 0.30,
    "knee": 1.50,
    "ankle_pitch": -0.23,
    "shoulder_pitch": -1.80,
    "shoulder_roll": 0.50,
}


def prone_joint_pos(like: torch.Tensor) -> torch.Tensor:
    values = []
    for name in E1_JOINT_NAMES:
        side, _, joint = name.partition("_")
        joint = joint.removesuffix("_joint")
        value = _PRONE_JOINTS.get(joint, 0.0)
        sign = -1 if side == "right" and joint in ("hip_roll", "shoulder_roll") else 1
        values.append(sign * value)
    return like.new_tensor(values)
