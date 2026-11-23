from dataclasses import dataclass

import torch

from mjlab.sensor import RayCastSensor, RayCastSensorCfg

from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

from prcf.tasks.mimic.config.e1.env_cfgs import TRACKING_BODIES

from ..terrains import amp_head

CONTACT_BODIES = {
    "left_foot": "left_ankle_roll_link", "right_foot": "right_ankle_roll_link",
    "left_hand": "left_wrist_pitch_link", "right_hand": "right_wrist_pitch_link",
    "left_knee": "left_knee_link", "right_knee": "right_knee_link",
}

CONTACT_FORCE_THRESHOLD = 5.0

AMP_FIELDS = ("dof_pos", "body_pos_local", "root_lin_vel", "root_ang_vel",
              "dof_vel", "contact_state", "dof_torque", "height_scan")

HEIGHT_GRID_X = tuple(i / 10 for i in range(-5, 16))

HEIGHT_GRID_Y = tuple(i / 10 for i in range(-5, 6))

HEIGHT_RAY_OFFSET = 2.0

HEIGHT_RAY_RANGE = 5.0

HEIGHT_MISS_VALUE = 5.0

SCAN_DIM = len(HEIGHT_GRID_X) * len(HEIGHT_GRID_Y)

AMP_DIM = 357

@dataclass
class AmpGridPattern:
    def generate_rays(self, mj_model, device):
        x, y = torch.meshgrid(torch.tensor(HEIGHT_GRID_X, device=device),
                              torch.tensor(HEIGHT_GRID_Y, device=device), indexing="ij")
        z = torch.full_like(x, HEIGHT_RAY_OFFSET)
        offsets = torch.stack((x, y, z), dim=-1).reshape(-1, 3)
        directions = torch.zeros_like(offsets)
        directions[:, 2] = -1.0
        return offsets, directions

class AmpRayCastSensor(RayCastSensor):
    def _extract_yaw_rotation(self, rot_mat):
        # Use the exported convention even near vertical pitch, where the stock
        # sensor switches to a lateral-axis heading.
        yaw = torch.atan2(rot_mat[:, 1, 0], rot_mat[:, 0, 0])
        c, s = yaw.cos(), yaw.sin()
        zero, one = torch.zeros_like(c), torch.ones_like(c)
        return torch.stack((c, -s, zero, s, c, zero, zero, zero, one), dim=-1).reshape(-1, 3, 3)

@dataclass
class AmpRayCastCfg(RayCastSensorCfg):
    def build(self):
        return AmpRayCastSensor(self)

def scan_height(env):
    distance = env.scene["amp_scan"].data.distances
    hits = distance >= 0
    return torch.where(hits, distance - HEIGHT_RAY_OFFSET, HEIGHT_MISS_VALUE)

def contact_force(env, part):
    sample = env.scene[f"contact_{part}"].data
    return torch.where(sample.found[:, 0] > 0, sample.force[:, 0].norm(dim=-1), 0.0)

def capture_torque(env):
    raw = env.scene["robot"].data.qfrc_actuator
    if not hasattr(env, "baseline_torque"):
        env.baseline_torque = torch.zeros_like(raw)
    if (env._sim_step_counter - 1) % env.cfg.decimation == 0:
        env.baseline_torque.zero_()
    env.baseline_torque.add_(raw / env.cfg.decimation)
    return raw.square().mean(-1)

class AmpFeatures:
    def __init__(self, cfg, env):
        robot = env.scene["robot"]
        self.body_ids = [robot.body_names.index(name) for name in TRACKING_BODIES[1:]]

    def __call__(self, env):
        data = env.scene["robot"].data
        heading = yaw_quat(data.root_link_quat_w)
        local = data.body_link_pos_w[:, self.body_ids] - data.root_link_pos_w[:, None]
        local = quat_apply_inverse(heading[:, None].expand(-1, local.shape[1], -1), local)
        contacts = torch.stack([contact_force(env, part) >= CONTACT_FORCE_THRESHOLD for part in CONTACT_BODIES], dim=-1)
        torque = getattr(env, "baseline_torque", None)
        if torque is None:
            torque = torch.zeros_like(data.joint_pos)
        return torch.cat((
            data.joint_pos, local.flatten(1), quat_apply_inverse(heading, data.root_link_lin_vel_w),
            quat_apply_inverse(heading, data.root_link_ang_vel_w), data.joint_vel, contacts,
            torque, scan_height(env),
        ), dim=-1)

def route(env):
    return amp_head(env).unsqueeze(-1).float()
