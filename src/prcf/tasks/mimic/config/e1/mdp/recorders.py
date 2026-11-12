"""Export-only recorder: refresh-safe state snapshots and applied substep forces."""

import numpy as np
import torch
from mjlab.managers.recorder_manager import RecorderTerm
from prcf.motion.recording import TrajectoryBuffer
from prcf.motion.schema import CONTACT_PARTS
from ..env_cfgs import CONTACT_BODIES


def _numpy(tensor):
    return tensor.detach().cpu().numpy().copy()


def _minimum_distance(sensor):
    data = sensor.data
    found = data.found[0] > 0
    return float(data.dist[0][found].min().item()) if found.any() else 0.0


def contact_snapshot(env, *, refresh=False):
    if refresh:
        # Per-substep recording populated sensor caches before env.step's final
        # forward(). Refresh snapshots without advancing physical time. These
        # recorder sensors have neither history buffers nor air-time tracking.
        for name in [*(f'contact_{part}' for part in CONTACT_PARTS),
                     'environment_collision', 'self_collision']:
            env.scene[name].update(0.0)
    robot = env.scene['robot']
    found, forces, positions, normals, velocities = [], [], [], [], []
    origin = env.scene.env_origins[0]
    for part, body in zip(CONTACT_PARTS, CONTACT_BODIES, strict=True):
        data = env.scene[f'contact_{part}'].data
        active = bool((data.found[0, 0] > 0).item())
        force, position, normal = data.force[0, 0], data.pos[0, 0], data.normal[0, 0]
        body_id = robot.body_names.index(body)
        velocity = robot.data.body_link_lin_vel_w[0, body_id] + torch.cross(
            robot.data.body_link_ang_vel_w[0, body_id],
            position - robot.data.body_link_pos_w[0, body_id], dim=-1,
        )
        found.append(active)
        forces.append(_numpy(force) if active else np.zeros(3))
        positions.append(_numpy(position - origin) if active else np.zeros(3))
        normals.append(_numpy(normal) if active else np.zeros(3))
        velocities.append(_numpy(velocity) if active else np.zeros(3))
    return {
        'contact_found': np.asarray(found), 'strongest_contact_force_w': np.stack(forces),
        'contact_position_w': np.stack(positions), 'contact_normal_w': np.stack(normals),
        'contact_point_velocity_w': np.stack(velocities),
        'min_environment_distance': np.asarray(_minimum_distance(env.scene['environment_collision'])),
        'min_self_distance': np.asarray(_minimum_distance(env.scene['self_collision'])),
    }


class TrajectoryRecorder(RecorderTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if env.num_envs != 1 or env.cfg.auto_reset:
            raise ValueError("Trajectory export requires num_envs=1 and auto_reset=False")
        self.buffer = TrajectoryBuffer()
        self._substeps = []

    def _state(self):
        env = self._env
        robot = env.scene['robot']
        command = env.command_manager.get_term('motion')
        return {
            'joint_pos': _numpy(robot.data.joint_pos[0]),
            'joint_vel': _numpy(robot.data.joint_vel[0]),
            'body_pos_w': _numpy(robot.data.body_link_pos_w[0] - env.scene.env_origins[0]),
            'body_quat_w': _numpy(robot.data.body_link_quat_w[0]),
            'body_lin_vel_w': _numpy(robot.data.body_link_lin_vel_w[0]),
            'body_ang_vel_w': _numpy(robot.data.body_link_ang_vel_w[0]),
            'time': np.asarray(len(self.buffer.states) * env.step_dt),
            'reference_frame': _numpy(command.time_steps[0]),
            'action': _numpy(env.action_manager.action[0]),
            'joint_position_target': _numpy(robot.data.joint_pos_target[0]),
            'qfrc_actuator_snapshot': _numpy(robot.data.qfrc_actuator[0]),
            **contact_snapshot(env, refresh=True),
        }

    def record_post_reset(self, env_ids):
        if 0 in env_ids:
            self.buffer = TrajectoryBuffer()
            self._substeps = []
            self.buffer.add_state(self._state())

    def sample_substep(self):
        # Called immediately after mj_step, before forward() recomputes PD forces.
        self._substeps.append({
            'qfrc_actuator_substep': _numpy(self._env.scene['robot'].data.qfrc_actuator[0]),
            **{f'{key}_substep': value for key, value in contact_snapshot(self._env).items()},
        })

    def record_post_step(self):
        if len(self._substeps) != self._env.cfg.decimation:
            raise RuntimeError("Physics recorder did not capture every substep")
        self.buffer.add_interval(self._substeps)
        self._substeps = []
        self.buffer.add_state(self._state())


def capture_substep(env):
    env.recorder_manager.get_term('trajectory').sample_substep()
    return torch.zeros(env.num_envs, device=env.device)
