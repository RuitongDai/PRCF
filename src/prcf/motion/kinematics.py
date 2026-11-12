"""Native MuJoCo replay/conversion utilities. These do NOT validate physical execution."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

def set_native_frame(model, state, motion, frame):
    state.qpos[:3] = motion['body_pos_w'][frame, 0]
    state.qpos[3:7] = motion['body_quat_w'][frame, 0]
    state.qpos[7:] = motion['joint_pos'][frame]
    state.qvel[:3] = motion['body_lin_vel_w'][frame, 0]
    rotation = Rotation.from_quat(state.qpos[3:7], scalar_first=True)
    state.qvel[3:6] = rotation.inv().apply(motion['body_ang_vel_w'][frame, 0])
    state.qvel[6:] = motion['joint_vel'][frame]
    mujoco.mj_forward(model, state)


def build_kinematic_motion(model, root_position, root_quaternion, joint_position, fps):
    frames = len(joint_position)
    if frames < 3:
        raise ValueError('Conversion needs at least three frames')
    rotation = Rotation.from_quat(root_quaternion, scalar_first=True)
    dt = 1.0 / fps
    linear = np.gradient(root_position, dt, axis=0)
    qvel = np.gradient(joint_position, dt, axis=0)
    angular = np.empty((frames, 3))
    angular[1:-1] = (rotation[2:] * rotation[:-2].inv()).as_rotvec() / (2 * dt)
    angular[0] = (rotation[1] * rotation[0].inv()).as_rotvec() / dt
    angular[-1] = (rotation[-1] * rotation[-2].inv()).as_rotvec() / dt
    result = {'joint_pos': joint_position, 'joint_vel': qvel,
              'body_pos_w': np.empty((frames, model.nbody - 1, 3)),
              'body_quat_w': np.empty((frames, model.nbody - 1, 4)),
              'body_lin_vel_w': np.empty((frames, model.nbody - 1, 3)),
              'body_ang_vel_w': np.empty((frames, model.nbody - 1, 3))}
    state, velocity = mujoco.MjData(model), np.empty(6)
    for i in range(frames):
        state.qpos[:3], state.qpos[3:7], state.qpos[7:] = root_position[i], root_quaternion[i], joint_position[i]
        state.qvel[:3], state.qvel[3:6], state.qvel[6:] = linear[i], rotation[i].inv().apply(angular[i]), qvel[i]
        mujoco.mj_forward(model, state)
        result['body_pos_w'][i] = state.xpos[1:]
        result['body_quat_w'][i] = state.xquat[1:]
        for body in range(1, model.nbody):
            mujoco.mj_objectVelocity(model, state, mujoco.mjtObj.mjOBJ_XBODY, body, velocity, 0)
            result['body_ang_vel_w'][i, body - 1] = velocity[:3]
            result['body_lin_vel_w'][i, body - 1] = velocity[3:]
    result['fps'] = np.asarray([fps])
    result['joint_names'] = np.asarray([model.joint(i).name for i in range(1, model.njnt)])
    result['body_names'] = np.asarray([model.body(i).name for i in range(1, model.nbody)])
    return result
