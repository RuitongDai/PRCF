"""Convert retargeted E1 CSV to mjlab NPZ using CPU kinematics; no W&B upload."""
import argparse
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from prcf.asset.robot.e1.e1_constants import e1_model


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-file', required=True)
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--input-fps', type=float, default=50)
    parser.add_argument('--output-fps', type=float, default=50)
    parser.add_argument('--quaternion-order', choices=('xyzw', 'wxyz'), default='xyzw')
    args = parser.parse_args()
    if not np.isfinite([args.input_fps, args.output_fps]).all() or min(args.input_fps, args.output_fps) <= 0:
        parser.error('frame rates must be positive and finite')
    destination = Path(args.output_file)
    if destination.suffix != '.npz':
        parser.error('--output-file must end with .npz')
    if destination.exists():
        parser.error('output already exists; choose a new path')
    csv = np.loadtxt(args.input_file, delimiter=',', ndmin=2)
    model, joint_names, _ = e1_model()
    if csv.shape[1] != 7 + len(joint_names) or len(csv) < 3 or not np.isfinite(csv).all():
        parser.error('CSV requires >=3 rows of root_xyz, root_quaternion, 25 E1 joints in XML order (32 columns)')
    quaternions = csv[:, 3:7]
    if not np.allclose(np.linalg.norm(quaternions, axis=1), 1, atol=1e-3):
        parser.error('CSV quaternions must have unit norm')
    original_times = np.arange(len(csv)) / args.input_fps
    times = np.arange(int(np.floor(original_times[-1] * args.output_fps + 1e-9)) + 1) / args.output_fps
    times = np.minimum(times, original_times[-1])
    rotations = Rotation.from_quat(quaternions, scalar_first=args.quaternion_order == 'wxyz')
    interpolated = Slerp(original_times, rotations)(times).as_quat(scalar_first=True)
    positions = np.stack([np.interp(times, original_times, csv[:, j]) for j in range(3)], axis=1)
    joints = np.stack([np.interp(times, original_times, csv[:, j]) for j in range(7, csv.shape[1])], axis=1)
    data = build_kinematic_motion(model, positions, interpolated, joints, args.output_fps)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **data)
    print(f'Saved {len(data['joint_pos'])} E1 frames to {destination}. This is a kinematic reference, not a physically validated rollout.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
