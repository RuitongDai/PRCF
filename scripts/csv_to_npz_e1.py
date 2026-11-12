"""Convert retargeted E1 CSV to mjlab NPZ using CPU kinematics; no W&B upload."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from prcf.motion.kinematics import build_kinematic_motion
from prcf.motion.validation import e1_model, validate_motion


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
    report = validate_motion(data, model=model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **data)
    print(f'Saved {report["frames"]} E1 frames to {destination}. This is a kinematic reference, not a physically validated rollout.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
