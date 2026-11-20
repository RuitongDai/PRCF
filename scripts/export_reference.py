"""Run a Mimic checkpoint and export one AMP observation NPZ per rollout."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np
import torch
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from prcf.asset.scene import SceneBuilder

if __package__:
    from .train import build_runner, dry_run, new_run_dir, prepare_env
else:
    from train import build_runner, dry_run, new_run_dir, prepare_env


# The channel order is part of the AMP observation layout.
CONTACT_BODIES = {
    'left_foot': 'left_ankle_roll_link',
    'right_foot': 'right_ankle_roll_link',
    'left_hand': 'left_wrist_pitch_link',
    'right_hand': 'right_wrist_pitch_link',
    'left_knee': 'left_knee_link',
    'right_knee': 'right_knee_link',
}
CONTACT_FORCE_THRESHOLD = 5.0


AMP_FIELDS = (
    'dof_pos', 'body_pos_local', 'root_lin_vel', 'root_ang_vel',
    'dof_vel', 'contact_state', 'dof_torque', 'height_scan',
)
HEIGHT_GRID_X = tuple(i / 10 for i in range(-5, 16))
HEIGHT_GRID_Y = tuple(i / 10 for i in range(-5, 6))
HEIGHT_RAY_OFFSET = 2.0
HEIGHT_RAY_RANGE = 5.0
HEIGHT_MISS_VALUE = 5.0


def heading_rotation(quaternion):
    """Local-to-world yaw rotation for the pelvis, with world Z as up."""
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    cos, sin = np.cos(yaw), np.sin(yaw)
    rotation = np.zeros((len(quaternion), 3, 3), dtype=np.float64)
    rotation[:, 0, 0] = rotation[:, 1, 1] = cos
    rotation[:, 0, 1] = -sin
    rotation[:, 1, 0] = sin
    rotation[:, 2, 2] = 1
    return rotation


def scan_static_scene(scene, root_position, heading):
    """Downward terrain-only rays in a yaw-aligned grid, returning height and hit mask."""
    x, y = np.meshgrid(HEIGHT_GRID_X, HEIGHT_GRID_Y, indexing='ij')
    grid = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, HEIGHT_RAY_OFFSET)))
    origins = root_position[:, None, :] + grid @ heading.transpose(0, 2, 1)

    spec = mujoco.MjSpec()
    SceneBuilder(scene)(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    down = np.array([0.0, 0.0, -1.0])
    distances = np.fromiter(
        (mujoco.mj_ray(model, data, origin, down, None, True, -1, None)
         for origin in origins.reshape(-1, 3)),
        dtype=np.float64, count=len(root_position) * len(grid),
    ).reshape(len(root_position), len(grid))
    hit = (distances >= 0) & (distances <= HEIGHT_RAY_RANGE)
    height = np.where(hit, distances - HEIGHT_RAY_OFFSET, HEIGHT_MISS_VALUE)
    return height, hit



def collect_rollout(env, policy, robot, body_indices, substep_torques, frames):
    """Record the actual state at each control interval's end."""
    raw = env.unwrapped
    command = raw.command_manager.get_term('motion')
    origin = raw.scene.env_origins[0]
    data = robot.data
    measurements = {name: [] for name in (
        'dof_pos', 'dof_vel', 'body_pos_w', 'root_pos_w', 'root_quat_w',
        'root_lin_vel_w', 'root_ang_vel_w', 'contact_state', 'dof_torque',
    )}
    root = robot.body_names.index('pelvis')
    sensors = [raw.scene[f'contact_{part}'] for part in CONTACT_BODIES]
    observation, _ = env.reset()
    termination = 'motion_finished'
    reference_frames = []
    for _ in range(frames - 1):
        substep_torques.clear()
        observation, _, _, _ = env.step(policy(observation))
        contacts = [sensor.data for sensor in sensors]
        active = torch.stack([
            (contact.found[0, 0] > 0)
            & (contact.force[0, 0].norm() >= CONTACT_FORCE_THRESHOLD)
            for contact in contacts
        ])
        step_data = {
            'dof_pos': data.joint_pos[0],
            'dof_vel': data.joint_vel[0],
            'body_pos_w': data.body_link_pos_w[0, body_indices] - origin,
            'root_pos_w': data.body_link_pos_w[0, root] - origin,
            'root_quat_w': data.body_link_quat_w[0, root],
            'root_lin_vel_w': data.body_link_lin_vel_w[0, root],
            'root_ang_vel_w': data.body_link_ang_vel_w[0, root],
            'contact_state': active,
            'dof_torque': torch.stack(substep_torques).mean(dim=0),
        }
        for name, value in step_data.items():
            measurements[name].append(value.clone())
        reference_frames.append(command.time_steps[0].clone())
        if bool(raw.reset_terminated[0]):
            termination = 'environment_termination'
            break
        if bool(command.finished[0]):
            break
        if bool(raw.reset_time_outs[0]):
            termination = 'time_out'
            break
    samples = {name: torch.stack(values).cpu().numpy() for name, values in measurements.items()}
    samples['reference_frame'] = torch.stack(reference_frames).cpu().numpy()
    return samples, termination


def write_trajectory(path, samples, *, fps, clip, joint_names, body_names, termination, expected_frames):
    root_position = samples['root_pos_w']
    heading = heading_rotation(samples['root_quat_w'])
    height, hit = scan_static_scene(clip.scene, root_position, heading)
    features = {
        'dof_pos': samples['dof_pos'],
        'body_pos_local': (samples['body_pos_w'] - root_position[:, None]) @ heading,
        'root_lin_vel': (samples['root_lin_vel_w'][:, None] @ heading)[:, 0],
        'root_ang_vel': (samples['root_ang_vel_w'][:, None] @ heading)[:, 0],
        'dof_vel': samples['dof_vel'],
        'contact_state': samples['contact_state'],
        'dof_torque': samples['dof_torque'],
        'height_scan': height,
    }
    features = {name: value.astype(np.float32) for name, value in features.items()}
    count = len(root_position)
    amp_obs = np.concatenate([features[name].reshape(count, -1) for name in AMP_FIELDS], axis=1)
    sizes = [features[name][0].size for name in AMP_FIELDS]
    path = Path(path)
    with path.open('xb') as stream:
        np.savez_compressed(
            stream, **features, amp_obs=amp_obs, schema_version=np.int32(1),
            root_pos_w=root_position, root_quat_w=samples['root_quat_w'],
            feature_names=np.asarray(AMP_FIELDS), feature_offsets=np.cumsum([0, *sizes]),
            fps=np.asarray([fps]), time=np.arange(1, count + 1) / fps,
            reference_frame=samples['reference_frame'],
            joint_names=np.asarray(joint_names), body_names=np.asarray(body_names),
            contact_parts=np.asarray(tuple(CONTACT_BODIES)),
            contact_force_threshold=np.float32(CONTACT_FORCE_THRESHOLD),
            coordinate_frame=np.str_('pelvis_yaw_world_z_up'),
            height_grid_x=np.asarray(HEIGHT_GRID_X), height_grid_y=np.asarray(HEIGHT_GRID_Y),
            height_hit=hit, height_ray_offset=np.float32(HEIGHT_RAY_OFFSET),
            height_ray_range=np.float32(HEIGHT_RAY_RANGE), height_miss_value=np.float32(HEIGHT_MISS_VALUE),
            clip_id=np.str_(clip.clip_id), scene_json=np.str_(json.dumps(clip.scene)),
            termination=np.str_(termination), expected_frames=np.int32(expected_frames),
        )
    print(f'Saved {path}  amp_obs={amp_obs.shape}  termination={termination}')


def export_rollouts(args, cfg, agent, clip, frames, fps):
    # Play configuration starts at frame 0. End each attempt at the first done.
    cfg.auto_reset = False
    cfg.commands['motion'].debug_vis = False
    cfg.scene.sensors += tuple(
        ContactSensorCfg(
            name=f'contact_{part}',
            primary=ContactMatch(mode='body', pattern=body, entity='robot'),
            secondary=ContactMatch(mode='body', pattern='world'),
            fields=('found', 'force'), reduce='maxforce',
        )
        for part, body in CONTACT_BODIES.items()
    )

    substep_torques = []

    def capture_torque(env):
        torque = env.scene['robot'].data.qfrc_actuator[0]
        substep_torques.append(torque.clone())
        return torque.new_zeros(env.num_envs)

    cfg.metrics = {'capture_torque': MetricsTermCfg(func=capture_torque, per_substep=True)}
    if args.dry_run:
        print(json.dumps(dry_run(cfg), indent=2))
        return
    if args.output_file:
        directory = args.output_file.parent
        directory.mkdir(parents=True, exist_ok=True)
        if args.output_file.exists():
            raise FileExistsError(args.output_file)
    else:
        directory = new_run_dir(args.log_root, args.task)

    env, runner = build_runner(args.task, cfg, agent, args.device, checkpoint=args.checkpoint)
    try:
        robot = env.unwrapped.scene['robot']
        command = env.unwrapped.command_manager.get_term('motion')
        body_names = tuple(name for name in command.cfg.body_names if name != 'pelvis')
        body_indices = [robot.body_names.index(name) for name in body_names]
        policy = runner.get_inference_policy(device=args.device)
        with torch.inference_mode():
            for attempt in range(args.attempts):
                samples, termination = collect_rollout(env, policy, robot, body_indices, substep_torques, frames)
                name = 'amp_obs.npz' if args.attempts == 1 else f'amp_obs_{attempt:04d}.npz'
                write_trajectory(
                    args.output_file or directory / name, samples, fps=fps, clip=clip,
                    joint_names=robot.joint_names, body_names=body_names, termination=termination,
                    expected_frames=frames - 1,
                )
                if hasattr(policy, 'reset'):
                    policy.reset(torch.ones(1, device=args.device, dtype=torch.bool))
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', nargs='?', default='PRCF-Mimic-E1-25DoF')
    parser.add_argument('--motion-file', '--clip', dest='motion_file', required=True,
                        help='motion NPZ or a JSON scene manifest')
    parser.add_argument('--checkpoint', help='local .pt checkpoint')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-root', '--output-root', dest='log_root', default='logs/exports')
    parser.add_argument('--output-file', type=Path, help='explicit NPZ path for a single attempt')
    parser.add_argument('--dry-run', action='store_true', help='compile the scene without running a policy')
    parser.add_argument('--attempts', type=int, default=1)
    parser.add_argument('--amp', action='store_true', help=argparse.SUPPRESS)  # Legacy command alias.
    args = parser.parse_args(argv)
    if not args.dry_run and not args.checkpoint:
        parser.error('--checkpoint is required for execution')
    if args.attempts < 1:
        parser.error('--attempts must be positive')
    if args.output_file and (args.attempts != 1 or args.output_file.suffix != '.npz'):
        parser.error('--output-file requires a .npz path and one attempt')

    with TemporaryDirectory(prefix='prcf-export-') as temp:
        cfg, agent, clip, frames, fps = prepare_env(
            args.task, args.motion_file, Path(temp) / 'reference.npz', play=True, seed=args.seed)
        export_rollouts(args, cfg, agent, clip, frames, fps)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
