"""Train a registered PRCF task with local TensorBoard logs."""

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
import json
import math
from pathlib import Path

import numpy as np
import torch

from prcf.asset.scene import SceneBuilder, load_aligned_motion, load_clip
from prcf.tasks.parkour.config.TIMP import TASK_ID as TIMP_TASK_ID, timp_env_cfg, timp_rl_cfg
from prcf.tasks.parkour.config.TIMP.rl.motion_loader import DEFAULT_POOLS, ReferencePools
from prcf.tasks.parkour.config.TIMP.terrains import TERRAINS


def jsonable(value):
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(x) for x in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, slice):
        return {'slice': [value.start, value.stop, value.step]}
    if isinstance(value, Enum):
        return jsonable(value.value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if callable(value):
        return f'{value.__module__}:{value.__qualname__}'
    return value


def write_json(path, data):
    Path(path).write_text(json.dumps(jsonable(data), indent=2, allow_nan=False) + '\n')


def new_run_dir(root, task_id):
    directory = Path(root) / task_id / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory.mkdir(parents=True, exist_ok=False)
    return directory.resolve()


def prepare_env(task_id, source, reference_path, *, play=False, num_envs=1, seed=42):
    """Apply the clip's scene and motion to the chosen task configuration."""
    import prcf.tasks  # Registers task IDs.
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    clip = load_clip(source)
    motion = load_aligned_motion(clip)
    fps = float(motion['fps'].item())
    frames = len(motion['joint_pos'])
    if fps <= 0 or not math.isfinite(fps) or frames < 2:
        raise ValueError('Motion requires a positive frame rate and at least two frames')

    cfg = load_env_cfg(task_id, play=play)
    cfg.scene.spec_fn = SceneBuilder(clip.scene)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.sim.mujoco.timestep = 1.0 / (fps * cfg.decimation)
    cfg.episode_length_s = (frames + 2) / fps
    np.savez_compressed(reference_path, **motion)
    cfg.commands['motion'].motion_file = str(reference_path.resolve())

    agent = load_rl_cfg(task_id)
    agent.seed = seed
    return cfg, agent, clip, frames, fps


def build_runner(task_id, cfg, agent_cfg, device, *, checkpoint=None, log_dir=None):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls
    if checkpoint is not None:
        source = Path(checkpoint).resolve().parent / 'config' / 'source.json'
        if source.exists():
            saved_task = json.loads(source.read_text()).get('provenance', {}).get('task_id')
            if saved_task and saved_task != task_id:
                raise ValueError(
                    f'Checkpoint belongs to {saved_task}, requested {task_id}. '
                    'Use its matching task ID; action meanings may differ.')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Check nvidia-smi or use --device cpu.')
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        train_cfg = asdict(agent_cfg)
        if checkpoint is not None and log_dir is None:
            saved = Path(checkpoint).resolve().parent / 'config' / 'rl.json'
            if saved.exists():
                train_cfg = json.loads(saved.read_text())
            if task_id == TIMP_TASK_ID:
                train_cfg['algorithm']['amp']['inference'] = True
        runner = load_runner_cls(task_id)(wrapped, train_cfg, log_dir=log_dir, device=device)
        if checkpoint is not None:
            kwargs = {} if log_dir is not None else {'load_cfg': {'actor': True}}
            runner.load(str(Path(checkpoint).resolve()), map_location=device, **kwargs)
        return wrapped, runner
    except BaseException:
        env.close()
        raise


def dry_run(cfg):
    from mjlab.scene import Scene
    scene = Scene(cfg.scene, device='cpu')
    model = scene.compile()
    cfg.sim.mujoco.apply(model)
    return {'native_scene_compiles': True, 'nq': model.nq, 'nv': model.nv,
            'nu': model.nu, 'sensors': model.nsensor, 'num_envs': cfg.scene.num_envs,
            'control_dt': cfg.sim.mujoco.timestep * cfg.decimation}


def prepare_timp(*, play=False, terrains=None, stage=2, reference_pools=None, num_envs=None, seed=42):
    names = tuple(terrains) if terrains is not None else None
    manifest = str(Path(reference_pools or DEFAULT_POOLS).resolve())
    cfg = timp_env_cfg(play=play, names=names, stage=stage, manifest=manifest, seed=seed)
    if num_envs is not None:
        cfg.scene.num_envs = num_envs
    active_names = tuple(cfg.scene.terrain.terrain_generator.sub_terrains)
    agent = timp_rl_cfg(play=play, stage=stage, manifest=manifest, names=active_names, seed=seed)
    return cfg, agent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', nargs='?', default='PRCF-Mimic-E1-25DoF')
    parser.add_argument('--motion-file', '--clip', dest='motion_file',
                        help='E1 NPZ for flat ground, or JSON pairing motion and obstacles')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-root', '--output-root', dest='log_root', default='logs',
                        help='root directory for training/evaluation artifacts (default: logs)')
    parser.add_argument('--dry-run', action='store_true',
                        help='load the motion and compile the native scene without training')
    parser.add_argument('--num-envs', type=int)
    parser.add_argument('--max-iterations', type=int)
    parser.add_argument('--checkpoint', help='resume a local .pt checkpoint')
    parser.add_argument('--terrains', nargs='+', choices=TERRAINS, help='TIMP terrain selection')
    parser.add_argument('--reference-pools', help='TIMP reference-pool JSON')
    parser.add_argument('--stage', type=int, choices=(2, 3), help='TIMP skill acquisition or support adaptation')
    args = parser.parse_args(argv)
    if args.max_iterations is not None and args.max_iterations < 1:
        parser.error('--max-iterations must be positive')
    if args.num_envs is not None and args.num_envs < 1:
        parser.error('--num-envs must be positive')
    timp = args.task == TIMP_TASK_ID
    if timp and args.motion_file:
        parser.error('TIMP uses --reference-pools')
    if timp and args.stage == 3 and not (args.checkpoint or args.dry_run):
        parser.error('TIMP Stage 3 requires --checkpoint from a TIMP run')
    if not timp and not args.motion_file:
        parser.error('--motion-file/--clip is required for Mimic')
    if not timp and (args.terrains or args.reference_pools or args.stage):
        parser.error('--terrains, --reference-pools and --stage apply to TIMP')
    directory = new_run_dir(args.log_root, args.task)
    config = directory / 'config'
    config.mkdir()
    if timp:
        cfg, agent = prepare_timp(terrains=args.terrains, stage=args.stage or 2, reference_pools=args.reference_pools,
                                  num_envs=args.num_envs, seed=args.seed)
        amp = agent.algorithm.amp
        pools = ReferencePools(amp['manifest'], 'cpu', heads=amp['heads'])
        source = {'stage': amp['stage'], 'reference_pools': amp['manifest'],
                  'terrains': tuple(cfg.scene.terrain.terrain_generator.sub_terrains),
                  'active_amp_heads': tuple(pools.clips)}
        write_json(config / 'reference_pools.json', json.loads(Path(amp['manifest']).read_text()))
    else:
        cfg, agent, clip, frames, fps = prepare_env(
            args.task, args.motion_file, config / 'reference.npz',
            num_envs=args.num_envs or 1024, seed=args.seed,
        )
        source = {'clip': clip, 'frames': frames, 'fps': fps}
    if args.max_iterations is not None:
        agent.max_iterations = args.max_iterations
    write_json(config / 'env.json', cfg)
    write_json(config / 'rl.json', agent)
    write_json(config / 'source.json', {
        **source,
        'provenance': {'task_id': args.task, 'seed': args.seed, 'checkpoint': args.checkpoint},
    })
    print(f'Run directory: {directory}')
    if args.dry_run:
        details = {'active_amp_heads': source['active_amp_heads']} if timp else {}
        print(json.dumps({**dry_run(cfg), **details}, indent=2))
        return 0
    env, runner = build_runner(args.task, cfg, agent, args.device,
                               checkpoint=args.checkpoint, log_dir=str(directory))
    try:
        runner.learn(num_learning_iterations=agent.max_iterations, init_at_random_ep_len=False)
    finally:
        env.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
