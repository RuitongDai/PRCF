"""Train the independent E1 25-DoF VDWL baseline."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

if __package__:
    from .train import dry_run, new_run_dir, write_json
else:
    from train import dry_run, new_run_dir, write_json

from prcf.tasks.parkour.config.baseline import TASK_ID, BaselineEnv, baseline_env_cfg, baseline_rl_cfg
from prcf.tasks.parkour.config.baseline.rl.motion_loader import DEFAULT_POOLS, ReferencePools
from prcf.tasks.parkour.config.baseline.rl.runner import BaselineRunner
from prcf.tasks.parkour.config.baseline.terrains import TERRAINS


def prepare_baseline(*, play=False, terrains=None, num_envs=None, reference_pools=None, seed=42):
    cfg = baseline_env_cfg(play=play, names=terrains, seed=seed)
    if num_envs is not None:
        cfg.scene.num_envs = num_envs
    names = tuple(cfg.scene.terrain.terrain_generator.sub_terrains)
    manifest = str(Path(reference_pools or DEFAULT_POOLS).resolve())
    agent = baseline_rl_cfg(play=play, names=names, manifest=manifest, seed=seed)
    return cfg, agent


def build_baseline_runner(cfg, agent, device, *, checkpoint=None, log_dir=None, cuda_debug=False):
    from mjlab.rl import RslRlVecEnvWrapper
    train_cfg = asdict(agent)
    inference = log_dir is None
    if checkpoint is not None:
        checkpoint = Path(checkpoint).resolve()
        source = checkpoint.parent / 'config/source.json'
        if source.exists() and json.loads(source.read_text()).get('task_id') != TASK_ID:
            raise ValueError('Use a checkpoint produced by the E1 25-DoF baseline task.')
        saved = checkpoint.parent / 'config/rl.json'
        if inference and saved.exists():
            train_cfg = json.loads(saved.read_text())
    train_cfg['algorithm']['amp']['inference'] = inference
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use --device cpu for a smoke test.')
    env = BaselineEnv(cfg, device=device, cuda_debug=cuda_debug)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = BaselineRunner(wrapped, train_cfg, log_dir=log_dir, device=device)
        if checkpoint is not None:
            load_cfg = {'actor': True} if inference else None
            runner.load(str(checkpoint), load_cfg=load_cfg, map_location=device)
        return wrapped, runner
    except BaseException:
        env.close()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-envs', type=int)
    parser.add_argument('--max-iterations', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--terrains', nargs='+', choices=TERRAINS)
    parser.add_argument('--reference-pools')
    parser.add_argument('--checkpoint')
    parser.add_argument('--log-root', default='logs')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--cuda-debug', action='store_true',
                        help='Synchronize CUDA at physics and sensor stages to locate GPU errors')
    args = parser.parse_args(argv)
    if args.num_envs is not None and args.num_envs < 1:
        parser.error('--num-envs must be positive')
    if args.max_iterations is not None and args.max_iterations < 1:
        parser.error('--max-iterations must be positive')
    if args.cuda_debug and not args.device.startswith('cuda'):
        parser.error('--cuda-debug requires a CUDA device')
    cfg, agent = prepare_baseline(terrains=args.terrains, num_envs=args.num_envs,
                                  reference_pools=args.reference_pools, seed=args.seed)
    pools = ReferencePools(agent.algorithm.amp['manifest'], 'cpu', heads=agent.algorithm.amp['heads'])
    if args.dry_run:
        print(json.dumps({**dry_run(cfg), 'amp_heads': tuple(pools.clips)}, indent=2))
        return 0
    if args.max_iterations is not None:
        agent.max_iterations = args.max_iterations
    directory = new_run_dir(args.log_root, TASK_ID)
    config = directory / 'config'
    config.mkdir()
    write_json(config / 'env.json', cfg)
    write_json(config / 'rl.json', agent)
    write_json(config / 'source.json', {'task_id': TASK_ID, 'seed': args.seed, 'checkpoint': args.checkpoint,
                                       'terrains': tuple(cfg.scene.terrain.terrain_generator.sub_terrains),
                                       'active_amp_heads': tuple(pools.clips)})
    write_json(config / 'reference_pools.json', json.loads(Path(agent.algorithm.amp['manifest']).read_text()))
    print(f'Run directory: {directory}', flush=True)
    env, runner = build_baseline_runner(cfg, agent, args.device, checkpoint=args.checkpoint,
                                        log_dir=str(directory), cuda_debug=args.cuda_debug)
    try:
        runner.learn(num_learning_iterations=agent.max_iterations, init_at_random_ep_len=True)
    finally:
        env.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
