"""Train a registered PRCF Mimic task with local TensorBoard logs."""

import argparse
import json
from pathlib import Path

from prcf.tasks.mimic.workflow import build_runner, dry_run, new_run_dir, prepare, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', nargs='?', default='PRCF-Mimic-E1-25DoF')
    parser.add_argument('--motion-file', '--clip', dest='motion_file', required=True,
                        help='E1 NPZ for flat ground, or JSON pairing motion and obstacles')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-root', '--output-root', dest='log_root', default='logs',
                        help='root directory for training/evaluation artifacts (default: logs)')
    parser.add_argument('--dry-run', action='store_true',
                        help='validate motion and compile the native scene without training')
    parser.add_argument('--num-envs', type=int, default=1024)
    parser.add_argument('--max-iterations', type=int, default=30000)
    parser.add_argument('--checkpoint', help='resume a local .pt checkpoint')
    args = parser.parse_args(argv)
    if args.max_iterations < 1:
        parser.error('--max-iterations must be positive')
    directory = new_run_dir(args.log_root, args.task)
    cfg, agent, *_ = prepare(args.task, args.motion_file, 'train', directory,
                            num_envs=args.num_envs, seed=args.seed)
    agent.max_iterations = args.max_iterations
    write_json(directory / 'config' / 'rl.json', agent)
    if args.checkpoint:
        from prcf.motion.validation import sha256
        write_json(directory / 'config' / 'resume.json', {
            'checkpoint': str(Path(args.checkpoint).resolve()),
            'checkpoint_sha256': sha256(args.checkpoint),
        })
    print(f'Run directory: {directory}')
    if args.dry_run:
        print(json.dumps(dry_run(cfg), indent=2))
        return 0
    env, runner = build_runner(args.task, cfg, agent, args.device,
                               checkpoint=args.checkpoint, log_dir=str(directory))
    try:
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
    finally:
        env.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
