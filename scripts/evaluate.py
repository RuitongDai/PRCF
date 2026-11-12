"""Evaluate a trained Mimic policy without exporting reference arrays."""

import argparse
import json

from prcf.tasks.mimic.workflow import dry_run, new_run_dir, prepare, run_evaluation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', nargs='?', default='PRCF-Mimic-E1-25DoF')
    parser.add_argument('--motion-file', '--clip', dest='motion_file', required=True,
                        help='E1 NPZ for flat ground, or motion/scene JSON')
    parser.add_argument('--checkpoint', help='local .pt checkpoint')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-root', '--output-root', dest='log_root', default='logs',
                        help='root directory for training/evaluation artifacts (default: logs)')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--attempts', type=int, default=1)
    args = parser.parse_args(argv)
    if not args.dry_run and not args.checkpoint:
        parser.error('--checkpoint is required for execution')
    if args.attempts < 1:
        parser.error('--attempts must be positive')
    directory = new_run_dir(args.log_root, args.task)
    cfg, agent, clip, reference, model, metadata = prepare(
        args.task, args.motion_file, 'export', directory, seed=args.seed)
    print(f'Run directory: {directory}')
    if args.dry_run:
        print(json.dumps(dry_run(cfg), indent=2))
        return 0
    summary = run_evaluation(
        args.task, cfg, agent, clip, reference, model, metadata, directory,
        args.checkpoint, args.device, args.attempts, save_rollouts=False)
    return 0 if summary['successes'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
