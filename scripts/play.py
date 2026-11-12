"""View a trained PRCF Mimic policy."""

import argparse
import json
from tempfile import TemporaryDirectory

from prcf.tasks.mimic.workflow import build_runner, dry_run, prepare


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', nargs='?', default='PRCF-Mimic-E1-25DoF')
    parser.add_argument('--motion-file', '--clip', dest='motion_file', required=True,
                        help='E1 NPZ for flat ground, or motion/scene JSON')
    parser.add_argument('--checkpoint', help='local .pt checkpoint')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-root', '--output-root', help='compatibility option; play does not create log directories')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--viewer', choices=('native', 'viser'), default='native')
    args = parser.parse_args(argv)
    if not args.dry_run and not args.checkpoint:
        parser.error('--checkpoint is required')
    # Aligned reference/config files are needed only while the viewer runs.
    # Keep them out of persistent training logs, including in --dry-run mode.
    with TemporaryDirectory(prefix='prcf-play-') as directory:
        cfg, agent, *_ = prepare(args.task, args.motion_file, 'play', directory, seed=args.seed)
        if args.dry_run:
            print(json.dumps(dry_run(cfg), indent=2))
            return 0
        from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
        env, runner = build_runner(args.task, cfg, agent, args.device, checkpoint=args.checkpoint)
        try:
            policy = runner.get_inference_policy(device=args.device)
            viewer_cls = NativeMujocoViewer if args.viewer == 'native' else ViserPlayViewer
            viewer_cls(env, policy).run()
        finally:
            env.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
