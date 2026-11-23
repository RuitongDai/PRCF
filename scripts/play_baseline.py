"""Play the independent E1 25-DoF VDWL baseline without creating training logs."""
import argparse
import json

if __package__:
    from .play import preview_terrain
    from .train import dry_run
    from .train_baseline import build_baseline_runner, prepare_baseline, TERRAINS
else:
    from play import preview_terrain
    from train import dry_run
    from train_baseline import build_baseline_runner, prepare_baseline, TERRAINS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-envs', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--terrains', nargs='+', choices=TERRAINS)
    parser.add_argument('--viewer', choices=('native', 'viser'), default='native')
    parser.add_argument('--preview-terrain', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if not (args.checkpoint or args.dry_run or args.preview_terrain):
        parser.error('--checkpoint is required for policy playback')
    if args.num_envs is not None and args.num_envs < 1:
        parser.error('--num-envs must be positive')
    cfg, agent = prepare_baseline(play=True, terrains=args.terrains, num_envs=args.num_envs, seed=args.seed)
    if args.preview_terrain:
        cfg.scene.entities = {}
        cfg.scene.sensors = ()
    if args.dry_run:
        print(json.dumps(dry_run(cfg), indent=2))
        return 0
    if args.preview_terrain:
        preview_terrain(cfg, args.viewer)
        return 0
    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
    env, runner = build_baseline_runner(cfg, agent, args.device, checkpoint=args.checkpoint)
    try:
        policy = runner.get_inference_policy(device=args.device)
        viewer = NativeMujocoViewer if args.viewer == 'native' else ViserPlayViewer
        viewer(env, policy).run()
    finally:
        env.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
