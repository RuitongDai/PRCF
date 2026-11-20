"""View a trained PRCF policy."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

if __package__:
    from .train import TIMP_TASK_ID, TERRAINS, build_runner, dry_run, prepare_env, prepare_timp
else:
    from train import TIMP_TASK_ID, TERRAINS, build_runner, dry_run, prepare_env, prepare_timp


def preview_terrain(cfg, viewer):
    import mujoco
    from mjlab.scene import Scene

    scene = Scene(cfg.scene, device='cpu')
    model = scene.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    origins = scene.terrain.terrain_origins.numpy()
    sx, sy = cfg.scene.terrain.terrain_generator.size
    radius = max(sx * origins.shape[0], sy * origins.shape[1])
    print('Terrain columns from low to high Y:', ', '.join(cfg.scene.terrain.terrain_generator.sub_terrains), flush=True)

    if viewer == 'native':
        from mujoco import viewer as native_viewer

        with native_viewer.launch_passive(model, data) as handle:
            handle.cam.lookat[:] = (0, 0, 0)
            handle.cam.distance = radius * 1.4
            handle.cam.azimuth = 140
            handle.cam.elevation = -65
            while handle.is_running():
                handle.sync()
                time.sleep(1 / 30)
        return

    import socket
    import viser
    from mjviser import ViserMujocoScene

    # Fail promptly if the port is unavailable.
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 8080))
    server = viser.ViserServer(host='127.0.0.1', port=8080, label='PRCF TIMP Terrain')
    try:
        server.scene.set_up_direction('+z')
        view = ViserMujocoScene(server, model, num_envs=1)
        view.update_from_mjdata(data)
        names = cfg.scene.terrain.terrain_generator.sub_terrains
        for index, name in enumerate(names):
            x, y, _ = origins[0, index]
            server.scene.add_label(f'/terrain/{name}', text=name, position=(x, y, 1.3))

        @server.on_client_connect
        def focus(client):
            client.camera.position = (-0.6 * radius, -0.6 * radius, 1.3 * radius)
            client.camera.look_at = (0, 0, 0)

        print('Terrain preview: http://127.0.0.1:8080', flush=True)
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', nargs='?', default='PRCF-Mimic-E1-25DoF')
    parser.add_argument('--motion-file', '--clip', dest='motion_file',
                        help='E1 NPZ for flat ground, or motion/scene JSON')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--checkpoint', help='local .pt checkpoint')
    mode.add_argument('--preview-terrain', action='store_true', help='view the static TIMP training terrain on CPU')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42, help='simulation reset seed')
    parser.add_argument('--log-root', '--output-root', help='compatibility option; play does not create log directories')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--viewer', choices=('native', 'viser'), default='native')
    parser.add_argument('--terrains', nargs='+', help='select TIMP terrain columns by name')
    parser.add_argument('--terrain-layout', choices=('training', 'copied'), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not (args.dry_run or args.preview_terrain or args.checkpoint):
        parser.error('--checkpoint is required')
    if args.task == TIMP_TASK_ID and args.motion_file:
        parser.error('Use --terrains to select a TIMP play scene')
    if args.task != TIMP_TASK_ID and not args.motion_file:
        parser.error('--motion-file/--clip is required for Mimic')
    if args.task != TIMP_TASK_ID and args.terrains:
        parser.error('--terrains applies to TIMP')
    if args.terrain_layout and not args.preview_terrain:
        parser.error('--terrain-layout requires --preview-terrain')
    if args.terrains and (len(set(args.terrains)) != len(args.terrains)
                         or any(name not in TERRAINS for name in args.terrains)):
        parser.error(f'--terrains requires unique names from {", ".join(TERRAINS)}')
    if args.preview_terrain:
        if args.task != TIMP_TASK_ID:
            parser.error('--preview-terrain applies to TIMP')
        cfg, _ = prepare_timp(play=True, terrains=args.terrains, num_envs=1, seed=args.seed)
        cfg.scene.entities = {}
        cfg.scene.sensors = ()
        generator = cfg.scene.terrain.terrain_generator
        generator.color_scheme = 'height'
        if args.dry_run:
            details = {'terrain_rows': generator.num_rows, 'terrain_columns': list(generator.sub_terrains)}
            print(json.dumps({**dry_run(cfg), **details}, indent=2))
        else:
            try:
                preview_terrain(cfg, args.viewer)
            except OSError as error:
                parser.exit(2, f'无法启动地形预览：{error}\n')
        return 0
    # Aligned reference/config files are needed only while the viewer runs.
    # Keep them out of persistent training logs, including in --dry-run mode.
    with TemporaryDirectory(prefix='prcf-play-') as directory:
        if args.task == TIMP_TASK_ID:
            cfg, agent = prepare_timp(play=True, terrains=args.terrains, seed=args.seed)
        else:
            cfg, agent, *_ = prepare_env(
                args.task, args.motion_file, Path(directory) / 'reference.npz', play=True, seed=args.seed,
            )
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
