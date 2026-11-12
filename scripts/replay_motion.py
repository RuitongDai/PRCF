"""Kinematic inspection of an input/output bundle, not policy execution."""

import argparse
from pathlib import Path
import time
import mujoco
from prcf.motion.kinematics import set_native_frame
from prcf.motion.schema import load_clip
from prcf.motion.scene import SceneBuilder
from prcf.motion.validation import load_motion, transform_motion, validate_motion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion-file', '--clip', dest='motion_file', required=True,
                        help='E1 NPZ, motion/scene JSON, or a directory in web mode')
    parser.add_argument('--viewer', choices=('viser', 'native'), default='viser',
                        help='web scene editor (default), or native replay')
    parser.add_argument('--host', default='127.0.0.1', help='web viewer bind address')
    parser.add_argument('--port', type=int, default=8080, help='web viewer port')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    if args.viewer == 'viser' and not args.headless:
        from prcf.tasks.mimic.replay import WebReplayEditor
        try:
            WebReplayEditor(args.motion_file, host=args.host, port=args.port).run()
        except (OSError, ValueError) as error:
            parser.exit(2, f'无法启动网页回放：{error}\n可使用 --port 指定其他端口，或 --viewer native。\n')
        return 0
    if Path(args.motion_file).is_dir():
        parser.error('native/headless replay requires a single NPZ or JSON file')
    clip = load_clip(args.motion_file)
    motion = transform_motion(load_motion(clip.motion_file), clip.motion_to_scene)
    report = validate_motion(motion, expected_fps=clip.metadata.get('declared_fps'))
    from mjlab.scene import Scene
    from prcf.tasks.mimic.config.e1.env_cfgs import e1_mimic_env_cfg
    cfg = e1_mimic_env_cfg('play')
    cfg.scene.spec_fn = SceneBuilder(clip.scene)
    scene = Scene(cfg.scene, device='cpu')
    model = scene.compile()
    state = mujoco.MjData(model)
    if args.headless:
        for frame in range(report['frames']):
            set_native_frame(model, state, motion, frame)
        print(f"Replayed {report['frames']} frames using native forward kinematics; no physics rollout performed.")
    else:
        from mujoco import viewer as native_viewer
        set_native_frame(model, state, motion, 0)
        with native_viewer.launch_passive(model, state) as viewer:
            for frame in range(report['frames']):
                if not viewer.is_running():
                    break
                start = time.monotonic()
                set_native_frame(model, state, motion, frame)
                viewer.sync()
                time.sleep(max(0, 1 / report['fps'] - (time.monotonic() - start)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
