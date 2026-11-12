"""Validate E1 motion and scene descriptions without GPU simulation."""

import argparse
import json

from prcf.motion.schema import load_clip
from prcf.motion.scene import SceneBuilder
from prcf.motion.validation import load_motion, transform_motion, validate_motion


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion-file', '--clip', dest='motion_file', required=True,
                        help='E1 NPZ or motion/scene JSON')
    args = parser.parse_args(argv)
    clip = load_clip(args.motion_file)
    data = load_motion(clip.motion_file)
    report = validate_motion(data, expected_fps=clip.metadata.get('declared_fps'))
    validate_motion(transform_motion(data, clip.motion_to_scene))
    SceneBuilder(clip.scene)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
