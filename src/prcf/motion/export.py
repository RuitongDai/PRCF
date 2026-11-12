"""Write accepted references and rejected diagnostics without overwriting source data."""

import json
from pathlib import Path
import numpy as np
from .schema import MOTION_FIELDS, CONTACT_PARTS


def save_attempt(directory, states, intervals, *, fps, joint_names, body_names, report, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    payload = {key: states[key] for key in MOTION_FIELDS}
    payload.update(fps=np.asarray([fps]), joint_names=np.asarray(joint_names), body_names=np.asarray(body_names))
    filename = 'motion.npz' if report['success'] else 'rollout.npz'
    np.savez_compressed(directory / filename, **payload)
    extras = {key: value for key, value in states.items() if key not in MOTION_FIELDS}
    extras.update(intervals)
    extras['contact_parts'] = np.asarray(CONTACT_PARTS)
    extras['action_valid'] = np.arange(len(states['joint_pos'])) > 0
    if intervals:
        substeps = len(next(iter(intervals.values()))[0])
        extras['physics_substep_time'] = (
            np.arange(len(states['joint_pos']) - 1)[:, None] / fps
            + np.arange(substeps)[None, :] / (fps * substeps)
        )
    np.savez_compressed(directory / 'extras.npz', **extras)
    (directory / 'quality.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    clip = {
        'schema_version': 1, 'clip_id': directory.name, 'robot_id': 'e1_25dof',
        'motion_file': filename, 'fps': fps,
        'coordinates': {'length_unit': 'm', 'up_axis': 'z', 'quaternion_order': 'wxyz'},
        'scene': metadata['scene'], 'quality': report['thresholds'],
        'metadata': {**metadata, 'accepted_reference': report['success'], 'quality_file': 'quality.json', 'extras_file': 'extras.npz'},
    }
    (directory / 'clip.json').write_text(json.dumps(clip, indent=2, allow_nan=False) + '\n')
    return directory
