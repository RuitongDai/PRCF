"""Explicit acceptance gates; training reward is never a trajectory success label."""

from dataclasses import asdict
import numpy as np
from .schema import CONTACT_PARTS, QualityConfig


def assess_trajectory(states, intervals, reference, config: QualityConfig, *, failure_reason=None, joint_limits=None):
    reasons = []
    if failure_reason:
        reasons.append(failure_reason)
    n = len(states['joint_pos'])
    expected = len(reference['joint_pos'])
    complete = n == expected and np.array_equal(states['reference_frame'], np.arange(expected))
    if not complete:
        reasons.append('incomplete_or_discontinuous_reference')
    finite = all(np.isfinite(v).all() for v in list(states.values()) + list(intervals.values()))
    if not finite:
        reasons.append('nonfinite_values')
        return {'success': False, 'reasons': reasons, 'frames': n, 'expected_frames': expected, 'thresholds': asdict(config)}
    index = np.clip(states['reference_frame'].astype(int), 0, expected - 1)
    root_error = np.linalg.norm(states['body_pos_w'][:, 0] - reference['body_pos_w'][index, 0], axis=-1)
    body_error = np.linalg.norm(states['body_pos_w'] - reference['body_pos_w'][index], axis=-1)
    joint_rmse = np.sqrt(np.mean((states['joint_pos'] - reference['joint_pos'][index]) ** 2, axis=-1))
    dot = np.abs(np.sum(states['body_quat_w'][:, 0] * reference['body_quat_w'][index, 0], axis=-1))
    orientation_error = 2 * np.arccos(np.clip(dot, 0, 1))
    env_dist = np.concatenate([states['min_environment_distance'].reshape(-1), intervals.get('min_environment_distance_substep', np.zeros(0)).reshape(-1)])
    self_dist = np.concatenate([states['min_self_distance'].reshape(-1), intervals.get('min_self_distance_substep', np.zeros(0)).reshape(-1)])
    found = states['contact_found']
    forces = states['strongest_contact_force_w']
    normals = states['contact_normal_w']
    velocities = states['contact_point_velocity_w']
    normal_component = np.sum(velocities * normals, axis=-1, keepdims=True) * normals
    slip = np.linalg.norm(velocities - normal_component, axis=-1)
    support = found & (np.linalg.norm(forces, axis=-1) >= config.contact_force_threshold)
    # Ignore isolated impacts: slip is assessed only for persistent loaded contacts.
    persistent = support.copy()
    persistent[0] = False
    persistent[1:] &= support[:-1]
    max_slip = float(slip[persistent].max()) if persistent.any() else 0.0
    metrics = {
        'max_root_error': float(root_error.max()), 'mean_root_error': float(root_error.mean()),
        'max_body_error': float(body_error.max()), 'max_joint_rmse': float(joint_rmse.max()),
        'max_root_orientation_error': float(orientation_error.max()),
        'final_root_error': float(np.linalg.norm(states['body_pos_w'][-1, 0] - reference['body_pos_w'][-1, 0])),
        'max_environment_penetration': max(0.0, -float(env_dist.min())),
        'max_self_penetration': max(0.0, -float(self_dist.min())),
        'max_contact_slip_speed': max_slip,
    }
    for key in ('max_root_error', 'max_body_error', 'max_joint_rmse', 'max_root_orientation_error',
                'final_root_error', 'max_environment_penetration', 'max_self_penetration', 'max_contact_slip_speed'):
        if metrics[key] > getattr(config, key):
            reasons.append(key)
    if joint_limits is not None:
        lo, hi = joint_limits[:, 0], joint_limits[:, 1]
        q = states['joint_pos']
        if np.any((q < lo - config.joint_limit_tolerance) | (q > hi + config.joint_limit_tolerance)):
            reasons.append('hard_joint_limit')
    contact_results = []
    for requirement in config.contact_requirements:
        end = requirement.end_frame
        if end >= expected:
            raise ValueError(f"Contact requirement extends beyond reference: {requirement}")
        part = CONTACT_PARTS.index(requirement.part)
        mask = (states['reference_frame'] >= requirement.start_frame) & (states['reference_frame'] <= end)
        fraction = float(support[mask, part].mean()) if mask.any() else 0.0
        contact_results.append({'part': requirement.part, 'fraction': fraction, 'required': requirement.min_fraction})
        if fraction < requirement.min_fraction:
            reasons.append(f'contact_requirement:{requirement.part}')
    return {
        'success': not reasons, 'reasons': reasons, 'frames': n, 'expected_frames': expected,
        'complete': complete, 'metrics': metrics, 'thresholds': asdict(config),
        'contact_requirements': contact_results,
        'contact_requirements_declared': bool(config.contact_requirements),
    }
