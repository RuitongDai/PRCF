"""CPU tests of model compatibility, scene pairing, continuity, and export gates."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import mujoco
import numpy as np
import torch

from prcf.motion.export import save_attempt
from prcf.motion.kinematics import build_kinematic_motion
from prcf.motion.quality import assess_trajectory
from prcf.motion.recording import TrajectoryBuffer
from prcf.motion.schema import ContactRequirement, QualityConfig, load_clip
from prcf.motion.scene import SceneBuilder, scene_boxes
from prcf.motion.validation import e1_model, load_motion, transform_motion, validate_motion


def fixture_motion(frames=12):
    model, joints, bodies = e1_model()
    qpos = model.key_qpos[0].copy()
    state = mujoco.MjData(model)
    state.qpos[:] = qpos
    mujoco.mj_kinematics(model, state)
    qpos[2] -= state.site_xpos[model.site('left_foot').id, 2]
    motion = build_kinematic_motion(
        model, np.tile(qpos[:3], (frames, 1)), np.tile(qpos[3:7], (frames, 1)),
        np.tile(qpos[7:], (frames, 1)), 50.0,
    )
    return model, motion


def fixture_recording(motion):
    frames = len(motion['joint_pos'])
    state = {key: motion[key].copy() for key in (
        'joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w')}
    state.update(reference_frame=np.arange(frames), time=np.arange(frames) / 50,
                 contact_found=np.ones((frames, 6), dtype=bool),
                 strongest_contact_force_w=np.tile([0, 0, 20.0], (frames, 6, 1)),
                 contact_normal_w=np.tile([0, 0, 1.0], (frames, 6, 1)),
                 contact_point_velocity_w=np.zeros((frames, 6, 3)),
                 min_environment_distance=np.zeros(frames), min_self_distance=np.zeros(frames))
    intervals = {'qfrc_actuator_substep': np.zeros((frames - 1, 4, 25)),
                 'min_environment_distance_substep': np.zeros((frames - 1, 4)),
                 'min_self_distance_substep': np.zeros((frames - 1, 4))}
    return state, intervals


class MotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.motion = fixture_motion()

    def setUp(self):
        self.data = {key: value.copy() for key, value in self.motion.items()}

    def test_valid_e1_and_true_root_velocity(self):
        report = validate_motion(self.data, model=self.model)
        self.assertEqual(len(report['joint_names']), 25)
        np.testing.assert_allclose(self.data['body_lin_vel_w'][:, 0], 0, atol=1e-10)

    def test_wrong_body_order_rejected_without_names(self):
        del self.data['body_names']
        self.data['body_pos_w'][:, [1, 2]] = self.data['body_pos_w'][:, [2, 1]]
        with self.assertWarns(UserWarning), self.assertRaisesRegex(ValueError, 'FK consistency'):
            validate_motion(self.data, model=self.model)

    def test_g1_or_21_joint_data_rejected(self):
        self.data['joint_pos'] = self.data['joint_pos'][:, :21]
        with self.assertRaisesRegex(ValueError, '25 joints'):
            validate_motion(self.data, model=self.model)

    def test_rate_and_nan_rejected(self):
        with self.assertRaisesRegex(ValueError, 'resample explicitly'):
            validate_motion(self.data, model=self.model, expected_fps=30)
        self.data['body_quat_w'][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            validate_motion(self.data, model=self.model)

    def test_rigid_scene_alignment_preserves_fk(self):
        theta = 0.4
        aligned = transform_motion(self.data, {'position': [2, -1, 0.1],
                                  'quaternion_wxyz': [np.cos(theta / 2), 0, 0, np.sin(theta / 2)]})
        validate_motion(aligned, model=self.model)
        np.testing.assert_allclose(aligned['joint_pos'], self.data['joint_pos'])

    def test_box_and_stairs_geometry(self):
        description = {'ground_height': 0, 'obstacles': [
            {'id': 'box', 'type': 'box', 'size_xyz': [1, 2, 0.6], 'position': [1, 0, 0.3]},
            {'id': 'stairs', 'type': 'stairs', 'step_height': 0.15,
             'step_depth': 0.3, 'width': 1, 'num_steps': 3, 'position': [3, 0, 0]},
        ]}
        spec = mujoco.MjSpec()
        SceneBuilder(description)(spec)
        model = spec.compile()
        np.testing.assert_allclose(model.geom('prcf_box').size, [0.5, 1, 0.3])
        boxes = scene_boxes(description)
        np.testing.assert_allclose(boxes[-1][2], [3.75, 0, 0.225])

    def test_bad_scene_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported obstacle'):
            SceneBuilder({'obstacles': [{'id': 'mesh', 'type': 'mesh'}]})
        with self.assertRaises(ValueError):
            SceneBuilder({'obstacles': [{'id': 'b', 'type': 'box', 'size_xyz': [1, 1, -1]}]})

    def test_saved_bundle_roundtrip(self):
        states, intervals = fixture_recording(self.data)
        report = assess_trajectory(states, intervals, self.data, QualityConfig())
        self.assertTrue(report['success'])
        with tempfile.TemporaryDirectory() as root:
            path = save_attempt(Path(root) / 'clip', states, intervals, fps=50,
                                joint_names=self.data['joint_names'].tolist(), body_names=self.data['body_names'].tolist(),
                                report=report, metadata={'scene': {'ground_height': 0}})
            clip = load_clip(path / 'clip.json')
            validate_motion(load_motion(clip.motion_file), model=self.model)
            with np.load(path / 'extras.npz') as extras:
                self.assertEqual(extras['physics_substep_time'].shape, (len(states['time']) - 1, 4))
                self.assertFalse(extras['action_valid'][0])


class QualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.motion = fixture_motion()

    def setUp(self):
        self.states, self.intervals = fixture_recording(self.motion)

    def test_discontinuous_or_reset_rollout_rejected(self):
        self.states['reference_frame'][5] = 0
        result = assess_trajectory(self.states, self.intervals, self.motion, QualityConfig())
        self.assertIn('incomplete_or_discontinuous_reference', result['reasons'])

    def test_failure_never_becomes_success(self):
        result = assess_trajectory(self.states, self.intervals, self.motion, QualityConfig(), failure_reason='fall')
        self.assertFalse(result['success'])

    def test_substep_penetration_not_hidden_by_final_state(self):
        self.intervals['min_environment_distance_substep'][3, 1] = -0.1
        result = assess_trajectory(self.states, self.intervals, self.motion, QualityConfig())
        self.assertIn('max_environment_penetration', result['reasons'])

    def test_contact_requirement_and_slip(self):
        config = QualityConfig(contact_requirements=[ContactRequirement('left_hand', 2, 8)])
        self.states['contact_found'][:, 2] = False
        self.states['contact_point_velocity_w'][:, 0, 0] = 2
        result = assess_trajectory(self.states, self.intervals, self.motion, config)
        self.assertIn('contact_requirement:left_hand', result['reasons'])
        self.assertIn('max_contact_slip_speed', result['reasons'])

    def test_buffer_requires_contiguous_intervals(self):
        buffer = TrajectoryBuffer()
        buffer.add_state({'q': [0]})
        buffer.add_state({'q': [1]})
        with self.assertRaisesRegex(RuntimeError, 'N-1'):
            buffer.arrays()


class CommandTests(unittest.TestCase):
    def test_adaptive_sampling_has_a_true_uniform_floor(self):
        from prcf.tasks.mimic.config.e1.mdp.commands import adaptive_bin_probabilities
        counts = torch.tensor([0., 0., 10000., 0., 0., 0., 0., 0., 0., 0.])
        kernel = torch.ones(1)
        probabilities = adaptive_bin_probabilities(counts, kernel, 0.5)
        self.assertAlmostEqual(float(probabilities.sum()), 1.)
        self.assertGreaterEqual(float(probabilities.min()), .05 - 1e-7)
        self.assertLessEqual(float(probabilities.max()), .55 + 1e-7)
        torch.testing.assert_close(probabilities, adaptive_bin_probabilities(counts * 100, kernel, .5))
        torch.testing.assert_close(adaptive_bin_probabilities(counts * 0, kernel, .5), torch.full((10,), .1))

    def test_nominal_reference_initialization_and_horizontal_failure(self):
        from prcf.tasks.mimic.config.e1.env_cfgs import e1_mimic_env_cfg
        cfg = e1_mimic_env_cfg('train')
        self.assertEqual(cfg.scene.entities['robot'].articulation.soft_joint_pos_limit_factor, 1.)
        self.assertEqual(cfg.commands['motion'].joint_position_range, (0., 0.))
        self.assertFalse(cfg.observations['actor'].enable_corruption)
        command = SimpleNamespace(anchor_pos_w=torch.tensor([[.6, 0., 1.]]),
                                  robot_anchor_pos_w=torch.tensor([[0., 0., 1.]]))
        env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))
        term = cfg.terminations['anchor_pos']
        self.assertTrue(bool(term.func(env, **term.params)[0]))
        command.robot_anchor_pos_w[0, 0] = .5
        self.assertFalse(bool(term.func(env, **term.params)[0]))

    def test_all_modes_have_serializable_config_snapshots(self):
        from prcf.tasks.mimic.workflow import jsonable
        from prcf.tasks.mimic.config.e1.env_cfgs import e1_mimic_env_cfg
        for mode in ('train', 'play', 'export'):
            cfg = e1_mimic_env_cfg(mode)
            self.assertTrue(hasattr(cfg.commands['motion'].viz, 'ghost_color'))
            json.dumps(jsonable(cfg), allow_nan=False)

    def test_reset_does_not_advance_and_eof_never_resamples(self):
        from prcf.tasks.mimic.config.e1.mdp.commands import FiniteMotionCommand
        command = object.__new__(FiniteMotionCommand)
        command._env = SimpleNamespace(num_envs=2, device='cpu')
        command.cfg = SimpleNamespace(sampling_mode='start')
        command.motion = SimpleNamespace(time_step_total=3)
        command.time_steps = torch.tensor([0, 0])
        command.finished = torch.zeros(2, dtype=torch.bool)
        command.time_left = torch.full((2,), 1e9)
        command._pending_forward = False
        command.update_relative_body_poses = Mock()
        command._update_metrics = Mock()
        command._resample_command = Mock()
        command.compute(0, torch.tensor([0, 1]))
        self.assertEqual(command.time_steps.tolist(), [0, 0])
        command.compute(torch.tensor([0.02, 0]))
        self.assertEqual(command.time_steps.tolist(), [1, 0])
        for _ in range(5):
            command.compute(0.02)
        self.assertEqual(command.time_steps.tolist(), [2, 2])
        self.assertTrue(command.finished.all())
        command._resample_command.assert_not_called()


if __name__ == '__main__':
    unittest.main()
