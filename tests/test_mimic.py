"""CPU tests of scene alignment and reference tracking commands."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import mujoco
import numpy as np
import torch

from prcf.asset.robot.e1.e1_constants import e1_model
from prcf.asset.scene import MotionClip, SceneBuilder, load_aligned_motion, scene_boxes
from scripts.csv_to_npz_e1 import build_kinematic_motion


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


class MotionTests(unittest.TestCase):
    def test_rigid_scene_alignment(self):
        _, motion = fixture_motion()
        shift = [2, -1, 0.1]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'motion.npz'
            np.savez_compressed(path, **motion)
            clip = MotionClip(path, 'fixture', motion_to_scene={
                'position': shift, 'quaternion_wxyz': [1, 0, 0, 0],
            })
            aligned = load_aligned_motion(clip)
        np.testing.assert_allclose(aligned['body_pos_w'], motion['body_pos_w'] + shift)
        np.testing.assert_allclose(aligned['joint_pos'], motion['joint_pos'])

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
        from scripts.train import jsonable
        from prcf.tasks.mimic.config.e1.env_cfgs import e1_mimic_env_cfg
        for mode in ('train', 'play'):
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
