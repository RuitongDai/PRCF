"""Regression checks for migrated camera history and routed AMP transitions."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from prcf.tasks.parkour.config.baseline.mdp.amp import AMP_DIM
from prcf.tasks.parkour.config.baseline.mdp.observations import DepthHistory
from prcf.tasks.parkour.config.baseline.mdp.stage2_commands import Stage2TargetVelocityCommand
from prcf.tasks.parkour.config.baseline.mdp.terminations import invalid_body_contacts
from prcf.tasks.parkour.config.baseline.rl.algorithm import BaselinePPO, WindowReplay
from prcf.tasks.parkour.config.baseline.rl.motion_loader import WINDOW


class BaselineTests(unittest.TestCase):
    def test_zero_stop_handles_vector_dt_and_partial_reset(self):
        command = object.__new__(Stage2TargetVelocityCommand)
        count = 4096
        vars(command).update(
            cfg=SimpleNamespace(zero_stop_probability=0.25, fixed_command=None),
            time_left=torch.ones(count),
            zero_stop_enabled=torch.zeros(count, dtype=torch.bool),
            zero_stop_active=torch.zeros(count, dtype=torch.bool),
            zero_stop_trigger_progress=torch.full((count,), 0.5),
            metrics={name: torch.zeros(count) for name in
                     ('target_progress', 'zero_stop_triggered', 'zero_stop_seconds')},
        )
        with patch.object(Stage2TargetVelocityCommand, 'stairs',
                          new_callable=lambda: property(lambda self: torch.ones(count, dtype=torch.bool))):
            dt = torch.full((count,), 0.02)
            dt[0] = 0.0
            command._update_zero_stop(dt)
            self.assertEqual(command.metrics['zero_stop_seconds'].sum().item(), 0.0)
            command.zero_stop_enabled[:3] = True
            command.metrics['target_progress'][:3] = 0.75
            command._update_zero_stop(dt)
            self.assertEqual(command.zero_stop_active[:4].tolist(), [False, True, True, False])
            torch.testing.assert_close(command.metrics['zero_stop_seconds'][:3], torch.tensor([0., .02, .02]))
            command._update_zero_stop(0.02, env_ids=torch.tensor([2]))
            torch.testing.assert_close(command.metrics['zero_stop_seconds'][:3], torch.tensor([0., .02, .04]))
            self.assertEqual(command.metrics['zero_stop_triggered'].sum().item(), 2.0)

    def test_box_allows_limb_support_but_preserves_torso_failure(self):
        sensor = SimpleNamespace(
            primary_names=('robot/pelvis', 'robot/left_knee_link', 'robot/left_elbow_link'),
            data=SimpleNamespace(force_history=torch.full((2, 3, 1, 3), 10.0)),
        )

        class Scene:
            terrain = SimpleNamespace(
                terrain_types=torch.tensor([0, 1]),
                cfg=SimpleNamespace(terrain_generator=SimpleNamespace(
                    sub_terrains={'perlin_rough': None, 'high_box': None})),
            )

            def __getitem__(self, name):
                return sensor

        env = SimpleNamespace(scene=Scene(), device='cpu')
        self.assertEqual(invalid_body_contacts(env, 'undesired_contact').tolist(),
                         [[True, True, True], [True, False, False]])

    def test_camera_history_waits_for_real_capture_and_refills_after_reset(self):
        sensor = SimpleNamespace(
            cfg=SimpleNamespace(height=36, width=64), update_period_steps=5, sample_id=1,
            data=SimpleNamespace(depth=torch.full((2, 36, 64, 1), 1.5)),
        )

        class Env:
            num_envs = 2
            device = 'cpu'
            common_step_counter = 0

            def __getitem__(self, name):
                return sensor

        env = Env()
        env.scene = env
        history = DepthHistory(None, env)

        def read():
            return history(env, crop_top=9, crop_bottom=9, crop_left=16, crop_right=16,
                           enable_noise=False).clone()

        first = read()
        sensor.data.depth.fill_(2.4)
        env.common_step_counter += 1
        torch.testing.assert_close(read(), first)
        sensor.sample_id += 1
        second = read()
        torch.testing.assert_close(second[:, 0], first[:, 1])
        torch.testing.assert_close(second[:, 1], torch.full((2, 18, 32), 0.8))
        history.reset(torch.tensor([0]))
        sensor.sample_id += 1
        sensor.data.depth.fill_(0.9)
        final = read()
        torch.testing.assert_close(final[0], torch.full((2, 18, 32), 0.3))
        torch.testing.assert_close(final[1, 0], second[1, 1])

    def test_amp_excludes_auto_reset_and_changed_head_boundaries(self):
        algorithm = object.__new__(BaselinePPO)
        vars(algorithm).update(
            history=torch.zeros(1, WINDOW, AMP_DIM), length=torch.zeros(1, dtype=torch.long),
            last_head=torch.full((1,), -1), amp_statistics=torch.zeros(3),
            replay={head: WindowReplay(8, 'cpu') for head in (0, 2)},
            discriminator=SimpleNamespace(reward=lambda windows, head: torch.ones(len(windows))),
        )

        def step(frame, head, done=False):
            obs = {'amp': torch.full((1, AMP_DIM), float(frame)), 'amp_route': torch.tensor([[head]])}
            reward = torch.ones(1)
            algorithm.process_env_step(obs, reward, torch.tensor([done]), {})
            return reward.item()

        with patch('rsl_rl.algorithms.PPO.process_env_step'):
            for frame in range(4):
                reward = step(frame, 0)
            self.assertAlmostEqual(reward, 1.0)
            self.assertEqual(algorithm.replay[0].count, 1)
            self.assertAlmostEqual(step(4, 0, done=True), 0.9)
            step(5, 0)
            for frame in range(6, 9):
                self.assertAlmostEqual(step(frame, 2), 0.9)
            self.assertAlmostEqual(step(9, 2), 1.0)
            torch.testing.assert_close(algorithm.replay[2].buffer[0, :, 0], torch.arange(6., 10.))
