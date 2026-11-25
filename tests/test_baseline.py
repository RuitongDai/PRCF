"""Regression checks for migrated camera history and routed AMP transitions."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from prcf.tasks.parkour.config.baseline.mdp.amp import AMP_DIM
from prcf.tasks.parkour.config.baseline.mdp.box_traversal import BoxTraversal
from prcf.tasks.parkour.config.baseline.mdp.observations import DepthHistory
from prcf.tasks.parkour.config.baseline.mdp.stage2_commands import Stage2TargetVelocityCommand
from prcf.tasks.parkour.config.baseline.mdp.terminations import (
    invalid_body_contacts, time_out_with_rough_boundary,
)
from prcf.tasks.parkour.config.baseline.rl.algorithm import BaselinePPO, WindowReplay
from prcf.tasks.parkour.config.baseline.rl.motion_loader import WINDOW
from prcf.tasks.parkour.config.baseline.terrains import SingleBoxTerrainCfg


class BaselineTests(unittest.TestCase):
    def test_rough_timeout_uses_local_tile_and_preserves_other_timeouts(self):
        origins = torch.tensor([[20., -28., 0.]]).repeat(6, 1)
        offsets = torch.tensor([
            [3.64, -3.64, 0.], [3.66, 0., 0.], [0., -3.66, 0.],
            [0., 0., 0.], [3.66, 0., 0.], [0., 0., 0.],
        ])
        command = SimpleNamespace(
            robot=SimpleNamespace(data=SimpleNamespace(root_link_pos_w=origins + offsets)),
            rough_terrain=torch.tensor([True, True, True, True, False, False]),
        )
        env = SimpleNamespace(
            scene=SimpleNamespace(env_origins=origins),
            cfg=SimpleNamespace(scene=SimpleNamespace(terrain=SimpleNamespace(
                terrain_generator=SimpleNamespace(size=(8., 8.)),
            ))),
            command_manager=SimpleNamespace(get_term=lambda name: command),
            episode_length_buf=torch.tensor([999, 999, 999, 1000, 1000, 2000]),
            max_episode_length=2000,
            step_dt=.02,
        )
        self.assertEqual(time_out_with_rough_boundary(env).tolist(),
                         [False, True, True, True, False, True])

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

    def test_box_traversal_requires_consecutive_top_support_and_stable_exit(self):
        tile = SingleBoxTerrainCfg(box_length=2.4)
        generator = SimpleNamespace(sub_terrains={'long_high_box': tile, 'perlin_rough': None},
                                    difficulty_range=(0.0, 1.0), num_rows=2)
        origin = torch.tensor([[2.0, 3.0, 0.0], [4.0, 5.0, 0.0], [6.0, 7.0, 0.0]])
        robot = SimpleNamespace(
            root_link_pos_w=origin + torch.tensor([[tile.front_x + .05, 0., .75],
                                                  [tile.front_x + .05, .85, .75],
                                                  [tile.front_x + .05, 0., .75]]),
            projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(3, 1),
            root_link_lin_vel_w=torch.zeros(3, 3),
        )
        position = origin[:, None, :] + torch.tensor([tile.front_x + .1, 0., 1.0])
        contacts = SimpleNamespace(pos=position.repeat(1, 8, 1), found=torch.ones(3, 8),
                                   force=torch.tensor([8., 0., 0.]).repeat(3, 8, 1),
                                   normal=torch.tensor([1., 0., 0.]).repeat(3, 8, 1))

        class Scene(dict):
            terrain = SimpleNamespace(terrain_types=torch.tensor([0, 0, 1]),
                                      terrain_levels=torch.tensor([1, 1, 0]),
                                      cfg=SimpleNamespace(terrain_generator=generator))
            env_origins = origin

        command = SimpleNamespace(is_standing_env=torch.zeros(3, dtype=torch.bool))
        env = SimpleNamespace(device='cpu', num_envs=3, step_dt=.02,
                              scene=Scene(robot=SimpleNamespace(data=robot), feet=SimpleNamespace(data=contacts)),
                              command_manager=SimpleNamespace(get_term=lambda name: command))
        route = BoxTraversal(None, env)

        def step():
            return route(env, command_name='base_velocity', sensor_name='feet',
                         contact_force_threshold=15., mount_seconds=.10,
                         landing_seconds=.30, bypass_seconds=.10)

        # A vertical face cannot count as top support. Leaving the corridor
        # for five steps is a failure only in the box environment.
        for _ in range(4):
            self.assertFalse(step().any())
        robot.root_link_pos_w[:, 2] += .008
        self.assertEqual(step().tolist(), [False, True, False])
        self.assertFalse(route.mounted.any())
        torch.testing.assert_close(route.lift, torch.tensor([1., 0., 0.]))

        contacts.normal[:] = torch.tensor([0., 0., 1.])
        contacts.found[0, 4:] = 0
        for _ in range(4):
            step()
            self.assertFalse(route.mounted.any())
        # Changing which foot touches restarts the dwell for the new foot.
        contacts.found[0, :4] = 0
        contacts.found[0, 4:] = 1
        for _ in range(4):
            step()
            self.assertFalse(route.mounted.any())
        step()
        self.assertEqual(route.mount_bonus.tolist(), [True, False, False])
        step()
        self.assertFalse(route.mount_bonus.any())

        robot.root_link_pos_w[0, 0] = origin[0, 0] + tile.end_x + .2
        contacts.pos[0, :, 0] = robot.root_link_pos_w[0, 0]
        contacts.pos[0, :, 2] = 0.
        robot.root_link_lin_vel_w[0, 2] = -.6
        step()
        self.assertFalse(route.passed.any())
        robot.root_link_lin_vel_w[0, 2] = 0.
        for _ in range(14):
            step()
            self.assertFalse(route.passed.any())
        step()
        self.assertEqual(route.exit_bonus.tolist(), [True, False, False])
        step()
        self.assertFalse(route.exit_bonus.any())
        route.reset(torch.tensor([1]))
        self.assertEqual(route.passed.tolist(), [True, False, False])
        route.reset(torch.tensor([0]))
        self.assertFalse(route.mounted.any())
        self.assertFalse(route.passed.any())

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
