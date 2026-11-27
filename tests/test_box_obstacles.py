"""Regression checks for box route assignment, rewards, and standing goals."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from prcf.tasks.parkour.config.box_obstacles.mdp.commands import BoxForwardCommandCfg
from prcf.tasks.parkour.config.box_obstacles.mdp.rewards import (
    BoxLimbProgress, BoxProgress, ascent_heading_error, excessive_terrain_force, head_contact,
)
from prcf.tasks.parkour.config.box_obstacles.mdp.terminations import BoxSuccess
from prcf.tasks.parkour.config.box_obstacles.terrains import BOX, BOX_HEIGHT, DOWN_GOAL, UP_GOAL


class Scene(dict):
    def __init__(self, robot):
        super().__init__(robot=SimpleNamespace(data=robot))
        self.env_origins = torch.tensor([[5.0, -4.0, 0.0], [-2.0, 3.0, 0.0]])
        self.terrain = SimpleNamespace(terrain_types=torch.tensor([0, 1]))


class BoxProgressTests(unittest.TestCase):
    def test_progress_uses_route_and_excludes_teleports_and_free_fall(self):
        robot = SimpleNamespace(
            indexing=SimpleNamespace(free_joint_q_adr=torch.arange(3)),
            data=SimpleNamespace(qpos=torch.zeros(2, 3)),
            root_link_lin_vel_w=torch.zeros(2, 3),
        )
        scene = Scene(robot)
        env = SimpleNamespace(scene=scene, num_envs=2, device="cpu", step_dt=0.02)
        qpos = robot.data.qpos
        qpos[:] = scene.env_origins + torch.tensor([[0.0, 0.0, 0.75], [BOX.end_x - 0.10, 0.0, 1.57]])
        course = SimpleNamespace(
            top_arm_support=torch.tensor([False, False]),
            top_foot_support=torch.tensor([False, True]),
            ground_foot_support=torch.tensor([False, False]),
        )
        progress = BoxProgress(None, env)
        progress.reset()

        with patch("prcf.tasks.parkour.config.box_obstacles.mdp.rewards.box_outcome", return_value=course):
            qpos[0, 0] += 0.01
            qpos[1, 2] -= 0.20
            result = progress(env, speed_reference=0.55)
            self.assertGreater(result[0], 0.0)
            self.assertEqual(result[1], 0.0)

            qpos[0, 0] -= 0.01
            qpos[1, 2] -= 0.01
            course.ground_foot_support[1] = True
            result = progress(env, speed_reference=0.55)
            self.assertLess(result[0], 0.0)
            self.assertGreater(result[1], 0.0)

            qpos[0] += torch.tensor([1.2, 0.0, 0.2])
            progress.reset(torch.tensor([0]))
            torch.testing.assert_close(progress(env, speed_reference=0.55), torch.zeros(2))

    def test_limb_progress_uses_reaching_and_load_transfer_with_reset_safe_deltas(self):
        robot = SimpleNamespace(
            root_link_pos_w=torch.zeros(2, 3),
            root_link_lin_vel_w=torch.zeros(2, 3),
            heading_w=torch.zeros(2),
            body_link_pos_w=torch.zeros(2, 4, 3),
        )
        scene = Scene(robot)
        scene["robot"].find_bodies = lambda names, **_: (
            [0, 1] if "wrist" in names[0] else [2, 3], names,
        )
        env = SimpleNamespace(scene=scene, num_envs=2, device="cpu", step_dt=0.02)
        robot.root_link_pos_w[:] = scene.env_origins + torch.tensor([
            [BOX.front_x - 0.45, 0.0, 0.75], [BOX.end_x + 0.10, 0.0, 1.30],
        ])
        course = SimpleNamespace(
            top_arm_support=torch.tensor([False, False]),
            top_foot_support=torch.tensor([False, True]),
            ground_foot_support=torch.tensor([False, False]),
            ground_foot_load=torch.zeros(2),
        )
        progress = BoxLimbProgress(None, env)
        robot.body_link_pos_w[:] = scene.env_origins[:, None, :]
        robot.body_link_pos_w[1, 2:] += progress.ground_goals + torch.tensor([0.0, 0.0, 0.25])

        with patch("prcf.tasks.parkour.config.box_obstacles.mdp.rewards.box_outcome", return_value=course):
            torch.testing.assert_close(progress(env, speed_reference=0.55), torch.zeros(2))
            robot.body_link_pos_w[0, :2, 0] += 0.02
            robot.body_link_pos_w[1, 2, 2] -= 0.02
            self.assertTrue((progress(env, speed_reference=0.55) > 0.0).all())

            robot.body_link_pos_w[1, 2, 2] -= 0.02
            course.top_foot_support[1] = False
            torch.testing.assert_close(progress(env, speed_reference=0.55), torch.zeros(2))

            course.top_arm_support[0] = True
            course.ground_foot_support[1] = True
            course.ground_foot_load[1] = 90.0
            robot.body_link_pos_w[0, 2:, 0] += 0.02
            self.assertTrue((progress(env, speed_reference=0.55) > 0.0).all())

            progress.reset(torch.tensor([0]))
            robot.body_link_pos_w[0, 2:, 0] += 0.20
            torch.testing.assert_close(progress(env, speed_reference=0.55), torch.zeros(2))

            robot.heading_w.fill_(torch.pi / 2)
            course.top_arm_support[0] = False
            torch.testing.assert_close(ascent_heading_error(env), torch.tensor([1.0, 0.0]))
            course.top_arm_support[0] = True
            torch.testing.assert_close(ascent_heading_error(env), torch.zeros(2))


class BoxContactRewardTests(unittest.TestCase):
    def test_contact_penalties_include_brief_impacts_and_ignore_normal_support(self):
        body = torch.zeros(3, 2, 4, 3)
        body[0, 0, 1, 2] = 430.0
        body[0, 1, 1, 2] = 430.0
        body[1, 1, 3, 1] = 825.0
        body[2, 0, 0, 2] = 1600.0
        head = torch.zeros(3, 1, 4, 3)
        head[:, 0, 2, 0] = torch.tensor([0.0, 6.0, 4.0])
        env = SimpleNamespace(scene={
            "body_terrain_contact": SimpleNamespace(data=SimpleNamespace(force_history=body)),
            "head_terrain_contact": SimpleNamespace(data=SimpleNamespace(force_history=head)),
        })

        torch.testing.assert_close(excessive_terrain_force(env, 550.0), torch.tensor([0.0, 0.25, 1.0]))
        torch.testing.assert_close(head_contact(env, 5.0), torch.tensor([0.0, 1.0, 0.0]))


class BoxStandingTests(unittest.TestCase):
    def setUp(self):
        self.robot = SimpleNamespace(
            root_link_lin_vel_w=torch.zeros(2, 3),
            root_link_ang_vel_w=torch.zeros(2, 3),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]] * 2),
        )
        scene = Scene(self.robot)
        self.robot.root_link_pos_w = scene.env_origins + torch.tensor([
            [UP_GOAL.x, 0.0, BOX_HEIGHT + 0.665], [DOWN_GOAL.x, 0.0, 0.665],
        ])
        feet = torch.tensor([
            [[UP_GOAL.x, -0.15, BOX_HEIGHT], [UP_GOAL.x, 0.15, BOX_HEIGHT]],
            [[DOWN_GOAL.x, -0.15, 0.0], [DOWN_GOAL.x, 0.15, 0.0]],
        ])
        self.contacts = SimpleNamespace(
            pos=scene.env_origins[:, None, :] + feet,
            found=torch.ones(2, 2),
            force=torch.tensor([[[25.0, 0.0, 0.0]] * 2] * 2),
            normal=torch.tensor([[[0.0, 0.0, 1.0]] * 2] * 2),
        )
        scene["box_feet"] = SimpleNamespace(data=self.contacts)
        scene["box_arms"] = SimpleNamespace(data=SimpleNamespace(
            **{**vars(self.contacts), "found": torch.zeros(2, 2)},
        ))
        self.env = SimpleNamespace(
            scene=scene, num_envs=2, device="cpu", step_dt=0.02,
            termination_manager=SimpleNamespace(terminated=torch.zeros(2, dtype=torch.bool)),
        )

    def test_each_route_needs_two_feet_in_its_own_area_for_full_dwell(self):
        success = BoxSuccess(SimpleNamespace(params={"standing_seconds": 0.30}), self.env)
        self.contacts.pos[1, 1, 1] += DOWN_GOAL.half_width
        for _ in range(14):
            self.assertEqual(success(self.env).tolist(), [False, False])
        self.assertEqual(success(self.env).tolist(), [True, False])
        torch.testing.assert_close(success.support_events, torch.zeros(2))

        self.contacts.pos[1, 1, 1] -= DOWN_GOAL.half_width
        self.robot.root_link_ang_vel_w[1, 2] = 1.0
        self.assertFalse(success(self.env)[1])
        self.robot.root_link_ang_vel_w[1, 2] = 0.0
        for _ in range(14):
            self.assertFalse(success(self.env)[1])
        self.assertTrue(success(self.env)[1])

        success.reset(torch.tensor([0]))
        self.assertEqual(success(self.env).tolist(), [False, True])

    def test_command_has_one_nonnegative_component_and_tapers_at_each_goal(self):
        cfg = BoxForwardCommandCfg(resampling_time_range=(20.0, 20.0), speed_range=(0.55, 0.55))
        command = cfg.build(self.env)
        command._resample_command(torch.arange(2))
        self.robot.root_link_pos_w[:, 0] = self.env.scene.env_origins[:, 0] + torch.tensor([0.0, BOX.end_x])
        command.compute(dt=0.0)
        torch.testing.assert_close(command.command, torch.tensor([[0.55], [0.25]]))

        self.robot.root_link_pos_w[1, 0] += 0.25
        command.compute(dt=0.0)
        torch.testing.assert_close(command.command, torch.tensor([[0.55], [0.25]]))

        self.robot.root_link_pos_w[:, 0] = self.env.scene.env_origins[:, 0] + torch.tensor([UP_GOAL.x, DOWN_GOAL.x])
        command.compute(dt=0.0)
        torch.testing.assert_close(command.command, torch.zeros(2, 1), atol=1e-6, rtol=0)
