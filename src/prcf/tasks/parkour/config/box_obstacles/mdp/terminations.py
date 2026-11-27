"""Contact milestones and sustained standing for the two box routes."""

import math

import torch

from ..terrains import BOX, BOX_HEIGHT, DOWN_GOAL, PATCH_SIZE, UP_GOAL, is_descent


def torso_collision(env):
    contacts = env.scene["torso_contact"].data
    force = torch.linalg.vector_norm(contacts.force, dim=-1)
    return ((contacts.found > 0) & (force > 10.0)).any(dim=1)


def low_pelvis(env):
    z = env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return z < 0.38


def left_box_route(env):
    robot = env.scene["robot"].data
    x, y = (robot.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]).unbind(dim=1)
    width = BOX.box_width / 2 + 0.15
    near_box = (x >= BOX.front_x - 0.40) & (x <= DOWN_GOAL.x + DOWN_GOAL.half_length)
    outside_patch = (x <= -BOX.spawn_x + 0.20) | (x >= PATCH_SIZE[0] - BOX.spawn_x - 0.20)
    return outside_patch | (y.abs() >= PATCH_SIZE[1] / 2 - 0.20) | (near_box & (y.abs() > width))


class BoxSuccess:
    def __init__(self, cfg, env):
        self.steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.dwell = math.ceil(cfg.params["standing_seconds"] / env.step_dt)
        self.arm_seen = torch.zeros_like(self.steps, dtype=torch.bool)
        self.foot_seen = torch.zeros_like(self.arm_seen)
        self.support_events = torch.zeros_like(self.steps, dtype=torch.float)
        self.top_arm_support = torch.zeros_like(self.arm_seen)
        self.top_foot_support = torch.zeros_like(self.arm_seen)
        self.ground_foot_support = torch.zeros_like(self.arm_seen)
        self.ground_foot_load = torch.zeros_like(self.steps, dtype=torch.float)
        self.settling = torch.zeros_like(self.arm_seen)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for state in (
            self.steps, self.arm_seen, self.foot_seen, self.support_events, self.top_arm_support,
            self.top_foot_support, self.ground_foot_support, self.ground_foot_load, self.settling,
        ):
            state[ids] = 0

    def __call__(self, env, **_):
        robot = env.scene["robot"].data
        x, y, z = (robot.root_link_pos_w - env.scene.env_origins).unbind(dim=1)
        down = is_descent(env)
        valid = ~env.termination_manager.terminated
        fx, fy, fz, force = self._contacts(env, "box_feet")
        ax, ay, az, arm_force = self._contacts(env, "box_arms")
        top_feet = self._on_box(fx, fy, fz)
        ground_feet = (fx >= BOX.end_x) & (fy.abs() <= BOX.box_width / 2) & (fz.abs() <= 0.04)
        has_top_foot = self._supported_sides(force, top_feet).any(dim=1)
        has_ground_foot = self._supported_sides(force, ground_feet).any(dim=1)
        has_top_arm = self._supported_sides(arm_force, self._on_box(ax, ay, az)).any(dim=1)

        arm = valid & ~down & has_top_arm
        foot = valid & torch.where(down, has_ground_foot, has_top_foot)
        self.support_events.copy_((arm & ~self.arm_seen).float() + (foot & ~self.foot_seen).float())
        self.arm_seen |= arm
        self.foot_seen |= foot
        self.top_arm_support.copy_(valid & has_top_arm)
        self.top_foot_support.copy_(valid & has_top_foot)
        self.ground_foot_support.copy_(valid & down & has_ground_foot)
        self.ground_foot_load.copy_(torch.where(ground_feet, force, 0.0).sum(dim=1) * self.ground_foot_support)

        top_goal_feet = top_feet & UP_GOAL.contains_xy(fx, fy)
        ground_goal_feet = ground_feet & DOWN_GOAL.contains_xy(fx, fy)
        goal_feet = torch.where(down[:, None], ground_goal_feet, top_goal_feet)
        both_feet = self._supported_sides(force, goal_feet).all(dim=1)
        in_goal = torch.where(down, DOWN_GOAL.contains_xy(x, y), UP_GOAL.contains_xy(x, y))
        pelvis_height = z - (~down) * BOX_HEIGHT
        self.settling.copy_(valid & in_goal & both_feet)
        velocity = robot.root_link_lin_vel_w
        standing = (
            self.settling & (pelvis_height >= 0.45) & (pelvis_height <= 0.85)
            & (-robot.projected_gravity_b[:, 2] > 0.90) & (velocity[:, 2].abs() < 0.25)
            & (torch.linalg.vector_norm(velocity[:, :2], dim=1) < 0.20)
            & (torch.linalg.vector_norm(robot.root_link_ang_vel_w, dim=1) < 0.70)
        )
        self.steps.copy_(torch.where(standing, self.steps + 1, 0))
        return self.steps >= self.dwell

    def _contacts(self, env, sensor):
        contacts = env.scene[sensor].data
        x, y, z = (contacts.pos - env.scene.env_origins[:, None, :]).unbind(dim=2)
        force = torch.where(
            (contacts.found > 0) & (contacts.normal[..., 2].abs() >= 0.7),
            contacts.force[..., 0].abs(), 0.0,
        )
        return x, y, z, force

    def _on_box(self, x, y, z):
        return ((x >= BOX.front_x) & (x <= BOX.end_x) & (y.abs() <= BOX.box_width / 2)
                & ((z - BOX_HEIGHT).abs() <= 0.04))

    def _supported_sides(self, force, region):
        return torch.where(region, force, 0.0).reshape(len(force), 2, -1).sum(dim=2) >= 15.0


def box_outcome(env):
    return env.termination_manager.get_term_cfg("box_success").func


def box_arm_support(env):
    return box_outcome(env).arm_seen.float()


def box_foot_support(env):
    return box_outcome(env).foot_seen.float()
