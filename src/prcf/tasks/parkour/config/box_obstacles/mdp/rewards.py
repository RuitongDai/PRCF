"""Approach, support transfer, and standing rewards for both box routes."""

import torch
from mjlab.managers.manager_base import ManagerTermBase

from ..terrains import BOX, BOX_HEIGHT, DOWN_GOAL, NOMINAL_PELVIS_HEIGHT, UP_GOAL, is_descent
from .terminations import box_outcome


def track_forward_speed(env, std):
    robot = env.scene["robot"].data
    target = env.command_manager.get_command("forward")[:, 0]
    x = robot.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    approach_end = torch.where(is_descent(env), BOX.end_x - 0.20, BOX.front_x - 0.40)
    vx = robot.root_link_lin_vel_w[:, 0]
    # Only score forward movement while approaching the obstacle.
    tracking = torch.exp(-((vx - target) / std).square())
    return tracking * (vx / target.clamp_min(0.10)).clamp(0.0, 1.0) * (x < approach_end)


def ascent_heading_error(env):
    robot = env.scene["robot"].data
    x = robot.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    course = box_outcome(env)
    near_front = (x >= BOX.front_x - 0.85) & (x <= BOX.front_x + 0.10)
    approaching = ~is_descent(env) & near_front & ~(course.top_arm_support | course.top_foot_support)
    return (1.0 - robot.heading_w.cos()) * approaching


class BoxProgress(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(env)
        self.robot = env.scene["robot"].data
        self.position_ids = self.robot.indexing.free_joint_q_adr[:3]
        self.previous = torch.zeros((env.num_envs, 3), device=env.device)
        self.up_goal = self.previous.new_tensor((UP_GOAL.x, 0.0, UP_GOAL.surface_z + NOMINAL_PELVIS_HEIGHT))
        self.down_goal = self.previous.new_tensor((DOWN_GOAL.x, 0.0, NOMINAL_PELVIS_HEIGHT))

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        # Reset events have written qpos. Derived link poses update in sim.forward().
        self.previous[ids] = self._position()[ids]

    def __call__(self, env, speed_reference):
        position = self._position()
        x = position[:, 0]
        previous_x = self.previous[:, 0]
        course = box_outcome(env)
        # Use the same target on both ends of a step so phase changes and resets
        # cannot create progress without motion.
        up_approach = self._distance_gain(position[:, :2], self.previous[:, :2], self.up_goal[:2])
        up_climb = self._distance_gain(position, self.previous, self.up_goal)
        on_box = course.top_arm_support | course.top_foot_support
        up_climb = torch.where(on_box, up_climb, up_climb.clamp_max(0.0))
        near_front = (x >= BOX.front_x - 0.45) & (previous_x >= BOX.front_x - 0.45)
        ascent = torch.where(near_front, up_climb, up_approach)

        down_approach = self._distance_gain(position[:, :2], self.previous[:, :2], self.down_goal[:2])
        down_transfer = self._distance_gain(position, self.previous, self.down_goal)
        # Positive pelvis progress past the rear edge requires ground-foot support.
        landed = course.ground_foot_support & (self.robot.root_link_lin_vel_w[:, 2] > -0.9)
        down_transfer = torch.where(landed, down_transfer, down_transfer.clamp_max(0.0))
        near_rear = (x >= BOX.end_x - 0.15) & (previous_x >= BOX.end_x - 0.15)
        descent = torch.where(near_rear, down_transfer, down_approach)

        self.previous.copy_(position)
        gain = torch.where(is_descent(env), descent, ascent)
        return (gain / (speed_reference * env.step_dt)).clamp(-1.0, 1.0)

    def _position(self):
        return self.robot.data.qpos[:, self.position_ids] - self._env.scene.env_origins

    @staticmethod
    def _distance_gain(position, previous, goal):
        return torch.linalg.vector_norm(previous - goal, dim=1) - torch.linalg.vector_norm(position - goal, dim=1)


class BoxLimbProgress(ManagerTermBase):
    """Score changes in wrist reach, foot placement, and acceptance of ground load."""

    def __init__(self, cfg, env):
        super().__init__(env)
        robot = env.scene["robot"]
        self.robot = robot.data
        self.wrists, _ = robot.find_bodies(("left_wrist_pitch_link", "right_wrist_pitch_link"), preserve_order=True)
        self.feet, _ = robot.find_bodies(("left_ankle_roll_link", "right_ankle_roll_link"), preserve_order=True)
        self.previous = torch.zeros((env.num_envs, 4), device=env.device)
        self.ready = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.hand_goals = self.previous.new_tensor([
            (BOX.front_x + 0.10, 0.24, BOX_HEIGHT + 0.05),
            (BOX.front_x + 0.10, -0.24, BOX_HEIGHT + 0.05),
        ])
        self.top_goals = self.previous.new_tensor([
            (BOX.front_x + 0.25, 0.16, BOX_HEIGHT + 0.065),
            (BOX.front_x + 0.25, -0.16, BOX_HEIGHT + 0.065),
        ])
        self.ground_goals = self.previous.new_tensor([
            (BOX.end_x + 0.27, 0.16, 0.065),
            (BOX.end_x + 0.27, -0.16, 0.065),
        ])

    def reset(self, env_ids=None):
        # Link poses are refreshed after reward reset. Start the baseline on the first step.
        self.ready[slice(None) if env_ids is None else env_ids] = False

    def __call__(self, env, speed_reference):
        course = box_outcome(env)
        origins = env.scene.env_origins
        x = self.robot.root_link_pos_w[:, 0] - origins[:, 0]
        wrists = self.robot.body_link_pos_w[:, self.wrists] - origins[:, None, :]
        feet = self.robot.body_link_pos_w[:, self.feet] - origins[:, None, :]
        hand_distance = torch.linalg.vector_norm(wrists - self.hand_goals, dim=-1).amin(dim=1)
        top_distance = torch.linalg.vector_norm(feet - self.top_goals, dim=-1).mean(dim=1)
        ground_distance = torch.linalg.vector_norm(feet - self.ground_goals, dim=-1).amin(dim=1)
        # Half the E1's weight is a meaningful transfer onto the first grounded foot.
        load_distance = 0.12 * (1.0 - (course.ground_foot_load / 143.0).clamp(0.0, 1.0))
        distance = torch.stack((hand_distance, top_distance, ground_distance, load_distance), dim=1)
        hand_gain, top_gain, ground_gain, load_gain = (self.previous - distance).unbind(dim=1)
        self.previous.copy_(distance)

        top_support = course.top_arm_support | course.top_foot_support
        near_front = (x >= BOX.front_x - 0.85) & (x <= BOX.end_x)
        ascent = torch.where(top_support, top_gain, hand_gain * self.robot.heading_w.cos().clamp_min(0.0))
        ascent = ascent * near_front

        near_rear = (x >= BOX.end_x - 0.35) & (x <= DOWN_GOAL.x + DOWN_GOAL.half_length)
        guiding = (top_support | course.ground_foot_support) & (self.robot.root_link_lin_vel_w[:, 2] > -0.9)
        foot_gain = torch.where(guiding, ground_gain, ground_gain.clamp_max(0.0))
        # After a foot reaches the floor, score the increase in ground load.
        descent = torch.where(course.ground_foot_support, torch.zeros_like(foot_gain), foot_gain) + load_gain
        descent = descent * near_rear

        gain = torch.where(is_descent(env), descent, ascent)
        progress = torch.where(self.ready, gain, 0.0)
        self.ready.fill_(True)
        return (progress / (speed_reference * env.step_dt)).clamp(-1.0, 1.0)


def support_bonus(env):
    return box_outcome(env).support_events


def settle_motion(env):
    robot = env.scene["robot"].data
    linear = (robot.root_link_lin_vel_w / 0.40).square().sum(dim=1)
    angular = robot.root_link_ang_vel_w.square().sum(dim=1)
    tilt = 4.0 * (1.0 + robot.projected_gravity_b[:, 2])
    return (linear + 0.25 * angular + tilt).clamp(0.0, 1.0) * box_outcome(env).settling


def descent_impact_speed(env):
    robot = env.scene["robot"].data
    x = robot.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    speed = (-robot.root_link_lin_vel_w[:, 2] - 0.9).clamp(0.0, 2.0)
    return speed.square() * is_descent(env) * (x >= BOX.end_x - 0.20)


def _peak_contact_force(env, sensor_name):
    history = env.scene[sensor_name].data.force_history
    return torch.linalg.vector_norm(history, dim=-1).amax(dim=(1, 2))


def head_contact(env, threshold):
    return (_peak_contact_force(env, "head_terrain_contact") > threshold).float()


def excessive_terrain_force(env, threshold):
    peak_force = _peak_contact_force(env, "body_terrain_contact")
    excess = (peak_force / threshold - 1.0).clamp(0.0, 1.0)
    return excess.square()


def standing_bonus(env):
    return env.termination_manager.get_term("box_success").float()


def failure(env):
    manager = env.termination_manager
    return (manager.terminated & ~manager.get_term("box_success")).float()
