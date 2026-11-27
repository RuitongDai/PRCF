"""APEX height and CoM ratchet with per-episode progress memory."""

import torch
from mjlab.managers.manager_base import ManagerTermBase

from ..poses import STANDING_PELVIS_HEIGHT
from ..sensors import SUPPORT_GEOMS
from ..terrains import BOX_EDGE_X
from .terminations import feet_on_ground, on_box, standing_on_ground, top_support

# These six source critic fields keep their reset values in the active climb-up
# task. The original command has update methods, but no active term calls them.
UP_PROGRESS_DEFAULTS = (3.14, 3.14, 2.0, 2.0, 3.14, 500.0)


class RatchetProgress(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(env)
        robot = env.scene["robot"]
        self.robot = robot.data
        self.descent = cfg.params["descent"]
        self.best = torch.zeros((env.num_envs, 2), device=env.device)
        self.ready = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.horizontal_penalty = torch.zeros(env.num_envs, device=env.device)
        self.at_goal = torch.zeros_like(self.ready)
        self.feet_near_ground = torch.zeros_like(self.ready)
        self.success = torch.zeros_like(self.ready)
        self.curriculum_up = torch.zeros_like(self.ready)
        self.curriculum_down = torch.zeros_like(self.ready)
        self.support_count = torch.zeros(env.num_envs, device=env.device)
        self.feet_ids, _ = robot.find_bodies(r".*_ankle_roll_link")
        self.upper_ids, _ = robot.find_bodies((r".*_wrist_.*", r".*_shoulder_.*", r".*_elbow_link"))
        support_names = env.scene["elbows_knees"].primary_names
        self.support_body_ids = [robot.body_names.index(name) for name in SUPPORT_GEOMS]
        self.support_geom_ids = torch.tensor(
            [[support_names.index(name) for name in pair] for pair in SUPPORT_GEOMS.values()],
            device=env.device,
        )
        self.extra_memory = self.best.new_tensor(UP_PROGRESS_DEFAULTS).expand(env.num_envs, -1)
        self.reset()

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.best[ids] = self.best.new_tensor((1.0, 1.0) if self.descent else (0.0, -2.0))
        for value in (
            self.ready, self.horizontal_penalty, self.at_goal, self.feet_near_ground,
            self.success, self.curriculum_up, self.curriculum_down, self.support_count,
        ):
            value[ids] = 0

    def _average_height(self, z, top):
        return z.minimum(top[:, None] + 0.02).mean(dim=1)

    def memory(self):
        if self.descent:
            return self.best
        # Observations follow mjlab's forward pass, including after a partial reset.
        height = self._average_height(self.robot.body_link_pos_w[..., 2], self._env.scene.env_origins[:, 2])
        self.best[:, 0] = torch.where(self.ready, self.best[:, 0], height)
        self.ready.fill_(True)
        return torch.cat((self.best, self.extra_memory), dim=1)

    def __call__(self, env, *, descent):
        origins = env.scene.env_origins
        bodies = self.robot.body_link_pos_w
        local = bodies - origins[:, None, :]
        x, _, z = local.unbind(dim=2)
        top = origins[:, 2]
        height = self._average_height(bodies[..., 2], top)
        root_x = self.robot.root_link_pos_w[:, 0] - origins[:, 0]
        root_z = self.robot.root_link_pos_w[:, 2]
        masses = self.robot.model.body_mass[:, self.robot.indexing.body_ids]
        done = env.termination_manager.terminated

        if descent:
            self.feet_near_ground.copy_((bodies[:, self.feet_ids, 2] < 0.12).all(dim=1))
            grounded = feet_on_ground(env).all(dim=1)
            outside = root_x < BOX_EDGE_X
            late = env.episode_length_buf > 200
            self.at_goal.copy_(self.feet_near_ground & outside & late)
            self.feet_near_ground.logical_and_(late)
            self.success.copy_(standing_on_ground(env, grounded & outside) & ~done)
            self.curriculum_up.copy_(grounded & (root_z > STANDING_PELVIS_HEIGHT - 0.10))
            self.curriculum_down.copy_((root_x > BOX_EDGE_X + 0.20) | (root_z < 0.40))

            clipped_x = x.clamp_min(BOX_EDGE_X)
            upper_z = self.robot.body_com_pos_w[:, self.upper_ids, 2] - top.clamp_max(0.78)[:, None]
            exit_reached = (clipped_x.amax(dim=1) < BOX_EDGE_X + 0.20) & (upper_z.amin(dim=1) > 0.05)
            feet_low = (bodies[:, self.feet_ids, 2] < 0.12).all(dim=1)
            reached = torch.stack((feet_low, exit_reached), dim=1)
            current = torch.stack((height, (clipped_x * masses).sum(dim=1) / masses.sum(dim=1)), dim=1)
            tolerance = self.best.new_tensor((1.0e-5, 1.0e-4))
            stalled = current >= self.best - tolerance
            self.best.copy_(torch.minimum(self.best, current))
        else:
            least_z, least_x = z.amin(dim=1), x.amin(dim=1)
            support = top_support(env, self.support_body_ids, self.support_geom_ids)
            self.support_count.copy_(support.sum(dim=1))
            self.at_goal.copy_(on_box(local))
            elbows, knees = support.split(2, dim=1)
            stable = self.robot.root_link_lin_vel_w.norm(dim=1) < 0.30
            self.success.copy_(self.at_goal & elbows.any(dim=1) & knees.any(dim=1) & stable & ~done)
            self.curriculum_up.copy_((least_z > 0.02) & (least_x > BOX_EDGE_X + 0.05))
            self.curriculum_down.copy_(done)

            reached = torch.stack((least_z > 0.0, least_x > BOX_EDGE_X), dim=1)
            current = torch.stack((height, (x * masses).sum(dim=1) / masses.sum(dim=1)), dim=1)
            tolerance = self.best.new_tensor((1.0e-5, 1.0e-3))
            stalled = current <= self.best + tolerance
            self.best.copy_(torch.maximum(self.best, current))

        penalty = stalled & ~reached
        self.horizontal_penalty.copy_(penalty[:, 1])
        return penalty[:, 0].float()


def progress_state(env):
    return env.reward_manager.get_term_cfg("height_progress").func


def horizontal_progress(env):
    return progress_state(env).horizontal_penalty


def goal_success(env):
    return progress_state(env).success.float()
