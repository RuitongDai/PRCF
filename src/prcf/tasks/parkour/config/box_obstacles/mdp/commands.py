"""A nonnegative world-X speed target for both box routes."""

from dataclasses import dataclass

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from ..terrains import BOX, DOWN_GOAL, DOWN_TILE, UP_GOAL, UP_TILE, is_descent


class BoxForwardCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.speed = torch.zeros(self.num_envs, device=self.device)
        self.target = self.speed[:, None].clone()

    @property
    def command(self):
        return self.target

    def _resample_command(self, env_ids):
        self.speed[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.speed_range)

    def _update_metrics(self):
        pass

    def _update_command(self, env_ids=None):
        env = self._env
        down = is_descent(env)
        x = env.scene["robot"].data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
        goal = torch.where(down, DOWN_GOAL.x, UP_GOAL.x)
        up_limit = 0.25 + (x - BOX.front_x).abs()
        down_limit = 0.25 + (BOX.end_x - x).clamp_min(0.0)
        edge_limit = torch.where(down, down_limit, up_limit)
        goal_limit = 1.6 * (goal - x).clamp_min(0.0)
        self.target[:, 0] = torch.minimum(self.speed, torch.minimum(edge_limit, goal_limit))

    def reset(self, env_ids):
        extras = super().reset(env_ids)
        env = self._env
        finished = (env.episode_length_buf[env_ids] > 0) & env.termination_manager.dones[env_ids]
        down = is_descent(env)[env_ids]
        success = env.termination_manager.get_term("box_success")[env_ids]
        for name, route in ((UP_TILE, ~down), (DOWN_TILE, down)):
            ids = finished & route
            if ids.any():
                extras[f"{name}_success_rate"] = success[ids].float()
        return extras


@dataclass(kw_only=True)
class BoxForwardCommandCfg(CommandTermCfg):
    speed_range: tuple[float, float]

    def build(self, env):
        return BoxForwardCommand(self, env)
