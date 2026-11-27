"""Hold the reset posture for the original APEX settling period."""

from dataclasses import dataclass

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


@dataclass(kw_only=True)
class ApexJointPositionActionCfg(JointPositionActionCfg):
    settle_steps: int = 15
    hold_reset_pose: bool = False

    def build(self, env):
        return ApexJointPositionAction(self, env)


class ApexJointPositionAction(JointPositionAction):
    cfg: ApexJointPositionActionCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._env = env
        self.nominal_pose = self._entity.data.default_joint_pos[0].clone()
        self.reset_target = torch.zeros_like(self.raw_action)
        self.previous_target = torch.zeros_like(self.raw_action)

    @property
    def target(self):
        return self._processed_actions

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        data = self._entity.data
        pose = data.joint_pos if self.cfg.hold_reset_pose else data.default_joint_pos
        joint_pos = pose[ids][:, self.target_ids]
        self.reset_target[ids] = joint_pos
        self.target[ids] = joint_pos
        self.previous_target[ids] = joint_pos

    def randomize_default_pose(self, env_ids):
        default = self._entity.data.default_joint_pos
        perturbation = torch.empty_like(default[env_ids]).uniform_(-0.01, 0.01)
        default[env_ids] = self.nominal_pose + perturbation
        self._offset[env_ids] = default[env_ids][:, self.target_ids]

    def process_actions(self, actions):
        self.previous_target.copy_(self.target)
        super().process_actions(actions)
        settle = self._env.episode_length_buf < self.cfg.settle_steps
        self.target.copy_(torch.where(settle[:, None], self.reset_target, self.target))
