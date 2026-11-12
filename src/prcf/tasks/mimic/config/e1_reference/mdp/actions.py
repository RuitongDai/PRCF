"""Reference-centred joint targets for offline physical-reference refinement."""
from dataclasses import dataclass

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


class ReferenceJointPositionAction(JointPositionAction):
    def process_actions(self, actions: torch.Tensor):
        command = self._env.command_manager.get_term(self.cfg.command_name)
        # Action targets may be ordered differently from the motion's XML joints.
        reference = command.joint_pos[:, self.target_ids]
        self._raw_actions[:] = actions
        self._processed_actions[:] = reference + actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions[:] = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1])


@dataclass(kw_only=True)
class ReferenceJointPositionActionCfg(JointPositionActionCfg):
    command_name: str = 'motion'

    def build(self, env):
        return ReferenceJointPositionAction(self, env)
