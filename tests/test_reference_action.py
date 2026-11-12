"""Reference residual targets and task identity must survive frame/order changes."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch

from prcf.tasks.mimic.config.e1_reference.mdp.actions import ReferenceJointPositionAction,ReferenceJointPositionActionCfg
from prcf.tasks.mimic.config.e1_reference.mdp.rewards import feet_world_position_tracking


class ReferenceActionTests(unittest.TestCase):
    def test_zero_residual_tracks_reference_and_joint_mapping(self):
        command=SimpleNamespace(joint_pos=torch.tensor([[.1,.2,.3],[.4,.5,.6]]))
        action=object.__new__(ReferenceJointPositionAction)
        action.cfg=ReferenceJointPositionActionCfg(entity_name='robot',actuator_names=('.*',))
        action._env=SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _:command))
        action._target_ids=torch.tensor([2,0,1])
        action._scale=torch.tensor([[.1,.2,.3]])
        action._raw_actions=torch.zeros(2,3)
        action._processed_actions=torch.zeros(2,3)
        action.process_actions(torch.zeros(2,3))
        torch.testing.assert_close(action._processed_actions,command.joint_pos[:,[2,0,1]])
        command.joint_pos+=.5
        action.process_actions(torch.ones(2,3))
        torch.testing.assert_close(action._processed_actions,command.joint_pos[:,[2,0,1]]+action._scale)

    def test_world_foot_reward_detects_horizontal_lag(self):
        command=SimpleNamespace(
            cfg=SimpleNamespace(body_names=('left_ankle_roll_link','right_ankle_roll_link')),
            body_pos_w=torch.zeros(1,2,3),robot_body_pos_w=torch.zeros(1,2,3))
        env=SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _:command))
        self.assertEqual(float(feet_world_position_tracking(env)),1.)
        command.robot_body_pos_w[:,:,0]+=.3
        self.assertLess(float(feet_world_position_tracking(env)),.02)

    def test_checkpoint_with_different_action_task_is_rejected(self):
        from prcf.tasks.mimic.workflow import build_runner
        with tempfile.TemporaryDirectory() as root:
            p=Path(root);(p/'config').mkdir()
            (p/'config/source.json').write_text(json.dumps({'provenance':{'task_id':'PRCF-Mimic-E1-25DoF'}}))
            with self.assertRaisesRegex(ValueError,'action meanings'):
                build_runner('PRCF-Mimic-E1-25DoF-Reference',None,None,'cpu',checkpoint=p/'model.pt')


if __name__=='__main__':unittest.main()
