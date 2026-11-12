"""Track each foot in the scene frame instead of diluting errors across all bodies."""
import torch


def feet_world_position_tracking(env, command_name='motion', std=0.15):
    command = env.command_manager.get_term(command_name)
    indexes = [command.cfg.body_names.index(name) for name in
               ('left_ankle_roll_link', 'right_ankle_roll_link')]
    error = (command.body_pos_w[:, indexes] - command.robot_body_pos_w[:, indexes]).square().sum(-1)
    return torch.exp(-error / std**2).mean(-1)
