"""Global errors, including horizontal drift hidden by relative tracking terms."""
import torch


def root_position_error(env):
    command = env.command_manager.get_term("motion")
    return torch.linalg.vector_norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=-1)


def body_position_error(env):
    command = env.command_manager.get_term("motion")
    return torch.linalg.vector_norm(command.body_pos_w - command.robot_body_pos_w, dim=-1).mean(-1)
