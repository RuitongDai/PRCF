"""Reference-state initialization for ascent and descent."""

import torch
from mjlab.envs.mdp.events import resolve_env_ids

from ..rl.motion_loader import BoxReferencePool
from ..terrains import BOX, BOX_HEIGHT, is_descent


def reset_box_spawn(env, env_ids):
    ids = resolve_env_ids(env, env_ids)
    robot = env.scene["robot"]
    down = is_descent(env)[ids]
    root = robot.data.default_root_state[ids].clone()
    root[:, :3] += env.scene.env_origins[ids]
    root[:, 0] += down * (BOX.end_x - 0.45)
    root[:, 2] += down * BOX_HEIGHT
    robot.write_root_state_to_sim(root, env_ids=ids)
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos[ids], robot.data.default_joint_vel[ids], env_ids=ids,
    )


class ReferenceReset:
    def __init__(self, cfg, env):
        self.states = BoxReferencePool(cfg.params["manifest"], env.device, load_states=True).reset_states()

    def __call__(self, env, env_ids, manifest, ascent_probability, descent_probability):
        ids = resolve_env_ids(env, env_ids)
        draws = torch.rand(len(ids), device=env.device)
        down = is_descent(env)[ids]
        ascent = ~down & (draws < ascent_probability)
        descent = down & (draws < descent_probability)
        robot = env.scene["robot"]
        for selected, pool in zip((ascent, descent), self.states, strict=True):
            reset_ids = ids[selected]
            if not reset_ids.numel():
                continue
            frame = torch.randint(len(pool), (len(reset_ids),), device=env.device)
            state = pool[frame]
            pose = state[:, :7].clone()
            pose[:, :3] += env.scene.env_origins[reset_ids]
            robot.write_root_link_pose_to_sim(pose, env_ids=reset_ids)
            robot.write_root_link_velocity_to_sim(state[:, 7:13], env_ids=reset_ids)
            joint_pos, joint_vel = state[:, 13:].chunk(2, dim=1)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=reset_ids)
