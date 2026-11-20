"""Training-only reference state initialization."""
import torch
from mjlab.envs.mdp.events import resolve_env_ids

from ..rl.motion_loader import ReferencePools
from ..terrains import SingleBoxTerrainCfg, amp_heads_for, terrain_difficulties


class ResetFromReference:
    def __init__(self, cfg, env):
        generator = env.scene.terrain.cfg.terrain_generator
        boxes = {name: tile for name, tile in generator.sub_terrains.items() if isinstance(tile, SingleBoxTerrainCfg)}
        self.pools = ReferencePools(cfg.params["manifest"], env.device, heads=amp_heads_for(boxes), load_states=True)
        self.reset_pools = {}
        difficulties = terrain_difficulties(generator)
        for column, (name, tile) in enumerate(generator.sub_terrains.items()):
            if name in boxes:
                for row, difficulty in enumerate(difficulties):
                    pool = self.pools.box_reset_pool(name, tile, tile.height_at(difficulty))
                    if pool[0]:
                        self.reset_pools[column, row] = pool

    def __call__(self, env, env_ids, manifest, probability):
        env_ids = resolve_env_ids(env, env_ids)
        robot = env.scene["robot"]
        terrain = env.scene.terrain
        columns, rows = terrain.terrain_types[env_ids], terrain.terrain_levels[env_ids]
        choose = torch.rand(len(env_ids), device=env.device) < probability
        for (column, row), pool in self.reset_pools.items():
            ids = env_ids[choose & (columns == column) & (rows == row)]
            if len(ids) == 0:
                continue
            state = self.pools.sample_states(pool, len(ids))
            state[:, :3] += env.scene.env_origins[ids]
            q, dq = state[:, 13:].chunk(2, dim=-1)
            robot.write_root_state_to_sim(state[:, :13], env_ids=ids)
            robot.write_joint_state_to_sim(q, dq, env_ids=ids)
