"""APEX probabilistic promotion and demotion between terrain rows."""

import torch

from .progress import progress_state


def terrain_height(env, env_ids, *, update_probability=0.8):
    terrain = env.scene.terrain
    state = progress_state(env)
    valid = env.episode_length_buf[env_ids] > 0
    promote = valid & state.curriculum_up[env_ids]
    demote = valid & state.curriculum_down[env_ids] & ~promote
    levels = terrain.terrain_levels[env_ids]
    advance = torch.rand_like(levels, dtype=torch.float32) < update_probability
    review = (torch.rand_like(levels, dtype=torch.float32) * (levels + 1)).long()
    proposed = torch.where(promote, torch.where(advance, levels + 1, review), levels)
    proposed = torch.where(demote, (levels - 1).clamp_min(0), proposed)
    overflow = proposed >= terrain.max_terrain_level
    proposed = torch.where(overflow, torch.randint_like(levels, terrain.max_terrain_level), proposed)
    terrain.terrain_levels[env_ids] = proposed
    terrain.env_origins[env_ids] = terrain.terrain_origins[proposed, terrain.terrain_types[env_ids]]
    return terrain.terrain_levels.float().mean()
