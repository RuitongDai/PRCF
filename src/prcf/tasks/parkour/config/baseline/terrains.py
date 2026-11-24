"""Baseline terrain map and training-time AMP routing."""

from dataclasses import dataclass, replace

import mujoco
import numpy as np
import torch
from mjlab.terrains import (
    BoxInvertedPyramidStairsTerrainCfg,
    BoxNestedRingsTerrainCfg,
    BoxPyramidStairsTerrainCfg,
    HfPerlinNoiseTerrainCfg,
    TerrainGeneratorCfg,
)
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput


HEADS = ("locomotion", "stairs", "box")
TERRAIN_HEADS = {
    "perlin_rough": "locomotion",
    "square_gaps": "stairs",
    "stairs_up": "stairs",
    "stairs_up_high": "stairs",
    "stairs_down": "stairs",
    "stairs_down_high": "stairs",
    "high_box": "box",
    "long_high_box": "box",
}
TERRAINS = tuple(TERRAIN_HEADS)


@dataclass(kw_only=True)
class SingleBoxTerrainCfg(SubTerrainCfg):
    """One solid box on a ground tile, with a common approach distance."""

    spawn_x: float = 3.0
    front_x: float = 1.53
    box_length: float = 1.20
    box_width: float = 1.20
    box_height_range: tuple[float, float] = (0.70, 1.0)

    @property
    def end_x(self):
        return self.front_x + self.box_length

    @property
    def center_x(self):
        return self.front_x + self.box_length / 2

    def height_at(self, difficulty):
        low, high = self.box_height_range
        return low + (high - low) * difficulty

    def function(self, difficulty, spec, rng):
        sx, sy = self.size
        height = self.height_at(difficulty)
        body = spec.body("terrain")
        ground = (0.35, 0.40, 0.44, 1.0)
        obstacle = (0.66, 0.48, 0.24, 1.0)
        friction = (0.85, 0.005, 0.0001)
        floor = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, group=0, size=(sx / 2, sy / 2, 0.075),
            pos=(sx / 2, sy / 2, -0.075), rgba=ground, friction=friction,
        )
        box = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, group=0,
            size=(self.box_length / 2, self.box_width / 2, height / 2),
            pos=(self.spawn_x + self.center_x, sy / 2, height / 2),
            rgba=obstacle, friction=friction,
        )
        return TerrainOutput(
            origin=np.array((self.spawn_x, sy / 2, 0.0)),
            geometries=[TerrainGeometry(geom=floor, color=ground), TerrainGeometry(geom=box, color=obstacle)],
        )


def e1_amp_parkour_terrain_cfg(names=None) -> TerrainGeneratorCfg:
    """Use the configured eight-column layout in training, playback, and preview."""
    high_box = SingleBoxTerrainCfg(proportion=0.16)
    sub_terrains = {
        "perlin_rough": HfPerlinNoiseTerrainCfg(
            proportion=0.20,
            height_range=(0.0, 0.10),
            octaves=2,
            persistence=0.25,
            lacunarity=2.0,
            scale=20.0,
            horizontal_scale=0.07,
            resolution=0.07,
            border_width=1.0,
        ),
        "square_gaps": BoxNestedRingsTerrainCfg(
            proportion=0.16,
            num_rings=4,
            ring_width_range=(0.30, 0.70),
            gap_range=(0.20, 0.70),
            height_range=(0.0, 0.0),
            platform_width=2.5,
            border_width=1.0,
            floor_depth=0.6,
        ),
        "stairs_up": BoxInvertedPyramidStairsTerrainCfg(
            proportion=0.16,
            step_height_range=(0.05, 0.23),
            step_width=0.30,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
        "stairs_up_high": BoxInvertedPyramidStairsTerrainCfg(
            proportion=0.16,
            step_height_range=(0.05, 0.45),
            step_width=1.0,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down": BoxPyramidStairsTerrainCfg(
            proportion=0.16,
            step_height_range=(0.05, 0.23),
            step_width=0.30,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down_high": BoxPyramidStairsTerrainCfg(
            proportion=0.16,
            step_height_range=(0.05, 0.45),
            step_width=1.0,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "high_box": high_box,
        "long_high_box": replace(high_box, box_length=2.40),
    }
    if names is not None:
        if not names or len(set(names)) != len(names) or any(name not in sub_terrains for name in names):
            raise ValueError(f"Choose unique terrain names from {', '.join(sub_terrains)}")
        sub_terrains = {name: sub_terrains[name] for name in names}
    return TerrainGeneratorCfg(
        seed=0,
        size=(8.0, 8.0),
        border_width=3.0,
        num_rows=10,
        num_cols=len(sub_terrains),
        curriculum=True,
        difficulty_range=(0.0, 1.0),
        add_lights=True,
        sub_terrains=sub_terrains,
    )


def terrain_configs(env):
    return env.scene.terrain.cfg.terrain_generator.sub_terrains


def terrain_names(env):
    return tuple(terrain_configs(env))


def on_terrain(env, name):
    names = terrain_names(env)
    return env.scene.terrain.terrain_types == (names.index(name) if name in names else -1)


def amp_heads_for(names):
    used = {TERRAIN_HEADS[name] for name in names}
    return tuple(head for head in HEADS if head in used)


def amp_head(env):
    heads = torch.tensor([HEADS.index(TERRAIN_HEADS[name]) for name in terrain_names(env)], device=env.device)
    return heads[env.scene.terrain.terrain_types]


def terrain_difficulties(generator):
    """Match mjlab's row interpolation, including a single-row layout."""
    return np.linspace(*generator.difficulty_range, generator.num_rows)
