"""Box geometry shared by both skills."""

from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput

BOX_EDGE_X = -1.0
BOX_HALF_WIDTH = 1.5
PATCH_SIZE = (4.0, 4.0)


@dataclass
class ApexBoxTerrainCfg(SubTerrainCfg):
    floating: bool = False
    height_range: tuple[float, float] = (0.55, 0.84)

    def function(self, difficulty, spec, rng):
        del rng
        length, width = self.size
        low, high = self.height_range
        top = low + difficulty * (high - low)
        bottom = top - 0.05 if self.floating else -1.0
        terrain = spec.body("terrain")
        floor = terrain.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, group=0,
            size=(length / 2, width / 2, 0.5), pos=(length / 2, width / 2, -0.5),
            friction=(0.8, 0.005, 0.0001),
        )
        box = terrain.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, group=0,
            size=(-BOX_EDGE_X, BOX_HALF_WIDTH, (top - bottom) / 2),
            pos=(length / 2, width / 2, (top + bottom) / 2),
            friction=(0.8, 0.005, 0.0001),
        )
        return TerrainOutput(
            # As in APEX, Z in the tile origin is the box top. Ground is world Z=0.
            origin=np.array((length / 2, width / 2, top)),
            geometries=[
                TerrainGeometry(geom=floor, color=(0.40, 0.43, 0.46, 1.0)),
                TerrainGeometry(geom=box, color=(0.69, 0.49, 0.29, 1.0)),
            ],
        )


def apex_terrain_cfg(*, descent: bool, play: bool) -> TerrainEntityCfg:
    sub_terrains = {"box": ApexBoxTerrainCfg()}
    if not descent:
        sub_terrains["floating_box"] = ApexBoxTerrainCfg(floating=True)
    return TerrainEntityCfg(
        terrain_type="generator",
        max_init_terrain_level=0 if play else 5,
        terrain_generator=TerrainGeneratorCfg(
            seed=0, size=PATCH_SIZE, border_width=2.0, num_rows=10,
            curriculum=True, color_scheme="none", sub_terrains=sub_terrains,
        ),
    )
