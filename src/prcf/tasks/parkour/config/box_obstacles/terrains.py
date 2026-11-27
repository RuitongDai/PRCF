"""Two copies of the reference box with route-specific standing areas."""

from dataclasses import dataclass

import mujoco

from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import TerrainGeometry

from ..baseline.terrains import SingleBoxTerrainCfg

BOX_HEIGHT = 0.905
BOX = SingleBoxTerrainCfg(box_height_range=(BOX_HEIGHT, BOX_HEIGHT))
PATCH_SIZE = (8.0, 8.0)
NOMINAL_PELVIS_HEIGHT = 0.665
BOX_LAYOUT = "up_down_standing"
UP_TILE, DOWN_TILE = "box_up", "box_down"


@dataclass(frozen=True)
class StandingArea:
    x: float
    surface_z: float
    half_length: float
    half_width: float

    def contains_xy(self, x, y):
        return ((x - self.x).abs() <= self.half_length) & (y.abs() <= self.half_width)


UP_GOAL = StandingArea(BOX.front_x + 0.35, BOX_HEIGHT, 0.32, 0.35)
DOWN_GOAL = StandingArea(BOX.end_x + 0.65, 0.0, 0.32, 0.40)


@dataclass(kw_only=True)
class GoalBoxTerrainCfg(SingleBoxTerrainCfg):
    goal: StandingArea
    color: tuple[float, float, float, float]

    def function(self, difficulty, spec, rng):
        output = super().function(difficulty, spec, rng)
        goal = self.goal
        marker = spec.body("terrain").add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, group=2, contype=0, conaffinity=0,
            size=(goal.half_length, goal.half_width, 0.002),
            pos=(self.spawn_x + goal.x, self.size[1] / 2, goal.surface_z + 0.002),
            rgba=self.color,
        )
        output.geometries.append(TerrainGeometry(geom=marker, color=self.color))
        return output


def box_terrain_cfg():
    generator = TerrainGeneratorCfg(
        seed=0,
        size=PATCH_SIZE,
        num_rows=1,
        num_cols=2,
        curriculum=True,  # One fixed column per task.
        sub_terrains={
            UP_TILE: GoalBoxTerrainCfg(
                box_height_range=BOX.box_height_range, goal=UP_GOAL, color=(0.20, 0.78, 0.43, 0.85),
            ),
            DOWN_TILE: GoalBoxTerrainCfg(
                box_height_range=BOX.box_height_range, goal=DOWN_GOAL, color=(0.24, 0.60, 0.95, 0.85),
            ),
        },
    )
    return TerrainEntityCfg(terrain_type="generator", terrain_generator=generator)


def is_descent(env):
    return env.scene.terrain.terrain_types == 1
