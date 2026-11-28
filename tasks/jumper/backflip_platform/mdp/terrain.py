"""A flat ground with one box on it, the robot spawned at the box's back edge.

The box is the whole point of the task: standing on it, a flip that drifts
backwards leaves from `height` above the ground it lands on, and the drop adds to
the time in the air. The spawn origin is on the box's top face, so every height
`jumper.backflip` measures from the env origin is measured from where the robot
stands, as it is on flat ground.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_plane


@dataclass(kw_only=True)
class PlatformTerrainCfg(SubTerrainCfg):
    height: float
    """Top of the box above the ground, metres."""
    back_margin: float
    """Box surface behind the spawn origin (towards -x, the robot's back), metres."""
    front_length: float
    """Box surface in front of the spawn origin, metres."""
    width: float
    """Box width across the robot (y), metres."""

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del difficulty, rng
        body = spec.body("terrain")
        ground = make_plane(body, self.size, 0.0, center_zero=False)[0]
        cx, cy = self.size[0] / 2, self.size[1] / 2
        length = self.back_margin + self.front_length
        box_x = cx + (self.front_length - self.back_margin) / 2
        box = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(length / 2, self.width / 2, self.height / 2),
            pos=(box_x, cy, self.height / 2),
        )
        return TerrainOutput(
            origin=np.array((cx, cy, self.height)),
            geometries=[
                TerrainGeometry(geom=ground, color=(0.5, 0.5, 0.5, 1.0)),
                TerrainGeometry(geom=box, color=(0.35, 0.35, 0.40, 1.0)),
            ],
        )


__all__ = ["PlatformTerrainCfg"]
