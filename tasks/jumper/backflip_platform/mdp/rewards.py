"""The drift penalty a platform needs: sideways only.

`jumper.ref_free_jump`'s `horizontal_drift` charges the base's distance from the
spawn in any direction, quadratically, at 0.10 m to the unit -- a vertical jump
has no direction to travel in. Off a platform the flip has one, backwards, and
clearing the back edge is about 0.25 m of it: (0.25 / 0.10)^2 = 6 a step at
weight -1, more than everything the flip earns. Across the robot, though, there
is still no reason to travel, so that axis keeps the original's kernel.
"""

from __future__ import annotations

import torch

from mjlab.entity import Entity


def lateral_drift(env, scale: float = 0.10) -> torch.Tensor:
    """(dy / scale)^2, dy the base's sideways distance from the env origin."""
    robot: Entity = env.scene["robot"]
    dy = robot.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return dy**2 / scale**2


__all__ = ["lateral_drift"]
