"""How an episode of jumper.backflip ends early: tipped over on the ground.

`jumper.ref_free_jump` ends an episode at 50 degrees of tilt, the controller's own
limit. In a backflip the body goes through 180 degrees on purpose, so the same
rule would end every flip at its first quarter-turn. It applies here only outside
the episode's own flight: before the push, and from the touchdown on, which is
where a fall is a fall.
"""

from __future__ import annotations

import torch

from mjlab.entity import Entity


def fell_over_grounded(
    env, command_name: str = "jump", limit_angle: float = 0.8727
) -> torch.Tensor:
    """True when the body is tilted past `limit_angle` radians and the episode is
    not in its own flight. The angle is the one `mjlab.tasks.velocity.mdp.bad_orientation`
    measures: gravity against the body's down axis."""
    cmd = env.command_manager.get_term(command_name)
    robot: Entity = env.scene["robot"]
    gz = robot.data.projected_gravity_b[:, 2]
    tilt = torch.acos((-gz).clamp(-1.0, 1.0)).abs()
    return (tilt > limit_angle) & ~cmd.in_flight


__all__ = ["fell_over_grounded"]
