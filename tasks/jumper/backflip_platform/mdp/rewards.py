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


def backward_flight(
    env, command_name: str = "jump", v_max: float = 1.0
) -> torch.Tensor:
    """Backward speed over the ground, clipped to [0, v_max] and scaled to [0, 1],
    paid every step of the episode's own flight.

    The platform only adds air time to a flight that clears the back edge, and
    nothing paid for clearing it: replaying the first platform run (`model_950`,
    32 envs, native CPU) not one episode left the block -- the base never came
    within 87 mm of the edge and the first touchdown was 43 mm *forward* of the
    spawn. This is the gradient towards the edge. The robot faces +x at the spawn
    and has no command to turn, so world -x is backwards.
    """
    cmd = env.command_manager.get_term(command_name)
    robot: Entity = env.scene["robot"]
    v = (-robot.data.root_link_lin_vel_w[:, 0]).clamp(0.0, v_max) / v_max
    return v * cmd.in_flight.float()


def illegal_contact_grounded(
    env, command_name: str = "jump", sensor_name: str = "body_ground_contact",
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """`ref_free_jump`'s `illegal_contact`, not charged during the episode's own
    flight: a calf brushing the block's edge on the way past it is the price of
    leaving the block, not a fall. On the ground before and after, it is charged
    as before."""
    cmd = env.command_manager.get_term(command_name)
    sensor = env.scene[sensor_name]
    f = sensor.data.force
    assert f is not None
    hit = (f.norm(dim=-1) > force_threshold).any(dim=1)
    return (hit & ~cmd.in_flight).float()


__all__ = ["backward_flight", "illegal_contact_grounded", "lateral_drift"]
