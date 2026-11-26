"""Rotation rewards for the backflip, and the one `ref_free_jump` term that cannot stay.

`jumper.ref_free_jump`'s `upright` is paid at every step. A backflip is, for about a
third of a second, upside down on purpose, so a term that pays for being upright
there would charge the flip for existing. `upright_grounded` is the same kernel
paid only while the robot is not in its own flight: standing before the push, and
standing after the landing.

Both flip terms read the bookkeeping in `mdp/commands.py`.
"""

from __future__ import annotations

import torch

from mjlab.entity import Entity


def flip_progress(env, command_name: str = "jump") -> torch.Tensor:
    """The change, this step, of the share of the target turn made in the episode's
    own flight. A potential: the whole turn pays the weight once, and a turn undone
    pays it back."""
    cmd = env.command_manager.get_term(command_name)
    return cmd.progress_delta


def flip_spin(
    env, command_name: str = "jump", omega_max: float = 20.0
) -> torch.Tensor:
    """Backward pitch rate, clipped to [0, omega_max] and scaled to [0, 1], paid
    every step of the episode's own flight.

    Not a potential, unlike `flip_progress`: a turn made is never paid back. In the
    air the body's angular momentum is conserved, so the spin is decided at the
    take-off and the policy cannot rotate forward again to undo it; the only way to
    collect more of this is to leave the ground spinning faster. That is the
    gradient a flip needs and the potential did not give: replaying the second run
    (`model_599`, 32 envs, native CPU) tilted back 0.107 turns in the air and
    landed with a net 0.003 -- worth nothing under the potential.
    """
    cmd = env.command_manager.get_term(command_name)
    robot: Entity = env.scene["robot"]
    omega = (-robot.data.root_link_ang_vel_b[:, 1]).clamp(0.0, omega_max) / omega_max
    return omega * cmd.in_flight.float()


def flip_landing(
    env, command_name: str = "jump", sigma_sq: float = 0.10
) -> torch.Tensor:
    """Paid every step after the jump has landed: the share of the target turn the
    flight made, times how upright the body is now.

    Both factors are needed. The turn alone would pay for ending on the back; the
    upright kernel alone is `ref_free_jump`'s landed apex, which a flat jump earns.
    Together it is nearly the whole of it only for a flip that came down on its
    feet.
    """
    cmd = env.command_manager.get_term(command_name)
    robot: Entity = env.scene["robot"]
    g = robot.data.projected_gravity_b[:, :2]
    upright = torch.exp(-(g**2).sum(dim=1) / sigma_sq)
    target = cmd.cfg.turn_target
    share = cmd.landed_turn.clamp(0.0, target) / target
    return cmd.landed.float() * share * upright


def upright_grounded(
    env, command_name: str = "jump", sigma_sq: float = 0.10
) -> torch.Tensor:
    """exp(-|g_xy|^2 / sigma_sq), paid while the robot is not in its own flight."""
    cmd = env.command_manager.get_term(command_name)
    robot: Entity = env.scene["robot"]
    g = robot.data.projected_gravity_b[:, :2]
    return torch.exp(-(g**2).sum(dim=1) / sigma_sq) * (~cmd.in_flight).float()


__all__ = ["flip_landing", "flip_progress", "flip_spin", "upright_grounded"]
