"""Critic-only observation for the backflip: how much of the turn has been made.

The actor sees its body and nothing else, as in `jumper.ref_free_jump`: gravity and
the gyro tell it its pose and spin, and it needs nothing counted. The critic is
asked to predict `flip_progress` and `flip_landing`, which are paid on a turn
accumulated over the whole flight and latched at the landing, and a value function
that is not shown the accumulator is fitting noise -- the reason
`ref_free_jump`'s critic is shown its own flight bookkeeping.
"""

from __future__ import annotations

import math

import torch


def flip_state(
    env, command_name: str = "jump", mirror_kind: str | None = None
) -> torch.Tensor:
    """(turn so far, turn at the landing), in whole turns. Both are zero for an
    episode with no flight of its own.

    `mirror_kind` is read off the params by the symmetry augmentation, not here.
    A backflip is symmetric about the body's sagittal plane, so the turn is a scalar
    that mirroring leaves alone.
    """
    del mirror_kind
    cmd = env.command_manager.get_term(command_name)
    return torch.stack([cmd.turn, cmd.landed_turn], dim=1) / (2.0 * math.pi)


__all__ = ["flip_state"]
