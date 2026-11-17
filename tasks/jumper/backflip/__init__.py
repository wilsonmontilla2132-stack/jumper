"""jumper.backflip -- a standing backflip, reference-free on the robot.

Built on `jumper.ref_free_jump`: the same recording-fed jump while training, the same
actor (its body and no clock) and the same way home, with the height objective
joined by a rotation objective -- how far the body turned backwards about its own
pitch axis during the flight it made, and landing on its feet afterwards.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.backflip",
    assets=JUMPER_ASSETS,
    description=(
        "jumper hexapod standing backflip, reference-free on the robot "
        "(ref_free_jump's feedforward and prior while training, plus a rotation objective)"
    ),
    tags=("locomotion", "jumper", "jump", "backflip", "action-prior"),
)
