"""jumper.backflip_platform -- the backflip from a raised platform.

`jumper.backflip` with the robot standing at the back edge of a block, so a flip
that carries it backwards falls past the edge and gets the drop's extra air time.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.backflip_platform",
    assets=JUMPER_ASSETS,
    description=(
        "jumper hexapod backflip off the back edge of a raised platform "
        "(jumper.backflip with a drop behind the robot for more air time)"
    ),
    tags=("locomotion", "jumper", "jump", "backflip", "action-prior"),
)
