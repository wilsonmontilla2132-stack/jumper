"""jumper.backflip_platform -- the backflip off the back edge of a raised block.

`jumper.backflip`'s environment, built by calling its `env_cfg()`, with the ground
replaced by a block the robot stands on and the drift penalty narrowed to the
sideways axis. Everything else -- the command, the flip rewards, the schedule,
the actor -- is that task's.

## Why a platform

Replaying `jumper.backflip`'s third run (`model_1499`, 16 envs, native CPU):
70% of the motors sit at 98% of `EFFORT_LIMIT` through the push, and the backward
pitch rate peaks at 8.2 rad/s in a flight of about 0.34 s. A whole turn in that
flight needs about 18.5 rad/s, and the push has little torque left to give. More
time in the air is the other lever: from `PLATFORM_HEIGHT` up, with the third
run's 1.4 m/s take-off, the flight is about 0.45 s and a turn needs about
14 rad/s. A first guess at a reachable target, not a measurement.

## The block

Measured at the stand pose: the rear feet sit at x = -0.146 m, the middle feet
at |y| = 0.205 m, the claws reach x = +0.206 m. `BACK_MARGIN` leaves 34 mm of
block behind the rear feet, so a stand never slips off and any real backward
travel clears it; `WIDTH` leaves 70 mm either side of the middle feet.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from ..backflip.env_cfg import env_cfg as _backflip_env_cfg
from ..ref_free_jump.env_cfg import DRIFT_SCALE, DRIFT_WEIGHT
from . import mdp as platform_mdp

#: Top of the block above the ground, metres. A guess; see the docstring.
PLATFORM_HEIGHT = 0.35
#: Block behind the spawn origin (the robot's back), metres.
BACK_MARGIN = 0.18
#: Block in front of the spawn origin, metres.
FRONT_LENGTH = 0.35
#: Block width across the robot, metres.
WIDTH = 0.55
#: Side of the square tile the block sits on. Every environment shares it; the
#: worlds do not collide with one another.
TILE = 3.0


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Backflip off the back edge of a raised block.

    Args:
        asset: path to the model XML, from `--model`. None uses the default jumper.xml.
        play: replay mode, as `jumper.backflip`'s.
    """
    cfg = _backflip_env_cfg(asset=asset, play=play)

    # ── The ground: one block on a flat tile ──────────────────────────────────
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        size=(TILE, TILE),
        num_rows=1,
        num_cols=1,
        border_width=0.0,
        sub_terrains={
            "platform": platform_mdp.terrain.PlatformTerrainCfg(
                size=(TILE, TILE),
                height=PLATFORM_HEIGHT,
                back_margin=BACK_MARGIN,
                front_length=FRONT_LENGTH,
                width=WIDTH,
            )
        },
    )

    # ── Drift: sideways only; backwards is where the flip goes ────────────────
    old = cfg.rewards.pop("horizontal_drift")
    assert old.weight == DRIFT_WEIGHT, "ref_free_jump's drift weight moved"
    cfg.rewards["lateral_drift"] = RewardTermCfg(
        func=platform_mdp.rewards.lateral_drift, weight=DRIFT_WEIGHT,
        params={"scale": DRIFT_SCALE},
    )

    return cfg


__all__ = ["env_cfg"]
