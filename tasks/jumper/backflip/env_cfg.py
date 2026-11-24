"""jumper.backflip -- a standing backflip, reference-free on the robot.

`jumper.ref_free_jump`'s environment with the parts a flip contradicts replaced, and
nothing else touched: it is built by calling that task's `env_cfg()` and editing
the result, so the physics, the actor's observation, the action term and its
fading feedforward, the domain randomisation and the way home are the ones that
task trained and validated.

## What is replaced, and why

- **The command** (`mdp/commands.py`): the same go and flight bookkeeping, plus the
  backward turn accumulated in the episode's own flight.
- **`upright`** -> `upright_grounded`: paid only outside the flight. The original
  would charge the flip for being upside down.
- **`fell_over`** -> `fell_over_grounded`: the 50 degree tilt limit applies outside the
  flight only. See `mdp/terminations.py`.
- **Two terms added**, `flip_progress` and `flip_landing`, cross-faded in on the
  same window as the height terms, and for the same reason `ref_free_jump` gives:
  paid from the start they would pull the policy off the recording's jump before it
  has it.
- **A critic observation**, `flip_state`.

## On the numbers below

**The reward weights, the schedule and the iteration budget are first guesses, not
measurements** -- nothing here has been trained at the time of writing. The
physics they rest on is measured (the recording's 1.556 m/s take-off, 0.32 s of
flight, and the model's 0.0188 kg m^2 pitch inertia at the stand pose: a whole
turn in that flight is about 19.8 rad/s and 3.7 J of rotation on top of the jump's
3.1 J; see the version note in the task's README once there is one). Where a run
sets one, the note beside it should say which checkpoint and how it was measured.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from ..ref_free_jump import mdp as ref_free_mdp
from ..ref_free_jump.env_cfg import env_cfg as _ref_free_env_cfg
from ..ref_free_jump.rl_cfg import NUM_STEPS_PER_ENV
from . import mdp as flip_mdp

# ── The flip ───────────────────────────────────────────────────────────────────
#: Backward turn, in radians, that counts as one flip.
TURN_TARGET = 2.0 * math.pi

#: `flip_progress`'s weight: what a whole turn pays in total, over the flight
#: (a potential; see `mdp/commands.py`). The flight is about 60 control steps at 200 Hz,
#: so this is about 3 a step in a flip made at the recording's pace. A guess.
FLIP_PROGRESS_WEIGHT = 800.0

#: `flip_landing`'s weight: paid every step after landing, in proportion to the
#: share of the turn made and how upright the body is. `ref_free_jump`'s landed apex
#: pays about 4 x 1.2 a step for a jump as high as the recording's, so this is set to
#: exceed what a flat jump earns, or the flat jump is the better policy. A guess.
FLIP_LANDING_WEIGHT = 15.0
FLIP_LANDING_SIGMA_SQ = 0.10

#: `ref_free_jump`'s weight for `upright`, which `upright_grounded` replaces; asserted below.
REF_UPRIGHT_WEIGHT = 1.0
#: `upright_grounded`'s weight, a fifth of that. Hypothesis, not a measurement: the
#: first run (1024 envs, T4, 1300 iterations) ended at 0.08 turns, and with the
#: full weight a failed flip, ending on the back, forfeits most of an episode's
#: upright and home-pose reward while a partial turn pays little, so not trying
#: pays more. Lowering this and raising `FLIP_PROGRESS_WEIGHT` (200 -> 800) is the
#: cheapest change that tests that reading.
UPRIGHT_WEIGHT = 0.2
UPRIGHT_SIGMA_SQ = 0.10

# ── The schedule ───────────────────────────────────────────────────────────────
#: `ref_free_jump` holds the recording's feedforward and prior until iteration 1000
#: and ends them at 3000. The jump itself is learnt in the first tens of iterations
#: with the feedforward on (`jumper.jump`, 1024 envs, T4: reward flat at 46 by
#: iteration 30), so the hold here is shorter, to give the flip more of the budget.
#: A guess, and the first thing to lengthen if the policy has no jump left when the
#: feedforward goes.
PRIOR_HOLD_ITERATIONS = 100
PRIOR_END_ITERATIONS = 350

#: The height terms' final weights. `ref_free_jump`'s are 4 and 2; halved, so that
#: height does not out-pay the turn. A guess.
APEX_END_WEIGHT = 2.0
FLIGHT_END_WEIGHT = 1.0

#: The tilt past which a grounded episode ends: `ref_free_jump`'s, the controller's.
FELL_OVER_RAD = math.radians(50.0)


def _reschedule(cfg: ManagerBasedRlEnvCfg) -> None:
    """Put the prior's window where this task wants it, in the action term and in
    every cross-fade at once.

    The action term's feedforward and the curriculum's weights count the same
    steps, and one moved without the other would fade the action and the reward on
    different days in silence -- which is why `ref_free_jump` writes the window
    once. It is written once here too, and these are its only readers.
    """
    hold = NUM_STEPS_PER_ENV * PRIOR_HOLD_ITERATIONS
    end = NUM_STEPS_PER_ENV * PRIOR_END_ITERATIONS
    action = cfg.actions["joint_pos"]
    action.hold_step = hold
    action.end_step = end
    for term in cfg.curriculum.values():
        if "start_step" in (term.params or {}):
            term.params = {**term.params, "start_step": hold, "end_step": end}


def _cross_fade(term: str, start: float, end: float) -> CurriculumTermCfg:
    return CurriculumTermCfg(
        func=ref_free_mdp.curriculum.anneal_reward_weight,
        params={
            "term_name": term,
            "start_weight": start,
            "end_weight": end,
            "start_step": NUM_STEPS_PER_ENV * PRIOR_HOLD_ITERATIONS,
            "end_step": NUM_STEPS_PER_ENV * PRIOR_END_ITERATIONS,
        },
    )


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Standing backflip.

    Args:
        asset: path to the model XML, from `--model`. None uses the default jumper.xml.
        play: replay mode -- noise and pushes off, every episode starts at HOME and
            jumps on the go, as the deployment will.
    """
    cfg = _ref_free_env_cfg(asset=asset, play=play)

    # ── The command: the same go, plus the turn ────────────────────────────────
    old = cfg.commands["jump"]
    cfg.commands["jump"] = flip_mdp.commands.FlipCommandCfg(
        **{f.name: getattr(old, f.name) for f in dataclasses.fields(old) if f.init},
        turn_target=TURN_TARGET,
    )

    # ── Rewards: the upright term gated, the flip terms added at weight 0 ──────
    old_upright = cfg.rewards.pop("upright")
    assert old_upright.weight == REF_UPRIGHT_WEIGHT, "ref_free_jump's upright weight moved"
    cfg.rewards["upright_grounded"] = RewardTermCfg(
        func=flip_mdp.rewards.upright_grounded, weight=UPRIGHT_WEIGHT,
        params={"command_name": "jump", "sigma_sq": UPRIGHT_SIGMA_SQ},
    )
    cfg.rewards["flip_progress"] = RewardTermCfg(
        func=flip_mdp.rewards.flip_progress, weight=0.0,
        params={"command_name": "jump"},
    )
    cfg.rewards["flip_landing"] = RewardTermCfg(
        func=flip_mdp.rewards.flip_landing, weight=0.0,
        params={"command_name": "jump", "sigma_sq": FLIP_LANDING_SIGMA_SQ},
    )

    # ── Curriculum: the same window for everything ─────────────────────────────
    cfg.curriculum["apex_height_weight"].params["end_weight"] = APEX_END_WEIGHT
    cfg.curriculum["flight_height_weight"].params["end_weight"] = FLIGHT_END_WEIGHT
    cfg.curriculum["flip_progress_weight"] = _cross_fade(
        "flip_progress", 0.0, FLIP_PROGRESS_WEIGHT
    )
    cfg.curriculum["flip_landing_weight"] = _cross_fade(
        "flip_landing", 0.0, FLIP_LANDING_WEIGHT
    )
    _reschedule(cfg)

    # ── Termination: tipped over, but not in the flight ────────────────────────
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=flip_mdp.terminations.fell_over_grounded,
        params={"command_name": "jump", "limit_angle": FELL_OVER_RAD},
    )

    # ── The critic is shown the turn ───────────────────────────────────────────
    cfg.observations["critic"].terms["flip_state"] = ObservationTermCfg(
        func=flip_mdp.observations.flip_state,
        params={"command_name": "jump", "mirror_kind": "scalar"},
    )

    return cfg


__all__ = ["env_cfg"]
