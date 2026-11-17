"""The go command of `jumper.ref_free_jump`, with the rotation bookkeeping a flip needs.

`RefFreeJumpCommand` already decides **whose flight it is**: a flight counts only if
the episode pushed it, so an episode spawned mid-air on the recording cannot collect a
jump it did not make. The turn is counted under the same rule -- only while the
episode's own flight is in progress -- for the same reason: a reward for rotation
that also paid reference-state initialisation for what it was handed would be
farmed by the spawns instead of learnt.

**Which way is a backflip.** Measured on the model (`assets/jumper/jumper.xml`, at the
stand pose): the front legs sit at +x (0.19 m), the left ones at +y, and +z is up. A
positive rotation about +y takes +x down, nose down: a forward flip. A backflip is
the negative one, so the turn accumulates `-omega_y * dt`, with `omega_y` the body-frame
angular velocity -- the quantity the robot's gyro reports.

**A potential, not a stream.** The dense term pays the *change* of
`clamp(turn, 0, target) / target` each step, so a whole turn pays its weight once
whatever route it took; turning back pays it back, and rotating to and fro for free is
not a trade. A flight that never lifted the robot (a hop, see `RefFreeJumpCommand`)
gives its turn back the same way, by dropping the potential to zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ...ref_free_jump.mdp.commands import RefFreeJumpCommand, RefFreeJumpCommandCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class FlipCommandCfg(RefFreeJumpCommandCfg):
    turn_target: float = 2.0 * math.pi
    """Radians of backward turn that count as a whole flip. 2 pi for a backflip;
    a smaller value is a partial flip, which is how a curriculum would be written."""

    def __post_init__(self):
        self.class_type = FlipCommand

    def build(self, env: ManagerBasedRlEnv) -> FlipCommand:
        return FlipCommand(self, env)


class FlipCommand(RefFreeJumpCommand):
    cfg: FlipCommandCfg

    def __init__(self, cfg: FlipCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        n = self.num_envs
        #: Backward turn so far in the episode's own flight, radians.
        self.turn = torch.zeros(n, device=self.device)
        #: The turn at the touchdown that ended the episode's jump, latched.
        self.landed_turn = torch.zeros(n, device=self.device)
        #: How much `clamp(turn, 0, target) / target` changed in the last update.
        self.progress_delta = torch.zeros(n, device=self.device)
        self._flip_turns = torch.zeros(n, device=self.device)
        self.metrics["flip_turns"] = self._flip_turns

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        self.turn[env_ids] = 0.0
        self.landed_turn[env_ids] = 0.0
        self.progress_delta[env_ids] = 0.0

    def _update_metrics(self) -> None:
        super()._update_metrics()
        self._flip_turns[:] = self.landed_turn / (2.0 * math.pi)
        self.metrics["flip_turns"] = self._flip_turns

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        # What the parent decides at this update -- take-off, touchdown, a jump
        # landed -- is decided on the state the last step ended in, so the interval
        # just integrated was a flight exactly when the episode was in flight
        # *before* the parent ran.
        was_flying = self.in_flight.clone()
        was_landed = self.landed.clone()
        super()._update_command(env_ids)
        if env_ids is not None:
            # Reset path: the spawn state was just written and nothing has flown.
            return

        target = self.cfg.turn_target
        before = self.turn.clamp(0.0, target) / target

        # In place, for the reason the parent gives: rollouts run under inference
        # mode, and rebinding the attribute would make it an inference tensor.
        omega_y = self.robot.data.root_link_ang_vel_b[:, 1]
        step = -omega_y * self._env.step_dt
        self.turn.copy_(torch.where(was_flying, self.turn + step, self.turn))

        # A hop that never lifted the robot is not the episode's jump: its turn goes.
        hop = was_flying & ~self.in_flight & ~self.landed
        self.turn.masked_fill_(hop, 0.0)

        just_landed = self.landed & ~was_landed
        self.landed_turn.copy_(torch.where(just_landed, self.turn, self.landed_turn))

        after = self.turn.clamp(0.0, target) / target
        self.progress_delta.copy_(after - before)


__all__ = ["FlipCommand", "FlipCommandCfg"]
