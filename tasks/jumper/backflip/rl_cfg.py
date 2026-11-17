"""PPO hyperparameters for jumper.backflip: jumper.ref_free_jump's, with its own budget."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..ref_free_jump.rl_cfg import NUM_STEPS_PER_ENV, agent_cfg as _ref_free_agent_cfg

#: The iterations to train for. `env_cfg.py` hands the recording over by iteration
#: 800; this leaves the objective, flip included, 3200 of its own. A guess.
MAX_ITERATIONS = 4000


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = _ref_free_agent_cfg()
    # The rollout length is `env_cfg.py`'s schedule's unit (iterations x steps), so
    # it is `ref_free_jump`'s and is asserted rather than assumed.
    assert cfg.num_steps_per_env == NUM_STEPS_PER_ENV
    cfg.experiment_name = "jumper.backflip"
    cfg.max_iterations = MAX_ITERATIONS
    return cfg
