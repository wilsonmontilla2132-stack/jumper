"""PPO hyperparameters for jumper.backflip_platform: jumper.backflip's."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..backflip.rl_cfg import agent_cfg as _backflip_agent_cfg


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = _backflip_agent_cfg()
    cfg.experiment_name = "jumper.backflip_platform"
    return cfg
