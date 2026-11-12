"""Reuse baseline PPO/network parameters to isolate the action/reward experiment."""
from ..e1.rl_cfg import e1_mimic_runner_cfg


def e1_reference_runner_cfg():
    cfg = e1_mimic_runner_cfg()
    cfg.experiment_name = 'e1_mimic_reference'
    return cfg
