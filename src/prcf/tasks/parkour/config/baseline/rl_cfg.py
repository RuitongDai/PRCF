"""VDWL/PPO settings adapted from the periodic 10 Hz E1 reference task."""
from dataclasses import dataclass, field

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from .rl.algorithm import BaselinePPO
from .rl.models import VDWLMLPActorModel
from .rl.motion_loader import DEFAULT_POOLS
from .terrains import TERRAINS, amp_heads_for


@dataclass
class BaselinePpoCfg(RslRlPpoAlgorithmCfg):
    amp: dict = field(default_factory=dict)
    symmetry_cfg: dict | None = None


def baseline_rl_cfg(*, play=False, names=TERRAINS, manifest=str(DEFAULT_POOLS), seed=42):
    algorithm = BaselinePpoCfg(
        class_name=f'{BaselinePPO.__module__}:BaselinePPO', learning_rate=1e-5,
        symmetry_cfg={
            'data_augmentation_func': 'prcf.tasks.parkour.config.baseline.symmetry:data_augmentation_func',
            'use_data_augmentation': False, 'use_mirror_loss': True, 'mirror_loss_coeff': 0.3,
        },
        amp={'manifest': manifest, 'heads': amp_heads_for(names), 'inference': play,
             'batch_size': 256, 'replay_capacity': 100_000},
    )
    return RslRlOnPolicyRunnerCfg(
        seed=seed, num_steps_per_env=24, max_iterations=50_000, save_interval=500,
        experiment_name='baseline', logger='tensorboard', upload_model=False, clip_actions=None,
        obs_groups={'actor': ('actor', 'depth'), 'critic': ('critic', 'critic_height_scan')},
        actor=RslRlModelCfg(class_name=f'{VDWLMLPActorModel.__module__}:VDWLMLPActorModel',
                          hidden_dims=(512, 256, 128), distribution_cfg={
                              'class_name': 'GaussianDistribution', 'init_std': 1., 'std_type': 'scalar'}),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128)), algorithm=algorithm,
    )
