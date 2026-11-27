"""PPO and single-discriminator AMP settings for E1 single-box traversal."""

from dataclasses import dataclass, field

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from .rl.algorithm import BoxAmpPPO
from .rl.motion_loader import DEFAULT_POOLS


@dataclass
class BoxPpoCfg(RslRlPpoAlgorithmCfg):
    amp: dict = field(default_factory=dict)


def box_rl_cfg(*, play=False, seed=42) -> RslRlOnPolicyRunnerCfg:
    actor = RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )
    critic = RslRlModelCfg(hidden_dims=(512, 256, 128))
    algorithm = BoxPpoCfg(
        class_name=f"{BoxAmpPPO.__module__}:{BoxAmpPPO.__name__}",
        amp={
            "manifest": str(DEFAULT_POOLS),
            "inference": play,
            "style_weight": 0.10,
            "batch_size": 256,
            "replay_capacity": 100_000,
        },
    )
    return RslRlOnPolicyRunnerCfg(
        seed=seed,
        num_steps_per_env=24,
        max_iterations=50_000,
        save_interval=500,
        experiment_name="box_obstacles",
        logger="tensorboard",
        upload_model=False,
        obs_groups={"actor": ("actor",), "critic": ("actor", "critic")},
        actor=actor,
        critic=critic,
        algorithm=algorithm,
    )
