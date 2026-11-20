"""Table II PPO and discriminator settings for the TIMP port."""
from dataclasses import dataclass, field

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from .rl.algorithm import TimpPPO
from .rl.models import TimpActor, TimpCritic
from .rl.motion_loader import DEFAULT_POOLS
from .terrains import TERRAINS, amp_heads_for


@dataclass
class TimpPpoCfg(RslRlPpoAlgorithmCfg):
    amp: dict = field(default_factory=dict)


def timp_rl_cfg(*, play=False, stage=2, manifest=str(DEFAULT_POOLS), names=TERRAINS, seed=42):
    heads = amp_heads_for(names)
    actor = RslRlModelCfg(
        class_name=f"{TimpActor.__module__}:TimpActor", hidden_dims=(256, 128, 64),
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "scalar"},
    )
    critic = RslRlModelCfg(class_name=f"{TimpCritic.__module__}:TimpCritic", hidden_dims=(256, 128, 64))
    algorithm = TimpPpoCfg(
        class_name=f"{TimpPPO.__module__}:TimpPPO", entropy_coef=0.01,
        amp={"stage": stage, "manifest": manifest, "heads": heads, "inference": play,
             "batch_size": 256, "replay_capacity": 50_000},
    )
    return RslRlOnPolicyRunnerCfg(
        seed=seed, num_steps_per_env=24, max_iterations=20_000 if stage == 2 else 10_000,
        obs_groups={"actor": ("actor", "depth"), "critic": ("actor", "critic")},
        save_interval=500, experiment_name="TIMP", logger="tensorboard", upload_model=False,
        clip_actions=5.0, actor=actor, critic=critic, algorithm=algorithm,
    )
