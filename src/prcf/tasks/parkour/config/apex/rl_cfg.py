"""PPO configuration shared by the two APEX policies."""

from dataclasses import dataclass

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class ApexPpoCfg(RslRlPpoAlgorithmCfg):
    symmetry_cfg: dict | None = None


def apex_rl_cfg(skill: str) -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128), obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        ),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), obs_normalization=True),
        algorithm=ApexPpoCfg(
            entropy_coef=0.001,
            symmetry_cfg={
                "use_data_augmentation": True, "use_mirror_loss": False,
                "data_augmentation_func": f"{__package__}.symmetry:mirror_apex", "mirror_loss_coeff": 0.0,
            },
        ),
        num_steps_per_env=24,
        max_iterations=19000,
        clip_actions=100.0,
        save_interval=250 if skill == "climb_up" else 500,
        experiment_name=f"apex_e1_{skill}",
        logger="tensorboard",
        upload_model=False,
    )
