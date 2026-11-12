"""Ordinary PPO for offline reference preparation; no AMP or distillation."""
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def e1_mimic_runner_cfg():
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128), activation='elu', obs_normalization=True,
            distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': 0.5, 'std_type': 'scalar'},
        ),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation='elu', obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(),
        experiment_name='e1_mimic', logger='tensorboard', upload_model=False,
        save_interval=500, num_steps_per_env=24, max_iterations=30000,
    )
