"""Reference-centred E1 Mimic experimental task registration."""
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task
from .env_cfgs import e1_reference_env_cfg
from .rl_cfg import e1_reference_runner_cfg

TASK_ID = 'PRCF-Mimic-E1-25DoF-Reference'
register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=e1_reference_env_cfg(),
    play_env_cfg=e1_reference_env_cfg(mode='play'),
    rl_cfg=e1_reference_runner_cfg(),
    runner_cls=MjlabOnPolicyRunner,
)
