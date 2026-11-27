"""Register the E1 single-box task."""

from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import box_env_cfg
from .rl_cfg import box_rl_cfg

TASK_ID = "PRCF-Parkour-E1-25DoF-BoxObstacles"

register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=box_env_cfg(),
    play_env_cfg=box_env_cfg(play=True),
    rl_cfg=box_rl_cfg(),
    runner_cls=MjlabOnPolicyRunner,
)
