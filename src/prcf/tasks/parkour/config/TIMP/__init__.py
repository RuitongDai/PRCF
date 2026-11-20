"""Register the E1 25-DoF terrain-indexed AMP baseline."""
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import timp_env_cfg
from .rl.runner import TimpRunner
from .rl_cfg import timp_rl_cfg

TASK_ID = "PRCF-Parkour-E1-25DoF-TIMP"

register_mjlab_task(
    task_id=TASK_ID, env_cfg=timp_env_cfg(), play_env_cfg=timp_env_cfg(play=True),
    rl_cfg=timp_rl_cfg(), runner_cls=TimpRunner,
)
