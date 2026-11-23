"""Register the independent E1 25-DoF VDWL multi-head AMP baseline."""
from mjlab.tasks.registry import register_mjlab_task
from .env import BaselineEnv
from .env_cfgs import baseline_env_cfg
from .rl.runner import BaselineRunner
from .rl_cfg import baseline_rl_cfg

TASK_ID = 'PRCF-Parkour-E1-25DoF-Baseline'
register_mjlab_task(
    task_id=TASK_ID, env_cfg=baseline_env_cfg(), play_env_cfg=baseline_env_cfg(play=True),
    rl_cfg=baseline_rl_cfg(), runner_cls=BaselineRunner,
)
