"""Register the two E1 APEX skills."""

from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import apex_climb_down_env_cfg, apex_climb_up_env_cfg
from .rl_cfg import apex_rl_cfg

CLIMB_UP_TASK_ID = "PRCF-Parkour-E1-25DoF-ApexClimbUp"
CLIMB_DOWN_TASK_ID = "PRCF-Parkour-E1-25DoF-ApexClimbDown"
TASK_IDS = (CLIMB_UP_TASK_ID, CLIMB_DOWN_TASK_ID)
APEX_LAYOUT = "apex_source_terms_v2"

for task_id, skill, env_cfg in (
    (CLIMB_UP_TASK_ID, "climb_up", apex_climb_up_env_cfg),
    (CLIMB_DOWN_TASK_ID, "climb_down", apex_climb_down_env_cfg),
):
    register_mjlab_task(
        task_id=task_id,
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(play=True),
        rl_cfg=apex_rl_cfg(skill),
        runner_cls=MjlabOnPolicyRunner,
    )
