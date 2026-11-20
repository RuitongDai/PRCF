"""E1 reference tracking with identical train/play/export observation dimensions."""
from copy import deepcopy
from dataclasses import fields
from typing import Literal

from mjlab.managers import TerminationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.tasks.tracking.mdp.terminations import bad_anchor_pos
from prcf.asset.robot.e1.e1_constants import E1_ACTION_SCALE, get_e1_robot_cfg
from prcf.asset.scene import SceneBuilder
from .mdp.commands import FiniteMotionCommandCfg
from .mdp.metrics import body_position_error, root_position_error
from .mdp.terminations import motion_finished


# Bodies used by the tracking command and AMP features.
# pelvis must be first because the motion command uses it to initialize the root.
TRACKING_BODIES = (
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_pitch_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_pitch_link",
)

def e1_mimic_env_cfg(mode: Literal['train', 'play'] = 'train'):
    cfg = make_tracking_env_cfg()
    cfg.scene.entities = {'robot': get_e1_robot_cfg()}
    # Keep reference initialization consistent with the model joint limits.
    cfg.scene.entities['robot'].articulation.soft_joint_pos_limit_factor = 1.0
    # Independent mjwarp worlds share a fixed scene and zero environment origins.
    cfg.scene.terrain = None
    cfg.scene.spec_fn = SceneBuilder({'ground_height': 0.0, 'obstacles': []})
    cfg.scene.num_envs = 1024 if mode == 'train' else 1
    cfg.actions['joint_pos'].scale = dict(E1_ACTION_SCALE)
    original = cfg.commands['motion']
    cfg.commands['motion'] = FiniteMotionCommandCfg(**{
        item.name: deepcopy(getattr(original, item.name)) for item in fields(original)
    })
    motion = cfg.commands['motion']
    motion.anchor_body_name = 'torso_link'
    motion.body_names = TRACKING_BODIES  # pelvis first: reference root initialization
    motion.joint_position_range = (0.0, 0.0)
    motion.pose_range = {}
    motion.velocity_range = {}
    motion.debug_vis = mode == 'play'
    # Half the resets cover the whole clip, half emphasize hard regions.
    motion.adaptive_uniform_ratio = 0.5
    # This task prepares nominal physical references, not a deployment policy.
    cfg.observations['actor'].enable_corruption = False
    cfg.events = {}  # Nominal physics references, without external pushes or DR.
    cfg.terminations['motion_finished'] = TerminationTermCfg(func=motion_finished, time_out=True)
    # Fixed obstacles require global alignment. Z-only checks let a standing
    # robot lag over half a metre behind a walking reference without failure.
    cfg.terminations['anchor_pos'].func = bad_anchor_pos
    cfg.terminations['anchor_pos'].params['threshold'] = 0.35
    cfg.terminations['ee_body_pos'].params['body_names'] = (
        'left_ankle_roll_link', 'right_ankle_roll_link',
        'left_wrist_pitch_link', 'right_wrist_pitch_link',
    )
    cfg.metrics = {
        'global_root_error': MetricsTermCfg(func=root_position_error),
        'global_body_error': MetricsTermCfg(func=body_position_error),
    }
    cfg.scene.sensors = (
        ContactSensorCfg(
            name='self_collision',
            primary=ContactMatch(mode='subtree', pattern='pelvis', entity='robot'),
            secondary=ContactMatch(mode='subtree', pattern='pelvis', entity='robot'),
            fields=('found', 'dist'), reduce='mindist',
        ),
    )
    cfg.viewer.body_name = 'torso_link'
    cfg.sim.mujoco.timestep = 0.005
    cfg.sim.mujoco.iterations = 20
    cfg.sim.nconmax = 256
    cfg.sim.njmax = 1024
    cfg.decimation = 4
    cfg.episode_length_s = 60.0  # Set from input motion before constructing the env.
    if mode != 'train':
        cfg.observations['actor'].enable_corruption = False
        motion.sampling_mode = 'start'
        motion.joint_position_range = (0.0, 0.0)
    return cfg
