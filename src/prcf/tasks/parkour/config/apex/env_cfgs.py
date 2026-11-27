"""Two independent APEX climbing environments using the E1 articulation."""

import re
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.managers import CurriculumTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from prcf.asset.robot.e1.e1_constants import E1_ACTION_SCALE, E1_JOINT_NAMES, E1_VELOCITY_LIMITS, get_e1_robot_cfg

from .mdp import observations, rewards, terminations
from .mdp.actions import ApexJointPositionActionCfg
from .mdp.curricula import terrain_height
from .mdp.events import apex_events
from .mdp.progress import RatchetProgress, goal_success, horizontal_progress
from .sensors import apex_sensors
from .terrains import apex_terrain_cfg


def _apex_env_cfg(*, descent: bool, play: bool) -> ManagerBasedRlEnvCfg:
    decimation = 4
    speed_limits = [
        0.9 * next(limit for pattern, limit in E1_VELOCITY_LIMITS.items() if re.fullmatch(pattern, name))
        for name in E1_JOINT_NAMES
    ]
    robot = get_e1_robot_cfg()
    # MuJoCo uses the higher-priority geom's friction and solver parameters.
    # Give each E1 collision geom authority over those values on terrain.
    robot.collisions = (replace(robot.collisions[0], condim=3, priority=1),)
    feet = SceneEntityCfg("robot", body_names=(r".*_ankle_roll_link",))
    return ManagerBasedRlEnvCfg(
        seed=42,
        decimation=decimation,
        episode_length_s=7.0,
        is_finite_horizon=not descent,
        scene=SceneCfg(
            num_envs=(1 if descent else 2) if play else 4096,
            terrain=apex_terrain_cfg(descent=descent, play=play),
            entities={"robot": robot},
            sensors=apex_sensors(descent=descent, play=play),
        ),
        sim=SimulationCfg(
            njmax=800, nconmax=96, contact_sensor_maxmatch=128,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_ROOT, entity_name="robot",
            distance=3.0, elevation=-15.0, azimuth=135.0,
        ),
        actions={
            "joint_pos": ApexJointPositionActionCfg(
                entity_name="robot", actuator_names=(".*",), scale=E1_ACTION_SCALE,
                use_default_offset=True, hold_reset_pose=descent,
            ),
        },
        observations=observations.observations_cfg(descent=descent, play=play),
        events=apex_events(descent=descent, play=play),
        curriculum={} if play else {"height": CurriculumTermCfg(func=terrain_height)},
        metrics={
            "success": MetricsTermCfg(func=goal_success, reduce="last"),
            "peak_contact_force": MetricsTermCfg(
                func=rewards.peak_force, params={"sensor_name": "body_forces"}, reduce="max",
            ),
        },
        rewards={
            "height_progress": RewardTermCfg(func=RatchetProgress, weight=-4, params={"descent": descent}),
            "horizontal_progress": RewardTermCfg(func=horizontal_progress, weight=-4),
            "alive": RewardTermCfg(func=mdp.is_alive, weight=15),
            "failure": RewardTermCfg(func=mdp.is_terminated, weight=-800),
            "wait": RewardTermCfg(func=rewards.wait_at_edge, weight=-1, params={"descent": descent}),
            "foot_pressure": RewardTermCfg(
                func=rewards.FootPressure, weight=-1, params={"descent": descent, "asset_cfg": feet},
            ),
            "contact_force": RewardTermCfg(
                func=rewards.excessive_force, weight=-1,
                params={"sensor_name": "body_forces", "threshold": 500.0, "gain": 0.005 if descent else 0.01},
            ),
            "head_force": RewardTermCfg(
                func=rewards.excessive_force, weight=-1,
                params={
                    "sensor_name": "body_forces", "threshold": 0.0, "gain": 0.10,
                    "asset_cfg": SceneEntityCfg("robot", body_names=("head_yaw_link",)),
                },
            ),
            "contact_slip": RewardTermCfg(
                func=rewards.contact_slip, weight=-0.50 if descent else -0.10, params={"descent": descent},
            ),
            "action_rate": RewardTermCfg(func=rewards.action_target_rate, weight=-0.20),
            "hip_joint_deviation": RewardTermCfg(
                func=rewards.hip_joint_deviation, weight=-6 if descent else -1, params={
                    "yaw_cfg": SceneEntityCfg("robot", joint_names=(r".*_hip_yaw_joint",)),
                    "roll_cfg": SceneEntityCfg("robot", joint_names=(r".*_hip_roll_joint",)),
                },
            ),
            "waist_joint_deviation": RewardTermCfg(
                func=rewards.waist_joint_deviation, weight=-6,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=("waist_yaw_joint",))},
            ),
            "joint_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10),
            "joint_speed_limits": RewardTermCfg(
                func=rewards.joint_speed_limit, weight=-10 if descent else -1, params={"limits": speed_limits},
            ),
            "joint_speed": RewardTermCfg(func=mdp.joint_vel_l2, weight=-0.001),
            "joint_speed_exp": RewardTermCfg(func=rewards.joint_speed_exp, weight=-1),
            "joint_acc": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.0e-8),
            "torque": RewardTermCfg(func=mdp.joint_torques_l2, weight=-1.5e-5),
            "power": RewardTermCfg(func=rewards.mechanical_power, weight=-1.0e-5),
            "angular_velocity_xy": RewardTermCfg(func=rewards.angular_velocity_xy, weight=-0.005),
            "torso_acceleration": RewardTermCfg(func=rewards.torso_acceleration, weight=-1.0e-4),
            "body_linear_acceleration": RewardTermCfg(func=rewards.body_linear_acceleration, weight=-2.0e-4),
        },
        terminations={
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
            "low_pelvis": TerminationTermCfg(func=terminations.low_pelvis),
            "outside_tile": TerminationTermCfg(func=terminations.outside_tile),
            "nan": TerminationTermCfg(func=mdp.nan_detection),
        },
    )


def apex_climb_up_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = _apex_env_cfg(descent=False, play=play)
    cfg.rewards.update(
        prone_pose=RewardTermCfg(func=rewards.prone_pose, weight=7),
        elbows_knees_on_top=RewardTermCfg(func=rewards.elbows_knees_on_top, weight=2),
        group_air_time=RewardTermCfg(func=rewards.group_air_time, weight=-1, params={
            "upper_cfg": SceneEntityCfg(
                "robot", body_names=(r".*_wrist_.*", r".*_shoulder_.*", r".*_elbow_link"),
            ),
            "lower_cfg": SceneEntityCfg("robot", body_names=(r".*_(hip_yaw|knee|ankle_roll)_link",)),
            "feet_cfg": SceneEntityCfg("robot", body_names=(r".*_ankle_roll_link",)),
        }),
    )
    cfg.terminations.update(
        returned_to_ground=TerminationTermCfg(
            func=terminations.returned_to_ground,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=(r".*_ankle_roll_link",))},
        ),
        too_far_back=TerminationTermCfg(func=terminations.too_far_back),
    )
    return cfg


def apex_climb_down_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = _apex_env_cfg(descent=True, play=play)
    torso = SceneEntityCfg("robot", body_names=("torso_link",))
    cfg.rewards.update(
        standing_joint=RewardTermCfg(func=rewards.standing_joint, weight=1),
        standing_orientation=RewardTermCfg(func=rewards.standing_orientation, weight=1),
        standing_linear_velocity=RewardTermCfg(
            func=rewards.standing_linear_velocity, weight=1, params={"asset_cfg": torso},
        ),
        standing_angular_velocity=RewardTermCfg(
            func=rewards.standing_angular_velocity, weight=1, params={"asset_cfg": torso},
        ),
        standing_height=RewardTermCfg(
            func=rewards.standing_height, weight=1,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=(r".*_shoulder_pitch_link",))},
        ),
    )
    cfg.terminations["knees_straight"] = TerminationTermCfg(
        func=terminations.knees_straight,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*_knee_joint",))},
    )
    return cfg
