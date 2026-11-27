"""Joint ascent and descent training on two copies of the same box."""

from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg,
)
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from prcf.asset.robot.e1.e1_constants import E1_ACTION_SCALE, get_e1_robot_cfg

from .mdp import observations as obs
from .mdp import rewards as box_rewards
from .mdp import terminations as done
from .mdp.commands import BoxForwardCommandCfg
from .mdp.events import ReferenceReset, reset_box_spawn
from .rl.motion_loader import DEFAULT_POOLS
from .terrains import box_terrain_cfg


def box_env_cfg(*, play=False, seed=42, num_envs=None, progress=True) -> ManagerBasedRlEnvCfg:
    decimation = 4
    scene = SceneCfg(
        num_envs=num_envs if num_envs is not None else (2 if play else 4096),
        terrain=box_terrain_cfg(),
        entities={"robot": get_e1_robot_cfg()},
        sensors=obs.box_sensors(contact_history=decimation if progress else 0),
    )

    observations = {
        "actor": ObservationGroupCfg(terms={
            "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel, scale=0.25),
            "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
            "command": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "forward"}),
            "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
            "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, scale=0.05),
            "last_action": ObservationTermCfg(func=mdp.last_action),
            "box_geometry": ObservationTermCfg(func=obs.box_geometry),
        }),
        "critic": ObservationGroupCfg(terms={
            "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
        }),
        "amp": ObservationGroupCfg(terms={"motion": ObservationTermCfg(func=obs.AmpFeatures)}),
    }

    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot", actuator_names=(".*",), scale=E1_ACTION_SCALE, use_default_offset=True,
        ),
    }

    commands = {
        "forward": BoxForwardCommandCfg(
            resampling_time_range=(20.0, 20.0),
            speed_range=(0.55, 0.55) if play else (0.45, 0.65),
        ),
    }

    events = {"reset": EventTermCfg(func=reset_box_spawn, mode="reset")}
    if not play:
        events["reference_state"] = EventTermCfg(
            func=ReferenceReset, mode="reset", params={
                "manifest": str(DEFAULT_POOLS),
                "ascent_probability": 0.40,
                "descent_probability": 0.25,
            },
        )

    rewards = {
        "track_forward_speed": RewardTermCfg(func=box_rewards.track_forward_speed, weight=2.0, params={"std": 0.35}),
    }
    if progress:
        rewards.update(
            box_progress=RewardTermCfg(func=box_rewards.BoxProgress, weight=5.0, params={"speed_reference": 0.55}),
            box_ascent_heading=RewardTermCfg(func=box_rewards.ascent_heading_error, weight=-4.0),
            box_limb_progress=RewardTermCfg(func=box_rewards.BoxLimbProgress, weight=4.0, params={"speed_reference": 0.55}),
            box_support_bonus=RewardTermCfg(func=box_rewards.support_bonus, weight=40.0),
            box_settle_motion=RewardTermCfg(func=box_rewards.settle_motion, weight=-1.0),
            box_descent_impact_speed=RewardTermCfg(func=box_rewards.descent_impact_speed, weight=-2.0),
            box_standing_bonus=RewardTermCfg(func=box_rewards.standing_bonus, weight=300.0),
            box_failure=RewardTermCfg(func=box_rewards.failure, weight=-40.0),
            action_rate=RewardTermCfg(func=mdp.action_rate_l2, weight=-0.02),
            joint_limits=RewardTermCfg(func=mdp.joint_pos_limits, weight=-5.0),
            head_contact=RewardTermCfg(func=box_rewards.head_contact, weight=-2.0, params={"threshold": 5.0}),
            # E1 weighs about 286 N. A 550 N per-link threshold leaves normal bracing unpenalized.
            excessive_terrain_force=RewardTermCfg(
                func=box_rewards.excessive_terrain_force, weight=-1.0, params={"threshold": 550.0},
            ),
        )

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "torso_collision": TerminationTermCfg(func=done.torso_collision),
        "low_pelvis": TerminationTermCfg(func=done.low_pelvis),
        "bad_orientation": TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": 1.77}),
        "left_box_route": TerminationTermCfg(func=done.left_box_route),
        "nan": TerminationTermCfg(func=mdp.nan_detection),
        "box_success": TerminationTermCfg(
            func=done.BoxSuccess,
            params={"standing_seconds": 0.30},
        ),
    }

    metrics = {
        "actuation": MetricsTermCfg(func=obs.capture_torque, per_substep=True),
        "box_arm_support": MetricsTermCfg(func=done.box_arm_support, reduce="last"),
        "box_foot_support": MetricsTermCfg(func=done.box_foot_support, reduce="last"),
    }

    sim = SimulationCfg(
        njmax=800, nconmax=96, contact_sensor_maxmatch=128,
        mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    )

    viewer = ViewerConfig(
        origin_type=ViewerConfig.OriginType.WORLD,
        lookat=(0.35, 0.0, 0.80), distance=9.0, elevation=-20.0, azimuth=145.0,
    )

    return ManagerBasedRlEnvCfg(
        seed=seed,
        scene=scene,
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        metrics=metrics,
        sim=sim,
        viewer=viewer,
        decimation=decimation,
        episode_length_s=20.0,
    )
