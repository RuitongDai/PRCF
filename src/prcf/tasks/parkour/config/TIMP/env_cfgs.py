"""Environment and sensor configuration for terrain-indexed E1 parkour."""
import math

from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from prcf.asset.robot.e1.e1_constants import E1_ACTION_SCALE, get_e1_robot_cfg, get_spec
from .mdp import events, observations as obs, rewards as rew
from .mdp.support import SupportReward
from .rl.motion_loader import DEFAULT_POOLS
from .terrains import e1_amp_parkour_terrain_cfg


def parkour_robot_spec():
    spec = get_spec()
    angle = math.radians(60)
    spec.body("pelvis").add_site(
        name="parkour_depth", pos=(0.15, 0.0, 0.20), size=(0.006,),
        xyaxes=(0.0, -1.0, 0.0, math.sin(angle), 0.0, math.cos(angle)),
    )
    return spec


def timp_env_cfg(*, play=False, names=None, stage=2, manifest=str(DEFAULT_POOLS), seed=42):
    generator = e1_amp_parkour_terrain_cfg(names)
    robot = get_e1_robot_cfg()
    robot.spec_fn = parkour_robot_spec
    command = UniformVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(4.0, 8.0),
        heading_command=True,
        rel_world_envs=1.0,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.65, 0.65) if play else (0.35, 0.90),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(0.0, 0.0),
        ),
    )
    observations = {
        "actor": ObservationGroupCfg(terms={"proprio": ObservationTermCfg(func=obs.proprioception)},
                                     history_length=5,
                                     flatten_history_dim=False),
        "depth": ObservationGroupCfg(terms={"camera": ObservationTermCfg(func=obs.DepthImage)}),
        "critic": ObservationGroupCfg(terms={"state": ObservationTermCfg(func=obs.PrivilegedState)}),
        "amp": ObservationGroupCfg(terms={"motion": ObservationTermCfg(func=obs.AmpFeatures)}),
        "amp_route": ObservationGroupCfg(terms={"id": ObservationTermCfg(func=obs.route)}),
    }
    event_terms = {"reset": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")}
    if not play:
        event_terms["reference"] = EventTermCfg(func=events.ResetFromReference, mode="reset",
                                                 params={"manifest": manifest, "probability": 0.8})
    # mjlab's reward manager already multiplies these weights by control dt.
    rewards = {
        "linear_command": RewardTermCfg(func=rew.track_planar_velocity,
                                        weight=0.75,
                                        params={"std": 0.50}),
        "angular_command": RewardTermCfg(func=rew.track_yaw_velocity,
                                         weight=0.25,
                                         params={"std": 0.50}),
        "action_rate": RewardTermCfg(func=mdp.action_rate_l2,
                                     weight=-0.025),
        "joint_torque": RewardTermCfg(func=mdp.joint_torques_l2,
                                      weight=-0.000025),
        "joint_limits": RewardTermCfg(func=mdp.joint_pos_limits,
                                      weight=-2.5),
        "torso_contact": RewardTermCfg(func=rew.penalize_torso_contact,
                                       weight=-0.5),
    }
    if stage == 3 and not play:
        rewards["geometric_support"] = RewardTermCfg(func=SupportReward, weight=0.5)
    return ManagerBasedRlEnvCfg(
        seed=seed,
        scene=SceneCfg(
            num_envs=len(generator.sub_terrains) if play else 4096,
            terrain=TerrainEntityCfg(terrain_type="generator",
                                     terrain_generator=generator,
                                     max_init_terrain_level=5),
            entities={"robot": robot},
            sensors=obs.parkour_sensors(),
        ),
        observations=observations,
        actions={"joint_pos": JointPositionActionCfg(entity_name="robot",
                                                     actuator_names=(".*",),
                                                     scale=dict(E1_ACTION_SCALE),
                                                     use_default_offset=True)},
        commands={"twist": command},
        events=event_terms,
        rewards=rewards,
        metrics={"actuation": MetricsTermCfg(func=obs.capture_torque,
                                             per_substep=True),
                 "traversal_success": MetricsTermCfg(func=rew.success_indicator,
                                                     reduce="last")},
        terminations={
            "time": TerminationTermCfg(func=mdp.time_out, time_out=True),
            "success": TerminationTermCfg(func=rew.CourseSuccess),
            "left_course": TerminationTermCfg(func=rew.LeftCourse),
            "fall": TerminationTermCfg(func=rew.bad_base_pose),
        },
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="torso_link"),
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.005, iterations=20), nconmax=256, njmax=1024),
        decimation=4,
        episode_length_s=20.0,
    )
