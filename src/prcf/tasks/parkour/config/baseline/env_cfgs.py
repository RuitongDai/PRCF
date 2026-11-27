"""Independent E1 25-DoF VDWL baseline with terrain-routed AMP."""
import math
import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from prcf.asset.robot.e1.e1_constants import E1_ACTION_SCALE, get_e1_robot_cfg, get_spec
from mjlab.managers.metrics_manager import MetricsTermCfg
from .mdp import amp
from .mdp.box_traversal import (
  BoxTraversal, box_exit_bonus, box_lift_progress, box_mount_bonus, box_task_succeeded, reset_box_pose,
)
from . import mdp
from .constants import MJLAB_JOINT_NAMES
from .mdp import Stage2TargetVelocityCommandCfg, LegacyGridPatternCfg
from .mdp.terminations import failure_termination, time_out_with_rough_boundary
from .sensors import PeriodicCameraSensorCfg
from .terrains import SingleBoxTerrainCfg, e1_amp_parkour_terrain_cfg


def baseline_robot_cfg():
  robot = get_e1_robot_cfg()
  robot.spec_fn = baseline_robot_spec
  return robot


def baseline_robot_spec():
  spec = get_spec()
  torso = spec.body("torso_link")
  angle = math.radians(70)
  torso.add_camera(name="depth_cam", pos=(0.09, 0.0, 0.18),
                   xyaxes=(0, -1, 0, math.sin(angle), 0, math.cos(angle)), fovy=58)
  spec.add_sensor(name="angular-velocity", type=mujoco.mjtSensor.mjSENS_GYRO,
                  objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu_in_torso")
  spec.add_sensor(name="orientation", type=mujoco.mjtSensor.mjSENS_FRAMEQUAT,
                  objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu_in_torso")
  return spec


def _terrain_scan(play: bool) -> RayCastSensorCfg:
  return RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="pelvis", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(2.0, 1.2), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=False,
    viz=RayCastSensorCfg.VizCfg(
      hit_color=(0.15, 0.65, 1.0, 0.85),
      miss_color=(1.0, 0.15, 0.15, 0.65),
      hit_sphere_color=(0.1, 0.85, 1.0, 1.0),
      hit_sphere_radius=0.16,
      show_rays=False,
      show_normals=False,
    ),
  )


def _foot_edge_scan(play: bool) -> TerrainHeightSensorCfg:
  return TerrainHeightSensorCfg(
    name="foot_edge_scan",
    frame=tuple(
      ObjRef(type="body", name=name, entity="robot")
      for name in ("left_ankle_roll_link", "right_ankle_roll_link")
    ),
    pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=6),
    ray_alignment="yaw",
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=False,
    viz=RayCastSensorCfg.VizCfg(
      hit_color=(1.0, 0.55, 0.05, 0.9),
      miss_color=(1.0, 0.1, 0.1, 0.7),
      hit_sphere_color=(1.0, 0.85, 0.1, 1.0),
      hit_sphere_radius=0.24,
      show_rays=False,
    ),
  )

def _vdwl_foot_scan(name: str, body: str, play: bool) -> RayCastSensorCfg:
  pattern = LegacyGridPatternCfg(
    tuple(round(i * 0.01, 2) for i in range(151)),
    (0.0,),
    z_offset=1.2,
  )
  return RayCastSensorCfg(
    name=name,
    frame=ObjRef(type="body", name=body, entity="robot"),
    pattern=pattern,
    ray_alignment="yaw",
    max_distance=2.5,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=False,
    viz=RayCastSensorCfg.VizCfg(
      hit_color=(0.1, 1.0, 0.2, 0.95),
      miss_color=(1.0, 0.1, 0.1, 0.7),
      hit_sphere_color=(0.1, 1.0, 0.25, 1.0),
      hit_sphere_radius=0.28,
      show_rays=False,
    ),
  )

def _vdwl_hold_scan(name: str, body: str, play: bool) -> RayCastSensorCfg:
  pattern = LegacyGridPatternCfg(tuple(round(-0.10 + i * 0.01, 2) for i in range(21)), tuple(round(-0.04 + i * 0.01, 2) for i in range(9)))
  return RayCastSensorCfg(
    name=name,
    frame=ObjRef(type="body", name=body, entity="robot"),
    pattern=pattern,
    ray_alignment="yaw",
    max_distance=2.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=False,
    viz=RayCastSensorCfg.VizCfg(
      hit_color=(0.75, 0.1, 1.0, 0.9),
      miss_color=(1.0, 0.1, 0.1, 0.7),
      hit_sphere_color=(0.95, 0.2, 1.0, 1.0),
      hit_sphere_radius=0.22,
      show_rays=False,
    ),
  )


def baseline_env_cfg(
  play: bool = False, names=None, seed=42,
) -> ManagerBasedRlEnvCfg:
  terrain_generator = e1_amp_parkour_terrain_cfg(names)
  has_box = any(isinstance(tile, SingleBoxTerrainCfg) for tile in terrain_generator.sub_terrains.values())
  all_joints = SceneEntityCfg(
    "robot", joint_names=MJLAB_JOINT_NAMES, preserve_order=True
  )
  feet = SceneEntityCfg(
    "robot",
    body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
    preserve_order=True,
  )
  leg_pose = SceneEntityCfg(
    "robot",
    joint_names=(
      ".*_hip_roll_joint", ".*_hip_yaw_joint"
    ),
  )
  actor_terms = {
    "gait_clock": ObservationTermCfg(
      func=mdp.gait_clock,
      params={"command_name": "base_velocity", "period": 1.2},
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/angular-velocity"},
      scale=0.2, noise=Unoise(n_min=-0.5, n_max=0.5)
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.torso_imu_projected_gravity,
      params={"sensor_name": "robot/orientation"},
      noise=Unoise(n_min=-0.05, n_max=0.05)
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "base_velocity"}
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"asset_cfg": all_joints},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={"asset_cfg": all_joints},
      scale=0.05,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "last_action": ObservationTermCfg(func=mdp.last_action),
  }
  critic_terms = {
    "gait_clock": ObservationTermCfg(
      func=mdp.gait_clock,
      params={"command_name": "base_velocity", "period": 1.2},
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/angular-velocity"},
      scale=0.2,
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.torso_imu_projected_gravity,
      params={"sensor_name": "robot/orientation"},
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "base_velocity"}
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel, params={"asset_cfg": all_joints}
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, params={"asset_cfg": all_joints}, scale=0.05
    ),
    "last_action": ObservationTermCfg(func=mdp.last_action),
    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
    "foot_contacts": ObservationTermCfg(
      func=mdp.foot_contacts, params={"sensor_name": "feet_ground_contact"}
    ),
    "foot_clearance": ObservationTermCfg(
      func=mdp.foot_clearance,
      params={"feet_cfg": feet, "height_sensor_name": "terrain_scan", "sole_offset": 0.06},
    ),
  }
  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=not play,
      history_length=15,
      flatten_history_dim=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=5,
      flatten_history_dim=True,
    ),
    "depth": ObservationGroupCfg(
      terms={
        "frames": ObservationTermCfg(
          func=mdp.DepthHistory,
          params={
            "sensor_name": "depth_camera",
            "history_length": 2,
            "history_stride": 5,
            "output_frames": 2,
            "delayed_frame_range": (0, 0),
            "crop_top": 9,
            "crop_bottom": 9,
            "crop_left": 16,
            "crop_right": 16,
            "min_distance": 0.1,
            "max_distance": 3.0,
            "enable_noise": not play,
            "depth_std": 0.01,
            "depth_std_multiplier": 0.015,
            "dropout_prob": 0.02,
            "dropout_value": -1.0,
          },
        )
      },
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "critic_height_scan": ObservationGroupCfg(
      terms={
        "height_scan": ObservationTermCfg(
          func=mdp.height_scan,
          params={"sensor_name": "terrain_scan", "offset": 0.5},
          clip=(-1.0, 1.0),
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "pie_targets": ObservationGroupCfg(
      terms={
        "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
        "height_map": ObservationTermCfg(
          func=mdp.height_scan,
          params={"sensor_name": "terrain_scan", "offset": 0.5},
          clip=(-1.0, 1.0),
        ),
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "amp": ObservationGroupCfg(
      terms={
        "state": ObservationTermCfg(
          func=amp.AmpFeatures,
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }
  observations["amp_route"] = ObservationGroupCfg(terms={"head": ObservationTermCfg(func=amp.route)})
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=E1_ACTION_SCALE,
      use_default_offset=True,
    )
  }
  commands: dict[str, CommandTermCfg] = {
    "base_velocity": Stage2TargetVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(20.0, 20.0),
      rel_standing_envs=0.05,
      complex_target_x=4.0,
      stair_forward_center_fraction=1.0,
      stair_forward_diagonal_fraction=0.0,
      stair_backward_diagonal_fraction=0.0,
      stair_lateral_target_range=1.50,
      square_gap_diagonal_fraction=0.0,
      square_gap_lateral_target_range=1.00,
      zero_stop_probability=0.0 if play else 0.25,
      debug_vis=play,
    )
  }
  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (0.0, 0.0)
        },
        "velocity_range": {
          "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
          "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        },
      },
    ),
    "reset_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (-1.0, 1.0),
        "asset_cfg": all_joints,
      },
    ),
  }
  if has_box:
    events["reset_box_pose"] = EventTermCfg(func=reset_box_pose, mode="reset")
  if not play:
    events.update({
      "push_robot": EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
      ),
      "base_mass": EventTermCfg(
        func=dr.body_mass,
        mode="startup",
        params={
          "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",)),
          "operation": "add",
          "ranges": (-2.0, 5.0),
        },
      ),
      "friction": EventTermCfg(
        func=dr.geom_friction,
        mode="startup",
        params={
          "asset_cfg": SceneEntityCfg(
            "robot", geom_names=r"^(left|right)_foot[1-7]_collision$"
          ),
          "operation": "abs",
          "ranges": (0.3, 1.0),
          "shared_random": True,
        },
      ),
      "camera_position": EventTermCfg(
        func=dr.cam_pos,
        mode="startup",
        params={
          "asset_cfg": SceneEntityCfg(
            "robot", camera_names=r".*depth_cam"
          ),
          "operation": "add",
          "ranges": {
            0: (-0.01, 0.01),
            1: (-0.01, 0.01),
            2: (-0.01, 0.01),
          },
        },
      ),
      "camera_orientation": EventTermCfg(
        func=dr.cam_quat,
        mode="startup",
        params={
          "asset_cfg": SceneEntityCfg(
            "robot", camera_names=r".*depth_cam"
          ),
          "roll_range": (-0.1, 0.1),
          "pitch_range": (-0.1, 0.1),
          "yaw_range": (-0.1, 0.1),
        },
      ),
      "camera_fovy": EventTermCfg(
        func=dr.cam_fovy,
        mode="startup",
        params={
          "asset_cfg": SceneEntityCfg("robot", camera_names=r".*depth_cam"),
          "operation": "abs",
          "ranges": (56.0, 60.0),
        },
      ),
    })

  rewards = {
    "track_lin_vel_xy": RewardTermCfg(
      func=mdp.track_lin_vel_xy,
      weight=3.0,
      params={"command_name": "base_velocity", "std": 0.5},
    ),
    "track_ang_vel_z": RewardTermCfg(
      func=mdp.track_ang_vel_z,
      weight=1.0,
      params={"command_name": "base_velocity", "std": 0.5},
    ),
    "navigation_heading_error": RewardTermCfg(
      func=mdp.navigation_heading_error,
      weight=-0.25,
      params={"command_name": "base_velocity"},
    ),
    "alive": RewardTermCfg(func=mdp.is_alive, weight=0.15),
    "lin_vel_z": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.0),
    "ang_vel_xy": RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05),
    "joint_vel": RewardTermCfg(
      func=mdp.joint_vel_l2, weight=-1.0e-3, params={"asset_cfg": all_joints}
    ),
    "joint_acc": RewardTermCfg(
      func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": all_joints}
    ),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    "joint_limits": RewardTermCfg(
      func=mdp.joint_pos_limits, weight=-5.0, params={"asset_cfg": all_joints}
    ),
    "energy": RewardTermCfg(func=mdp.energy, weight=-2.0e-5),
    "leg_joint_position": RewardTermCfg(
      func=mdp.joint_deviation_exp,
      weight=0.5,
      params={"asset_cfg": leg_pose, "tolerance": 0.1, "scale": 3.0},
    ),
    "flat_orientation": RewardTermCfg(
      func=mdp.flat_orientation_l2, weight=-10.0
    ),
    "feet_slide": RewardTermCfg(
      func=mdp.feet_slide,
      weight=-0.4,
      params={
        "sensor_name": "feet_ground_contact",
        "feet_cfg": feet,
        "force_threshold": 1.0,
      },
    ),
    "feet_stumble": RewardTermCfg(
      func=mdp.feet_stumble,
      weight=-10.0,
      params={"sensor_name": "feet_ground_contact", "force_ratio": 5.0},
    ),
    "foot_impact_velocity": RewardTermCfg(
      func=mdp.foot_impact_velocity,
      weight=-0.05,
      params={"feet_cfg": feet, "threshold": 0.4},
    ),
    "feet_lateral_distance_limit": RewardTermCfg(
      func=mdp.feet_lateral_distance_limit,
      weight=-1.0,
      params={
        "feet_cfg": feet,
        "minimum": 0.18,
        "maximum": 0.28,
        "scale": 10.0,
      },
    ),
    "feet_hold_placement_exp": RewardTermCfg(func=mdp.FootholdPlacementExp, weight=1.0, params={"feet_cfg": feet, "sensor_name": "foot_forward_scan"}),
    "feet_hold": RewardTermCfg(func=mdp.FootHoldPenalty, weight=-2.0, params={"sensor_name": "foot_hold_scan"}),
    "feet_flat_ori": RewardTermCfg(
      func=mdp.feet_orientation_contact,
      weight=-2.0,
      params={
        "sensor_name": "feet_ground_contact",
        "feet_cfg": feet,
        "contact_force_threshold": 1.0,
      },
    ),
    "dont_wait": RewardTermCfg(
      func=mdp.dont_wait,
      weight=-0.5,
      params={"command_name": "base_velocity"},
    ),
    "undesired_contacts": RewardTermCfg(
      func=mdp.undesired_contact_count,
      weight=-1.0,
      params={"sensor_name": "undesired_contact", "threshold": 1.0},
    ),
    "termination": RewardTermCfg(func=failure_termination, weight=-200.0),
  }

  rewards["feet_gait_contact"] = RewardTermCfg(
    func=mdp.feet_gait_contact_isaacgym, weight=0.5,
    params={"sensor_name": "feet_ground_contact", "period": 1.2,
            "phase_offset": 0.5, "stance_ratio": 0.65,
            "command_name": "base_velocity"},
  )
  rewards["feet_gait_stance_mask"] = RewardTermCfg(
    func=mdp.feet_gait_stance_mask_isaacgym, weight=-2.0,
    params={"feet_cfg": feet, "period": 1.2,
            "phase_offset": 0.5, "stance_ratio": 0.65,
            "fast_command_threshold": 0.3, "fast_command_scale": 0.2,
            "command_name": "base_velocity"},
  )
  rewards["feet_gait_swing_mask"] = RewardTermCfg(
    func=mdp.feet_gait_swing_mask_isaacgym, weight=-2.0,
    params={"sensor_name": "feet_ground_contact", "period": 1.2,
            "phase_offset": 0.5, "stance_ratio": 0.65,
            "fast_command_threshold": 0.3, "fast_command_scale": 0.2,
            "command_name": "base_velocity"},
  )
  rewards["feet_both_air_downstairs"] = RewardTermCfg(
    func=mdp.feet_both_air_downstairs, weight=-4.0,
    params={"sensor_name": "feet_ground_contact",
            "contact_force_threshold": 5.0, "downward_speed_threshold": 0.15,
            "command_x_threshold": 0.2, "command_name": "base_velocity"},
  )
  rewards["feet_roll_downstairs"] = RewardTermCfg(
    func=mdp.feet_roll_downstairs, weight=-8.0,
    params={"feet_cfg": feet, "command_name": "base_velocity",
            "contact_sensor_name": "feet_ground_contact",
            "contact_force_threshold": 1.0,
            "downward_speed_threshold": 0.05,
            "roll_tolerance": 0.08,
            "command_x_threshold": 0.2,
            "terrain_names": ("stairs_down", "stairs_down_high")},
  )

  for name in ("flat_orientation", "leg_joint_position", "lin_vel_z", "ang_vel_xy",
               "feet_gait_contact", "feet_gait_stance_mask", "feet_gait_swing_mask",
               "feet_lateral_distance_limit", "feet_hold_placement_exp", "feet_hold",
               "feet_flat_ori", "dont_wait"):
    term = rewards[name]
    rewards[name] = RewardTermCfg(func=mdp.LocomotionReward, weight=term.weight,
                                 params={"reward_fn": term.func, **term.params})

  if has_box:
    rewards.update(
      box_lift_progress=RewardTermCfg(func=box_lift_progress, weight=3.0),
      box_mount_bonus=RewardTermCfg(func=box_mount_bonus, weight=20.0),
      box_exit_bonus=RewardTermCfg(func=box_exit_bonus, weight=30.0),
    )

  sensors = (
    PeriodicCameraSensorCfg(
      name="depth_camera",
      camera_name="robot/depth_cam",
      width=64,
      height=36,
      data_types=("depth",),
      use_textures=False,
      use_shadows=False,
      enabled_geom_groups=(0, 2),
      clone_data=False,
      update_period_steps=5,
    ),
    _terrain_scan(play),
    amp.AmpRayCastCfg(name="amp_scan", frame=ObjRef(type="body", name="pelvis", entity="robot"),
                      ray_alignment="yaw", pattern=amp.AmpGridPattern(),
                      max_distance=amp.HEIGHT_RAY_RANGE, include_geom_groups=(0,)),
    *(ContactSensorCfg(name=f"contact_{part}",
                       primary=ContactMatch(mode="body", pattern=body, entity="robot"),
                       secondary=ContactMatch(mode="body", pattern="terrain"),
                       fields=("found", "force"), reduce="maxforce")
      for part, body in amp.CONTACT_BODIES.items()),
    _foot_edge_scan(play),
    _vdwl_foot_scan("left_foot_forward_scan", "left_ankle_roll_link", play),
    _vdwl_foot_scan("right_foot_forward_scan", "right_ankle_roll_link", play),
    _vdwl_hold_scan("left_foot_hold_scan", "left_ankle_roll_link", play),
    _vdwl_hold_scan("right_foot_hold_scan", "right_ankle_roll_link", play),
    ContactSensorCfg(
      name="feet_ground_contact",
      primary=ContactMatch(
        mode="subtree",
        pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
        entity="robot",
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
      history_length=4,
      track_air_time=True,
    ),
    ContactSensorCfg(
      name="undesired_contact",
      primary=ContactMatch(
        mode="body",
        pattern=(
          r"^(pelvis|torso_link|waist_yaw_link|left_knee_link|right_knee_link|"
          r"left_shoulder_roll_link|right_shoulder_roll_link|"
          r"left_elbow_link|right_elbow_link)$"
        ),
        entity="robot",
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
      history_length=4,
    ),
  )
  if has_box:
    sensors += (ContactSensorCfg(
      name="box_foot_contacts",
      primary=ContactMatch(
        mode="subtree",
        pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
        entity="robot",
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force", "pos", "normal"),
      reduce="maxforce",
      num_slots=4,
    ),)
  return ManagerBasedRlEnvCfg(
    seed=seed,
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=terrain_generator,
        max_init_terrain_level=0 if play else 5,
        debug_vis=play,
      ),
      entities={"robot": baseline_robot_cfg()},
      sensors=sensors,
      num_envs=len(terrain_generator.sub_terrains) if play else 4096,
      extent=2.0,
    ),
    observations=observations,
    metrics={"actuation": MetricsTermCfg(func=amp.capture_torque, per_substep=True)},
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations={
      "time_out": TerminationTermCfg(
        func=time_out_with_rough_boundary,
        time_out=True,
        params={
          "command_name": "base_velocity",
          "rough_episode_s": 20.0,
          "edge_margin": 0.35,
        },
      ),
      "illegal_contact": TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": "undesired_contact", "threshold": 1.0},
      ),
      "gap_base_height": TerminationTermCfg(
        func=mdp.gap_base_height,
        params={
          "minimum_height": 0.5,
          "gap_terrain_name": "square_gaps",
        },
      ),
      "bad_orientation": TerminationTermCfg(
        func=mdp.traversal_orientation,
        params={"limit_angle": 1.0},
      ),
      "unstable_velocity": TerminationTermCfg(
        func=mdp.unstable_velocity,
        params={"asset_cfg": all_joints},
      ),
      "nan": TerminationTermCfg(func=mdp.nan_detection),
      **({
        "box_bypass": TerminationTermCfg(
          func=BoxTraversal,
          params={
            "command_name": "base_velocity",
            "sensor_name": "box_foot_contacts",
            "contact_force_threshold": 15.0,
            "mount_seconds": 0.10,
            "landing_seconds": 0.30,
            "bypass_seconds": 0.10,
          },
        ),
        "box_success": TerminationTermCfg(func=box_task_succeeded),
      } if has_box else {}),
      **({
        "target_reached": TerminationTermCfg(
          func=mdp.target_reached,
          params={"command_name": "base_velocity"},
        )}if play else {}),
    },
    curriculum={
      "terrain_tracking_progress": CurriculumTermCfg(
        func=mdp.terrain_tracking_progress,
        params={
          "command_name": "base_velocity",
          "linear_thresholds": (0.0, 0.0) if play else (0.30, 0.60),
          "yaw_thresholds": (0.0, 0.0) if play else (0.30, 0.50),
          "normalize_by_elapsed": play,
          "require_target_reached": play,
        },
      )
    },
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.WORLD,
      entity_name=None,
      body_name=None,
      lookat=(0.0, 0.75, 0.0),
      distance=4.12,
      elevation=-14.0,
      azimuth=180.0,
    ),
    sim=SimulationCfg(
      njmax=800,
      nconmax=96,
      contact_sensor_maxmatch=128,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    decimation=4,
    episode_length_s=int(1e9) if play else 20.0,
  )
