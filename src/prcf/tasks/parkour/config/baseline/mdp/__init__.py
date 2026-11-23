"""MDP terms owned by the E1 25-DoF AMP Walk task."""

from mjlab.envs.mdp import *  # noqa: F403

from .commands import ParkourTargetVelocityCommandCfg as ParkourTargetVelocityCommandCfg
from .stage2_commands import Stage2TargetVelocityCommandCfg as Stage2TargetVelocityCommandCfg
from .curricula import terrain_tracking_progress as terrain_tracking_progress
from .observations import DepthHistory as DepthHistory
from .observations import torso_imu_projected_gravity as torso_imu_projected_gravity
from .observations import LegacyGridPatternCfg as LegacyGridPatternCfg
from .observations import foot_contacts as foot_contacts
from .observations import foot_clearance as foot_clearance
from .observations import gait_clock as gait_clock
from .observations import isaacgym_gait_phase as isaacgym_gait_phase
from .rewards import *  # noqa: F403
from .terminations import illegal_contact as illegal_contact
from .terminations import undesired_contact_count as undesired_contact_count
from .terminations import terrain_out_of_bounds as terrain_out_of_bounds
from .terminations import gap_base_height as gap_base_height
from .terminations import target_reached as target_reached
from .terminations import unstable_velocity as unstable_velocity

from .terminations import traversal_orientation as traversal_orientation
