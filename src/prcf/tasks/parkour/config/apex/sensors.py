"""Height measurements and contact sensors for the APEX skills."""

from dataclasses import dataclass
from typing import Literal

import torch
import warp as wp
from mjlab.sensor import (
    BuiltinSensorCfg, ContactMatch, ContactSensor, ContactSensorCfg, GridPatternCfg, ObjRef,
    RayCastSensor, RayCastSensorCfg,
)

SCAN_OFFSET = 2.0
SUPPORT_GEOMS = {
    "left_elbow_link": ("left_elbow_yaw_collision", "left_shoulder_yaw_collision"),
    "right_elbow_link": ("right_elbow_yaw_collision", "right_shoulder_yaw_collision"),
    "left_knee_link": ("left_shin_collision", "left_thigh_collision"),
    "right_knee_link": ("right_shin_collision", "right_thigh_collision"),
}


@dataclass
class ApexHeightGrid(GridPatternCfg):
    resolution: float = 0.04

    def generate_rays(self, mj_model, device):
        offsets, directions = super().generate_rays(mj_model, device)
        offsets[:, 2] = SCAN_OFFSET
        return offsets, directions


@dataclass(kw_only=True)
class ApexHeightScanCfg(RayCastSensorCfg):
    apply_drift: bool = True
    drift_mode: Literal["uniform", "split_xy_z"] = "uniform"
    drift_range: tuple[float, float] = (0.0, 0.0)
    drift_range_xy: tuple[float, float] = (0.0, 0.0)
    drift_range_z: tuple[float, float] = (0.0, 0.0)

    def build(self):
        return ApexHeightScan(self)


class ApexHeightScan(RayCastSensor):
    cfg: ApexHeightScanCfg

    def initialize(self, mj_model, model, data, device):
        super().initialize(mj_model, model, data, device)
        self.drift = torch.zeros((data.nworld, 3), device=device)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        if self.cfg.drift_mode == "uniform":
            self.drift[ids] = torch.empty_like(self.drift[ids]).uniform_(*self.cfg.drift_range)
        else:
            xy = torch.empty_like(self.drift[ids, :2]).uniform_(*self.cfg.drift_range_xy)
            z = torch.empty_like(self.drift[ids, 2:]).uniform_(*self.cfg.drift_range_z)
            self.drift[ids] = torch.cat((xy, z), dim=1)

    def prepare_rays(self):
        super().prepare_rays()
        ranges = (self.cfg.drift_range,) if self.cfg.drift_mode == "uniform" else (
            self.cfg.drift_range_xy, self.cfg.drift_range_z,
        )
        if self.cfg.apply_drift and any(bounds != (0.0, 0.0) for bounds in ranges):
            self._cached_world_origins.add_(self.drift[:, None, :])
            self._cached_frame_pos.add_(self.drift[:, None, :])
            wp.to_torch(self._ray_pnt).view_as(self._cached_world_origins).copy_(self._cached_world_origins)


@dataclass
class GroundContactCfg(ContactSensorCfg):
    def build(self):
        return GroundContact(self)


class GroundContact(ContactSensor):
    def _update_air_time_tracking(self, dt):
        # MuJoCo's netforce reduction reports world-frame force. The source
        # GroundSensorZ counts only upward support above 1 N as ground contact.
        supported = self._extract_sensor_data().force[..., 2] > 1.0
        state = self._air_time_state
        landed = supported & (state.current_air_time > 0.0)
        detached = ~supported & (state.current_contact_time > 0.0)
        state.last_air_time.copy_(torch.where(landed, state.current_air_time + dt, state.last_air_time))
        state.last_contact_time.copy_(torch.where(detached, state.current_contact_time + dt, state.last_contact_time))
        state.current_air_time.copy_(torch.where(supported, 0.0, state.current_air_time + dt))
        state.current_contact_time.copy_(torch.where(supported, state.current_contact_time + dt, 0.0))


def apex_sensors(*, descent, play):
    terrain = ContactMatch(mode="body", pattern="terrain")
    bodies = ContactMatch(mode="body", pattern=".*", entity="robot")
    return (
        ApexHeightScanCfg(
            name="height_scan", frame=ObjRef(type="body", name="torso_link", entity="robot"),
            pattern=ApexHeightGrid(), ray_alignment="yaw", max_distance=5.0,
            include_geom_groups=(0,), apply_drift=not (descent or play),
            drift_mode="split_xy_z" if descent else "uniform", debug_vis=not descent,
        ),
        ContactSensorCfg(
            name="feet",
            primary=ContactMatch(mode="body", pattern=r".*_ankle_roll_link", entity="robot"),
            secondary=terrain, fields=("found", "force", "pos", "normal"),
            num_slots=4,
        ),
        ContactSensorCfg(
            name="elbows_knees",
            primary=ContactMatch(
                mode="geom", pattern=tuple(name for pair in SUPPORT_GEOMS.values() for name in pair),
                entity="robot",
            ),
            secondary=terrain, fields=("found", "force", "pos", "normal"),
            num_slots=2,
        ),
        GroundContactCfg(
            name="terrain_forces", primary=bodies, secondary=terrain,
            fields=("found", "force"), reduce="netforce", track_air_time=not descent,
        ),
        ContactSensorCfg(
            name="body_forces", primary=bodies, fields=("force",), reduce="netforce", history_length=3,
        ),
        BuiltinSensorCfg(
            name="torso_lin_acc", sensor_type="framelinacc",
            obj=ObjRef(type="xbody", name="torso_link", entity="robot"),
        ),
        BuiltinSensorCfg(
            name="torso_ang_acc", sensor_type="frameangacc",
            obj=ObjRef(type="xbody", name="torso_link", entity="robot"),
        ),
    )
