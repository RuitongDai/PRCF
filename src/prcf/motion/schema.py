"""Versioned, local-only motion/scene bundles. All paths resolve from clip.json."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

MOTION_FIELDS = (
    "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
    "body_lin_vel_w", "body_ang_vel_w",
)
CONTACT_PARTS = ("left_foot", "right_foot", "left_hand", "right_hand", "left_knee", "right_knee")


@dataclass
class ContactRequirement:
    part: str
    start_frame: int
    end_frame: int  # inclusive
    min_fraction: float = 0.5

    def __post_init__(self):
        if self.part not in CONTACT_PARTS:
            raise ValueError(f"Unknown contact part: {self.part}")
        if not 0 <= self.start_frame <= self.end_frame or not 0 <= self.min_fraction <= 1:
            raise ValueError("Invalid contact requirement frame range or fraction")


@dataclass
class QualityConfig:
    # Starting thresholds, to be chosen and reported per motion, not universal guarantees.
    max_root_error: float = 0.25
    max_body_error: float = 0.25
    max_joint_rmse: float = 0.35
    max_root_orientation_error: float = 0.8
    final_root_error: float = 0.15
    max_environment_penetration: float = 0.015
    max_self_penetration: float = 0.01
    max_contact_slip_speed: float = 0.5
    contact_force_threshold: float = 5.0
    joint_limit_tolerance: float = 0.02
    contact_requirements: list[ContactRequirement] = field(default_factory=list)

    def __post_init__(self):
        self.contact_requirements = [
            ContactRequirement(**x) if isinstance(x, dict) else x
            for x in self.contact_requirements
        ]
        for key, value in asdict(self).items():
            if key != "contact_requirements" and (not np.isfinite(value) or value < 0):
                raise ValueError(f"Quality threshold {key} must be finite and nonnegative")


@dataclass
class Clip:
    clip_id: str
    motion_file: Path
    scene: dict[str, Any] = field(default_factory=lambda: {"ground_height": 0.0, "obstacles": []})
    quality: QualityConfig = field(default_factory=QualityConfig)
    motion_to_scene: dict[str, Any] = field(default_factory=lambda: {
        "position": [0.0, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    })
    manifest_file: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def load_clip(path: str | Path) -> Clip:
    path = Path(path).expanduser().resolve()
    if path.suffix == ".npz":
        return Clip(clip_id=path.stem, motion_file=path)
    with path.open() as stream:
        data = json.load(stream)
    allowed = {"schema_version", "clip_id", "robot_id", "motion_file", "fps", "coordinates",
               "scene", "quality", "motion_to_scene", "metadata"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown clip fields: {sorted(unknown)}")
    if data.get("schema_version") != 1:
        raise ValueError("clip.json requires schema_version: 1")
    if data.get("robot_id", "e1_25dof") != "e1_25dof":
        raise ValueError("This task currently supports robot_id=e1_25dof only")
    coordinates = data.get("coordinates", {})
    for key, expected in {"length_unit": "m", "up_axis": "z", "quaternion_order": "wxyz"}.items():
        if coordinates.get(key, expected) != expected:
            raise ValueError(f"Unsupported coordinate convention: {key}")
    motion = (path.parent / data["motion_file"]).resolve()
    metadata = dict(data.get("metadata", {}))
    if "fps" in data:
        metadata["declared_fps"] = data["fps"]
    clip = Clip(
        clip_id=data["clip_id"], motion_file=motion,
        scene=data.get("scene", {"ground_height": 0.0, "obstacles": []}),
        quality=QualityConfig(**data.get("quality", {})),
        manifest_file=path, metadata=metadata,
    )
    clip.motion_to_scene = data.get("motion_to_scene", clip.motion_to_scene)
    return clip
