"""Load scene manifests and build their MuJoCo geometry for the script entry points."""

from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def _vector(value, length, label):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain {length} finite numbers")
    return vector


def scene_boxes(scene):
    unknown = set(scene) - {"ground_height", "ground_friction", "obstacles"}
    if unknown:
        raise ValueError(f"Unsupported scene fields: {sorted(unknown)}")
    ground = scene.get("ground_height", 0.0)
    if ground is not None and not np.isfinite(ground):
        raise ValueError("ground_height must be finite or null")
    ground_friction = _vector(scene.get("ground_friction", [0.8, 0.005, 0.0001]), 3, "ground_friction")
    if np.any(ground_friction < 0):
        raise ValueError("Friction coefficients cannot be negative")
    boxes, ids = [], set()
    for obstacle in scene.get("obstacles", []):
        allowed = {"id", "type", "size_xyz", "position", "quaternion_wxyz", "fixed", "friction",
                   "step_height", "step_depth", "num_steps", "width"}
        if set(obstacle) - allowed:
            raise ValueError(f"Unsupported obstacle fields: {sorted(set(obstacle) - allowed)}")
        name = obstacle["id"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or name in ids:
            raise ValueError(f"Invalid or repeated obstacle id: {name}")
        ids.add(name)
        if obstacle.get("fixed", True) is not True:
            raise ValueError("Scene v1 supports fixed obstacles only")
        position = _vector(obstacle.get("position", [0, 0, 0]), 3, "position")
        quat = _vector(obstacle.get("quaternion_wxyz", [1, 0, 0, 0]), 4, "quaternion_wxyz")
        if not np.isclose(np.linalg.norm(quat), 1, atol=1e-5):
            raise ValueError("Obstacle quaternion must have unit norm")
        friction = _vector(obstacle.get("friction", [0.8, 0.005, 0.0001]), 3, "friction")
        if np.any(friction < 0):
            raise ValueError("Friction coefficients cannot be negative")
        kind = obstacle["type"]
        if kind == "box":
            size = _vector(obstacle["size_xyz"], 3, "size_xyz")
            if np.any(size <= 0):
                raise ValueError("Box dimensions must be positive")
            boxes.append((name, size, position, quat, friction))
        elif kind == "stairs":
            count = obstacle["num_steps"]
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError("num_steps must be a positive integer")
            height, depth, width = (float(obstacle[k]) for k in ("step_height", "step_depth", "width"))
            if not np.isfinite([height, depth, width]).all() or min(height, depth, width) <= 0:
                raise ValueError("Stair dimensions must be finite and positive")
            rotation = Rotation.from_quat(quat, scalar_first=True)
            if not np.allclose(rotation.apply([0, 0, 1]), [0, 0, 1], atol=1e-5):
                raise ValueError("Stairs support yaw rotation only")
            for i in range(count):
                size = np.array([depth, width, (i + 1) * height])
                center = position + rotation.apply([(i + 0.5) * depth, 0, size[2] / 2])
                boxes.append((f"{name}_step_{i}", size, center, quat, friction))
        else:
            raise ValueError(f"Unsupported obstacle type {kind!r}; v1 supports box/stairs, not arbitrary OBJ meshes")
    return boxes


@dataclass
class SceneBuilder:
    description: dict

    def __post_init__(self):
        self.description = deepcopy(self.description)
        scene_boxes(self.description)

    def __call__(self, spec: mujoco.MjSpec):
        ground = self.description.get("ground_height", 0.0)
        if ground is not None:
            spec.worldbody.add_geom(
                name="prcf_ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
                pos=[0, 0, ground], size=[0, 0, 0.1], contype=1, conaffinity=1,
                condim=3, friction=self.description.get("ground_friction", [0.8, 0.005, 0.0001]),
                # Scene surfaces must be visible with default viewer/camera options;
                # group 3 is the hidden robot collision-debug layer.
                rgba=[0.45, 0.48, 0.52, 1], group=0,
            )
        for name, size, position, quat, friction in scene_boxes(self.description):
            spec.worldbody.add_geom(
                name=f"prcf_{name}", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(size / 2).tolist(), pos=position.tolist(), quat=quat.tolist(),
                contype=1, conaffinity=1, condim=3, friction=friction.tolist(),
                rgba=[0.6, 0.55, 0.4, 1], group=0,
            )
        spec.worldbody.add_light(name="prcf_light", pos=[0, 0, 5], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)


@dataclass
class MotionClip:
    motion_file: Path
    clip_id: str
    scene: dict = field(default_factory=lambda: {'ground_height': 0.0, 'obstacles': []})
    motion_to_scene: dict = field(default_factory=dict)
    manifest_file: Path | None = None


def load_clip(source):
    path = Path(source).expanduser().resolve()
    if path.suffix == '.npz':
        return MotionClip(path, path.stem)
    payload = json.loads(path.read_text())
    return MotionClip(
        (path.parent / payload['motion_file']).resolve(),
        payload.get('clip_id', path.stem),
        scene=payload.get('scene', {'ground_height': 0.0, 'obstacles': []}),
        motion_to_scene=payload.get('motion_to_scene', {}),
        manifest_file=path,
    )


def load_aligned_motion(clip):
    with np.load(clip.motion_file, allow_pickle=False) as archive:
        motion = dict(archive)
    return align_motion(motion, clip.motion_to_scene)


def align_motion(motion, transform):
    if not transform:
        return motion
    shift = transform.get('position', [0, 0, 0])
    quat = transform.get('quaternion_wxyz', [1, 0, 0, 0])
    rotation = Rotation.from_quat(quat, scalar_first=True)
    positions = motion['body_pos_w']
    motion['body_pos_w'] = rotation.apply(positions.reshape(-1, 3)).reshape(positions.shape) + shift
    for key in ('body_lin_vel_w', 'body_ang_vel_w'):
        velocity = motion[key]
        motion[key] = rotation.apply(velocity.reshape(-1, 3)).reshape(velocity.shape)
    orientation = motion['body_quat_w']
    bodies = Rotation.from_quat(orientation.reshape(-1, 4), scalar_first=True)
    motion['body_quat_w'] = (rotation * bodies).as_quat(scalar_first=True).reshape(orientation.shape)
    return motion
