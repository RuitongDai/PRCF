"""Validate mjlab arrays against the actual E1 kinematic model before simulation."""

from __future__ import annotations

import hashlib
from pathlib import Path
import warnings

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .schema import MOTION_FIELDS


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def e1_model():
    from mjlab.entity import Entity
    from prcf.asset.robot.e1.e1_constants import get_e1_robot_cfg
    robot = Entity(get_e1_robot_cfg())
    return robot.spec.compile(), robot.joint_names, robot.body_names


def load_motion(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        missing = set(MOTION_FIELDS + ("fps",)) - set(source.files)
        if missing:
            raise ValueError(f"Motion missing fields: {sorted(missing)}")
        return {key: source[key].copy() for key in source.files}


def transform_motion(data, transform):
    unknown = set(transform) - {"position", "quaternion_wxyz"}
    if unknown:
        raise ValueError(f"Unknown motion_to_scene fields: {sorted(unknown)}")
    position = np.asarray(transform.get("position", [0, 0, 0]), dtype=float)
    quat = np.asarray(transform.get("quaternion_wxyz", [1, 0, 0, 0]), dtype=float)
    if position.shape != (3,) or quat.shape != (4,) or not np.isfinite(position).all() or not np.isfinite(quat).all():
        raise ValueError("Invalid motion_to_scene transform")
    if not np.isclose(np.linalg.norm(quat), 1, atol=1e-5):
        raise ValueError("motion_to_scene quaternion must have unit norm")
    rotation = Rotation.from_quat(quat, scalar_first=True)
    result = {key: value.copy() for key, value in data.items()}
    result["body_pos_w"] = rotation.apply(data["body_pos_w"].reshape(-1, 3)).reshape(data["body_pos_w"].shape) + position
    for key in ("body_lin_vel_w", "body_ang_vel_w"):
        result[key] = rotation.apply(data[key].reshape(-1, 3)).reshape(data[key].shape)
    original = Rotation.from_quat(data["body_quat_w"].reshape(-1, 4), scalar_first=True)
    result["body_quat_w"] = (rotation * original).as_quat(scalar_first=True).reshape(data["body_quat_w"].shape)
    return result


def validate_motion(data, *, model=None, expected_fps=None, check_fk=True):
    if model is None:
        model, joint_names, body_names = e1_model()
    else:
        joint_names = tuple(model.joint(i).name for i in range(model.njnt)
                            if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE)
        body_names = tuple(model.body(i).name for i in range(1, model.nbody))
    fps_values = np.asarray(data["fps"])
    if fps_values.size != 1 or not np.isfinite(fps_values).all() or float(fps_values.item()) <= 0:
        raise ValueError("fps must be one positive finite number")
    fps = float(fps_values.item())
    if expected_fps is not None and not np.isclose(fps, expected_fps, atol=1e-6):
        raise ValueError(f"Motion fps={fps} does not match expected fps={expected_fps}; resample explicitly")
    q = np.asarray(data["joint_pos"])
    if q.ndim != 2 or q.shape[0] < 2:
        raise ValueError("joint_pos must have shape (T, J), T >= 2")
    frames = len(q)
    shapes = {"joint_pos": (frames, len(joint_names)), "joint_vel": (frames, len(joint_names)),
              "body_pos_w": (frames, len(body_names), 3), "body_quat_w": (frames, len(body_names), 4),
              "body_lin_vel_w": (frames, len(body_names), 3), "body_ang_vel_w": (frames, len(body_names), 3)}
    for key, shape in shapes.items():
        array = np.asarray(data[key])
        if array.shape != shape:
            raise ValueError(f"{key}: expected {shape}, got {array.shape}. E1 requires 25 joints and its own body ordering.")
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError(f"{key}: values must be finite numeric data")
    if not np.allclose(np.linalg.norm(data["body_quat_w"], axis=-1), 1, atol=1e-3):
        raise ValueError("body_quat_w must contain unit quaternions in wxyz order")
    named = all(name in data for name in ("joint_names", "body_names"))
    for key, expected in (("joint_names", joint_names), ("body_names", body_names)):
        if key in data and tuple(data[key].astype(str).tolist()) != tuple(expected):
            raise ValueError(f"{key} does not match the current robot model")
    if not named:
        warnings.warn("NPZ lacks joint/body names; checking sampled forward kinematics. Preserve names in future exports.", stacklevel=2)
    limited = model.jnt_limited[1:].astype(bool)
    lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
    if np.any((q[:, limited] < lo[limited] - 0.02) | (q[:, limited] > hi[limited] + 0.02)):
        raise ValueError("Reference joint positions exceed E1 hard joint limits")
    if check_fk:
        state = mujoco.MjData(model)
        for frame in np.unique(np.linspace(0, frames - 1, min(7, frames), dtype=int)):
            state.qpos[:3] = data["body_pos_w"][frame, 0]
            state.qpos[3:7] = data["body_quat_w"][frame, 0]
            state.qpos[7:] = q[frame]
            mujoco.mj_kinematics(model, state)
            if np.max(np.linalg.norm(state.xpos[1:] - data["body_pos_w"][frame], axis=-1)) > 0.002:
                raise ValueError(f"Body positions fail E1 FK consistency at frame {frame}; check model and body/joint ordering")
            dot = np.abs(np.sum(state.xquat[1:] * data["body_quat_w"][frame], axis=-1))
            if np.max(2 * np.arccos(np.clip(dot, 0, 1))) > 0.01:
                raise ValueError(f"Body orientations fail E1 FK consistency at frame {frame}")
    return {"frames": frames, "fps": fps, "duration_s": (frames - 1) / fps,
            "joint_names": list(joint_names), "body_names": list(body_names), "names_present": named}
