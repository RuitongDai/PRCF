"""Load the exported E1 box clips for AMP and reference resets."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply, yaw_quat

from prcf.asset.robot.e1.e1_constants import E1_JOINT_NAMES
from ...baseline.mdp.amp import (
    AMP_DIM, AMP_FIELDS, CONTACT_BODIES, CONTACT_FORCE_THRESHOLD,
    HEIGHT_GRID_X, HEIGHT_GRID_Y, HEIGHT_MISS_VALUE, HEIGHT_RAY_OFFSET, HEIGHT_RAY_RANGE, TRACKING_BODIES,
)
from ..terrains import BOX, BOX_HEIGHT

DEFAULT_POOLS = Path(__file__).resolve().parents[1] / "reference_pools.json"
WINDOW = 4


@dataclass
class ReferenceClip:
    motion: torch.Tensor
    state: torch.Tensor | None
    weight: float
    reset_ranges: dict[str, slice]
    box: dict | None
    reset_box_edge: str | None


class BoxReferencePool:
    def __init__(self, manifest, device, *, load_states=False):
        path = Path(manifest)
        config = json.loads(path.read_text())
        directory = Path(config["motion_dir"])
        if not directory.is_absolute():
            repo = next((p for p in path.resolve().parents if (p / "pyproject.toml").is_file()),
                        Path.cwd())
            directory = repo / directory
        self.device = device
        self.offsets = torch.arange(WINDOW, device=device)
        unknown_heads = config["pools"].keys() - {"box"}
        if unknown_heads:
            raise ValueError(f"Unknown BoxObstacles AMP heads in {path}: {', '.join(sorted(unknown_heads))}")
        entries = config["pools"].get("box", [])
        self.clips = [self._read(directory / entry["file"], entry, load_states) for entry in entries]
        if [clip.reset_box_edge for clip in self.clips] != ["front", "rear"]:
            raise ValueError("The box pool must contain an ascent clip followed by a descent clip.")
        self.weights = torch.tensor([clip.weight for clip in self.clips], device=device)

    def _read(self, path, entry, load_states):
        reset_box_edge = entry.get("reset_box_edge")
        if reset_box_edge is not None and reset_box_edge not in ("front", "rear"):
            raise ValueError(f"Invalid reset_box_edge for {path}. Use 'front' or 'rear' in the box pool.")
        with np.load(path, allow_pickle=False) as data:
            motion = data["amp_obs"]
            layout = (tuple(data["feature_names"]), tuple(data["joint_names"]), tuple(data["body_names"]),
                      tuple(data["contact_parts"]))
            expected = (AMP_FIELDS, E1_JOINT_NAMES, TRACKING_BODIES[1:], tuple(CONTACT_BODIES))
            if motion.ndim != 2 or motion.shape[1] != AMP_DIM or layout != expected or data["fps"].item() != 50:
                raise ValueError(f"{path} does not match the BoxObstacles 50 Hz, 357-dimensional AMP layout")
            scan = (tuple(data["height_grid_x"]), tuple(data["height_grid_y"]), data["height_ray_offset"].item(),
                    data["height_ray_range"].item(), data["height_miss_value"].item())
            if scan != (HEIGHT_GRID_X, HEIGHT_GRID_Y, HEIGHT_RAY_OFFSET, HEIGHT_RAY_RANGE, HEIGHT_MISS_VALUE):
                raise ValueError(f"Height scan does not match BoxObstacles in {path}")
            if data["contact_force_threshold"].item() != CONTACT_FORCE_THRESHOLD:
                raise ValueError(f"Contact force threshold does not match BoxObstacles in {path}")
            start, stop = entry.get("frame_range", (0, len(motion)))
            if not 0 <= start <= stop - WINDOW or stop > len(motion):
                raise ValueError(f"Invalid four-frame range [{start}, {stop}) in {path}")
            motion = motion[start:stop]
            if not np.isfinite(motion).all() or not np.all(np.diff(data["reference_frame"][start:stop]) == 1):
                raise ValueError(f"Invalid or nonconsecutive motion in {path}")
            state = self._read_state(data, slice(start, stop)) if load_states else None
            box = None
            if load_states and reset_box_edge is not None:
                scene = json.loads(data["scene_json"].item())
                box = next((obstacle for obstacle in scene["obstacles"] if obstacle["type"] == "box"), None)
            reset_ranges = {}
            for terrain, (first, last) in entry.get("reset_ranges", {}).items():
                if not start <= first < last <= stop:
                    raise ValueError(f"Reset range [{first}, {last}) is outside [{start}, {stop}) in {path}")
                reset_ranges[terrain] = slice(first - start, last - start)
        weight = float(entry.get("weight", 1.0))
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError(f"Invalid reference weight in {path}")
        return ReferenceClip(torch.as_tensor(motion, device=self.device, dtype=torch.float32), state, weight,
                             reset_ranges, box, reset_box_edge)

    def _read_state(self, data, rows):
        pos, quat, lin, ang, joint_pos, joint_vel = [
            torch.as_tensor(data[key][rows].copy(), device=self.device, dtype=torch.float32)
            for key in ("root_pos_w", "root_quat_w", "root_lin_vel", "root_ang_vel", "dof_pos", "dof_vel")
        ]
        heading = yaw_quat(quat)
        state = torch.cat((pos, quat, quat_apply(heading, lin), quat_apply(heading, ang), joint_pos, joint_vel), dim=1)
        if not torch.isfinite(state).all():
            raise ValueError("Nonfinite reference reset state")
        return state

    def reset_states(self):
        """Match box height and align the front edge for ascent or rear edge for descent."""
        datasets = []
        for clip in self.clips:
            if clip.box is None or clip.state is None:
                continue
            length, width, ref_height = clip.box["size_xyz"]
            if abs(BOX_HEIGHT - ref_height) > 0.015 or abs(BOX.box_width - width) > 0.01:
                continue
            data = clip.state[clip.reset_ranges.get("box", slice(None))].clone()
            ref_x, ref_y, _ = clip.box["position"]
            if clip.reset_box_edge == "front":
                data[:, 0] += BOX.front_x - (ref_x - length / 2)
            else:
                data[:, 0] += BOX.end_x - (ref_x + length / 2)
            data[:, 1] -= ref_y
            datasets.append(data)
        if len(datasets) != 2:
            raise ValueError("Both box reference scenes must match the fixed box height and width.")
        return datasets

    def sample(self, count):
        choices = torch.multinomial(self.weights, count, replacement=True)
        batch = torch.empty((count, WINDOW, AMP_DIM), device=self.device)
        for index, clip in enumerate(self.clips):
            ids = (choices == index).nonzero(as_tuple=True)[0]
            starts = torch.randint(len(clip.motion) - WINDOW + 1, (len(ids),), device=self.device)
            batch[ids] = clip.motion[starts[:, None] + self.offsets]
        return batch
