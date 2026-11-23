"""Load export-compatible weighted motion pools for AMP."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from prcf.asset.robot.e1.e1_constants import E1_JOINT_NAMES
from ..mdp.amp import (
    AMP_DIM, AMP_FIELDS, CONTACT_BODIES, CONTACT_FORCE_THRESHOLD,
    HEIGHT_GRID_X, HEIGHT_GRID_Y, HEIGHT_MISS_VALUE, HEIGHT_RAY_OFFSET, HEIGHT_RAY_RANGE, TRACKING_BODIES,
)
from ..terrains import HEADS

DEFAULT_POOLS = Path(__file__).resolve().parents[1] / "reference_pools.json"
WINDOW = 4


@dataclass
class ReferenceClip:
    motion: torch.Tensor
    weight: float


class ReferencePools:
    def __init__(self, manifest, device, *, heads=HEADS):
        path = Path(manifest)
        config = json.loads(path.read_text())
        directory = Path(config["motion_dir"])
        if not directory.is_absolute():
            repo = next((p for p in path.resolve().parents if (p / "pyproject.toml").is_file()),
                        Path.cwd())
            directory = repo / directory
        self.device = device
        self.clips = {}
        self.weights = {}
        self.offsets = torch.arange(WINDOW, device=device)
        unknown_heads = config["pools"].keys() - set(HEADS)
        if unknown_heads:
            raise ValueError(f"Unknown baseline AMP heads in {path}: {', '.join(sorted(unknown_heads))}")
        for head in heads:
            entries = config["pools"].get(head, [])
            if not entries:
                raise ValueError(f"No reference motions for active head {head}")
            if entries:
                clips = [self._read(directory / entry["file"], entry) for entry in entries]
                self.clips[head] = clips
                self.weights[head] = torch.tensor([clip.weight for clip in clips], device=device)

    def _read(self, path, entry):
        with np.load(path, allow_pickle=False) as data:
            motion = data["amp_obs"]
            layout = (tuple(data["feature_names"]), tuple(data["joint_names"]), tuple(data["body_names"]),
                      tuple(data["contact_parts"]))
            expected = (AMP_FIELDS, E1_JOINT_NAMES, TRACKING_BODIES[1:], tuple(CONTACT_BODIES))
            if motion.ndim != 2 or motion.shape[1] != AMP_DIM or layout != expected or data["fps"].item() != 50:
                raise ValueError(f"{path} does not match the baseline 50 Hz, 357-dimensional AMP layout")
            scan = (tuple(data["height_grid_x"]), tuple(data["height_grid_y"]), data["height_ray_offset"].item(),
                    data["height_ray_range"].item(), data["height_miss_value"].item())
            if scan != (HEIGHT_GRID_X, HEIGHT_GRID_Y, HEIGHT_RAY_OFFSET, HEIGHT_RAY_RANGE, HEIGHT_MISS_VALUE):
                raise ValueError(f"Height scan does not match baseline in {path}")
            if data["contact_force_threshold"].item() != CONTACT_FORCE_THRESHOLD:
                raise ValueError(f"Contact force threshold does not match baseline in {path}")
            start, stop = entry.get("frame_range", (0, len(motion)))
            if not 0 <= start <= stop - WINDOW or stop > len(motion):
                raise ValueError(f"Invalid four-frame range [{start}, {stop}) in {path}")
            motion = motion[start:stop]
            if not np.isfinite(motion).all() or not np.all(np.diff(data["reference_frame"][start:stop]) == 1):
                raise ValueError(f"Invalid or nonconsecutive motion in {path}")
        weight = float(entry.get("weight", 1.0))
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError(f"Invalid reference weight in {path}")
        return ReferenceClip(torch.as_tensor(motion, device=self.device, dtype=torch.float32), weight)

    def sample(self, head, count):
        return self._sample([clip.motion for clip in self.clips[head]], self.weights[head], count)

    def _sample(self, datasets, weights, count):
        choices = torch.multinomial(weights, count, replacement=True)
        shape = (count, WINDOW, AMP_DIM)
        batch = torch.empty(shape, device=self.device)
        for index, data in enumerate(datasets):
            ids = (choices == index).nonzero(as_tuple=True)[0]
            length = len(data) - WINDOW + 1
            starts = torch.randint(length, (len(ids),), device=self.device)
            batch[ids] = data[starts[:, None] + self.offsets]
        return batch
