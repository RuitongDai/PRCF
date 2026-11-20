"""Hand and knee support reward on the rectangular high-box top."""
import itertools
import math

import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

from .observations import CONTACT_BODIES, contact_force
from ..terrains import SingleBoxTerrainCfg, terrain_configs, terrain_difficulties

RAMP_STEPS = 96_000


def box_support_candidates(box_size, footprint, area_threshold):
    """Score planar footprints over the finite box top, then apply 0.1 m NMS."""
    half = np.array(box_size) / 2
    points = np.array(list(itertools.product(
        np.arange(-half[0], half[0] + 1e-6, 0.025),
        np.arange(-half[1], half[1] + 1e-6, 0.025),
    )))
    offsets = np.array(list(itertools.product(
        np.linspace(-footprint[0] / 2, footprint[0] / 2, 5),
        np.linspace(-footprint[1] / 2, footprint[1] / 2, 5),
    )))
    coverage = np.zeros(len(points))
    for yaw in (-math.pi / 6, 0.0, math.pi / 6):
        c, s = math.cos(yaw), math.sin(yaw)
        rotated = offsets @ np.array(((c, s), (-s, c)))
        inside = (np.abs(points[:, None] + rotated) <= half + 1e-9).all(axis=-1)
        coverage = np.maximum(coverage, inside.mean(axis=-1))
    edge = np.min(half - np.abs(points), axis=-1)
    scores = coverage * np.clip(edge / 0.11, 0, 1)
    valid = coverage >= area_threshold
    chosen = []
    for index in np.argsort(-scores, kind="stable"):
        if valid[index] and all(np.linalg.norm(points[index] - points[i]) >= 0.10 for i in chosen):
            chosen.append(index)
            if len(chosen) == 32:
                break
    centers = np.column_stack((points[chosen], np.zeros(len(chosen))))
    return centers, scores[chosen]


class SupportReward:
    """Privileged attraction before contact, placement quality, and retention."""

    def __init__(self, cfg, env):
        self.parts = ("left_hand", "right_hand", "left_knee", "right_knee")
        anchors = ("left_shoulder_roll_link", "right_shoulder_roll_link",
                   "left_hip_roll_link", "right_hip_roll_link")
        bodies = env.scene["robot"].body_names
        self.anchors = [bodies.index(name) for name in anchors]
        self.endpoints = [bodies.index(CONTACT_BODIES[part]) for part in self.parts]
        self.feet = [bodies.index(CONTACT_BODIES[part]) for part in ("left_foot", "right_foot")]
        difficulties = terrain_difficulties(env.scene.terrain.cfg.terrain_generator)
        self.boxes = {}
        slots = 8
        for column, tile in enumerate(terrain_configs(env).values()):
            if isinstance(tile, SingleBoxTerrainCfg):
                size = (tile.box_length, tile.box_width)
                candidates = (box_support_candidates(size, (0.12, 0.08), 0.85),
                              box_support_candidates(size, (0.10, 0.07), 0.92))
                points = [torch.as_tensor(x, device=env.device, dtype=torch.float32) for x, _ in candidates]
                scores = [torch.as_tensor(s, device=env.device, dtype=torch.float32) for _, s in candidates]
                slots = min(slots, *(len(group) for group in points))
                heights = torch.as_tensor(tile.height_at(difficulties), device=env.device, dtype=torch.float32)
                self.boxes[column] = (tile, points, scores, heights)
        self.centers = torch.tensor(((0.15, 0, -0.15), (0.12, 0, -0.18)), device=env.device)
        self.axes = torch.tensor(((0.48, 0.42, 0.50), (0.38, 0.285, 0.43)), device=env.device)
        self.forces_active = torch.zeros((env.num_envs, 4), dtype=torch.bool, device=env.device)
        self.previous = torch.zeros_like(self.forces_active)
        self.holding = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.released = torch.zeros_like(self.holding)
        self.swing_steps = torch.zeros((env.num_envs, 4), dtype=torch.int32, device=env.device)
        self.candidates = torch.zeros((env.num_envs, 4, slots, 3), device=env.device)
        self.quality = torch.zeros((env.num_envs, 4, slots), device=env.device)
        self.refresh = torch.ones_like(self.holding)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.forces_active[env_ids] = False
        self.previous[env_ids] = False
        self.holding[env_ids] = False
        self.released[env_ids] = False
        self.swing_steps[env_ids] = 0
        self.refresh[env_ids] = True

    def __call__(self, env):
        result = torch.zeros(env.num_envs, device=env.device)
        types = env.scene.terrain.terrain_types
        for column, box in self.boxes.items():
            ids = (types == column).nonzero(as_tuple=True)[0]
            if len(ids):
                result[ids] = self._box_reward(env, ids, *box)
        return result

    def _box_reward(self, env, ids, tile, points_by_limb, scores_by_limb, heights):
        robot = env.scene["robot"].data
        center = env.scene.env_origins[ids].clone()
        center[:, 0] += tile.center_x
        center[:, 2] += heights[env.scene.terrain.terrain_levels[ids]]
        half_x = tile.box_length / 2
        position = robot.body_link_pos_w[ids][:, self.endpoints] - center[:, None]
        velocity = robot.body_link_lin_vel_w[ids][:, self.endpoints]
        base = robot.root_link_pos_w[ids] - center
        # Use box-relative phases for the solid box in this port.
        feet = robot.body_link_pos_w[ids][:, self.feet] - center[:, None]
        landed = torch.stack([contact_force(env, part)[ids] >= 5 for part in ("left_foot", "right_foot")], dim=-1)
        half = feet.new_tensor((half_x, tile.box_width / 2))
        feet_on_top = (feet[..., :2].abs() < half).all(-1)
        feet_on_top &= feet[..., 2].abs() < 0.14
        stable_on_top = (feet_on_top & landed).all(-1) & (robot.projected_gravity_b[ids, 2] < -0.85)
        self.released[ids] |= stable_on_top | (base[:, 0] > half_x + 0.15)
        at_box = (base[:, 0] >= -half_x - 0.60) & ~self.released[ids]
        near_top = (base[:, 0] >= -half_x - 0.15) & (base[:, 2] >= -0.1)
        active = torch.stack([contact_force(env, p)[ids] for p in self.parts], dim=-1)
        self.forces_active[ids] = active >= torch.where(self.forces_active[ids], 3.0, 5.0)
        edge = (half - position[..., :2].abs()).amin(-1)
        valid = self.forces_active[ids] & (edge >= 0.055)
        valid &= position[..., 2].abs() <= 0.08
        hands = valid[:, :2].any(-1)
        valid[:, 2:] &= hands[:, None]
        self.holding[ids] |= near_top & hands
        hold = self.holding[ids] & at_box

        update = self.refresh[ids] | (env.common_step_counter % 5 == 0)
        update_ids = ids[update]
        if len(update_ids):
            quat = yaw_quat(robot.root_link_quat_w[update_ids])
            anchors = robot.body_link_pos_w[update_ids][:, self.anchors] - center[update, None]
            for limb, anchor in enumerate(anchors.unbind(dim=1)):
                limb_type = limb // 2
                points = points_by_limb[limb_type]
                delta = points[None] - anchor[:, None]
                local = quat_apply_inverse(quat[:, None].expand(-1, len(points), -1), delta)
                distance = ((local - self.centers[limb_type]) / self.axes[limb_type]).square().sum(-1)
                score = scores_by_limb[limb_type] * torch.exp(-distance) * (distance <= 1.0)
                values, index = score.topk(self.quality.shape[-1], dim=-1)
                self.candidates[update_ids, limb] = points[index]
                self.quality[update_ids, limb] = values
            self.refresh[update_ids] = False

        quality = self.quality[ids]
        eligible = quality > 0
        weights = torch.softmax((quality / 0.15).masked_fill(~eligible, -1e6), dim=-1) * eligible
        delta = position[:, :, None] + 0.12 * velocity[:, :, None] - self.candidates[ids]
        variance = position.new_tensor((0.12, 0.12, 0.10, 0.10)).square()[None, :, None]
        attraction = (weights * torch.exp(-delta.square().sum(-1) / variance)).sum(-1)
        attraction *= ~valid
        attraction[:, 2:] *= hands[:, None]
        actual = (edge / 0.11).clamp(0.0, 1.0) * valid
        placement = (
            0.15 * attraction[:, :2].mean(-1) + 0.35 * actual[:, :2].mean(-1)
            + 0.10 * attraction[:, 2:].mean(-1) + 0.25 * actual[:, 2:].mean(-1)
        ) * eligible.flatten(1).any(-1)

        contacts = valid.sum(-1)
        previous = self.previous[ids]
        lost = previous & ~valid
        touchdown = ~previous & valid
        swing = torch.where(~valid & hold[:, None], self.swing_steps[ids] + 1, 0)
        self.swing_steps[ids] = swing
        self.previous[ids] = valid
        holding_reward = 0.15 * ((contacts >= 2) & hands) + 0.025 * (contacts - 2).clamp(min=0)
        holding_penalty = (0.35 * (contacts < 2) + 0.35 * ~hands
                           + 0.10 * lost[:, :2].float().mean(-1) + 0.05 * lost[:, 2:].float().mean(-1)
                           + 0.10 * (swing > position.new_tensor((10, 10, 25, 25))).float().mean(-1)
                           + 0.20 * (lost[:, :2].any(-1) & (contacts < 2))
                           + 0.35 * (previous[:, :2].all(-1) & ~hands)
                           + 0.15 * (touchdown & (velocity[..., 2] < -0.5)).float().mean(-1))
        elapsed = env.common_step_counter - getattr(env, "timp_stage_start", 0)
        alpha = 0.1 + 0.9 * min(max(elapsed, 0) / RAMP_STEPS, 1.0)
        return alpha * (placement * at_box + (holding_reward - holding_penalty) * hold)
