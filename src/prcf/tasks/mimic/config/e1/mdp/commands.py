"""Advance only on physics steps; never teleport at the end of a reference."""

from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg


def adaptive_bin_probabilities(failures, kernel, uniform_ratio):
    """An explicit uniform mixture, independent of failure-count magnitude."""
    if not 0 <= uniform_ratio <= 1:
        raise ValueError('adaptive_uniform_ratio must be between 0 and 1')
    weights = F.conv1d(
        F.pad(failures.clamp_min(0)[None, None], (0, len(kernel) - 1), mode='replicate'),
        kernel[None, None],
    ).flatten()
    uniform = torch.full_like(weights, 1.0 / len(weights))
    total = weights.sum()
    difficulty = torch.where(total > 0, weights / total.clamp_min(torch.finfo(weights.dtype).tiny), uniform)
    return (1 - uniform_ratio) * difficulty + uniform_ratio * uniform


class FiniteMotionCommand(MotionCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._advance_ids = torch.empty(0, dtype=torch.long, device=self.device)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self.finished[env_ids] = False

    def _adaptive_sampling(self, env_ids):
        failed = self._env.termination_manager.terminated[env_ids]
        bins = torch.clamp(
            (self.time_steps[env_ids] * self.bin_count) // self.motion.time_step_total,
            0, self.bin_count - 1,
        )
        # Counts per parallel world keep EMA values comparable across env counts.
        # Accumulate if the command is reset multiple times before compute().
        counts = torch.bincount(bins[failed], minlength=self.bin_count).to(self.bin_failed_count.dtype)
        self._current_bin_failed.add_(counts / self.num_envs)
        probabilities = adaptive_bin_probabilities(
            self.bin_failed_count, self.kernel, self.cfg.adaptive_uniform_ratio)
        sampled_bins = torch.multinomial(probabilities, len(env_ids), replacement=True)
        fraction = torch.rand(len(env_ids), device=self.device)
        self.time_steps[env_ids] = (
            (sampled_bins + fraction) / self.bin_count * (self.motion.time_step_total - 1)
        ).long()
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        self.metrics['sampling_entropy'][:] = entropy / math.log(self.bin_count) if self.bin_count > 1 else 1.0
        maximum, index = probabilities.max(dim=0)
        self.metrics['sampling_top1_prob'][:] = maximum
        self.metrics['sampling_top1_bin'][:] = index.float() / self.bin_count

    def compute(self, dt, env_ids=None):
        # Reset dt=0 leaves frame zero intact; a vector dt handles auto-reset.
        if isinstance(dt, torch.Tensor):
            self._advance_ids = torch.where(dt > 0)[0]
        elif dt > 0:
            self._advance_ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        else:
            self._advance_ids = torch.empty(0, dtype=torch.long, device=self.device)
        self.update_relative_body_poses()
        super().compute(dt, env_ids)

    def _update_command(self, env_ids=None):
        ids = self._advance_ids
        self.time_steps[ids] = torch.clamp(self.time_steps[ids] + 1, max=self.motion.time_step_total - 1)
        self.finished[ids] = self.time_steps[ids] >= self.motion.time_step_total - 1
        if self._pending_forward:
            self._pending_forward = False
            self._env.sim.forward()
        self.update_relative_body_poses()
        if env_ids is None and self.cfg.sampling_mode == "adaptive":
            self.bin_failed_count = (
                self.cfg.adaptive_alpha * self._current_bin_failed
                + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
            )
            self._current_bin_failed.zero_()


@dataclass(kw_only=True)
class FiniteMotionCommandCfg(MotionCommandCfg):
    def build(self, env):
        return FiniteMotionCommand(self, env)
