"""Isaac-Gym VDWL all-MLP visual actor adapted to E1 25-DoF observations."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from rsl_rl.modules.distribution import GaussianDistribution


DEPTH_SHAPE = (2, 18, 32)
PROPRIO_TERM_WIDTHS = (2, 3, 3, 3, 25, 25, 25)
PROPRIO_FRAME_DIM = sum(PROPRIO_TERM_WIDTHS)
LONG_HISTORY_LENGTH = 15
SHORT_HISTORY_LENGTH = 5


class _MLP(nn.Sequential):
  def __init__(self, dims, final_activation=False):
    layers = []
    for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
      layers.append(nn.Linear(in_dim, out_dim))
      if index < len(dims) - 2 or final_activation:
        layers.append(nn.ELU())
    super().__init__(*layers)


class VDWLMLPActorModel(nn.Module):
  """Strict VDWL encoder/decoder topology with 25-DoF-sized proprioception."""

  is_recurrent = False

  def __init__(self, obs, obs_groups, obs_set, output_dim, distribution_cfg=None, **kwargs):
    super().__init__()
    del obs_groups, obs_set, kwargs
    self.history_dim = int(obs["actor"].shape[-1])
    expected_history_dim = PROPRIO_FRAME_DIM * LONG_HISTORY_LENGTH
    if self.history_dim != expected_history_dim:
      raise ValueError(
        f"Expected {LONG_HISTORY_LENGTH}x{PROPRIO_FRAME_DIM}={expected_history_dim} "
        f"term-major proprio values, got {self.history_dim}."
      )
    if tuple(obs["depth"].shape[-3:]) != DEPTH_SHAPE:
      raise ValueError(f"Expected depth shape {DEPTH_SHAPE}, got {tuple(obs['depth'].shape[-3:])}.")

    self.proprio_dim = self.history_dim
    self.current_proprio_dim = PROPRIO_FRAME_DIM
    self.short_proprio_dim = PROPRIO_FRAME_DIM * SHORT_HISTORY_LENGTH
    self.height_map_dim = int(obs["critic_height_scan"].shape[-1])

    self.depth_encoder = _MLP([2 * 18 * 32, 512, 256, 128], final_activation=True)
    self.prop_encoder = _MLP([self.proprio_dim, 256, 128, 64])
    self.fusion_encoder = _MLP([192, 256, 128, 51])
    self.obs_decoder = _MLP([16, 64, 128, self.current_proprio_dim])
    self.height_decoder = _MLP([32, 128, 256, self.height_map_dim])

    cfg = copy.deepcopy(distribution_cfg or {})
    cfg.pop("class_name", None)
    self.distribution = GaussianDistribution(output_dim, **cfg)
    self.policy = _MLP([self.short_proprio_dim + 51, 512, 256, 128, self.distribution.input_dim])
    self.distribution.init_mlp_weights(self.policy)
    self.last_estimates = None

  @staticmethod
  def _extract_recent_frames(proprio, frame_count):
    """Convert MJLab term-major history into a recent frame-major vector."""
    chunks = []
    start = 0
    for width in PROPRIO_TERM_WIDTHS:
      block = proprio[..., start:start + width * LONG_HISTORY_LENGTH]
      block = block.reshape(*block.shape[:-1], LONG_HISTORY_LENGTH, width)
      chunks.append(block[..., -frame_count:, :])
      start += width * LONG_HISTORY_LENGTH
    return torch.cat(chunks, dim=-1).flatten(start_dim=-2)

  def extract_short_proprio(self, proprio):
    return self._extract_recent_frames(proprio, SHORT_HISTORY_LENGTH)

  def extract_current_proprio(self, proprio):
    return self._extract_recent_frames(proprio, 1)

  def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
    del masks, hidden_state
    proprio = obs["actor"]
    depth_latent = self.depth_encoder(obs["depth"].flatten(start_dim=1))
    prop_latent = self.prop_encoder(proprio)
    fusion = self.fusion_encoder(torch.cat((prop_latent, depth_latent), dim=-1))
    short_proprio = self.extract_short_proprio(proprio)
    distribution_input = self.policy(torch.cat((short_proprio, fusion), dim=-1))
    self.distribution.update(distribution_input)
    self.last_estimates = {
      "velocity": fusion[:, 0:3],
      "observation": self.obs_decoder(fusion[:, 3:19]),
      "height_map": self.height_decoder(fusion[:, 19:51]),
    }
    if stochastic_output:
      return self.distribution.sample()
    return self.distribution.deterministic_output(distribution_input)

  def reset(self, dones=None, hidden_state=None): del dones, hidden_state
  def get_hidden_state(self): return None
  def detach_hidden_state(self, dones=None): del dones
  def update_normalization(self, obs): del obs
  @property
  def output_entropy(self): return self.distribution.entropy
  @property
  def output_mean(self): return self.distribution.mean
  @property
  def output_std(self): return self.distribution.std
  @property
  def output_distribution_params(self): return self.distribution.params
  def get_output_log_prob(self, outputs): return self.distribution.log_prob(outputs)
  def get_kl_divergence(self, old_params, new_params): return self.distribution.kl_divergence(old_params, new_params)
  def as_onnx(self, verbose=False):
    del verbose
    return _Onnx(self)


class _Onnx(nn.Module):
  def __init__(self, model):
    super().__init__()
    model_for_export = copy.copy(model)
    model_for_export.last_estimates = None
    if hasattr(model_for_export.distribution, "_distribution"):
      model_for_export.distribution = copy.copy(model_for_export.distribution)
      model_for_export.distribution._distribution = None
    self.model = copy.deepcopy(model_for_export).eval()

  def forward(self, proprio, depth):
    return self.model({"actor": proprio, "depth": depth})

  def get_dummy_inputs(self):
    return torch.zeros(1, self.model.history_dim), torch.zeros(1, *DEPTH_SHAPE)

  @property
  def input_names(self): return ["proprio", "depth"]
  @property
  def output_names(self): return ["actions"]
