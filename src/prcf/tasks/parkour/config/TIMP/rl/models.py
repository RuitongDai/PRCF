"""Four-expert actor/critic and a separately optimized HIM estimator."""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP

PROPRIO_DIM = 84
HISTORY = 5
DEPTH_LATENT = 32
HIM_LATENT = 16


class GatedExperts(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, activation):
        super().__init__()
        self.experts = nn.ModuleList([
            MLP(input_dim, output_dim, hidden_dims, activation) for _ in range(4)
        ])
        self.gate = nn.Sequential(nn.Linear(input_dim, 128), nn.ELU(), nn.Linear(128, 4))

    def forward(self, value):
        weights = self.gate(value).softmax(dim=-1)
        predictions = torch.stack([expert(value) for expert in self.experts], dim=-1)
        return (predictions * weights.unsqueeze(-2)).sum(dim=-1)


class HistoryEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MLP(PROPRIO_DIM * HISTORY, 3 + HIM_LATENT, (512, 256, 128), "elu")

    def forward(self, history):
        estimate = self.encoder(history.flatten(1))
        return estimate[:, :3], F.normalize(estimate[:, 3:], dim=-1)


class ActorFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.him = HistoryEstimator()
        self.depth = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2), nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.Flatten(), nn.Linear(32 * 6 * 6, DEPTH_LATENT), nn.ELU(),
        )

    def forward(self, history, depth):
        with torch.no_grad():
            velocity, response = self.him(history)
        visual = self.depth((depth - 0.3) / 3.7)
        return torch.cat((history[:, -1], visual, velocity, response), dim=-1)


class TimpActor(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims, activation, **kwargs):
        super().__init__(obs, obs_groups, obs_set, output_dim, (1,), activation, **kwargs)
        self.features = ActorFeatures()
        self.mlp = GatedExperts(self._get_latent_dim(), self.distribution.input_dim, hidden_dims, activation)

    def _get_obs_dim(self, obs, obs_groups, obs_set):
        return obs_groups[obs_set], PROPRIO_DIM

    def _get_latent_dim(self):
        return PROPRIO_DIM + DEPTH_LATENT + 3 + HIM_LATENT

    def get_latent(self, obs, masks=None, hidden_state=None):
        return self.features(obs["actor"], obs["depth"])

    def as_jit(self):
        return TimpInference(self)

    def as_onnx(self, verbose=False):
        return TimpInference(self)


class TimpCritic(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims, activation, **kwargs):
        super().__init__(obs, obs_groups, obs_set, output_dim, (1,), activation, **kwargs)
        self.mlp = GatedExperts(self.obs_dim, output_dim, hidden_dims, activation)

    def _get_obs_dim(self, obs, obs_groups, obs_set):
        return obs_groups[obs_set], PROPRIO_DIM + obs["critic"].shape[-1]

    def get_latent(self, obs, masks=None, hidden_state=None):
        return torch.cat((obs["actor"][:, -1], obs["critic"]), dim=-1)


class TimpInference(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = deepcopy(model.features)
        self.experts = deepcopy(model.mlp)
        self.output = model.distribution.as_deterministic_output_module()

    def forward(self, proprio_history, depth):
        return self.output(self.experts(self.features(proprio_history, depth)))

    @torch.jit.export
    def reset(self):
        pass

    @property
    def input_names(self):
        return ["proprio_history", "depth"]

    @property
    def output_names(self):
        return ["actions"]

    def get_dummy_inputs(self):
        return torch.zeros(1, HISTORY, PROPRIO_DIM), torch.ones(1, 1, 24, 24)


class HimObjective(nn.Module):
    """Velocity regression plus swapped prototype prediction of the next observation."""

    def __init__(self):
        super().__init__()
        self.target = MLP(PROPRIO_DIM, HIM_LATENT, (128, 64), "elu")
        self.prototypes = nn.Parameter(torch.randn(32, HIM_LATENT))

    @staticmethod
    @torch.no_grad()
    def assignments(logits):
        balanced = torch.softmax(logits / 0.05, dim=-1).T.clamp_min(1e-12)
        for _ in range(3):
            balanced = balanced / balanced.sum(dim=1, keepdim=True)
            balanced = balanced / balanced.sum(dim=0, keepdim=True)
        return balanced.T

    def forward(self, estimator, history, successor, velocity):
        estimated_velocity, source = estimator(history)
        target = F.normalize(self.target(successor), dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        p_src, p_tgt = source @ prototypes.T, target @ prototypes.T
        q_src, q_tgt = self.assignments(p_src), self.assignments(p_tgt)
        swap = -0.5 * ((q_src * F.log_softmax(p_tgt / 0.1, dim=-1)).sum(-1)
                       + (q_tgt * F.log_softmax(p_src / 0.1, dim=-1)).sum(-1)).mean()
        return F.mse_loss(estimated_velocity, velocity), swap
