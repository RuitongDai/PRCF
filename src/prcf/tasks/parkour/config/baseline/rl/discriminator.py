"""Least-squares AMP with a shared trunk and three terrain-specific heads."""
import torch
from torch import nn
from rsl_rl.modules import EmpiricalNormalization

from ..mdp.amp import AMP_DIM
from ..terrains import HEADS
from .motion_loader import WINDOW


class TerrainDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalizer = EmpiricalNormalization(AMP_DIM)
        self.trunk = nn.Sequential(
            nn.Linear(WINDOW * AMP_DIM, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(256, 1) for _ in HEADS])

    def forward(self, normalized, head):
        return self.heads[head](self.trunk(normalized.flatten(1))).squeeze(-1)

    def normalized(self, windows):
        return self.normalizer(windows).clamp(-10.0, 10.0)

    @torch.no_grad()
    def reward(self, windows, head):
        prediction = self(self.normalized(windows), head)
        return (1 - 0.25 * (prediction - 1).square()).clamp_min(0)

    def loss(self, expert, policy, head):
        expert = self.normalized(expert).detach().requires_grad_()
        policy = self.normalized(policy).detach()
        expert_prediction = self(expert, head)
        policy_prediction = self(policy, head)
        adversarial = 0.5 * ((expert_prediction - 1).square().mean()
                             + (policy_prediction + 1).square().mean())
        gradient, = torch.autograd.grad(expert_prediction.sum(), expert, create_graph=True)
        penalty = gradient.flatten(1).square().sum(-1).mean()
        return adversarial + 10.0 * penalty
