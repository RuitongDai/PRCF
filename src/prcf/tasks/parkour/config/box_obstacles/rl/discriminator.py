"""Least-squares AMP with a single discriminator output."""
import torch
from torch import nn
from rsl_rl.modules import EmpiricalNormalization

from ...baseline.mdp.amp import AMP_DIM
from .motion_loader import WINDOW


class BoxDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalizer = EmpiricalNormalization(AMP_DIM)
        self.trunk = nn.Sequential(
            nn.Linear(WINDOW * AMP_DIM, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(),
        )
        self.output = nn.Linear(256, 1)

    def forward(self, normalized):
        return self.output(self.trunk(normalized.flatten(1))).squeeze(-1)

    def normalized(self, windows):
        return self.normalizer(windows).clamp(-10.0, 10.0)

    @torch.no_grad()
    def reward(self, windows):
        prediction = self(self.normalized(windows))
        return (1 - 0.25 * (prediction - 1).square()).clamp_min(0)

    def loss(self, expert, policy):
        expert = self.normalized(expert).detach().requires_grad_()
        policy = self.normalized(policy).detach()
        expert_prediction = self(expert)
        policy_prediction = self(policy)
        adversarial = 0.5 * ((expert_prediction - 1).square().mean()
                             + (policy_prediction + 1).square().mean())
        gradient, = torch.autograd.grad(expert_prediction.sum(), expert, create_graph=True)
        penalty = gradient.flatten(1).square().sum(-1).mean()
        return adversarial + 10.0 * penalty
