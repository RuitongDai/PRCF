"""Shared terrain-indexed AMP discriminator, equations 3–6 of TIMP."""
import math

import torch
from torch import nn
from rsl_rl.modules import EmpiricalNormalization

from ..mdp.observations import AMP_DIM, SCAN_DIM
from ..terrains import HEADS
from .motion_loader import WINDOW


class TerrainDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalizer = EmpiricalNormalization(AMP_DIM)
        self.scan_encoder = nn.Sequential(
            nn.Linear(SCAN_DIM, 64), nn.LeakyReLU(), nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, 16),
        )
        self.trunk = nn.Sequential(
            nn.Linear(WINDOW * (AMP_DIM - SCAN_DIM + 16), 512), nn.LeakyReLU(),
            nn.Linear(512, 256), nn.LeakyReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(256, 1) for _ in HEADS])

    def normalized(self, windows):
        return self.normalizer(windows).clamp(-10, 10)

    def forward(self, normalized, head):
        motion = normalized[..., :-SCAN_DIM]
        scan = self.scan_encoder(normalized[..., -SCAN_DIM:])
        return self.heads[head](self.trunk(torch.cat((motion, scan), dim=-1).flatten(1))).squeeze(-1)

    @torch.no_grad()
    def reward(self, windows, head):
        value = self(self.normalized(windows), head)
        return (0.3 * value).tanh().exp() - math.exp(-1)

    def loss(self, expert, policy, head):
        expert = self.normalized(expert).detach().requires_grad_()
        policy = self.normalized(policy).detach().requires_grad_()
        logit_e, logit_p = self(expert, head), self(policy, head)
        adversarial = 0.5 * ((0.3 * logit_p).tanh().mean() - (0.3 * logit_e).tanh().mean())
        grad_e, grad_p = torch.autograd.grad(
            logit_e.sum() + logit_p.sum(), (expert, policy), create_graph=True,
        )
        # kappa / 2 = 1 and p = 6. Squared norm cubed avoids a singular norm derivative.
        penalty = grad_e.flatten(1).square().sum(-1).pow(3).mean()
        penalty += grad_p.flatten(1).square().sum(-1).pow(3).mean()
        return adversarial + penalty
