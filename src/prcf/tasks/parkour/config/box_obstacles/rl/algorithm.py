"""RSL-RL PPO with a single AMP replay buffer and discriminator."""

import torch
from rsl_rl.algorithms import PPO

from ...baseline.mdp.amp import AMP_DIM
from ...baseline.rl.algorithm import WindowReplay
from .discriminator import BoxDiscriminator
from .motion_loader import BoxReferencePool, WINDOW


class BoxAmpPPO(PPO):
    def __init__(self, *args, amp, **kwargs):
        super().__init__(*args, **kwargs)
        self.pools = None
        if amp["inference"]:
            return
        if self.is_multi_gpu:
            raise ValueError("Box AMP currently supports single-process training.")
        self.pools = BoxReferencePool(amp["manifest"], self.device)
        self.discriminator = BoxDiscriminator().to(self.device)
        self.discriminator_optimizer = torch.optim.Adam([
            {"params": self.discriminator.trunk.parameters(), "weight_decay": 1e-3},
            {"params": self.discriminator.output.parameters(), "weight_decay": 0.1},
        ], lr=self.learning_rate)
        self.batch_size = amp["batch_size"]
        self.style_weight = amp["style_weight"]
        self.replay = WindowReplay(amp["replay_capacity"], self.device)
        self.history = torch.zeros(self.storage.num_envs, WINDOW, AMP_DIM, device=self.device)
        self.length = torch.zeros(self.storage.num_envs, dtype=torch.long, device=self.device)
        self.amp_statistics = torch.zeros(3, device=self.device)

    def process_env_step(self, obs, rewards, dones, extras):
        done = dones.reshape(-1).bool()
        self.history[:, :-1] = self.history[:, 1:].clone()
        self.history[:, -1] = obs["amp"]
        # The observation of a done environment already belongs to its next reset.
        self.length.copy_(torch.where(done, 0, (self.length + 1).clamp_max(WINDOW)))
        ready = self.length == WINDOW
        windows = self.history[ready]
        style = torch.zeros_like(rewards)
        if len(windows):
            self.replay.append(windows)
            style[ready] = self.discriminator.reward(windows)
        task = torch.nan_to_num(rewards, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        mixed_reward = (1.0 - self.style_weight) * task + self.style_weight * style
        self.amp_statistics += torch.stack((style.mean(), ready.float().mean(), mixed_reward.mean()))
        super().process_env_step(obs, mixed_reward, dones, extras)

    def update(self):
        losses = super().update()
        if self.replay.count:
            logs = []
            for _ in range(self.num_learning_epochs * self.num_mini_batches):
                expert = self.pools.sample(self.batch_size)
                policy = self.replay.sample(self.batch_size)
                with torch.no_grad():
                    features = torch.cat((expert.flatten(0, 1), policy.flatten(0, 1)))
                    self.discriminator.normalizer.update(features)
                loss = self.discriminator.loss(expert, policy)
                self.discriminator_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
                self.discriminator_optimizer.step()
                logs.append(loss.detach())
            losses["amp_discriminator"] = torch.stack(logs).mean().item()
        style, ready, total = (self.amp_statistics / self.storage.num_transitions_per_env).tolist()
        losses.update(amp_style=style, amp_window_ready=ready, combined_reward=total)
        self.amp_statistics.zero_()
        return losses

    def save(self):
        state = super().save()
        if self.pools is not None:
            state["box_amp"] = {
                "discriminator": self.discriminator.state_dict(),
                "optimizer": self.discriminator_optimizer.state_dict(),
            }
        return state

    def load(self, loaded_dict, load_cfg, strict):
        resume = self.pools is not None and load_cfg is None
        if resume and "box_amp" not in loaded_dict:
            raise ValueError("Resume requires a checkpoint from the single-box AMP task.")
        result = super().load(loaded_dict, load_cfg, strict)
        if resume:
            amp = loaded_dict["box_amp"]
            self.discriminator.load_state_dict(amp["discriminator"], strict=strict)
            self.discriminator_optimizer.load_state_dict(amp["optimizer"])
        return result
