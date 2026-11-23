"""RSL-RL PPO with VDWL auxiliary learning and terrain-indexed AMP."""
import torch
from rsl_rl.algorithms import PPO

from ..mdp.amp import AMP_DIM
from ..terrains import HEADS
from .discriminator import TerrainDiscriminator
from .motion_loader import ReferencePools, WINDOW


class WindowReplay:
    def __init__(self, capacity, device):
        self.buffer = torch.empty((capacity, WINDOW, AMP_DIM), device=device)
        self.count = 0
        self.cursor = 0

    def append(self, windows):
        windows = windows[-len(self.buffer):]
        slots = (torch.arange(len(windows), device=windows.device) + self.cursor) % len(self.buffer)
        self.buffer[slots] = windows
        self.cursor = (self.cursor + len(windows)) % len(self.buffer)
        self.count = min(self.count + len(windows), len(self.buffer))

    def sample(self, count):
        return self.buffer[torch.randint(self.count, (count,), device=self.buffer.device)]


class BaselinePPO(PPO):
    def __init__(self, *args, amp, **kwargs):
        super().__init__(*args, **kwargs)
        self.pools = None
        self.vdwl_obs = None
        self.vdwl_targets = None
        if amp['inference']:
            return
        if self.is_multi_gpu:
            raise ValueError('Baseline AMP currently supports single-process training.')
        self.pools = ReferencePools(amp['manifest'], self.device, heads=amp['heads'])
        self.discriminator = TerrainDiscriminator().to(self.device)
        self.discriminator_optimizer = torch.optim.Adam([
            {'params': self.discriminator.trunk.parameters(), 'weight_decay': 1e-3},
            {'params': self.discriminator.heads.parameters(), 'weight_decay': 0.1},
        ], lr=self.learning_rate)
        self.batch_size = amp['batch_size']
        self.replay = {HEADS.index(head): WindowReplay(amp['replay_capacity'] // len(self.pools.clips), self.device)
                       for head in self.pools.clips}
        self.history = torch.zeros(self.storage.num_envs, WINDOW, AMP_DIM, device=self.device)
        self.length = torch.zeros(self.storage.num_envs, dtype=torch.long, device=self.device)
        self.last_head = torch.full_like(self.length, -1)
        self.amp_statistics = torch.zeros(3, device=self.device)

    def act(self, obs):
        if self.pools is not None:
            self.vdwl_obs = {key: obs[key].detach().clone() for key in ('actor', 'depth')}
            self.vdwl_targets = obs['pie_targets'].detach().clone()
        return super().act(obs)

    def process_env_step(self, obs, rewards, dones, extras):
        done = dones.reshape(-1).bool()
        heads = obs['amp_route'].reshape(-1).long()
        self.history[:, :-1] = self.history[:, 1:].clone()
        self.history[:, -1] = obs['amp']
        self.length = torch.where(done, 0, torch.where(heads == self.last_head, self.length + 1, 1)).clamp(max=WINDOW)
        self.last_head = torch.where(done, -1, heads)
        style = torch.zeros_like(rewards)
        ready = self.length == WINDOW
        for head, replay in self.replay.items():
            ids = ready & (heads == head)
            if ids.any():
                windows = self.history[ids]
                replay.append(windows)
                style[ids] = self.discriminator.reward(windows, head)
        task = torch.nan_to_num(rewards.clone(), nan=0., posinf=10., neginf=-10.).clamp(-10., 10.)
        rewards.copy_(0.9 * task + 0.1 * style)
        self.amp_statistics += torch.stack((style.mean(), ready.float().mean(), rewards.mean()))
        super().process_env_step(obs, rewards, dones, extras)

    def update(self):
        losses = super().update()
        totals = []
        updates = self.num_learning_epochs * self.num_mini_batches
        for _ in range(updates):
            batches = [(head, self.pools.sample(HEADS[head], self.batch_size), replay.sample(self.batch_size))
                       for head, replay in self.replay.items() if replay.count]
            if not batches:
                break
            with torch.no_grad():
                self.discriminator.normalizer.update(torch.cat([x.flatten(0, 1) for _, e, p in batches for x in (e, p)]))
            loss = torch.stack([self.discriminator.loss(e, p, head) for head, e, p in batches]).mean()
            self.discriminator_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.discriminator_optimizer.step()
            totals.append(loss.detach())
        if totals:
            losses['amp_discriminator'] = torch.stack(totals).mean().item()
        if self.vdwl_targets is not None:
            observations = {name: value.clone() for name, value in self.vdwl_obs.items()}
            targets = self.vdwl_targets.clone()
            self.actor(observations, stochastic_output=False)
            estimates = self.actor.last_estimates
            current = self.actor.extract_current_proprio(observations['actor'])
            parts = {
                'vdwl_velocity': torch.nn.functional.mse_loss(estimates['velocity'], targets[:, :3]),
                'vdwl_height_map': torch.nn.functional.mse_loss(estimates['height_map'], targets[:, 3:]),
                'vdwl_observation': torch.nn.functional.mse_loss(estimates['observation'], current),
            }
            self.optimizer.zero_grad()
            sum(parts.values()).backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizer.step()
            losses.update({name: value.item() for name, value in parts.items()})
        with torch.no_grad():
            self.actor.distribution.std_param.clamp_(min=0.05)
        style, ready, total = (self.amp_statistics / self.storage.num_transitions_per_env).tolist()
        losses.update(amp_style=style, amp_window_ready=ready, combined_reward=total)
        self.amp_statistics.zero_()
        return losses

    def save(self):
        state = super().save()
        if self.pools is not None:
            state['baseline_amp'] = {
                'heads': HEADS,
                'discriminator': self.discriminator.state_dict(),
                'optimizer': self.discriminator_optimizer.state_dict(),
            }
        return state

    def load(self, loaded_dict, load_cfg, strict):
        resume = self.pools is not None and load_cfg is None
        if resume:
            amp = loaded_dict.get('baseline_amp', {})
            if tuple(amp.get('heads', ())) != HEADS:
                raise ValueError('Resume requires a matching baseline three-head AMP checkpoint.')
        result = super().load(loaded_dict, load_cfg, strict)
        if resume:
            self.discriminator.load_state_dict(amp['discriminator'], strict=strict)
            self.discriminator_optimizer.load_state_dict(amp['optimizer'])
        return result
