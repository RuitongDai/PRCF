"""RSL-RL PPO with routed AMP, replay, and a separate HIM update."""
import torch
from rsl_rl.algorithms import PPO

from ..mdp.observations import AMP_DIM
from ..mdp.support import RAMP_STEPS
from ..terrains import HEADS
from .discriminator import TerrainDiscriminator
from .models import HimObjective
from .motion_loader import ReferencePools, WINDOW


class WindowReplay:
    def __init__(self, capacity, device):
        self.buffer = torch.empty((capacity, WINDOW, AMP_DIM), device=device)
        self.cursor = 0
        self.count = 0

    def append(self, windows):
        windows = windows[-len(self.buffer):]
        count = len(windows)
        slots = (torch.arange(count, device=windows.device) + self.cursor) % len(self.buffer)
        self.buffer[slots] = windows
        self.cursor = (self.cursor + count) % len(self.buffer)
        self.count = min(self.count + count, len(self.buffer))

    def sample(self, count):
        return self.buffer[torch.randint(self.count, (count,), device=self.buffer.device)]


class TimpPPO(PPO):
    def __init__(self, *args, amp, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage = amp["stage"]
        self.stage_step = 0
        self.batch_size = amp["batch_size"]
        self.him_pairs = []
        self.amp_statistics = torch.zeros(3, device=self.device)
        self.him_objective = None
        self.pools = None
        if amp["inference"]:
            return
        if self.is_multi_gpu:
            raise ValueError("TIMP requires a single training process for its AMP and HIM updates")
        self.pools = ReferencePools(amp["manifest"], self.device, heads=amp["heads"])
        self.discriminator = TerrainDiscriminator().to(self.device)
        self.discriminator_optimizer = torch.optim.Adam(
            [
                {"params": list(self.discriminator.scan_encoder.parameters())
                           + list(self.discriminator.trunk.parameters()), "weight_decay": 1e-3},
                {"params": self.discriminator.heads.parameters(), "weight_decay": 0.1},
            ],
            lr=5e-6,
        )
        heads = self.pools.clips
        slots = amp["replay_capacity"] // max(1, len(heads))
        self.replay = {HEADS.index(head): WindowReplay(slots, self.device) for head in heads}
        self.history = torch.zeros((self.storage.num_envs, WINDOW, AMP_DIM), device=self.device)
        self.last_head = torch.full((self.storage.num_envs,), -1, device=self.device, dtype=torch.long)
        self.window_length = torch.zeros_like(self.last_head)
        self.him_objective = HimObjective().to(self.device)
        self.him_optimizer = torch.optim.Adam(
            list(self.actor.features.him.parameters()) + list(self.him_objective.parameters()), lr=1e-3,
        )

    @property
    def amp_weight(self):
        return 1.0 if self.stage == 2 else 1.0 - 0.5 * min(self.stage_step / RAMP_STEPS, 1.0)

    def process_env_step(self, obs, rewards, dones, extras):
        done = dones.reshape(-1).bool()
        heads = obs["amp_route"].reshape(-1).long()
        self.history[:, :-1] = self.history[:, 1:].clone()
        self.history[:, -1] = obs["amp"]
        # The returned observation on a done step is already the new reset state.
        self.window_length = torch.where(done, 0, torch.where(
            heads == self.last_head, self.window_length + 1, 1,
        )).clamp(max=WINDOW)
        self.last_head = torch.where(done, -1, heads)

        style = torch.zeros_like(rewards)
        ready = self.window_length == WINDOW
        scored = torch.zeros_like(ready)
        for head, replay in self.replay.items():
            ids = (ready & (heads == head)).nonzero(as_tuple=True)[0]
            if len(ids):
                windows = self.history[ids]
                replay.append(windows)
                style[ids] = self.discriminator.reward(windows, head)
                scored[ids] = True

        # Capture bounded pairs from a single episode for the auxiliary HIM update.
        ids = (~done).nonzero(as_tuple=True)[0]
        if len(ids):
            ids = ids[torch.randperm(len(ids), device=self.device)[:128]]
            prev_obs = self.transition.observations
            self.him_pairs.append((prev_obs["actor"][ids], obs["actor"][ids, -1], prev_obs["critic"][ids, :3]))

        total = rewards + self.amp_weight * style
        self.amp_statistics[0] += style.mean()
        self.amp_statistics[1] += scored.float().mean()
        self.amp_statistics[2] += total.mean()
        self.stage_step += 1
        super().process_env_step(obs, total, dones, extras)

    def update(self):
        results = super().update()
        batches = []
        for head, replay in self.replay.items():
            if replay.count:
                expert = self.pools.sample(HEADS[head], self.batch_size)
                policy = replay.sample(self.batch_size)
                batches.append((head, expert, policy))
        if batches:
            with torch.no_grad():
                frames = torch.cat([windows.flatten(0, 1) for _, e, p in batches for windows in (e, p)])
                self.discriminator.normalizer.update(frames)
            loss = torch.stack([self.discriminator.loss(e, p, h) for h, e, p in batches]).sum()
            self.discriminator_optimizer.zero_grad()
            loss.backward()
            self.discriminator_optimizer.step()
            results["amp_discriminator"] = loss.item()
        if self.him_pairs:
            history, successor, velocity = [torch.cat(batch) for batch in zip(*self.him_pairs)]
            self.him_pairs.clear()
            velocity_loss, swap_loss = self.him_objective(self.actor.features.him, history, successor, velocity)
            self.him_optimizer.zero_grad()
            (velocity_loss + swap_loss).backward()
            self.him_optimizer.step()
            results["him_velocity"] = velocity_loss.item()
            results["him_swap"] = swap_loss.item()
        style, coverage, total = (self.amp_statistics / self.storage.num_transitions_per_env).tolist()
        results.update(amp_style=style, amp_window_ready=coverage, combined_reward=total, amp_weight=self.amp_weight)
        self.amp_statistics.zero_()
        return results

    def save(self):
        state = super().save()
        if self.pools is not None:
            state["timp"] = {
                "stage": self.stage, "step": self.stage_step,
                "heads": HEADS,
                "discriminator": self.discriminator.state_dict(),
                "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
                "him": self.him_objective.state_dict(), "him_optimizer": self.him_optimizer.state_dict(),
            }
        return state

    def load(self, loaded_dict, load_cfg, strict):
        state = loaded_dict.get("timp")
        resume = self.pools is not None and load_cfg is None
        if resume:
            if state is None:
                raise ValueError("Resuming TIMP training requires a TIMP training checkpoint")
            if tuple(state.get("heads", ())) != HEADS:
                raise ValueError(f"Checkpoint uses different AMP heads. Start a new TIMP run with {', '.join(HEADS)}.")
            if self.stage < state["stage"]:
                raise ValueError("A Stage 3 checkpoint cannot initialize Stage 2 training")
        restore_iteration = super().load(loaded_dict, load_cfg, strict)
        if resume:
            self.discriminator.load_state_dict(state["discriminator"], strict=strict)
            self.discriminator_optimizer.load_state_dict(state["discriminator_optimizer"])
            self.him_objective.load_state_dict(state["him"], strict=strict)
            self.him_optimizer.load_state_dict(state["him_optimizer"])
            self.stage_step = state["step"] if self.stage == state["stage"] else 0
        return restore_iteration
