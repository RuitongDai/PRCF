"""Left-right symmetry for the XML-order E1 visual AMP task."""

from __future__ import annotations

import torch

from .constants import (
  ACTOR_HISTORY,
  CRITIC_HISTORY,
  NUM_JOINTS,
  PROPRIO_TERM_WIDTHS,
)


_HISTORY_LENGTH = ACTOR_HISTORY
_CRITIC_HISTORY_LENGTH = CRITIC_HISTORY
_ACTOR_FRAME_DIM = 86
_HEIGHT_ROWS = 13
_HEIGHT_COLUMNS = 21

# XML order: left leg, right leg, waist, left arm, right arm.
_JOINT_INDICES = (
  6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12,
  19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18,
)
_JOINT_SIGNS = (
  1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1,
  1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1,
)
_POLAR_SIGNS = (1, -1, 1)
_ANGULAR_SIGNS = (-1, 1, -1)
_COMMAND_SIGNS = (1, -1, -1)


def _sign(values, tensor):
  return tensor.new_tensor(values)


def mirror_joints(values):
  if values.shape[-1] != NUM_JOINTS:
    raise ValueError(f"Expected {NUM_JOINTS} joints, got {values.shape[-1]}.")
  return values[..., _JOINT_INDICES] * _sign(_JOINT_SIGNS, values)


def _mirror_actor(actor):
  mirrored = actor.clone()
  start = 0
  transforms = (
    lambda block: -block,
    lambda block: block * _sign(_ANGULAR_SIGNS, block),
    lambda block: block * _sign(_POLAR_SIGNS, block),
    lambda block: block * _sign(_COMMAND_SIGNS, block),
    mirror_joints,
    mirror_joints,
    mirror_joints,
  )
  for width, transform in zip(PROPRIO_TERM_WIDTHS, transforms):
    stop = start + width * _HISTORY_LENGTH
    block = actor[..., start:stop].reshape(
      *actor.shape[:-1], _HISTORY_LENGTH, width
    )
    mirrored[..., start:stop] = transform(block).reshape_as(
      mirrored[..., start:stop]
    )
    start = stop
  return mirrored


def _mirror_critic(critic):
  mirrored = critic.clone()
  widths = (*PROPRIO_TERM_WIDTHS, 3, 2, 2)
  transforms = (
    lambda block: -block,
    lambda block: block * _sign(_ANGULAR_SIGNS, block),
    lambda block: block * _sign(_POLAR_SIGNS, block),
    lambda block: block * _sign(_COMMAND_SIGNS, block),
    mirror_joints,
    mirror_joints,
    mirror_joints,
    lambda block: block * _sign(_POLAR_SIGNS, block),
    lambda block: block[..., (1, 0)],
    lambda block: block[..., (1, 0)],
  )
  start = 0
  for width, transform in zip(widths, transforms):
    stop = start + width * _CRITIC_HISTORY_LENGTH
    block = critic[..., start:stop].reshape(
      *critic.shape[:-1], _CRITIC_HISTORY_LENGTH, width
    )
    mirrored[..., start:stop] = transform(block).reshape_as(
      mirrored[..., start:stop]
    )
    start = stop
  return mirrored


def _mirror_height_scan(scan):
  expected = _HEIGHT_ROWS * _HEIGHT_COLUMNS
  if scan.shape[-1] != expected:
    raise ValueError(f"Expected {expected} height rays, got {scan.shape[-1]}.")
  values = scan.reshape(*scan.shape[:-1], _HEIGHT_ROWS, _HEIGHT_COLUMNS)
  return values.flip(-2).reshape_as(scan)


def mirror_actor_observation(actor):
  """Return the left/right mirrored term-major actor observation."""
  expected = _HISTORY_LENGTH * _ACTOR_FRAME_DIM
  if actor.shape[-1] != expected:
    raise ValueError(f"Expected actor width {expected}, got {actor.shape[-1]}.")
  return _mirror_actor(actor)


def data_augmentation_func(obs=None, actions=None, env=None, obs_type=None):
  """Append one mirrored PPO sample for every original sample."""
  del env, obs_type
  obs_aug = None
  if obs is not None:
    mirrored = obs.clone()
    if "actor" in mirrored.keys():
      mirrored["actor"] = _mirror_actor(obs["actor"])
    if "critic" in mirrored.keys():
      mirrored["critic"] = _mirror_critic(obs["critic"])
    if "critic_height_scan" in mirrored.keys():
      mirrored["critic_height_scan"] = _mirror_height_scan(
        obs["critic_height_scan"]
      )
    if "depth" in mirrored.keys():
      mirrored["depth"] = obs["depth"].flip(-1)
    obs_aug = torch.cat((obs, mirrored), dim=0)

  actions_aug = None
  if actions is not None:
    actions_aug = torch.cat((actions, mirror_joints(actions)), dim=0)
  return obs_aug, actions_aug


__all__ = [
  "data_augmentation_func",
  "mirror_actor_observation",
  "mirror_joints",
]
