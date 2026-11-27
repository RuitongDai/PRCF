"""E1 reset and dynamics randomization for the climbing skills."""

import math

import torch
from mjlab.envs import mdp
from mjlab.managers import EventTermCfg, SceneEntityCfg
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

from ..poses import PRONE_PELVIS_HEIGHT, PRONE_PITCH, STANDING_PELVIS_HEIGHT, prone_joint_pos
from ..terrains import BOX_EDGE_X


def reset_at_edge(env, env_ids, *, descent, randomize):
    ids = mdp.resolve_env_ids(env, env_ids)
    robot = env.scene["robot"]
    root = robot.data.default_root_state[ids].clone()
    origin = env.scene.env_origins[ids]
    angles = torch.zeros_like(root[:, :3])
    root[:, :2] = origin[:, :2]
    if descent:
        root[:, 0] += BOX_EDGE_X + 0.33
        root[:, 2] = origin[:, 2] + PRONE_PELVIS_HEIGHT
        joint_pos = prone_joint_pos(root).expand(len(ids), -1).clone()
        position_noise = (0.075, 0.60)
        angle_noise = (math.radians(72), 0.0, math.radians(15))
    else:
        root[:, 0] += BOX_EDGE_X - 0.275
        root[:, 2] = STANDING_PELVIS_HEIGHT
        joint_pos = robot.data.default_joint_pos[ids].clone()
        position_noise = (0.075, 1.35)
        angle_noise = (0.0, 0.0, math.radians(60))
    if randomize:
        root[:, :2] += (2 * torch.rand_like(root[:, :2]) - 1) * root.new_tensor(position_noise)
        angles += (2 * torch.rand_like(angles) - 1) * root.new_tensor(angle_noise)
        joint_pos += torch.empty_like(joint_pos).uniform_(-0.10, 0.10)
    limits = robot.data.soft_joint_pos_limits[ids]
    joint_pos.clamp_(min=limits[..., 0], max=limits[..., 1])
    if descent:
        # APEX composes local roll/yaw perturbations after a 90-degree prone
        # pitch. The extra inclination is part of the E1-calibrated pose.
        angles[:, 1] = PRONE_PITCH - math.pi / 2
        base = root.new_tensor((math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0))
        root[:, 3:7] = quat_mul(base.expand(len(ids), -1), quat_from_euler_xyz(*angles.unbind(dim=1)))
    else:
        root[:, 3:7] = quat_from_euler_xyz(*angles.unbind(dim=1))
    root[:, 7:] = 0.0
    robot.write_root_state_to_sim(root, env_ids=ids)
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=ids)


def randomize_default_pose(env, env_ids):
    env.action_manager.get_term("joint_pos").randomize_default_pose(mdp.resolve_env_ids(env, env_ids))


@requires_model_fields("body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
def randomize_torso_mass(env, env_ids, *, asset_cfg):
    ids = mdp.resolve_env_ids(env, env_ids)
    body_ids = env.scene[asset_cfg.name].indexing.body_ids[asset_cfg.body_ids]
    nominal_mass = env.sim.get_default_field("body_mass")[body_ids]
    nominal_inertia = env.sim.get_default_field("body_inertia")[body_ids]
    delta = nominal_mass.new_empty((len(ids), len(body_ids))).uniform_(-1.0, 1.0)
    mass = nominal_mass + delta
    inertia = nominal_inertia * (mass / nominal_mass)[..., None]
    env.sim.model.body_mass[ids[:, None], body_ids] = mass
    env.sim.model.body_inertia[ids[:, None], body_ids] = inertia


@requires_model_fields("geom_friction", "geom_solref")
class ContactMaterial(ManagerTermBase):
    """Sample the source material buckets, then map them to MuJoCo contacts."""

    def __init__(self, cfg, env):
        super().__init__(env)
        robot = env.scene["robot"]
        self.geom_local_ids = cfg.params["asset_cfg"].geom_ids
        self.geom_ids = robot.indexing.geom_ids[self.geom_local_ids]
        self.static = torch.empty((env.num_envs, len(self.geom_ids)), device=env.device)
        self.dynamic = torch.empty_like(self.static)

    def __call__(self, env, env_ids, *, asset_cfg):
        ids = mdp.resolve_env_ids(env, env_ids)
        buckets = self.static.new_empty((64, 3)).uniform_()
        buckets = buckets * buckets.new_tensor((1.3, 0.9, 0.5)) + buckets.new_tensor((0.3, 0.3, 0.0))
        indices = torch.randint(len(buckets), (len(ids), len(self.geom_ids)), device=env.device)
        static, dynamic, restitution = buckets[indices].unbind(dim=2)
        self.static[ids] = static
        self.dynamic[ids] = dynamic.minimum(static)
        self.set_at_rest(env, ids)
        # solref uses damping ratio, so map the coefficient of restitution
        # through the equivalent damped-oscillator relation.
        log_e = restitution.clamp_min(1.0e-6).log()
        ratio = -log_e / (math.pi**2 + log_e.square()).sqrt()
        env.sim.model.geom_solref[ids[:, None], self.geom_ids, 1] = torch.where(restitution > 0.0, ratio, 1.0)

    def set_at_rest(self, env, ids):
        env.sim.model.geom_friction[ids[:, None], self.geom_ids, 0] = self.static[ids]

    def update_friction(self, env):
        data = env.scene["robot"].data
        speed = data.geom_lin_vel_w[:, self.geom_local_ids, :2].norm(dim=-1)
        sliding = (speed / 0.05).clamp(0.0, 1.0)
        env.sim.model.geom_friction[:, self.geom_ids, 0] = self.static.lerp(self.dynamic, sliding)


def reset_contact_material(env, env_ids):
    material = env.event_manager.get_term_cfg("physics_material").func
    material.set_at_rest(env, mdp.resolve_env_ids(env, env_ids))


def update_contact_friction(env, env_ids):
    env.event_manager.get_term_cfg("physics_material").func.update_friction(env)


def push_root_com(env, env_ids, *, velocity_range):
    ids = mdp.resolve_env_ids(env, env_ids)
    robot = env.scene["robot"]
    velocity = robot.data.root_com_vel_w[ids]
    extent = velocity.new_tensor(velocity_range)
    delta = (2 * torch.rand_like(velocity) - 1) * extent
    robot.write_root_com_velocity_to_sim(velocity + delta, env_ids=ids)


def apex_events(*, descent, play):
    terms = {
        "reset_at_edge": EventTermCfg(
            func=reset_at_edge, mode="reset", params={"descent": descent, "randomize": not play},
        ),
    }
    if play:
        return terms
    torso = SceneEntityCfg("robot", body_names=("torso_link",))
    terms["torso_mass"] = EventTermCfg(
        func=randomize_torso_mass, mode="startup", params={"asset_cfg": torso},
    )
    terms["torso_com"] = EventTermCfg(
        func=mdp.dr.body_com_offset, mode="startup",
        params={"asset_cfg": torso, "ranges": {0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)}},
    )
    terms["physics_material"] = EventTermCfg(
        func=ContactMaterial, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", geom_names=(r".*_collision",))},
    )
    terms["joint_default_pos"] = EventTermCfg(func=randomize_default_pose, mode="startup")
    terms["reset_material"] = EventTermCfg(func=reset_contact_material, mode="reset")
    terms["contact_friction"] = EventTermCfg(func=update_contact_friction, mode="step")
    if descent:
        terms["push"] = EventTermCfg(
            func=push_root_com, mode="interval", interval_range_s=(1.0, 3.0),
            params={"velocity_range": (0.5, 0.5, 0.2, 0.52, 0.52, 0.78)},
        )
    return terms
