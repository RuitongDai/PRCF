"""Failure conditions and the two support goals."""

from ..poses import STANDING_PELVIS_HEIGHT
from ..terrains import BOX_EDGE_X, BOX_HALF_WIDTH, PATCH_SIZE


def low_pelvis(env):
    return env.scene["robot"].data.root_link_pos_w[:, 2] < 0.40


def returned_to_ground(env, *, asset_cfg):
    data = env.scene[asset_cfg.name].data
    feet_low = (data.body_link_pos_w[:, asset_cfg.body_ids, 2] < 0.26).all(dim=1)
    return (env.episode_length_buf > 150) & feet_low


def too_far_back(env):
    x = env.scene["robot"].data.body_link_pos_w[..., 0] - env.scene.env_origins[:, None, 0]
    return (env.episode_length_buf > 100) & (x.amin(dim=1) < BOX_EDGE_X - 0.40)


def knees_straight(env, *, asset_cfg):
    joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    return ((env.episode_length_buf < 50) & (joint_pos < 0.30).all(dim=1)) | (joint_pos < 0.10).all(dim=1)


def outside_tile(env):
    xy = env.scene["robot"].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return (xy.abs() >= xy.new_tensor(PATCH_SIZE) / 2).any(dim=1)


def horizontal_contacts(env, sensor_name):
    contacts = env.scene[sensor_name].data
    pos = contacts.pos - env.scene.env_origins[:, None, :]
    active = (contacts.found > 0) & (contacts.normal[..., 2].abs() >= 0.7) & (contacts.force[..., 0].abs() > 5.0)
    return pos, active


def on_box(link_pos_local):
    x, y, z = link_pos_local.unbind(dim=2)
    return (
        (x.amin(dim=1) > BOX_EDGE_X + 0.02)
        & (x.amax(dim=1) < -BOX_EDGE_X + 0.02)
        & (y.abs().amax(dim=1) < BOX_HALF_WIDTH + 0.02)
        & (z.amin(dim=1) > -0.01)
    )


def top_support(env, body_ids, geom_ids):
    pos, active = horizontal_contacts(env, "elbows_knees")
    x, y, z = pos.unbind(dim=2)
    on_top = active & (z.abs() < 0.035) & (x.abs() <= -BOX_EDGE_X) & (y.abs() <= BOX_HALF_WIDTH)
    slots = env.scene["elbows_knees"].cfg.num_slots
    # Upper-arm and thigh capsules reach the E1 elbow and knee joints. Count
    # their load only when the contact point is close to the relevant joint.
    geom_pos = pos.unflatten(1, (-1, slots))[:, geom_ids]
    body_pos = env.scene["robot"].data.body_link_pos_w[:, body_ids] - env.scene.env_origins[:, None, :]
    near_joint = (geom_pos - body_pos[:, :, None, None, :]).norm(dim=-1) < 0.105
    contact = on_top.unflatten(1, (-1, slots))[:, geom_ids] & near_joint
    return contact.any(dim=(2, 3))


def feet_on_ground(env):
    pos, active = horizontal_contacts(env, "feet")
    x, _, z = pos.unbind(dim=2)
    ground_z = z + env.scene.env_origins[:, None, 2]
    ground = active & (ground_z.abs() < 0.035) & (x < BOX_EDGE_X + 0.02)
    return ground.reshape(len(x), 2, -1).any(dim=2)


def standing_on_ground(env, grounded):
    data = env.scene["robot"].data
    linear = data.root_link_lin_vel_w.norm(dim=1)
    angular = data.root_link_ang_vel_w.norm(dim=1)
    return (
        grounded
        & (data.root_link_pos_w[:, 2] > STANDING_PELVIS_HEIGHT - 0.10)
        & (-data.projected_gravity_b[:, 2] > 0.90)
        & (linear < 0.30)
        & (angular < 0.80)
    )
