"""Box geometry for the policy and export-compatible features for AMP."""

import torch
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef

from ...baseline.mdp.amp import (
    AmpFeatures as AmpFeatures,
    AmpGridPattern,
    AmpRayCastCfg,
    capture_torque as capture_torque,
    CONTACT_BODIES,
    HEIGHT_RAY_RANGE,
)
from ..terrains import BOX, BOX_HEIGHT, NOMINAL_PELVIS_HEIGHT, is_descent


def box_sensors(*, contact_history=0):
    sensors = (
        AmpRayCastCfg(
            name="amp_scan", frame=ObjRef(type="body", name="pelvis", entity="robot"),
            ray_alignment="yaw", pattern=AmpGridPattern(),
            max_distance=HEIGHT_RAY_RANGE, include_geom_groups=(0,),
        ),
        *(ContactSensorCfg(
            name=f"contact_{part}",
            primary=ContactMatch(mode="body", pattern=body, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="maxforce",
        ) for part, body in CONTACT_BODIES.items()),
        ContactSensorCfg(
            name="box_feet",
            primary=ContactMatch(
                mode="subtree", pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force", "pos", "normal"), reduce="maxforce", num_slots=4,
        ),
        ContactSensorCfg(
            name="box_arms",
            primary=ContactMatch(mode="subtree", pattern=r"^(left_elbow_link|right_elbow_link)$", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force", "pos", "normal"), reduce="maxforce", num_slots=4,
        ),
        ContactSensorCfg(
            name="torso_contact",
            primary=ContactMatch(mode="body", pattern=r"^(pelvis|torso_link|waist_yaw_link)$", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="maxforce",
        ),
    )
    if not contact_history:
        return sensors
    return (
        *sensors,
        ContactSensorCfg(
            name="head_terrain_contact",
            primary=ContactMatch(mode="body", pattern="head_yaw_link", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("force",), reduce="maxforce", history_length=contact_history,
        ),
        ContactSensorCfg(
            name="body_terrain_contact",
            primary=ContactMatch(
                mode="body", pattern=".*", entity="robot", exclude=("head_yaw_link",),
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("force",), reduce="netforce", history_length=contact_history,
        ),
    )


def box_geometry(env):
    robot = env.scene["robot"].data
    x, y, z = (robot.root_link_pos_w - env.scene.env_origins).unbind(dim=1)
    heading = robot.heading_w
    down = is_descent(env)
    target_z = NOMINAL_PELVIS_HEIGHT + (~down) * BOX_HEIGHT
    return torch.stack((
        BOX.front_x - x, BOX.end_x - x, -y, target_z - z,
        heading.cos(), heading.sin(), down.float(),
    ), dim=1)
