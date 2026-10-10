"""Swing: microduck strapped into a seat on two elastic cords pumps itself up from a
standstill with its head and legs."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from types import ModuleType

import mjswan
import mujoco
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import matrix_from_quat
from mjlab.utils.string import resolve_expr
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjswan.mjlab import apply_mjlab_sim_options, build_single_entity_trace_env

from mjswan_playground.microduck.main import CONTROL_DT, ENTITY, VELOCITY_ENV, _padded

from . import _common

POLICY_REPO_ID = "HannesVonEssen/microduck-swing"
POLICY_REVISION = "e7a603fa89a74b589608c5a026a7720339cb22e0"

CONSTANTS_PY = "src/mjlab_microduck/robot/microduck_constants.py"

#: ``make_microduck_swing_env_cfg``'s ``joint_pos`` action scale.
ACTION_SCALE = 0.7
#: In place of the XML's 0.0048 N*m: BAM's load-dependent Coulomb friction, up to about
#: 0.17 N*m with the servos saturated as they are through most of the pump. Without it
#: the swing passes 180 degrees and slackens the cords at the top.
SERVO_FRICTIONLOSS = 0.1


def swing_plane_cue(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg(ENTITY)
) -> torch.Tensor:
    """Upstream's ``swing_plane_heading_observation``, ``[0, R[0, 1], R[2, 1]]``.

    The x and z of the trunk's y axis in the still-start frame, which the identity reset
    makes the world's: zero while the swing stays in its plane."""
    body_y = matrix_from_quat(env.scene[asset_cfg.name].data.root_link_quat_w)[:, :, 1]
    return torch.stack(
        (torch.zeros_like(body_y[:, 0]), body_y[:, 0], body_y[:, 2]), dim=-1
    )


def _upstream(root: Path) -> ModuleType:
    """``microduck_constants`` with what this mjlab cannot build stubbed out (BAM, and
    1.3.0's ``CollisionCfg``), its swing built on ``scene.xml``: the same robot plus a
    floor and lights."""
    constants = _common.load_upstream(
        root,
        CONSTANTS_PY,
        stubs={
            "mjlab_microduck.actuator": dict.fromkeys(
                ("FrictionDRBamActuatorCfg", "BacklashEncoderBamActuatorCfg"), dict
            ),
            "mjlab.utils.spec_config": {"CollisionCfg": dict},
        },
    )
    constants.MICRODUCK_ALLCOLLISIONS_XML = root / _common.ALLCOLLISIONS_SCENE_XML
    return constants


def _seated_pose(root: Path) -> dict[str, float]:
    """``SWING_SEATED_FRAME``'s joints by name: the reset pose, what actions offset from
    and what ``joint_pos`` reads against."""
    constants = _upstream(root)
    names = tuple(_common.servo_joints(constants.get_swing_spec()))
    pose = resolve_expr(constants.SWING_SEATED_FRAME.joint_pos, names)
    return dict(zip(names, pose, strict=True))


def _scene_spec(
    root: Path, seated: dict[str, float], *, tracing: bool = False
) -> mujoco.MjSpec:
    """``get_swing_spec``'s seat, A-frame and cords, with mjlab's velocity sim settings
    and the servo stand-ins; ``tracing`` leaves those and the mesh copies out.

    Collisions stay as the XML has them, not ``_common.full_collision``: mjlab 1.3.0's
    ``FULL_COLLISION`` disables the geoms it does not match by looking their names up,
    and ``""`` finds one visual geom, so this XML's unnamed hulls (trunk, hips, legs,
    jaw) kept colliding in training. The pump presses each leg against the trunk about a
    third of the time.
    """
    constants = _upstream(root)
    spec = constants.get_swing_spec()
    if not tracing:
        apply_mjlab_sim_options(spec, VELOCITY_ENV.sim)
        _common.filter_servos(spec)
        for joint in spec.joints:
            if joint.type == mujoco.mjtJoint.mjJNT_HINGE:
                joint.frictionloss = SERVO_FRICTIONLOSS
        _common.own_collision_meshes(spec)
    root_pose = (0.0, 0.0, constants.SWING_BOTTOM_TRUNK_Z, 1.0, 0.0, 0.0, 0.0)
    _common.set_only_keyframe(spec, seated, root_pose)
    return spec


def add_scenes(project: mjswan.ProjectHandle, root: Path) -> None:
    seated = _seated_pose(root)
    spec = _scene_spec(root, seated)
    joint_names = _common.servo_joints(spec)

    scene = project.add_scene(name="Swing", spec=spec, control_dt=CONTROL_DT)
    scene.add_attribution("3d-models", license=root / "LICENSE-HARDWARE")
    # The swing env's own camera, side-on to the swing plane, pulled back from 1.15 m so
    # a browser frame holds the whole +/-80 degree arc.
    scene.set_viewer(
        mjswan.ViewerConfig(
            origin_type=mjswan.ViewerConfig.OriginType.WORLD,
            lookat=(0.0, 0.0, 0.39),
            distance=1.5,
            elevation=-13.0,
            azimuth=100.0,
        )
    )
    scene.set_trace_env(
        build_single_entity_trace_env(
            partial(_scene_spec, root, seated, tracing=True), entity_name=ENTITY
        )
    )
    scene.add_policy(
        name="Pump",
        policy=_common.hub_policy(POLICY_REPO_ID, "policy.onnx", POLICY_REVISION),
        # robotd's 61 values, the twist slot carrying the swing-plane cue instead.
        observations=ObservationGroupCfg(
            terms={
                **_common.proprioception(_common.joints_cfg(joint_names)),
                "command": ObservationTermCfg(func=swing_plane_cue),
                "head_command": _padded(4),
                "body_command": _padded(6),
            }
        ),
        actions=_common.servo_action(ACTION_SCALE),
        policy_joint_names=joint_names,
        default_joint_pos=[seated[name] for name in joint_names],
    )
