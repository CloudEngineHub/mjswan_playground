"""Backflip: microduck jumps backwards off a 0.8 m platform, turns over once in the air
and lands on a crash mat."""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from types import ModuleType

import mjswan
import mujoco
import torch
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjswan.managers.termination_manager import TerminationTermCfg
from mjswan.mjlab import apply_mjlab_sim_options, build_single_entity_trace_env

from mjswan_playground.microduck.main import (
    CONTROL_DT,
    ENTITY,
    VELOCITY_ENV,
    _padded,
    _stand_pose,
)

from . import _common

POLICY_REPO_ID = "HannesVonEssen/microduck-backflip"
POLICY_REVISION = "0aff1c74c8ad3953f331193a5a671257ef5aab37"

ROBOT_PY = "src/mjlab_microduck/robot/long_jump_robot.py"
STAGE_PY = "src/mjlab_microduck/robot/flip_stage.py"

#: The release profile's drop, platform top above mat top (``MICRODUCK_FLIP_H_*``).
DROP = 0.8
#: ``flip_command_obs`` at that drop, backwards: ``[(drop - 0.7) / 0.2, -1, 0, 0, 0, 0]``.
BODY_COMMAND = ((DROP - 0.7) / 0.2, -1.0, 0.0, 0.0, 0.0, 0.0)
#: ``reset_flip_state``'s standing spawn at the middle of its draws: the trunk 2 cm
#: behind a 3 to 8 cm edge distance, 11.8 cm above the platform top, facing -x.
SPAWN_X = -(0.055 + 0.02)
SPAWN_DZ = 0.118
#: Longer than ``_common.SERVO_FILTER_S``: at 30 ms about half the landings wobble off
#: the mat within 12 s, at 45 ms one in ten.
SERVO_FILTER_S = 0.045
#: The release render's length. The start is fixed, so each episode is the same flip.
EPISODE_S = 12.0

_STAGE_GEOM_FIELDS = (
    "type", "size", "rgba", "friction", "condim", "priority", "solref", "solimp",
    "contype", "conaffinity", "group",
)  # fmt: skip


def body_command(env, *, values: tuple[float, ...], **_) -> torch.Tensor:
    return torch.tensor(values, device=env.device).expand(env.num_envs, -1)


def fell(
    env, *, limit_angle: float, mat_top: float, platform_top: float, **_
) -> torch.Tensor:
    """Stateless stand-in for ``flip_fell``, a contact phase machine with a latch:
    tipped past ``limit_angle`` lying on the mat or on the platform, or the trunk below
    the mat top. In flight the trunk only tips that far well clear of both."""
    asset = env.scene[ENTITY]
    x, z = asset.data.root_link_pos_w[:, 0], asset.data.root_link_pos_w[:, 2]
    tipped = -asset.data.projected_gravity_b[:, 2] < math.cos(limit_angle)
    on_mat = z < mat_top + 0.2
    on_platform = (x < 0.0) & (z > platform_top - 0.1) & (z < platform_top + 0.08)
    return (tipped & (on_mat | on_platform)) | (z < mat_top - 0.1)


def _add_stage(spec: mujoco.MjSpec, stage: ModuleType) -> None:
    """The platform and the mat as world geoms, fixed where the release profile parks
    them every reset."""
    for body_spec, center in (
        (stage.platform_spec(), stage.platform_center(DROP)),
        (stage.mat_spec(), stage.mat_center()),
    ):
        for geom in body_spec.bodies[1].geoms:
            pos = [c + p for c, p in zip(center, geom.pos)]
            copy = spec.worldbody.add_geom(name=geom.name, pos=pos)
            for field in _STAGE_GEOM_FIELDS:
                setattr(copy, field, getattr(geom, field))


def _scene_spec(
    root: Path, stand_pose: dict[str, float], *, tracing: bool = False
) -> mujoco.MjSpec:
    """``scene.xml`` (every collision mesh and a floor) with ``long_jump_robot``'s hull
    on each visible part, mjlab's velocity sim settings, the servo filters and the stage;
    ``tracing`` leaves out the last three and the mesh copies, as nothing traced reads
    them."""
    robot = _common.load_upstream(root, ROBOT_PY)
    stage = _common.load_upstream(root, STAGE_PY)
    spec = mujoco.MjSpec.from_file(str(root / _common.ALLCOLLISIONS_SCENE_XML))
    robot.add_full_collision_geoms(spec)
    robot.add_leg_fold_pairs(spec)
    _common.full_collision(spec)
    if not tracing:
        apply_mjlab_sim_options(spec, VELOCITY_ENV.sim)
        _common.filter_servos(spec, SERVO_FILTER_S)
        _add_stage(spec, stage)
        _common.own_collision_meshes(spec)
    spawn_z = stage.MAT_THICKNESS_M + DROP + SPAWN_DZ
    # Yaw pi: the platform edge is behind it.
    _common.set_only_keyframe(spec, stand_pose, (SPAWN_X, 0, spawn_z, 0, 0, 0, 1))
    return spec


def add_scenes(project: mjswan.ProjectHandle, root: Path) -> None:
    stand_pose = _stand_pose(root / _common.ALLCOLLISIONS_SCENE_XML)
    spec = _scene_spec(root, stand_pose)
    joint_names = _common.servo_joints(spec)
    mat_top = _common.load_upstream(root, STAGE_PY).MAT_THICKNESS_M

    scene = project.add_scene(name="Backflip", spec=spec, control_dt=CONTROL_DT)
    scene.add_attribution("3d-models", license=root / "LICENSE-HARDWARE")
    scene.set_viewer(_common.follow_cam(distance=1.7, elevation=-12.0, azimuth=145.0))
    scene.set_trace_env(
        build_single_entity_trace_env(
            partial(_scene_spec, root, stand_pose, tracing=True),
            entity_name=ENTITY,
            control_dt=CONTROL_DT,
            episode_length_s=EPISODE_S,
        )
    )
    scene.add_policy(
        name="Backflip",
        policy=_common.hub_policy(POLICY_REPO_ID, "policy.onnx", POLICY_REVISION),
        commands={},
        # The running policy's 61 values, with the twist and head slots padded and the
        # body slot carrying the stage.
        observations=ObservationGroupCfg(
            terms={
                **_common.proprioception(_common.joints_cfg(joint_names)),
                "command": _padded(3),
                "head_command": _padded(4),
                "body_command": ObservationTermCfg(
                    func=body_command, params={"values": BODY_COMMAND}
                ),
            }
        ),
        actions=_common.servo_action(),
        terminations={
            "fell": TerminationTermCfg(
                func=fell,
                params={
                    "limit_angle": math.radians(60.0),
                    "mat_top": mat_top,
                    "platform_top": mat_top + DROP,
                },
            ),
            "time_out": VELOCITY_ENV.terminations["time_out"],
        },
        policy_joint_names=joint_names,
        default_joint_pos=[stand_pose[name] for name in joint_names],
    )
