"""Fast running: microduck sprinting at up to about 2 m/s on flat ground."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import mjswan
import mujoco
from mjswan.managers.observation_manager import ObservationGroupCfg
from mjswan.mjlab import apply_mjlab_sim_options, build_single_entity_trace_env

from mjswan_playground._trace import CommandValues
from mjswan_playground.microduck.main import (
    CONTROL_DT,
    ENTITY,
    STAND_HEIGHT,
    VELOCITY_ENV,
    _driven,
    _padded,
    _stand_pose,
)

from . import _common

POLICY_REPO_ID = "HannesVonEssen/microduck-running"
POLICY_REVISION = "9f45af1f4b04252f3aa1f4c056fa7fa0d4b35e4c"

#: Up to the 2.2 m/s the policy trained through, starting at the play config's 1.0.
#: Lateral and yaw trained on +/-0.02 m/s and +/-0.05 rad/s only, so they read zero.
FORWARD = mjswan.SliderConfig(
    name="lin_vel_x", label="Forward (m/s)", range=(0.0, 2.2), default=1.0, step=0.05
)


def _scene_spec(
    root: Path, stand_pose: dict[str, float], *, tracing: bool = False
) -> mujoco.MjSpec:
    """``scene_walk.xml``, the model the running task trains on, with mjlab's velocity
    sim settings, the servo filters and the collision mesh copies; ``tracing`` leaves
    them out, as nothing traced reads them."""
    spec = mujoco.MjSpec.from_file(str(root / _common.WALK_SCENE_XML))
    if not tracing:
        apply_mjlab_sim_options(spec, VELOCITY_ENV.sim)
        _common.filter_servos(spec)
        _common.own_collision_meshes(spec)
    _common.set_only_keyframe(spec, stand_pose, (0, 0, STAND_HEIGHT, 1, 0, 0, 0))
    return spec


def add_scenes(project: mjswan.ProjectHandle, root: Path) -> None:
    stand_pose = _stand_pose(root / _common.WALK_SCENE_XML)
    spec = _scene_spec(root, stand_pose)
    joint_names = _common.servo_joints(spec)

    scene = project.add_scene(name="Running", spec=spec, control_dt=CONTROL_DT)
    scene.add_attribution("3d-models", license=root / "LICENSE-HARDWARE")
    scene.set_viewer(_common.follow_cam())
    scene.set_trace_env(
        build_single_entity_trace_env(
            partial(_scene_spec, root, stand_pose, tracing=True),
            entity_name=ENTITY,
            commands={"twist": CommandValues(1)},
        )
    )
    scene.add_policy(
        name="Run",
        policy=_common.hub_policy(POLICY_REPO_ID, "policy.onnx", POLICY_REVISION),
        commands={"twist": mjswan.ui_command([FORWARD])},
        # robotd's 61 values: 48 of proprioception, then twist(3), head(4), body(6).
        observations=ObservationGroupCfg(
            terms={
                **_common.proprioception(_common.joints_cfg(joint_names)),
                "forward": _driven("twist"),
                "twist_pad": _padded(2),
                "head_command": _padded(4),
                "body_command": _padded(6),
            }
        ),
        actions=_common.servo_action(),
        terminations={"fell_over": VELOCITY_ENV.terminations["fell_over"]},
        policy_joint_names=joint_names,
        default_joint_pos=[stand_pose[name] for name in joint_names],
    )
