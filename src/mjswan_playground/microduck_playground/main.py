"""Microduck Playground's running policy: microduck sprinting at up to about 2 m/s.

The same robot and 61-value contract as ``microduck``, so the scene compiles from the
XML the running task trained on, and the policy drives its ``<position>`` servos. See
``README.md``.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import mjswan
import mujoco
import onnx
from huggingface_hub import hf_hub_download
from mjlab.envs.mdp import observations as obs_fns
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjswan.envs.mdp.actions import JointPositionActionCfg
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjswan.mjlab import apply_mjlab_sim_options, build_single_entity_trace_env

from mjswan_playground._deps import ensure_repo
from mjswan_playground._trace import CommandValues
from mjswan_playground.microduck.main import (
    CONTROL_DT,
    ENTITY,
    ROOT_JOINT,
    STAND_HEIGHT,
    STAND_KEY,
    TRACKED_BODY,
    VELOCITY_ENV,
    _driven,
    _free_joint_adr,
    _padded,
    _servo_joints,
    _stand_pose,
)

REPO_URL = "https://github.com/Vottivott/microduck-playground.git"
REPO_COMMIT = "f010ef1cefcad49c05c20a4a78f840af83a643b9"
#: The model the running task trains on (``get_walk_spec``).
SCENE_XML = "src/mjlab_microduck/robot/microduck/scene_walk.xml"

POLICY_REPO_ID = "HannesVonEssen/microduck-running"
POLICY_REVISION = "9f45af1f4b04252f3aa1f4c056fa7fa0d4b35e4c"
POLICY_FILENAME = "policy.onnx"

#: Up to the 2.2 m/s the policy trained through, starting at the play config's 1.0.
#: Lateral and yaw trained on +/-0.02 m/s and +/-0.05 rad/s only, so they read zero.
FORWARD = mjswan.SliderConfig(
    name="lin_vel_x", label="Forward (m/s)", range=(0.0, 2.2), default=1.0, step=0.05
)

#: Training delays each servo command by 3 to 6 physics steps (15 to 30 ms) through BAM,
#: which the browser lacks; a first-order filter on each servo stands in for it.
SERVO_FILTER_S = 0.03
#: ``joint_vel`` reads one control step back, as the servo firmware's velocity does.
JOINT_VEL_LAG = 1


def _resolve_root() -> Path:
    return ensure_repo(
        name="microduck_playground",
        url=REPO_URL,
        commit=REPO_COMMIT,
        marker=SCENE_XML,
        root_env_var="MJSWAN_MICRODUCK_PLAYGROUND_ROOT",
    )


def _policy() -> onnx.ModelProto:
    return onnx.load(
        hf_hub_download(POLICY_REPO_ID, POLICY_FILENAME, revision=POLICY_REVISION)
    )


def _scene_spec(
    root: Path, stand_pose: dict[str, float], *, tracing: bool = False
) -> mujoco.MjSpec:
    """``scene_walk.xml`` with mjlab's velocity sim settings, the servo filters and
    STAND as its only keyframe, which the browser and the tracing env both reset to.

    ``tracing`` leaves the sim settings and filters out: nothing traced reads them.
    """
    spec = mujoco.MjSpec.from_file(str(root / SCENE_XML))
    if not tracing:
        apply_mjlab_sim_options(spec, VELOCITY_ENV.sim)
        for actuator in spec.actuators:
            actuator.dyntype = mujoco.mjtDyn.mjDYN_FILTEREXACT
            actuator.dynprm[0] = SERVO_FILTER_S

    model = spec.compile()
    qpos = model.qpos0.copy()
    qpos[_free_joint_adr(model, ROOT_JOINT)] = [0.0, 0.0, STAND_HEIGHT, 1, 0, 0, 0]
    for joint in range(model.njnt):
        name = model.joint(joint).name
        if name in stand_pose:
            qpos[model.jnt_qposadr[joint]] = stand_pose[name]
    ctrl = [stand_pose[name] for name in _servo_joints(model)]

    for key in list(spec.keys):
        spec.delete(key)
    # The filters start settled at STAND, as the servos are.
    act = ctrl if model.na else []
    spec.add_key(name=STAND_KEY, qpos=qpos.tolist(), ctrl=ctrl, act=act)
    return spec


def setup_builder() -> mjswan.Builder:
    root = _resolve_root()
    stand_pose = _stand_pose(root / SCENE_XML)

    builder = mjswan.Builder()
    project = builder.add_project(name="Microduck Playground", license=root / "LICENSE")
    project.set_notice(root / "NOTICE")

    spec = _scene_spec(root, stand_pose)
    joint_names = _servo_joints(spec.compile())
    joints = SceneEntityCfg(
        name=ENTITY, joint_names=tuple(joint_names), preserve_order=True
    )

    scene = project.add_scene(name="Running", spec=spec, control_dt=CONTROL_DT)
    scene.add_attribution("3d-models", license=root / "LICENSE-HARDWARE")
    scene.set_viewer(
        mjswan.ViewerConfig(
            origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY,
            body_name=TRACKED_BODY,
            distance=0.8,
            elevation=-12.0,
            azimuth=40.0,
        )
    )
    scene.set_trace_env(
        build_single_entity_trace_env(
            partial(_scene_spec, root, stand_pose, tracing=True),
            entity_name=ENTITY,
            commands={"twist": CommandValues(1)},
        )
    )

    scene.add_policy(
        name="Run",
        policy=_policy(),
        commands={"twist": mjswan.ui_command([FORWARD])},
        # robotd's 61 values: 48 of proprioception, then twist(3), head(4), body(6).
        observations=ObservationGroupCfg(
            terms={
                "base_ang_vel": ObservationTermCfg(func=obs_fns.base_ang_vel),
                "projected_gravity": ObservationTermCfg(func=obs_fns.projected_gravity),
                "joint_pos": ObservationTermCfg(
                    func=obs_fns.joint_pos_rel, params={"asset_cfg": joints}
                ),
                "joint_vel": ObservationTermCfg(
                    func=obs_fns.joint_vel_rel,
                    params={"asset_cfg": joints},
                    history_steps=(JOINT_VEL_LAG,),
                ),
                "actions": ObservationTermCfg(func=obs_fns.last_action),
                "forward": _driven("twist"),
                "twist_pad": _padded(2),
                "head_command": _padded(4),
                "body_command": _padded(6),
            }
        ),
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="",
                actuator_names=(".*",),
                scale=1.0,
                use_default_offset=True,
            )
        },
        terminations={"fell_over": VELOCITY_ENV.terminations["fell_over"]},
        policy_joint_names=joint_names,
        default_joint_pos=[stand_pose[name] for name in joint_names],
    )
    return builder
