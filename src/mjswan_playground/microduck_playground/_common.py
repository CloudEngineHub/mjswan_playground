"""What every experiment's scene shares: the pinned checkout, the robot's servos and the
proprioception every policy here reads."""

from __future__ import annotations

from pathlib import Path

import mjswan
import mujoco
import onnx
from huggingface_hub import hf_hub_download
from mjlab.envs.mdp import observations as obs_fns
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjswan.envs.mdp.actions import JointPositionActionCfg
from mjswan.managers.observation_manager import ObservationTermCfg

from mjswan_playground._deps import ensure_repo
from mjswan_playground.microduck.main import (
    ENTITY,
    ROOT_JOINT,
    STAND_KEY,
    TRACKED_BODY,
    _free_joint_adr,
    _servo_joints,
)

REPO_URL = "https://github.com/Vottivott/microduck-playground.git"
REPO_COMMIT = "f010ef1cefcad49c05c20a4a78f840af83a643b9"
ROBOT_DIR = "src/mjlab_microduck/robot/microduck"
#: The walking model and its STAND keyframe, the pose actions offset from.
WALK_SCENE_XML = f"{ROBOT_DIR}/scene_walk.xml"

#: Training delays each servo command by 3 to 6 physics steps (15 to 30 ms) through BAM,
#: which the browser lacks; a first-order filter on each servo stands in for it.
SERVO_FILTER_S = 0.03
#: ``joint_vel`` reads one control step back, as the servo firmware's velocity does.
JOINT_VEL_LAG = 1


def resolve_root() -> Path:
    return ensure_repo(
        name="microduck_playground",
        url=REPO_URL,
        commit=REPO_COMMIT,
        marker=WALK_SCENE_XML,
        root_env_var="MJSWAN_MICRODUCK_PLAYGROUND_ROOT",
    )


def hub_policy(repo_id: str, filename: str, revision: str) -> onnx.ModelProto:
    return onnx.load(hf_hub_download(repo_id, filename, revision=revision))


def filter_servos(spec: mujoco.MjSpec, time_constant: float = SERVO_FILTER_S) -> None:
    for actuator in spec.actuators:
        actuator.dyntype = mujoco.mjtDyn.mjDYN_FILTEREXACT
        actuator.dynprm[0] = time_constant


def set_only_keyframe(
    spec: mujoco.MjSpec,
    pose: dict[str, float],
    root: tuple[float, ...],
    extra_qpos: dict[str, tuple[float, ...]] | None = None,
) -> None:
    """Make ``pose`` at ``root`` (``x y z qw qx qy qz``) the spec's only keyframe, which
    the browser and the tracing env both reset to, with any filters settled at it.

    ``extra_qpos`` places other free joints (props) by name.
    """
    model = spec.compile()
    qpos = model.qpos0.copy()
    qpos[_free_joint_adr(model, ROOT_JOINT)] = root
    for name, value in (extra_qpos or {}).items():
        qpos[_free_joint_adr(model, name)] = value
    for joint in range(model.njnt):
        name = model.joint(joint).name
        if name in pose:
            qpos[model.jnt_qposadr[joint]] = pose[name]
    ctrl = [pose[name] for name in _servo_joints(model)]
    for key in list(spec.keys):
        spec.delete(key)
    spec.add_key(
        name=STAND_KEY, qpos=qpos.tolist(), ctrl=ctrl, act=ctrl if model.na else []
    )


def servo_joints(spec: mujoco.MjSpec) -> list[str]:
    return _servo_joints(spec.compile())


def joints_cfg(joint_names: list[str]) -> SceneEntityCfg:
    return SceneEntityCfg(
        name=ENTITY, joint_names=tuple(joint_names), preserve_order=True
    )


def proprioception(joints: SceneEntityCfg) -> dict[str, ObservationTermCfg]:
    """The 48 values every policy here opens with: gyro, gravity, joint positions
    against the default pose, lagged joint velocities, and the last action."""
    return {
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
    }


def servo_action() -> dict[str, JointPositionActionCfg]:
    """``ctrl = default + action`` into the XML's own ``<position>`` servos."""
    return {
        "joint_pos": JointPositionActionCfg(
            entity_name="",
            actuator_names=(".*",),
            scale=1.0,
            use_default_offset=True,
        )
    }


def follow_cam(distance: float = 0.8, elevation: float = -12.0, azimuth: float = 40.0):
    return mjswan.ViewerConfig(
        origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY,
        body_name=TRACKED_BODY,
        distance=distance,
        elevation=elevation,
        azimuth=azimuth,
    )
