"""HUSKY humanoid-skateboarding demo. See ``README.md``."""

from __future__ import annotations

import math
from pathlib import Path

import mjswan
import mujoco
import onnx
from mjlab.envs.mdp import observations as obs_fns
from mjlab.envs.mdp import terminations as term_fns
from mjlab.sim import MujocoCfg, SimulationCfg
from mjswan.envs.mdp.actions import JointPositionActionCfg
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjswan.managers.termination_manager import TerminationTermCfg
from mjswan.mjlab import apply_mjlab_sim_options, build_single_entity_trace_env

from mjswan_playground._deps import ensure_repo
from mjswan_playground._trace import CommandValues

from . import terms

HUSKY_REPO_URL = "https://github.com/TeleHuman/humanoid_skateboarding.git"
HUSKY_REPO_COMMIT = "d93833e80deff7f927c0b80ef9c435d8b5c488fe"

SCENE_XML = "test_scene/mjlab_scene.xml"
ROBOT_XML = "src/mjlab_husky/asset_zoo/robots/skateboard/xmls/g1.xml"
POLICY_ONNX = "ckpts/test.onnx"

ENTITY = "robot"
#: Copied from `unitree_g1_skater_env_cfg`: importing it needs upstream's own rsl_rl.
SIM = SimulationCfg(
    mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20, ccd_iterations=50)
)
DECIMATION = 4
CONTROL_DT = SIM.mujoco.timestep * DECIMATION
#: `G1SkaterManagerBasedRlEnvCfg.cycle_time`: seconds per push -> steer cycle.
CYCLE_TIME = 6.0
#: `G1SkaterManagerBasedRlEnvCfg`: frames the observation group stacks.
HISTORY_LENGTH = 5


def _resolve_husky_root() -> Path:
    return ensure_repo(
        name="humanoid_skateboarding",
        url=HUSKY_REPO_URL,
        commit=HUSKY_REPO_COMMIT,
        marker=SCENE_XML,
        root_env_var="MJSWAN_HUSKY_ROOT",
    )


def _robot_joints(model: mujoco.MjModel) -> tuple[list[str], list[float]]:
    """The policy's joints in model order (*not* actuator order, which is why upstream's
    ``sim.py`` reindexes), posed at the scene's ``init_state`` keyframe."""
    key_qpos = model.key_qpos[0]
    names: list[str] = []
    defaults: list[float] = []
    for joint in range(model.njnt):
        name = model.joint(joint).name
        if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if not name.startswith(f"{ENTITY}/"):
            continue
        names.append(name)
        defaults.append(float(key_qpos[model.jnt_qposadr[joint]]))
    return names, defaults


def _action_scale(model: mujoco.MjModel, joint_names: list[str]) -> list[float]:
    """Upstream's ``G1_23Dof_ACTION_SCALE``, recovered from the model rather than copied:
    ``0.25 * effort_limit / stiffness``, from ``forcerange`` and ``gainprm[0]``."""
    by_joint: dict[str, float] = {}
    for actuator in range(model.nu):
        if model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        stiffness = float(model.actuator_gainprm[actuator, 0])
        effort_limit = float(model.actuator_forcerange[actuator, 1])
        if stiffness <= 0.0 or effort_limit <= 0.0:
            continue
        joint = model.joint(int(model.actuator_trnid[actuator, 0])).name
        by_joint[joint] = 0.25 * effort_limit / stiffness
    missing = [name for name in joint_names if name not in by_joint]
    if missing:
        raise ValueError(
            "No position actuator with both a stiffness and an effort limit for "
            f"{missing}; the action scale mjlab derives from them cannot be recovered."
        )
    return [by_joint[name] for name in joint_names]


def _trace_spec(robot_xml: Path, default_joint_pos: list[float]) -> mujoco.MjSpec:
    """The robot alone, keyframed at the skater's initial pose.

    ``build_single_entity_trace_env`` reads ``default_joint_pos`` off the first keyframe
    and ``joint_pos_rel`` bakes it in; the standalone robot XML has no keyframe, so
    without this the policy sees joint positions relative to zero.
    """
    spec = mujoco.MjSpec.from_file(str(robot_xml))
    root_pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # never read: a dynamic slot
    # Not "init_state": mjlab's `Entity` adds a keyframe under that name from the
    # `init_state` this one resolves to, and MuJoCo rejects the repeat.
    spec.add_key(name="default_pose", qpos=root_pose + list(default_joint_pos))
    return spec


def setup_builder() -> mjswan.Builder:
    root = _resolve_husky_root()

    spec = mujoco.MjSpec.from_file(str(root / SCENE_XML))
    apply_mjlab_sim_options(spec, SIM)
    model = spec.compile()
    joint_names, default_joint_pos = _robot_joints(model)

    builder = mjswan.Builder()
    project = builder.add_project(
        name="HUSKY Skateboarding", license=root / "LICENSE-CC-BY-NC-4.0.md"
    )
    scene = project.add_scene(
        name="Unitree G1 on a Skateboard",
        spec=spec,
        control_dt=CONTROL_DT,
    )
    scene.add_attribution(
        "unitree_g1",
        license="BSD-3-Clause",
        copyright='HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")',
    )
    # No mjlab task to trace against: the task lives in the HUSKY package. The robot
    # alone covers every term below, plus a width per command.
    scene.set_trace_env(
        build_single_entity_trace_env(
            lambda: _trace_spec(root / ROBOT_XML, default_joint_pos),
            entity_name=ENTITY,
            commands={
                "skate": CommandValues(2),  # push speed, heading
                "phase": CommandValues(1),
            },
        )
    )
    scene.set_viewer(
        mjswan.ViewerConfig(
            origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY,
            entity_name=ENTITY,
            body_name="torso_link",
            distance=4.0,
            elevation=-10.0,
            azimuth=210.0,
        )
    )

    scene.add_policy(
        name="HUSKY Skater",
        policy=onnx.load(str(root / POLICY_ONNX)),
        commands={
            "skate": mjswan.ui_command(
                [
                    mjswan.SliderConfig(
                        name="lin_vel_x",
                        label="Push Speed",
                        range=(0.0, 1.5),
                        default=1.0,
                        step=0.1,
                    ),
                    mjswan.SliderConfig(
                        name="heading",
                        label="Heading",
                        range=(-math.pi / 4, math.pi / 4),
                        default=0.0,
                        step=0.02,
                    ),
                ]
            ),
            "phase": terms.PhaseCommandCfg(
                cycle_time=CYCLE_TIME,
                control_dt=CONTROL_DT,
            ),
        },
        observations=ObservationGroupCfg(
            history_length=HISTORY_LENGTH,
            terms={
                "command": ObservationTermCfg(
                    func=obs_fns.generated_commands,
                    params={"command_name": "skate"},
                    scale=(2.0, 1.0),
                ),
                "heading": ObservationTermCfg(func=terms.heading, scale=1.0 / math.pi),
                "base_ang_vel": ObservationTermCfg(
                    func=obs_fns.builtin_sensor,
                    params={"sensor_name": f"{ENTITY}/imu_ang_vel"},
                    scale=0.25,
                ),
                "projected_gravity": ObservationTermCfg(func=obs_fns.projected_gravity),
                "joint_pos": ObservationTermCfg(func=obs_fns.joint_pos_rel),
                "joint_vel": ObservationTermCfg(func=obs_fns.joint_vel_rel, scale=0.05),
                "actions": ObservationTermCfg(func=obs_fns.last_action),
                "phase": ObservationTermCfg(
                    func=obs_fns.generated_commands, params={"command_name": "phase"}
                ),
            },
        ),
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name=ENTITY,
                actuator_names=(".*",),
                scale=_action_scale(model, joint_names),
                use_default_offset=True,
            )
        },
        terminations={
            "fell_over": TerminationTermCfg(
                func=term_fns.bad_orientation,
                params={"limit_angle": math.radians(70.0)},
            ),
        },
        policy_joint_names=joint_names,
        default_joint_pos=default_joint_pos,
    )

    return builder
