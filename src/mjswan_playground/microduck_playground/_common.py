"""What every experiment's scene shares: the pinned checkout, the robot's servos and the
proprioception every policy here reads."""

from __future__ import annotations

import contextlib
import importlib.util
import re
import sys
import types
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
#: Every collision mesh, plus a floor.
ALLCOLLISIONS_SCENE_XML = f"{ROBOT_DIR}/scene.xml"

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


@contextlib.contextmanager
def _stub_modules(stubs: dict[str, dict[str, object]]):
    saved = {name: sys.modules.get(name) for name in stubs}
    for name, attrs in stubs.items():
        module = types.ModuleType(name)
        module.__path__ = []
        module.__dict__.update(attrs)
        sys.modules[name] = module
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_upstream(
    root: Path, relpath: str, stubs: dict[str, dict[str, object]] | None = None
) -> types.ModuleType:
    """Run one upstream module from the checkout with ``mjlab_microduck`` stubbed out:
    its package imports BAM and builds mjlab 1.3.0 configs, neither of which loads
    against this mjlab. ``stubs`` adds or overrides stubbed modules and what they hold.
    Writes no bytecode into the shared checkout."""
    constants = {
        "MICRODUCK_WALK_XML": root / ROBOT_DIR / "robot_walk.xml",
        "MICRODUCK_ALLCOLLISIONS_XML": root / ROBOT_DIR / "robot_allcollisions.xml",
        "MICRODUCK_WALK_ROBOT_CFG": types.SimpleNamespace(
            spec_fn=lambda: mujoco.MjSpec.from_file(str(root / WALK_SCENE_XML))
        ),
        "FULL_COLLISION": None,
        "HOME_FRAME": None,
        "actuators": None,
    }
    all_stubs = {
        "mjlab_microduck": {},
        "mjlab_microduck.robot": {},
        "mjlab_microduck.robot.microduck_constants": constants,
        **(stubs or {}),
    }
    name = "mjlab_microduck." + relpath.removeprefix("src/mjlab_microduck/")
    spec = importlib.util.spec_from_file_location(
        name.removesuffix(".py").replace("/", "."), root / relpath
    )
    module = importlib.util.module_from_spec(spec)
    dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        with _stub_modules(all_stubs):
            spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = dont_write
    return module


_FOOT_COLLISION = re.compile(r"^(left|right)_foot_collision$")


def full_collision(spec: mujoco.MjSpec, robot_body: str = TRACKED_BODY) -> None:
    """Upstream's ``FULL_COLLISION`` under mjlab 1.3.0, which this mjlab can no longer
    build: in the robot, only ``*_collision`` geoms collide, at condim 1, and the feet
    at condim 3, priority 1 and friction 1.

    They also go to group 3, hidden: the hulls ``long_jump_robot`` adds set their class
    after creation, so they keep group 0 and would be drawn over the visual meshes.
    """
    stack = [spec.body(robot_body)]
    while stack:
        body = stack.pop()
        stack.extend(body.bodies)
        for geom in body.geoms:
            colliding = geom.name.endswith("_collision")
            geom.contype = geom.conaffinity = int(colliding)
            if not colliding:
                continue
            geom.group = 3
            foot = bool(_FOOT_COLLISION.match(geom.name))
            geom.condim = 3 if foot else 1
            geom.priority = int(foot)
            if foot:
                geom.friction[0] = 1.0


_MESH_FIELDS = (
    "file", "content_type", "scale", "refpos", "refquat", "inertia", "maxhullvert",
    "smoothnormal", "uservert", "usernormal", "userface", "userfacenormal",
)  # fmt: skip


def own_collision_meshes(spec: mujoco.MjSpec) -> None:
    """Give each collision geom a mesh no rendered geom shares.

    mjswan's renderer turns the vertices of every mesh a rendered geom (group < 3) uses
    to three.js's y-up in place, in the model the physics reads, so a collision geom on
    the same mesh collides with a hull rotated 90 degrees.
    """
    # ponytail: drop once mjswan copies the vertices before turning them.
    rendered = {
        geom.meshname
        for geom in spec.geoms
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH and geom.group < 3
    }
    copies: dict[str, str] = {}
    for geom in spec.geoms:
        if geom.type != mujoco.mjtGeom.mjGEOM_MESH or geom.meshname not in rendered:
            continue
        if not (geom.contype or geom.conaffinity):
            continue
        if geom.group < 3:
            raise ValueError(
                f"Geom {geom.name!r} is rendered and collides, so mjswan's renderer "
                "turns the hull it collides with; split it into a visual and a "
                "collision geom."
            )
        if geom.meshname not in copies:
            source = spec.mesh(geom.meshname)
            copy = spec.add_mesh(name=f"{geom.meshname}_collision")
            for field in _MESH_FIELDS:
                setattr(copy, field, getattr(source, field))
            copies[geom.meshname] = copy.name
        geom.meshname = copies[geom.meshname]


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


def servo_action(scale: float = 1.0) -> dict[str, JointPositionActionCfg]:
    """``ctrl = default + scale * action`` into the XML's own ``<position>`` servos."""
    return {
        "joint_pos": JointPositionActionCfg(
            entity_name="",
            actuator_names=(".*",),
            scale=scale,
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
