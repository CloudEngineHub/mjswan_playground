"""Desk climb: microduck climbs a 27-tread ladder from the floor onto a desk 66 cm up,
tumbles onto the desktop, and a second policy stands it up there."""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import json
import math
import os
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import mjswan
import mujoco
import numpy as np
import onnx
import torch
from mjlab.envs.mdp import events as event_fns
from mjlab.envs.mdp import terminations as term_fns
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, sample_uniform
from mjswan.managers.event_manager import EventTermCfg
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjswan.managers.termination_manager import TerminationTermCfg
from mjswan.mjlab import apply_mjlab_sim_options, build_single_entity_trace_env
from onnx import TensorProto, helper, numpy_helper

from mjswan_playground.microduck.main import (
    CONTROL_DT,
    ENTITY,
    VELOCITY_ENV,
    _stand_pose,
)

from . import _common

POLICY_REPO_ID = "HannesVonEssen/microduck-climb"
POLICY_REVISION = "15723878fa5cc31e326ce522ab16e25d5142632e"

EXPERIMENT = "experiments/desk-climb"
#: The experiment's own robot, with a ``passive_mouth`` hinge the shared one lacks. Its
#: keyframes predate that hinge, so they are dropped.
SCENE_XML = f"{EXPERIMENT}/source/src/mjlab_microduck/robot/microduck/scene.xml"
#: The environment ``training/run.py`` sets for every recipe, which the geometry patch
#: chain reads at import.
GEOMETRY_ENV = {"ENDING_ROLE": "above", "LADDER_SHIFT": ".06"}

#: ``make_floor_desk``'s one level, and ``reset_floor_desk``'s floor spawn: tread 0's rear
#: edge ``FLOOR_GAP`` ahead of the toe, the trunk 0.12 m plus 2 mm above the floor.
RISER = 0.024
ANGLE = math.radians(62.0)
FLOOR_GAP = 0.04
SPAWN_HEIGHT = 0.122
#: ``geometry_previous`` parks treads 27 to 29 under the floor.
ACTIVE_TREADS = 27

#: The play spawn's noise as the ladder sees it: ``reset_stair_ladder`` solves the ladder
#: origin from the robot's own xy noise, so only ``ladder_y_noise`` moves the robot
#: against the stage. Yaw 5 degrees, pitch ``tilt_noise_deg``, roll half of it.
SPAWN_NOISE = (0.01, math.radians(1.0), math.radians(2.0), math.radians(5.0))
JOINT_NOISE = 0.03

#: ``evaluate_sequence.py``'s ``supported_root`` supervisor with run.py's
#: ``SWITCH_MARGIN``: the root this far inside the desk's near edge, 5 cm inside its
#: other edges, and above the desktop.
SWITCH_MARGIN = 0.04
EDGE_MARGIN = 0.05
#: RUNTIME.md's get-up contract: the executed offset smoothed per joint, and the servo
#: gain at 0.8 of the climber's.
HEAD_ALPHA = 0.5
LEG_ALPHA = 0.7
GETUP_KP_RATIO = 0.8
#: mjswan's recurrent carry, ``adapt_hx``.
CARRY = 128

#: Training's 3 to 6 physics step servo delay (15 to 30 ms) as a filter, half of
#: ``_common.SERVO_FILTER_S``: from 22.5 ms on, the dive off the top tread lands short of
#: the desk.
SERVO_FILTER_S = 0.015
#: The XL330 friction model upstream's ``BamActuator`` trains with, ``params/xl330/m6.json``
#: in Rhoban/bam at 62bd8ce (the commit upstream's ``uv.lock`` pins), and the stiffer
#: friction constraint it gives each servo.
FRICTION_BASE = 0.004771183165566
FRICTION_STRIBECK = 0.004676345799486616
LOAD_FRICTION_MOTOR = 0.2667860954283698
DTHETA_STRIBECK = 2.890372094130307
STRIBECK_ALPHA = 8.683259907618984
STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)
STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)

#: Upstream's ``bad_orientation`` limit.
TIPPED = math.radians(65.0)
#: Upstream evaluates 60 s. Most attempts that miss the desk lie across its edge for the
#: rest of it, so those restart at 30 s instead: by then nearly every handoff has come.
EPISODE_S = 60.0
SHORT_OF_DESK_S = 30.0
#: The recorded handoff (``evidence/switches.json``) "Get up" starts from, one the get-up
#: stands from: not every one does.
GETUP_FROM = 9


@contextlib.contextmanager
def _upstream_geometry(root: Path):
    """Import upstream's geometry patch chain, which loads its modules by top-level name
    from ``training/`` and ``source/src`` and patches ``mjlab_microduck.tasks.mdp``,
    stubbed here since its package imports BAM. Undone on exit, bytecode included."""
    exp = root / EXPERIMENT

    def owned(name: str) -> bool:
        return name.startswith(("mjlab_microduck", "geometry_"))

    saved = {name: sys.modules.pop(name) for name in list(sys.modules) if owned(name)}
    saved_env = {key: os.environ.get(key) for key in GEOMETRY_ENV}
    saved_path, dont_write = list(sys.path), sys.dont_write_bytecode
    mdp = SimpleNamespace(reset_floor_desk=None, floor_desk_status=None)
    sys.modules["mjlab_microduck.tasks"] = SimpleNamespace(mdp=mdp, __path__=[])
    sys.modules["mjlab_microduck.tasks.mdp"] = mdp
    sys.path[:0] = [str(exp / "training"), str(exp / "source/src")]
    os.environ.update(GEOMETRY_ENV)
    sys.dont_write_bytecode = True
    try:
        yield importlib.import_module("geometry_patch")
    finally:
        sys.path[:], sys.dont_write_bytecode = saved_path, dont_write
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for name in [name for name in sys.modules if owned(name)]:
            del sys.modules[name]
        sys.modules.update(saved)


def _stage(root: Path) -> SimpleNamespace:
    """The floor-to-desk structure and its 27 live treads, relative to the robot's spawn.

    ``reset_floor_desk`` places the ladder origin from the floor spawn, and the patch
    chain's resets then move the treads onto the curved top (``geometry_base``) and the
    ladder and robot together ``SHIFT`` toward the fixed desk (``geometry_shift``).
    """
    with _upstream_geometry(root) as chain:
        floor_desk, base = chain.floor_desk, chain.base
        ladder = importlib.import_module("mjlab_microduck.robot.ladder")
        geometry = floor_desk.GEOMETRY
        one = torch.ones(1, dtype=torch.float64)
        centres = ladder.tread_layout(
            geometry, RISER * one, ANGLE * one, 0 * one, 0 * one
        )[0][0].numpy()
        u_floor = -(
            ladder.SOLE_TOE_AHEAD_OF_SITE_M
            + FLOOR_GAP
            + geometry.tread_depth_m
            - RISER / math.tan(ANGLE)
        )
        treads = []
        for i in range(ACTIVE_TREADS):
            centre = centres[i] + (
                base.DX[i] - u_floor,
                0.0,
                base.NEW_Z[i] - base.OLD_Z[i],
            )
            pitch = base.PITCH[i]
            quat = (math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0)
            treads.append((ladder.make_tread_spec(geometry, i), centre, quat))
        return SimpleNamespace(
            structure=floor_desk.spec_for("structure"),
            desk=floor_desk.spec_for("desk"),
            treads=treads,
            # The structure's origin, which stays put; its ladder records carry SHIFT.
            origin=np.array([-u_floor - chain.SHIFT, 0.0, 0.0]),
        )


def _add_stage(spec: mujoco.MjSpec, stage: SimpleNamespace) -> None:
    """Every stage geom as a static world geom. A mesh one collides hidden under a
    visual twin, as mjswan's renderer turns the vertices of a mesh it draws."""
    placed = [
        (source, geom.name, geom, stage.origin, geom.quat)
        for source in (stage.structure, stage.desk)
        for geom in source.geoms
    ]
    placed += [
        (tread, f"tread_{i:02d}", tread.geoms[0], pos, quat)
        for i, (tread, pos, quat) in enumerate(stage.treads)
    ]
    for source, name, geom, offset, quat in placed:
        mesh = geom.type == mujoco.mjtGeom.mjGEOM_MESH
        if mesh:
            spec.add_mesh(
                name=geom.meshname, uservert=source.mesh(geom.meshname).uservert
            )
        look = dict(
            type=geom.type,
            size=geom.size,
            pos=np.asarray(geom.pos) + offset,
            quat=quat,
            meshname=geom.meshname,
            rgba=geom.rgba,
        )
        spec.worldbody.add_geom(
            name=name,
            friction=geom.friction,
            condim=geom.condim,
            priority=geom.priority,
            solref=geom.solref,
            solimp=geom.solimp,
            group=3 if mesh else 0,
            **look,
        )
        if mesh:
            spec.worldbody.add_geom(
                name=f"{name}_visual", contype=0, conaffinity=0, **look
            )


def _scene_spec(
    root: Path,
    stand_pose: dict[str, float],
    stage: SimpleNamespace,
    *,
    tracing: bool = False,
) -> mujoco.MjSpec:
    """The experiment's ``scene.xml`` under ``FULL_COLLISION``, on the stage, with a
    contact sensor per foot on the desktop, the ladder env's sim settings, the servo
    filters and BAM's stiff servo friction; ``tracing`` leaves out the last three and
    the mesh copies."""
    spec = mujoco.MjSpec.from_file(str(root / SCENE_XML))
    for key in list(spec.keys):
        spec.delete(key)
    _common.full_collision(spec)
    _add_stage(spec, stage)
    for side in ("left", "right"):
        spec.add_sensor(
            name=f"{side}_foot_on_desk",
            type=mujoco.mjtSensor.mjSENS_CONTACT,
            objtype=mujoco.mjtObj.mjOBJ_GEOM,
            objname=f"{side}_foot_collision",
            reftype=mujoco.mjtObj.mjOBJ_GEOM,
            refname="desktop",
            intprm=[1, 0, 1],  # found, unreduced, one contact
        )
    if not tracing:
        # ``make_microduck_ladder_env_cfg``'s solver iterations over the velocity env's.
        sim = VELOCITY_ENV.sim
        sim = dataclasses.replace(
            sim,
            mujoco=dataclasses.replace(sim.mujoco, iterations=30, ls_iterations=50),
        )
        apply_mjlab_sim_options(spec, sim)
        _common.filter_servos(spec, SERVO_FILTER_S)
        for name in _common.servo_joints(spec):
            spec.joint(name).solref_friction = STIFF_SOLREF_FRICTION
            spec.joint(name).solimp_friction = STIFF_SOLIMP_FRICTION
        _common.own_collision_meshes(spec)
    _common.set_only_keyframe(spec, stand_pose, (0, 0, SPAWN_HEIGHT, 1, 0, 0, 0))
    return spec


def _handoff_policy(
    getup: onnx.ModelProto, climber: onnx.ModelProto | None = None
) -> onnx.ModelProto:
    """Both actors and upstream's handoff as one recurrent graph, since picking a policy
    in the browser restarts the sim.

    It reads ``actor`` (the 34 proprioceptive values), ``supervisor`` (1.0 once the
    supervisor would switch) and the ``[1, 128]`` carry: the switch latch, the previous
    raw output and the previous executed offset. Both actors see those 34, the previous
    raw output and 13 zeros. After the switch the get-up's output is smoothed from the
    last executed offset and pulled ``1 - GETUP_KP_RATIO`` of the way back to the
    measured joint position, which stands in for the lower servo gain. With no climber
    it is the get-up alone, switched from the start, and reads no ``supervisor``.
    """
    names = next(p.value for p in getup.metadata_props if p.key == "joint_names")
    alpha = [
        HEAD_ALPHA if name.startswith(("neck", "head")) else LEG_ALPHA
        for name in names.split(",")
    ]
    constants = {
        "alpha": np.array([alpha], np.float32),
        "kp_ratio": np.float32(GETUP_KP_RATIO),
        "always": np.ones((1, 1), np.float32),
        "zeros13": np.zeros((1, 13), np.float32),
        "zeros99": np.zeros((1, CARRY - 29), np.float32),
        "axis": np.array([1]),
        **{f"at{i}": np.array([i]) for i in (1, 6, 15, 20, 29)},
    }
    nodes = []

    def node(op, inputs, output, **attrs):
        nodes.append(helper.make_node(op, inputs, [output], **attrs))
        return output

    def cut(source, start, end, output):
        return node("Slice", [source, f"at{start}", f"at{end}", "axis"], output)

    previous_raw = cut("adapt_hx", 1, 15, "previous_raw")
    previous_executed = cut("adapt_hx", 15, 29, "previous_executed")
    joint_pos = cut("actor", 6, 20, "joint_pos")
    obs = node("Concat", ["actor", previous_raw, "zeros13"], "obs", axis=1)
    actors = [onnx.compose.add_prefix(getup, "getup/")]
    if climber is not None:
        actors.append(onnx.compose.add_prefix(climber, "climber/"))
    for actor in actors:
        node("Identity", [obs], actor.graph.input[0].name)
        nodes.extend(actor.graph.node)
    raw = actors[0].graph.output[0].name
    previous_weight = node("Sub", ["always", "alpha"], "previous_weight")
    executed = node(
        "Add",
        [
            node("Mul", ["alpha", raw], "smoothed_raw"),
            node("Mul", [previous_weight, previous_executed], "smoothed_previous"),
        ],
        "smoothed",
    )
    error = node("Sub", [executed, joint_pos], "error")
    action = node(
        "Add", [joint_pos, node("Mul", ["kp_ratio", error], "softened")], "target"
    )
    inputs = [helper.make_tensor_value_info("actor", TensorProto.FLOAT, [1, 34])]
    switched = "always"
    if climber is not None:
        inputs.append(
            helper.make_tensor_value_info("supervisor", TensorProto.FLOAT, [1, 1])
        )
        constants.update(at0=np.array([0]), half=np.float32(0.5))
        was_switched = cut("adapt_hx", 0, 1, "was_switched")
        switched = node("Max", [was_switched, "supervisor"], "switched")
        on = node("Greater", [switched, "half"], "on")
        climbed = actors[1].graph.output[0].name
        raw, executed, action = (
            node("Where", [on, value, climbed], f"{value}_or_climbed")
            for value in (raw, executed, action)
        )
    node("Concat", [switched, raw, executed, "zeros99"], "next_adapt_hx", axis=1)
    inputs.append(
        helper.make_tensor_value_info("adapt_hx", TensorProto.FLOAT, [1, CARRY])
    )
    graph = helper.make_graph(
        nodes,
        "desk_climb_handoff",
        inputs,
        [
            helper.make_tensor_value_info(action, TensorProto.FLOAT, [1, 14]),
            helper.make_tensor_value_info(
                "next_adapt_hx", TensorProto.FLOAT, [1, CARRY]
            ),
        ],
        [
            *(i for actor in actors for i in actor.graph.initializer),
            *(numpy_helper.from_array(np.asarray(v), k) for k, v in constants.items()),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=getup.opset_import, ir_version=getup.ir_version
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def servo_load_friction(env, env_ids, *, asset_cfg: SceneEntityCfg, **_) -> None:
    """BAM's gearbox friction on each servo's own torque, as its ``frictionloss``.

    Training's ``BamActuator`` rewrites it every physics step, this every control step.
    Its terms on the external load are left out: they read ``qfrc_bias`` and
    ``qfrc_constraint``, which neither mjlab's entity data nor the browser serves."""
    asset = env.scene[asset_cfg.name]
    joints = asset_cfg.joint_ids
    speed = asset.data.joint_vel[:, joints].abs()
    friction = (
        FRICTION_BASE
        + FRICTION_STRIBECK * torch.exp(-((speed / DTHETA_STRIBECK) ** STRIBECK_ALPHA))
        + LOAD_FRICTION_MOTOR * asset.data.qfrc_actuator[:, joints].abs()
    )
    dofs = asset.indexing.joint_v_adr[joints]
    env.sim.model.dof_frictionloss[env_ids[:, None], dofs] = friction[env_ids]


def desk_handoff(
    env,
    *,
    asset_cfg: SceneEntityCfg,
    sensor_rows: tuple[int, int],
    x_range: tuple[float, float],
    y_max: float,
    z_min: float,
    **_,
) -> torch.Tensor:
    """The supervisor's flag: a foot on the desktop, read off the two contact sensors,
    while the root is inside ``x_range``, within ``y_max`` of the centre line and above
    ``z_min``. Privileged: only the composite's switch reads it, never an actor."""
    root = env.scene[asset_cfg.name].data.root_link_pos_w
    on_desk = (env.sim.data.sensordata[:, list(sensor_rows)] > 0).any(dim=1)
    inside = (
        (root[:, 0] > x_range[0])
        & (root[:, 0] < x_range[1])
        & (root[:, 1].abs() < y_max)
        & (root[:, 2] > z_min)
    )
    return (on_desk & inside).float().unsqueeze(1)


def spawn(env, env_ids, *, asset_cfg: SceneEntityCfg, pos, noise, **_) -> None:
    """The floor spawn at ``pos``, at rest, with uniform ``noise`` (half-ranges of y,
    roll, pitch and yaw). Not mjlab's ``reset_root_state_uniform``: it offsets the
    trace env's default root pose, which is the origin rather than the keyframe's."""
    env_ids = event_fns.resolve_env_ids(env, env_ids)
    n = len(env_ids)
    half = torch.tensor(noise, device=env.device)
    draw = sample_uniform(-half, half, (n, len(noise)), env.device)
    position = torch.tensor(pos, device=env.device).repeat(n, 1)
    position[:, 1] += draw[:, 0]
    quat = quat_from_euler_xyz(draw[:, 1], draw[:, 2], draw[:, 3])
    asset = env.scene[asset_cfg.name]
    asset.write_root_link_pose_to_sim(torch.cat([position, quat], -1), env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(
        torch.zeros(n, 6, device=env.device), env_ids=env_ids
    )


def fell(env, *, limit_angle: float, height: float, asset_cfg: SceneEntityCfg, **_):
    """Tipped past ``limit_angle`` with the root below ``height``: the dive onto the
    desk tips it further, but above the desktop."""
    return term_fns.bad_orientation(
        env, limit_angle, asset_cfg
    ) & term_fns.root_height_below_minimum(env, height, asset_cfg)


def time_out(
    env, *, seconds: float, short_s: float, x_min: float, asset_cfg: SceneEntityCfg, **_
):
    """mjlab's ``time_out`` at ``seconds``, or at ``short_s`` for an attempt whose root
    is still short of ``x_min`` by then: draped over the desk edge, stalled on the
    ladder or standing on the floor."""
    elapsed = env.episode_length_buf * env.step_dt
    short = env.scene[asset_cfg.name].data.root_link_pos_w[:, 0] < x_min
    return (elapsed >= seconds) | ((elapsed >= short_s) & short)


def add_scenes(project: mjswan.ProjectHandle, root: Path) -> None:
    stand_pose = _stand_pose(root / _common.WALK_SCENE_XML)
    stage = _stage(root)
    spec = _scene_spec(root, stand_pose, stage)
    model = spec.compile()
    joint_names = _common.servo_joints(spec)
    joints = _common.joints_cfg(joint_names)
    robot = SceneEntityCfg(ENTITY)
    desk = model.geom("desktop")
    near, far = desk.pos[0] - desk.size[0], desk.pos[0] + desk.size[0]
    desk_top = float(desk.pos[2] + desk.size[2])
    switch_line = near + SWITCH_MARGIN

    scene = project.add_scene(name="Desk Climb", spec=spec, control_dt=CONTROL_DT)
    scene.add_attribution("3d-models", license=root / "LICENSE-HARDWARE")
    scene.set_viewer(_common.follow_cam(distance=1.2, elevation=-10.0, azimuth=115.0))
    scene.set_trace_env(
        build_single_entity_trace_env(
            partial(_scene_spec, root, stand_pose, stage, tracing=True),
            entity_name=ENTITY,
            control_dt=CONTROL_DT,
        )
    )
    getup = _common.hub_policy(POLICY_REPO_ID, "models/getup.onnx", POLICY_REVISION)
    climber = _common.hub_policy(POLICY_REPO_ID, "models/climber.onnx", POLICY_REVISION)
    # The 48 of proprioception less the last action, which the composite carries.
    actor = ObservationGroupCfg(
        terms={
            name: term
            for name, term in _common.proprioception(joints).items()
            if name != "actions"
        }
    )
    servo_friction = EventTermCfg(
        func=servo_load_friction,
        mode="interval",
        interval_range_s=(CONTROL_DT, CONTROL_DT),
        params={"asset_cfg": joints},
        label="Servo gearbox friction",
    )
    shared = dict(
        out_keys=["action", ["next", "adapt_hx"]],
        commands={},
        actions=_common.servo_action(),
        terminations={
            "fell": TerminationTermCfg(
                func=fell,
                params={"limit_angle": TIPPED, "height": desk_top, "asset_cfg": robot},
            ),
            "time_out": TerminationTermCfg(
                func=time_out,
                time_out=True,
                params={
                    "seconds": EPISODE_S,
                    "short_s": SHORT_OF_DESK_S,
                    "x_min": switch_line,
                    "asset_cfg": robot,
                },
            ),
        },
        policy_joint_names=joint_names,
        default_joint_pos=[stand_pose[name] for name in joint_names],
    )
    scene.add_policy(
        name="Climb",
        policy=_handoff_policy(getup, climber),
        in_keys=["actor", "supervisor", "adapt_hx"],
        observations={
            "actor": actor,
            "supervisor": ObservationGroupCfg(
                terms={
                    "handoff": ObservationTermCfg(
                        func=desk_handoff,
                        params={
                            "asset_cfg": robot,
                            "sensor_rows": tuple(
                                int(model.sensor_adr[model.sensor(name).id])
                                for name in ("left_foot_on_desk", "right_foot_on_desk")
                            ),
                            "x_range": (switch_line, far - EDGE_MARGIN),
                            "y_max": float(desk.size[1]) - EDGE_MARGIN,
                            "z_min": desk_top,
                        },
                    )
                }
            ),
        },
        events={
            "spawn": EventTermCfg(
                func=spawn,
                params={
                    "asset_cfg": robot,
                    "pos": (0.0, 0.0, SPAWN_HEIGHT),
                    "noise": SPAWN_NOISE,
                },
            ),
            "joints": EventTermCfg(
                func=event_fns.reset_joints_by_offset,
                params={
                    "position_range": (-JOINT_NOISE, JOINT_NOISE),
                    "velocity_range": (0.0, 0.0),
                    "asset_cfg": joints,
                },
            ),
            "servo_friction": servo_friction,
        },
        **shared,
    )
    # One of upstream's recorded handoffs, with the root moved from ladder to scene
    # coordinates.
    handoff = json.loads((root / EXPERIMENT / "evidence/switches.json").read_text())[
        GETUP_FROM
    ]
    qpos = np.array(handoff["qpos"])
    qpos[:3] = np.add(handoff["root_ladder_relative"], stage.origin)
    scene.add_policy(
        name="Get up",
        policy=_handoff_policy(getup),
        in_keys=["actor", "adapt_hx"],
        observations={"actor": actor},
        events={"servo_friction": servo_friction},
        initial_qpos=qpos.tolist(),
        initial_qvel=handoff["qvel"],
        **shared,
    )
