"""Basketball: microduck balancing on top of a free-rolling size-7 basketball it cannot
see."""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import mjswan
import mujoco
import numpy as np
import onnx
from mjlab.envs.mdp import terminations as term_fns
from mjswan.managers.observation_manager import ObservationGroupCfg
from mjswan.managers.termination_manager import TerminationTermCfg
from mjswan.mjlab import apply_mjlab_sim_options, build_single_entity_trace_env
from onnx import TensorProto, helper, numpy_helper

from mjswan_playground._trace import CommandValues
from mjswan_playground.microduck.main import (
    CONTROL_DT,
    ENTITY,
    VELOCITY_ENV,
    _driven,
    _padded,
    _sliders,
    _stand_pose,
)

from . import _common

POLICY_REPO_ID = "HannesVonEssen/microduck-basketball"
POLICY_REVISION = "6e61a733aeff338bdd352148819bc99cf9a51491"

COLOURWAY_PY = "src/mjlab_microduck/robot/basketball.py"
BALL_ASSETS = "src/mjlab_microduck/robot/assets/basketball"

#: ``microduck_basketball_env_cfg``'s ball, pinned where upstream reads the radius from
#: ``MICRODUCK_BB_BALL_RADIUS``.
BALL_RADIUS = 0.12
BALL_MASS = 0.62
BALL_FRICTION = (1.2, 0.01, 0.001)
BALL_JOINT = "ball_free"
#: ``reset_basketball``'s spawn: the ball 1 mm above the floor, the trunk ``ROOT_HEIGHT``
#: plus 3 mm above its apex.
ROOT_HEIGHT = 0.125
BALL_POSE = (0.0, 0.0, BALL_RADIUS + 0.001, 1.0, 0.0, 0.0, 0.0)
ROOT_POSE = (0.0, 0.0, 2 * BALL_RADIUS + ROOT_HEIGHT + 0.003, 1.0, 0.0, 0.0, 0.0)

#: Shorter than ``_common.SERVO_FILTER_S``: under upstream's eval pushes the balance
#: holds at 10 to 20 ms and mostly fails from 30 ms.
SERVO_FILTER_S = 0.02

#: The play env's twist ranges.
TWIST_SLIDERS = (
    ("lin_vel_x", "Forward (m/s)", (-0.15, 0.15)),
    ("lin_vel_y", "Sideways (m/s)", (-0.1, 0.1)),
    ("ang_vel_z", "Turn (rad/s)", (-0.5, 0.5)),
)

#: ``basketball_fell`` measures from the ball, which the single-entity trace env lacks.
#: The ball's centre stays at its radius on the floor, so its 5 cm margin over the ball
#: becomes a trunk height; its 2D offset check is left out.
TERMINATIONS = {
    "tilted": TerminationTermCfg(
        func=term_fns.bad_orientation, params={"limit_angle": math.radians(55.0)}
    ),
    "off_ball": TerminationTermCfg(
        func=term_fns.root_height_below_minimum,
        params={"minimum_height": 2 * BALL_RADIUS + 0.05},
    ),
}


def _single_carry(policy: onnx.ModelProto) -> onnx.ModelProto:
    """The LSTM's ``(h, c)`` as the one ``[1, 2 * hidden]`` carry mjswan feeds back, h
    first.

    mjswan opens every episode with ``is_init`` and a ``[1, 128]`` zero carry, so the
    carry is zero-padded to width and zeroed while ``is_init`` holds."""
    graph = policy.graph
    obs, h_in, c_in = graph.input
    actions, h_out, c_out = graph.output
    hidden = h_in.type.tensor_type.shape.dim[-1].dim_value
    width = 2 * hidden
    constants = {
        "carry_zeros": np.zeros((1, width), np.float32),
        "carry_start": np.array([0]),
        "carry_end": np.array([width]),
        "carry_axis": np.array([1]),
        "carry_hc_shape": np.array([2, 1, 1, hidden]),
        "carry_shape": np.array([1, width]),
        "carry_h": np.array(0),
        "carry_c": np.array(1),
    }
    nodes = [
        helper.make_node(
            "Concat", ["adapt_hx", "carry_zeros"], ["carry_padded"], axis=1
        ),
        helper.make_node(
            "Slice",
            ["carry_padded", "carry_start", "carry_end", "carry_axis"],
            ["carry_sized"],
        ),
        helper.make_node("Not", ["is_init"], ["carry_kept"]),
        helper.make_node("Cast", ["carry_kept"], ["carry_keep"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["carry_sized", "carry_keep"], ["carry"]),
        helper.make_node("Reshape", ["carry", "carry_hc_shape"], ["carry_hc"]),
        helper.make_node("Gather", ["carry_hc", "carry_h"], [h_in.name], axis=0),
        helper.make_node("Gather", ["carry_hc", "carry_c"], [c_in.name], axis=0),
        *graph.node,
        helper.make_node("Concat", [h_out.name, c_out.name], ["carry_out"], axis=0),
        helper.make_node("Reshape", ["carry_out", "carry_shape"], ["next_adapt_hx"]),
    ]
    wrapped = helper.make_model(
        helper.make_graph(
            nodes,
            graph.name,
            [
                obs,
                helper.make_tensor_value_info("is_init", TensorProto.BOOL, [1]),
                helper.make_tensor_value_info(
                    "adapt_hx", TensorProto.FLOAT, [1, "carry"]
                ),
            ],
            [
                actions,
                helper.make_tensor_value_info(
                    "next_adapt_hx", TensorProto.FLOAT, [1, width]
                ),
            ],
            initializer=[
                *graph.initializer,
                *(numpy_helper.from_array(v, k) for k, v in constants.items()),
            ],
        ),
        opset_imports=policy.opset_import,
        ir_version=policy.ir_version,
    )
    helper.set_model_props(wrapped, {p.key: p.value for p in policy.metadata_props})
    onnx.checker.check_model(wrapped, full_check=True)
    return wrapped


def _add_ball(spec: mujoco.MjSpec, root: Path) -> None:
    """``_ball_spec``: an invisible collision sphere whose friction wins against the floor
    and the feet, under a textured mesh that carries the look."""
    assets = root / BALL_ASSETS
    spec.add_texture(
        name="basketball_tex",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        file=str(assets / "basketball.png"),
    )
    material = spec.add_material(
        name="basketball_mat", specular=0.15, shininess=0.15, reflectance=0.0
    )
    textures = list(material.textures)
    textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "basketball_tex"
    material.textures = textures
    spec.add_mesh(name="basketball_mesh", file=str(assets / "basketball.obj"))

    body = spec.worldbody.add_body(name="ball", pos=[0.0, 0.0, BALL_RADIUS])
    body.add_freejoint(name=BALL_JOINT)
    body.add_geom(
        name="ball_sphere",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[BALL_RADIUS, 0.0, 0.0],
        mass=BALL_MASS,
        rgba=[1.0, 1.0, 1.0, 0.0],
        group=3,
        friction=BALL_FRICTION,
        priority=1,
        condim=4,
    )
    body.add_geom(
        name="ball_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="basketball_mesh",
        material="basketball_mat",
        contype=0,
        conaffinity=0,
        mass=0.0,
    )


def _scene_spec(
    root: Path, stand_pose: dict[str, float], *, tracing: bool = False
) -> mujoco.MjSpec:
    """``scene.xml``, the all-collision model the play env loads, in the release's cream
    colourway, on the ball, with mjlab's velocity sim settings and the servo filters;
    ``tracing`` leaves out all of that but the model, as an mjlab entity is one free
    joint and nothing traced reads the rest.

    Collisions stay as the XML has them, as in ``swing``: mjlab 1.3.0's
    ``FULL_COLLISION`` disabled none of the unnamed hulls a falling duck hits the ball
    with.
    """
    spec = mujoco.MjSpec.from_file(str(root / _common.ALLCOLLISIONS_SCENE_XML))
    ball = None
    if not tracing:
        apply_mjlab_sim_options(spec, VELOCITY_ENV.sim)
        _common.filter_servos(spec, SERVO_FILTER_S)
        _common.own_collision_meshes(spec)
        colourway = _common.load_upstream(root, COLOURWAY_PY)
        colourway.basketball_robot_cfg(SimpleNamespace(spec_fn=lambda: spec)).spec_fn()
        # After the robot, which has to stay the scene's first body.
        _add_ball(spec, root)
        ball = {BALL_JOINT: BALL_POSE}
    _common.set_only_keyframe(spec, stand_pose, ROOT_POSE, extra_qpos=ball)
    return spec


def add_scenes(project: mjswan.ProjectHandle, root: Path) -> None:
    stand_pose = _stand_pose(root / _common.ALLCOLLISIONS_SCENE_XML)
    spec = _scene_spec(root, stand_pose)
    joint_names = _common.servo_joints(spec)

    scene = project.add_scene(name="Basketball", spec=spec, control_dt=CONTROL_DT)
    scene.add_attribution("3d-models", license=root / "LICENSE-HARDWARE")
    scene.set_viewer(_common.follow_cam(distance=1.3))
    scene.set_trace_env(
        build_single_entity_trace_env(
            partial(_scene_spec, root, stand_pose, tracing=True),
            entity_name=ENTITY,
            commands={"twist": CommandValues(3)},
        )
    )
    policy = _common.hub_policy(POLICY_REPO_ID, "policy.onnx", POLICY_REVISION)
    scene.add_policy(
        name="Balance",
        policy=_single_carry(policy),
        in_keys=["actor", "is_init", "adapt_hx"],
        out_keys=["action", ["next", "adapt_hx"]],
        commands={"twist": _sliders(TWIST_SLIDERS)},
        # robotd's 61 values: 48 of proprioception, then twist(3), head(4), body(6).
        observations=ObservationGroupCfg(
            terms={
                **_common.proprioception(_common.joints_cfg(joint_names)),
                "twist": _driven("twist"),
                "head_command": _padded(4),
                "body_command": _padded(6),
            }
        ),
        actions=_common.servo_action(),
        terminations=TERMINATIONS,
        policy_joint_names=joint_names,
        default_joint_pos=[stand_pose[name] for name in joint_names],
    )
