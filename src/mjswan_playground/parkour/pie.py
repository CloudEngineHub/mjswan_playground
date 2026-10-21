"""PIE's Go2: stairs, a velocity command and a front depth camera."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import mjswan
import numpy as np
import onnx
from mjlab.sensor import ObjRef, RayCastSensorCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from onnx import TensorProto, helper, numpy_helper

from . import terms, upstream

#: Upstream's crop: ten columns off each side of the 106 x 60 image.
ROWS, COLS = (0, 60), (10, 96)
FRAME = (ROWS[1] - ROWS[0]) * (COLS[1] - COLS[0])
HIDDEN = 128


def env_cfg() -> tuple[Any, dict[str, int]]:
    """Upstream's play config as the browser runs it, and its ``DepthHistory`` params.

    The policy's three groups are renamed: mjswan refuses an actor its runner gives
    three groups, while the exported policy takes them as three inputs."""
    cfg = load_env_cfg(upstream.PIE, play=True)
    cfg.commands["twist"] = _mjlab_twist(cfg.commands["twist"])
    # Play draws a new terrain each run; the browser's is drawn once, with this seed.
    cfg.scene.terrain.terrain_generator.seed = 0
    camera = next(s for s in cfg.scene.sensors if s.name == "front_depth")
    cfg.scene.sensors = (
        *(s for s in cfg.scene.sensors if s is not camera),
        RayCastSensorCfg(
            name=camera.name,
            frame=ObjRef(type="body", name="base_link", entity="robot"),
            pattern=terms.CameraRaysCfg(
                pos=camera.pos,
                quat=camera.quat,
                fovy=camera.fovy,
                width=camera.width,
                height=camera.height,
                rows=ROWS,
                cols=COLS,
            ),
            ray_alignment="base",
            # The terrain alone: in mjlab the camera never sees the Go2 from its nose.
            include_geom_groups=(0,),
            max_distance=100.0,
        ),
    )
    depth = cfg.observations["camera"].terms["front_depth"]
    history = {
        key: depth.params[key]
        for key in ("frame_history_length", "update_period_steps")
    }
    depth.func = terms.pie_depth
    depth.params = {
        "sensor_name": camera.name,
        "shape": (ROWS[1] - ROWS[0], COLS[1] - COLS[0]),
        "cutoff_distance": depth.params["cutoff_distance"],
        "gaussian_blur": depth.params["gaussian_blur"],
    }
    groups = cfg.observations
    cfg.observations = {
        "proprio": groups["actor"],
        "history": groups["proprio_history"],
        "depth": groups["camera"],
    }
    return cfg, history


def _mjlab_twist(twist: Any) -> UniformVelocityCommandCfg:
    """Upstream's ``TerrainAwareVelocityCommandCfg`` as mjlab's class, with its ranges,
    for the env the build traces against: upstream's predates mjlab 1.6's commands."""
    fields = {field.name for field in dataclasses.fields(UniformVelocityCommandCfg)}
    values = {
        field.name: getattr(twist, field.name)
        for field in dataclasses.fields(twist)
        if field.name in fields and field.name not in ("ranges", "viz")
    }
    return UniformVelocityCommandCfg(
        **values,
        ranges=UniformVelocityCommandCfg.Ranges(**dataclasses.asdict(twist.ranges)),
    )


def twist(env_cfg: Any) -> Any:
    """The operator's twist, as upstream's own sim2sim drives PIE (`--lin-vel-x 0.5`,
    a gamepad or keys after): sliders over play's ranges, ``lin_vel_y`` held at zero."""
    ranges = env_cfg.commands["twist"].ranges
    return mjswan.velocity_command(
        lin_vel_x=ranges.lin_vel_x,
        lin_vel_y=ranges.lin_vel_y,
        ang_vel_z=ranges.ang_vel_z,
        default_lin_vel_x=0.5,
    )


def policy(root: Path, frame_history_length: int, update_period_steps: int) -> Any:
    """``policy.onnx`` with ``DepthHistory`` and the GRU state in mjswan's carry.

    It reads the current depth frame, and keeps the GRU's hidden state, the frames it
    holds and the control steps since the episode began in ``adapt_hx``, which mjswan
    zeroes at each reset. As ``DepthHistory``, it fills every frame on the first step
    and shifts a new one in every ``update_period_steps``."""
    assert frame_history_length == 2
    model = onnx.load(str(root / upstream.PIE_POLICY))
    graph = model.graph
    frames = slice(HIDDEN, HIDDEN + 2 * FRAME)
    carry = frames.stop + 1

    def node(op: str, inputs: list[str], output: str, **attrs: Any) -> Any:
        return helper.make_node(op, inputs, [output], **attrs)

    def const(name: str, value: Any, dtype: Any = np.float32) -> None:
        graph.initializer.append(
            numpy_helper.from_array(np.asarray(value, dtype), name)
        )

    for name, value in {
        "hx_start": [0],
        "frames_start": [frames.start],
        "newer_start": [frames.start + FRAME],
        "frames_stop": [frames.stop],
        "carry_stop": [carry],
        "axis": [1],
        "hidden_shape": [1, 1, HIDDEN],
        "flat_hidden": [1, HIDDEN],
        "history_shape": [1, 2, ROWS[1] - ROWS[0], COLS[1] - COLS[0]],
        "init_shape": [1, 1],
    }.items():
        const(name, value, np.int64)
    const("one", 1.0)
    const("period", float(update_period_steps))

    # Two of upstream's inputs become this wrapper's intermediates.
    inputs = {i.name: i for i in graph.input}
    for name in ("depth_history", "memory_h_in"):
        graph.input.remove(inputs[name])
    graph.input.extend(
        [
            helper.make_tensor_value_info("depth", TensorProto.FLOAT, [1, FRAME]),
            helper.make_tensor_value_info("is_init", TensorProto.BOOL, [1]),
            helper.make_tensor_value_info("adapt_hx", TensorProto.FLOAT, [1, carry]),
        ]
    )
    wrapper = [
        node("Not", ["is_init"], "running"),
        node("Cast", ["running"], "keep", to=TensorProto.FLOAT),
        node("Reshape", ["is_init", "init_shape"], "first"),
        node("Slice", ["adapt_hx", "frames_stop", "carry_stop", "axis"], "step_before"),
        node("Add", ["step_before", "one"], "step_next"),
        node("Mul", ["step_next", "keep"], "step"),
        node("Div", ["step", "period"], "periods"),
        node("Floor", ["periods"], "whole_periods"),
        node("Equal", ["periods", "whole_periods"], "update"),
        node("Slice", ["adapt_hx", "frames_start", "newer_start", "axis"], "older_was"),
        node("Slice", ["adapt_hx", "newer_start", "frames_stop", "axis"], "newer_was"),
        node("Where", ["update", "newer_was", "older_was"], "older_shifted"),
        node("Where", ["first", "depth", "older_shifted"], "older"),
        node("Where", ["update", "depth", "newer_was"], "newer"),
        node("Concat", ["older", "newer"], "frames", axis=1),
        node("Reshape", ["frames", "history_shape"], "depth_history"),
        node("Slice", ["adapt_hx", "hx_start", "frames_start", "axis"], "hidden_was"),
        node("Mul", ["hidden_was", "keep"], "hidden"),
        node("Reshape", ["hidden", "hidden_shape"], "memory_h_in"),
    ]
    outputs = {o.name: o for o in graph.output}
    graph.output.remove(outputs["memory_h_out"])
    graph.output.append(
        helper.make_tensor_value_info("next_adapt_hx", TensorProto.FLOAT, [1, carry])
    )
    graph.node.extend(
        [
            node("Reshape", ["memory_h_out", "flat_hidden"], "hidden_out"),
            node("Concat", ["hidden_out", "frames", "step"], "next_adapt_hx", axis=1),
        ]
    )
    for i, wrapped in enumerate(wrapper):
        graph.node.insert(i, wrapped)
    onnx.checker.check_model(model)
    return model
