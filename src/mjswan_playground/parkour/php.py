"""PHP's G1: five teachers tracking a clip over its boxes, and the depth student."""

from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from mjlab.entity import EntityCfg
from mjlab.sensor import ObjRef, RayCastSensorCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from onnx import TensorProto, helper, numpy_helper

from mjswan_playground._gait_clock import GaitClockCommandCfg
from mjswan_playground._trace import nan_detection

from . import terms, upstream

#: The student's camera crop, rows ``2:`` and columns ``4:-4`` of 60 x 106.
STUDENT_ROWS, STUDENT_COLS = (2, 60), (4, 102)
CLIP_CLOCK = "clip_clock"


def env_cfg(task_id: str, clip: Path) -> Any:
    """Upstream's play config as the browser runs it: mjlab's ``MotionCommand`` over the
    clip in its layout, from the first frame, and boxes where the clip's terrain variant
    0 puts them, as play's ``fixed_variant_id=0`` does."""
    from src.tasks.php.dataset import read_terrain
    from src.tasks.php.mdp.terrain import box_spec

    cfg = load_env_cfg(task_id, play=True)
    php = cfg.commands["motion"]
    robot = cfg.scene.entities["robot"].spec_fn().compile()
    npz = upstream.clip_npz(
        clip,
        [robot.joint(i).name for i in range(robot.njnt) if robot.jnt_type[i] != 0],
        [robot.body(i).name for i in range(1, robot.nbody)],
    )
    cfg.commands["motion"] = MotionCommandCfg(
        entity_name=php.entity_name,
        motion_file=str(npz),
        anchor_body_name=php.anchor_body_name,
        body_names=php.body_names,
        resampling_time_range=php.resampling_time_range,
        debug_vis=php.debug_vis,
        pose_range=php.pose_range,
        velocity_range=php.velocity_range,
        joint_position_range=php.joint_position_range,
        sampling_mode="start",
        viz=php.viz,
    )
    for name in [n for n in cfg.scene.entities if n.startswith("php_box_")]:
        del cfg.scene.entities[name]
    for i, box in enumerate(read_terrain(clip)[:, 0]):
        cfg.scene.entities[f"php_box_{i}"] = EntityCfg(
            spec_fn=functools.partial(_box, box_spec, tuple(box[7:] / 2)),
            init_state=EntityCfg.InitialStateCfg(
                pos=tuple(box[:3]), rot=tuple(box[3:7])
            ),
        )
    # Declares the box fields `place_boxes` resizes; static boxes resize nothing.
    cfg.events.pop("php_box_fields")
    # Upstream's clip end, a time-out: mjswan's tracking command restarts the episode
    # when its clip loops.
    cfg.terminations.pop("motion_end")
    cfg.terminations["nan"].func = nan_detection
    return cfg


def _box(box_spec: Any, half_size: tuple[float, ...]) -> Any:
    spec = box_spec()
    spec.geoms[0].size = half_size
    return spec


def student_cfg(cfg: Any, clip: Path) -> Any:
    """The student's camera as raycasts, and its ``vel_cmd`` read off a clip clock.

    Its two groups are renamed ``proprio`` and ``depth``: mjswan refuses an actor its
    runner gives two groups, while the exported student takes them as two inputs."""
    from src.tasks.php.mdp.depth import FOVY

    camera = next(s for s in cfg.scene.sensors if s.name == "depth_camera")
    cfg.scene.sensors = (
        *(s for s in cfg.scene.sensors if s is not camera),
        RayCastSensorCfg(
            name=camera.name,
            frame=ObjRef(type="body", name="torso_link", entity="robot"),
            pattern=terms.CameraRaysCfg(
                pos=camera.pos,
                quat=camera.quat,
                fovy=FOVY,
                width=camera.width,
                height=camera.height,
                rows=STUDENT_ROWS,
                cols=STUDENT_COLS,
            ),
            ray_alignment="base",
            # The camera renders every geom in its groups, its own body's included.
            exclude_parent_body=False,
            include_geom_groups=camera.enabled_geom_groups,
            max_distance=100.0,
        ),
    )
    shape = (STUDENT_ROWS[1] - STUDENT_ROWS[0], STUDENT_COLS[1] - STUDENT_COLS[0])
    focal = 0.5 * camera.height / math.tan(math.radians(FOVY) / 2)
    y = (torch.arange(*STUDENT_ROWS) + 0.5 - camera.height / 2) / focal
    x = (torch.arange(*STUDENT_COLS) + 0.5 - camera.width / 2) / focal
    depth = cfg.observations["student_depth"].terms["depth"]
    depth.func = terms.php_depth
    depth.params = {
        "sensor_name": camera.name,
        "shape": shape,
        "ray_to_z": torch.rsqrt(1 + y[:, None] ** 2 + x[None, :] ** 2),
        "resize": terms.php_resize(shape[1], 87),
        "gaussian_blur": depth.params["gaussian_blur"],
    }
    with np.load(clip) as src:
        rows = src["vel_cmd"][:, :11].astype(np.float32)
    # The clock reads 0 on an episode's first step, where upstream's frame is already 1.
    table = torch.from_numpy(np.concatenate([rows[1:], rows[-1:]]))
    cfg.commands[CLIP_CLOCK] = GaitClockCommandCfg(steps_per_period=len(table))
    velocity = cfg.observations["student_proprio"].terms["velocity_command"]
    velocity.func = terms.clip_velocity_command
    velocity.params = {"clock_name": CLIP_CLOCK, "table": table}
    groups = cfg.observations
    cfg.observations = {
        "proprio": groups["student_proprio"],
        "depth": groups["student_depth"],
    }
    return cfg


def student_policy(root: Path, lag: int) -> Any:
    """``model_23000.onnx`` reading the depth frame ``lag`` steps old, out of the frames
    it keeps in mjswan's carry, as ``PhpDepthImage`` keeps them: on an episode's first
    step every frame is the current one."""
    model = onnx.load(str(root / upstream.STUDENT_POLICY))
    graph = model.graph
    frame = 58 * 87
    for name, value in {
        "frame_end": [frame],
        "carry_end": [lag * frame],
        "start": [0],
        "axis": [1],
        "image_shape": [1, 1, 58, 87],
        "init_shape": [1, 1],
        "repeats": [1, lag],
    }.items():
        graph.initializer.append(numpy_helper.from_array(np.asarray(value), name))
    inputs = {i.name: i for i in graph.input}
    graph.input.remove(inputs["student_depth"])
    graph.input.extend(
        [
            helper.make_tensor_value_info("depth", TensorProto.FLOAT, [1, frame]),
            helper.make_tensor_value_info("is_init", TensorProto.BOOL, [1]),
            helper.make_tensor_value_info(
                "adapt_hx", TensorProto.FLOAT, [1, lag * frame]
            ),
        ]
    )
    wrapper = [
        ("Reshape", ["is_init", "init_shape"], "first"),
        ("Slice", ["adapt_hx", "start", "frame_end", "axis"], "oldest"),
        ("Slice", ["adapt_hx", "frame_end", "carry_end", "axis"], "newer"),
        ("Where", ["first", "depth", "oldest"], "lagged"),
        ("Reshape", ["lagged", "image_shape"], "student_depth"),
        ("Tile", ["depth", "repeats"], "primed"),
        ("Concat", ["newer", "depth"], "rolled"),
        ("Where", ["first", "primed", "rolled"], "next_adapt_hx"),
    ]
    for i, (op, args, out) in enumerate(wrapper):
        attrs = {"axis": 1} if op == "Concat" else {}
        graph.node.insert(i, helper.make_node(op, args, [out], **attrs))
    graph.output.append(
        helper.make_tensor_value_info(
            "next_adapt_hx", TensorProto.FLOAT, [1, lag * frame]
        )
    )
    onnx.checker.check_model(model)
    return model
