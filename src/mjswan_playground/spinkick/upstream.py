"""The spinkick checkout: the deployed checkpoint, and the clip it carries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper

from mjswan_playground._deps import CACHE_DIR, ensure_repo

REPO_URL = "https://github.com/mujocolab/g1_spinkick_example.git"
#: The commit `pyproject.toml` pins the `spinkick` extra to.
REPO_COMMIT = "f1c4b230bd1a76c44e801e49f079212b6c6487da"
POLICY_ONNX = "spinkick_safe.onnx"

#: mjlab's motion `.npz` fields, which its tracking exporter also names the outputs.
CLIP_FIELDS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
_CACHE = CACHE_DIR / "spinkick"


def resolve_root() -> Path:
    return ensure_repo(
        name="g1_spinkick_example",
        url=REPO_URL,
        commit=REPO_COMMIT,
        marker=POLICY_ONNX,
        root_env_var="MJSWAN_SPINKICK_ROOT",
    )


def ensure_clip(policy: onnx.ModelProto, env_cfg: Any) -> Path:
    """The reference motion as an mjlab motion ``.npz``, converted once into the cache.

    Upstream publishes the clip only to a W&B registry, but mjlab's tracking exporter
    bakes it into the ONNX: one table per field, every frame, the 14 tracked bodies
    only. ``MotionLoader`` indexes the robot's full body list, so the rest are rebuilt
    the way ``csv_to_npz`` builds a clip, kept only if its tracked bodies match.
    """
    path = _CACHE / f"{Path(POLICY_ONNX).stem}-{REPO_COMMIT[:7]}.npz"
    if path.exists():
        return path
    baked = _baked_clip(policy)
    clip, tracked = _replay(baked, env_cfg)
    error = max(
        float(np.abs(clip[name][:, tracked] - baked[name]).max())
        for name in CLIP_FIELDS
        if name.startswith("body_")
    )
    if error > 1e-4:
        raise RuntimeError(
            f"Rebuilt clip disagrees with the exported one by {error:.2e}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **clip)
    return path


def _baked_clip(policy: onnx.ModelProto) -> dict[str, np.ndarray]:
    """Each motion output's table: the constant its ``Gather`` indexes by time step."""
    producers = {out: node for node in policy.graph.node for out in node.output}
    constants = {
        init.name: numpy_helper.to_array(init) for init in policy.graph.initializer
    }
    for node in policy.graph.node:
        if node.op_type == "Constant":
            constants[node.output[0]] = numpy_helper.to_array(
                helper.get_attribute_value(node.attribute[0])
            )
    return {name: constants[producers[name].input[0]] for name in CLIP_FIELDS}


def _replay(
    baked: dict[str, np.ndarray], env_cfg: Any
) -> tuple[dict[str, Any], list[int]]:
    """``csv_to_npz``'s loop over the baked frames, and the tracked bodies' indices."""
    from mjlab.scene import Scene
    from mjlab.sim.sim import Simulation, SimulationCfg

    scene = Scene(env_cfg.scene, device="cpu")
    sim = Simulation(
        num_envs=1, cfg=SimulationCfg(), model=scene.compile(), device="cpu"
    )
    scene.initialize(sim.mj_model, sim.model, sim.data)
    scene.reset()
    robot = scene["robot"]

    frames = {name: torch.tensor(table) for name, table in baked.items()}
    log: dict[str, list[np.ndarray]] = {name: [] for name in CLIP_FIELDS}
    for frame in range(len(frames["joint_pos"])):
        # Body 0 of the tracked list is the pelvis, the robot's root.
        root = robot.data.default_root_state.clone()
        root[:, 0:3] = frames["body_pos_w"][frame, 0]
        root[:, 3:7] = frames["body_quat_w"][frame, 0]
        root[:, 7:10] = frames["body_lin_vel_w"][frame, 0]
        root[:, 10:13] = frames["body_ang_vel_w"][frame, 0]
        robot.write_root_state_to_sim(root)
        robot.write_joint_state_to_sim(
            frames["joint_pos"][frame : frame + 1],
            frames["joint_vel"][frame : frame + 1],
        )
        sim.forward()
        data = robot.data
        for name, value in (
            ("joint_pos", data.joint_pos),
            ("joint_vel", data.joint_vel),
            ("body_pos_w", data.body_link_pos_w),
            ("body_quat_w", data.body_link_quat_w),
            ("body_lin_vel_w", data.body_link_lin_vel_w),
            ("body_ang_vel_w", data.body_link_ang_vel_w),
        ):
            log[name].append(value[0].cpu().numpy().copy())

    clip = {
        "fps": [round(1.0 / (env_cfg.sim.mujoco.timestep * env_cfg.decimation))],
        **{name: np.stack(values) for name, values in log.items()},
    }
    motion = env_cfg.commands["motion"]
    return clip, robot.find_bodies(motion.body_names, preserve_order=True)[0]
