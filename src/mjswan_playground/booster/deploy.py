"""The booster_deploy checkout and its K1 tasks, the booster_assets checkout and its K1
model, and the checkpoints and clips converted for mjlab's joint and body order."""

from __future__ import annotations

import copy
import importlib
import inspect
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from mjswan_playground._deps import CACHE_DIR, ensure_repo

DEPLOY_URL = "https://github.com/BoosterRobotics/booster_deploy.git"
DEPLOY_COMMIT = "7bb1462e2ac742eba5c01659b815e2e99fbe617f"
ASSETS_URL = "https://github.com/BoosterRobotics/booster_assets.git"
ASSETS_COMMIT = "12a97516f57469078707afb587a5f84c1cb110d1"
_CACHE = CACHE_DIR / "booster"


def resolve_roots() -> tuple[Path, Path]:
    deploy = ensure_repo(
        name="booster_deploy",
        url=DEPLOY_URL,
        commit=DEPLOY_COMMIT,
        marker="tasks/beyond_mimic/beyond_mimic.py",
        root_env_var="MJSWAN_BOOSTER_DEPLOY_ROOT",
    )
    assets = ensure_repo(
        name="booster_assets",
        url=ASSETS_URL,
        commit=ASSETS_COMMIT,
        marker="robots/K1/K1_22dof.xml",
        root_env_var="MJSWAN_BOOSTER_ASSETS_ROOT",
    )
    return deploy, assets


def task_cfg(root: Path, name: str) -> Any:
    """A K1 task booster_deploy registers, with its gains and its policy's paths."""
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    for kind in ("beyond_mimic", "locomotion"):
        importlib.import_module(f"tasks.{kind}.robots.k1")
    from booster_deploy.utils.registry import get_task

    return get_task(name)


def _task_dir(task: Any) -> Path:
    """Where booster_deploy's policy resolves the task's checkpoint and clip paths."""
    return Path(inspect.getfile(task.policy.constructor)).parent


def _mlp(state: dict[str, np.ndarray], x: str) -> tuple[list, list, str]:
    """The checkpoint's `actor` on `x`: linear layers with an ELU between each two."""
    nodes, inits = [], []
    layers = sorted({int(k.split(".")[1]) for k in state if k.startswith("actor.")})
    for i, layer in enumerate(layers):
        inits += [
            numpy_helper.from_array(state[f"actor.{layer}.weight"], f"w{i}"),
            numpy_helper.from_array(state[f"actor.{layer}.bias"], f"b{i}"),
        ]
        nodes.append(
            helper.make_node("Gemm", [x, f"w{i}", f"b{i}"], [f"g{i}"], transB=1)
        )
        x = f"g{i}"
        if i < len(layers) - 1:
            nodes.append(helper.make_node("Elu", [x], [f"h{i}"], alpha=1.0))
            x = f"h{i}"
    return nodes, inits, x


def _io(name: str, shape: list, dtype: int = TensorProto.FLOAT) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def _model(
    task: Any, change: str, nodes: list, inits: list, inputs: list, outputs: list
) -> onnx.ModelProto:
    model = helper.make_model(
        helper.make_graph(nodes, "policy", inputs, outputs, inits),
        # IR 8 and opset 17, as torch exports them: onnxruntime reads no IR past 10.
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=8,
        # Apache-2.0 asks a changed file to say so.
        doc_string=(
            f"Booster Robotics' {Path(task.policy.checkpoint_path).name} "
            f"(booster_deploy, Apache-2.0), changed by mjswan_playground: {change}."
        ),
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def tracking_onnx(task: Any) -> onnx.ModelProto:
    """The TorchScript checkpoint as ONNX, reading and writing joints in the model's
    order: Isaac Lab's, which it was trained in, sits only between two `Gather`s."""
    robot = task.robot
    to_sim = [robot.joint_names.index(name) for name in robot.sim_joint_names]
    to_model = [robot.sim_joint_names.index(name) for name in robot.joint_names]
    n = len(to_sim)

    def joints(start: int) -> list[int]:
        return [start + j for j in to_sim]

    # The clip's joint positions and velocities, the anchor's orientation (6) and the
    # trunk's angular velocity (3), then joint positions, velocities and the last action.
    obs_order = [
        *joints(0),
        *joints(n),
        *range(2 * n, 2 * n + 9),
        *joints(2 * n + 9),
        *joints(3 * n + 9),
        *joints(4 * n + 9),
    ]

    module = torch.jit.load(_task_dir(task) / task.policy.checkpoint_path, "cpu")
    state = {k: v.numpy() for k, v in module.state_dict().items()}
    # `normalizer` or `obs_normalizer`, by exporter version: (x - mean) / (std + eps).
    norm = next(name for name, _ in module.named_children() if name != "actor")
    mean = state[f"{norm}._mean"].reshape(-1)
    scale = state[f"{norm}._std"].reshape(-1) + float(getattr(module, norm).eps)
    if len(obs_order) != mean.size:
        raise ValueError(
            f"Expected a {len(obs_order)}-wide observation, got {mean.size}."
        )

    mlp, weights, out = _mlp(state, "x")
    nodes = [
        helper.make_node("Gather", ["obs", "obs_order"], ["x0"], axis=1),
        helper.make_node("Sub", ["x0", "obs_mean"], ["x1"]),
        helper.make_node("Div", ["x1", "obs_scale"], ["x"]),
        *mlp,
        helper.make_node("Gather", [out, "action_order"], ["actions"], axis=1),
    ]
    inits = [
        numpy_helper.from_array(np.asarray(obs_order, np.int64), "obs_order"),
        numpy_helper.from_array(mean, "obs_mean"),
        numpy_helper.from_array(scale, "obs_scale"),
        numpy_helper.from_array(np.asarray(to_model, np.int64), "action_order"),
        *weights,
    ]
    return _model(
        task,
        "rebuilt as ONNX that reads and writes joints in the model's order",
        nodes,
        inits,
        [_io("obs", [1, mean.size])],
        [_io("actions", [1, n])],
    )


def walk_onnx(task: Any) -> onnx.ModelProto:
    """`LocomotionPolicy` as one recurrent ONNX. It reads one frame of the observation
    without the last action, and keeps the history, the last action and the filtered
    action in `adapt_hx`, which mjswan zeroes at each reset and seeds 128 wide."""
    cfg, robot = task.policy, task.robot
    # K1's config: the first frame fills the history, and `use_actor_module` skips the
    # checkpoint's normalizer, an identity anyway.
    assert cfg.history_init == "repeat" and cfg.use_actor_module
    names = list(cfg.policy_joint_names)
    n = len(names)
    frame = 9 + 3 * n
    width = cfg.actor_obs_history_length * frame
    carry = width + 2 * n
    fix = np.zeros(n, np.float32)
    fix[names.index(cfg.arm_action_fix_joint_name)] = cfg.arm_action_fix_offset

    def scalar(value: float) -> np.ndarray:
        return np.asarray(value, np.float32)

    def index(*values: int) -> np.ndarray:
        return np.asarray(values, np.int64)

    constants = {
        "carry_zeros": np.zeros((1, carry), np.float32),
        "carry_start": index(0),
        "carry_end": index(carry),
        "axis": index(1),
        "oldest_end": index(frame),
        "history_end": index(width),
        "last_end": index(width + n),
        # The angular velocity, gravity, command and joint positions, then velocities.
        "obs_scale": np.r_[np.ones(9 + n), np.full(n, cfg.obs_dof_vel_scale)].astype(
            np.float32
        ),
        "obs_min": scalar(-cfg.clip_observation),
        "obs_max": scalar(cfg.clip_observation),
        "repeats": index(1, cfg.actor_obs_history_length),
        "action_min": scalar(-cfg.clip_action),
        "action_max": scalar(cfg.clip_action),
        "fix": fix,
        "filter": scalar(cfg.action_filter),
        "filter_rest": scalar(1.0 - cfg.action_filter),
        "rest": np.zeros((1, 1), np.float32),
        # Each model joint's action: the policy's, or the rest pose for the head.
        "joint_order": np.asarray(
            [names.index(j) if j in names else n for j in robot.joint_names], np.int64
        ),
    }

    module = torch.jit.load(_task_dir(task) / cfg.checkpoint_path, "cpu")
    state = {k: v.numpy() for k, v in module.state_dict().items()}
    mlp, weights, out = _mlp(state, "history")

    def node(op: str, inputs: list[str], output: str, **attrs: Any) -> onnx.NodeProto:
        return helper.make_node(op, inputs, [output], **attrs)

    nodes = [
        node("Not", ["is_init"], "running"),
        node("Cast", ["running"], "keep", to=TensorProto.FLOAT),
        node("Concat", ["adapt_hx", "carry_zeros"], "hx_padded", axis=1),
        node("Slice", ["hx_padded", "carry_start", "carry_end", "axis"], "hx_sized"),
        node("Mul", ["hx_sized", "keep"], "hx"),
        node("Slice", ["hx", "oldest_end", "history_end", "axis"], "kept"),
        node("Slice", ["hx", "history_end", "last_end", "axis"], "last_action"),
        node("Slice", ["hx", "last_end", "carry_end", "axis"], "filtered_before"),
        node("Mul", ["obs", "obs_scale"], "scaled"),
        node("Concat", ["scaled", "last_action"], "frame_raw", axis=1),
        node("Clip", ["frame_raw", "obs_min", "obs_max"], "frame"),
        node("Concat", ["kept", "frame"], "rolled", axis=1),
        node("Tile", ["frame", "repeats"], "repeated"),
        node("Where", ["is_init", "repeated", "rolled"], "history"),
        *mlp,
        node("Clip", [out, "action_min", "action_max"], "raw_action"),
        node("Sub", ["raw_action", "fix"], "target"),
        node("Mul", ["target", "filter"], "target_part"),
        node("Mul", ["filtered_before", "filter_rest"], "filtered_part"),
        node("Add", ["target_part", "filtered_part"], "filtered"),
        node("Concat", ["filtered", "rest"], "with_rest", axis=1),
        node("Gather", ["with_rest", "joint_order"], "action", axis=1),
        node(
            "Concat",
            ["history", "raw_action", "filtered"],
            "next_adapt_hx",
            axis=1,
        ),
    ]
    inits = [
        *(numpy_helper.from_array(v, k) for k, v in constants.items()),
        *weights,
    ]
    return _model(
        task,
        "rebuilt as ONNX that keeps LocomotionPolicy's observation history, arm offset "
        "and action filter, and writes every model joint",
        nodes,
        inits,
        [
            _io("obs", [1, 9 + 2 * n]),
            _io("is_init", [1], TensorProto.BOOL),
            _io("adapt_hx", [1, "carry"]),
        ],
        [
            _io("action", [1, len(robot.joint_names)]),
            _io("next_adapt_hx", [1, carry]),
        ],
    )


def clip_npz(task: Any, env_cfg: Any) -> Path:
    """The task's clip replayed through mjlab, as `mjlab.scripts.csv_to_npz` does, so
    its joints and bodies come in the model's order; cached by commit."""
    source = _task_dir(task) / task.policy.motion_path
    out = _CACHE / f"{source.stem}-{DEPLOY_COMMIT[:7]}.npz"
    if not out.exists():
        _CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(out, **_replay(np.load(source), task.robot, env_cfg))
    return out


def _replay(clip: Any, robot_cfg: Any, env_cfg: Any) -> dict[str, np.ndarray]:
    from mjlab.scene import Scene
    from mjlab.sim.sim import Simulation
    from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul, yaw_quat

    fps = float(clip["fps"][0])
    # booster_deploy's clips carry Isaac Lab's joint and body order, the root body first.
    to_model = [robot_cfg.sim_joint_names.index(name) for name in robot_cfg.joint_names]
    joint_pos = torch.from_numpy(clip["joint_pos"][:, to_model])
    joint_vel = torch.from_numpy(clip["joint_vel"][:, to_model])
    root_pos = torch.from_numpy(clip["body_pos_w"][:, 0])
    root_quat = torch.from_numpy(clip["body_quat_w"][:, 0])
    root_lin_vel = torch.from_numpy(clip["body_lin_vel_w"][:, 0])
    root_ang_vel = torch.from_numpy(clip["body_ang_vel_w"][:, 0])

    # As booster_deploy loads it (`align_to_first_frame`): from the origin, facing +x.
    yaw_inv = quat_inv(yaw_quat(root_quat[:1])).repeat(root_quat.shape[0], 1)
    origin = root_pos[:1].clone()
    origin[:, 2] = 0.0
    root_pos = quat_apply(yaw_inv, root_pos - origin)
    root_quat = quat_mul(yaw_inv, root_quat)
    root_lin_vel = quat_apply(yaw_inv, root_lin_vel)
    root_ang_vel = quat_apply(yaw_inv, root_ang_vel)

    scene_cfg = copy.deepcopy(env_cfg.scene)
    scene_cfg.num_envs = 1
    scene = Scene(scene_cfg, device="cpu")
    sim = Simulation(num_envs=1, cfg=env_cfg.sim, model=scene.compile(), device="cpu")
    scene.initialize(sim.mj_model, sim.model, sim.data)
    robot = scene["robot"]

    log: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        )
    }
    for t in range(joint_pos.shape[0]):
        root_state = robot.data.default_root_state.clone()
        root_state[:, 0:3] = root_pos[t]
        root_state[:, :2] += scene.env_origins[:, :2]
        root_state[:, 3:7] = root_quat[t]
        root_state[:, 7:10] = root_lin_vel[t]
        root_state[:, 10:] = root_ang_vel[t]
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(joint_pos[t : t + 1], joint_vel[t : t + 1])
        sim.forward()
        scene.update(sim.mj_model.opt.timestep)
        data = robot.data
        log["joint_pos"].append(data.joint_pos[0].numpy().copy())
        log["joint_vel"].append(data.joint_vel[0].numpy().copy())
        log["body_pos_w"].append(data.body_link_pos_w[0].numpy().copy())
        log["body_quat_w"].append(data.body_link_quat_w[0].numpy().copy())
        log["body_lin_vel_w"].append(data.body_link_lin_vel_w[0].numpy().copy())
        log["body_ang_vel_w"].append(data.body_link_ang_vel_w[0].numpy().copy())
    return {"fps": np.asarray([fps]), **{k: np.stack(v) for k, v in log.items()}}
