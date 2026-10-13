"""Upstream's terms that mjswan cannot trace as written, restated so it can. ``main.py``
puts them in the env config, so the build and parity both run them."""

from __future__ import annotations

import functools

import numpy as np
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform


def nan_detection(env) -> torch.Tensor:
    """mjlab's ``NanGuard.detect_nans``, without the in-place ``|=`` that
    ``torch.onnx.export`` has no symbolic for."""
    data = env.sim.data
    nan_mask = torch.zeros(
        data.qpos.shape[0], dtype=torch.bool, device=data.qpos.device
    )
    for t in (data.qpos, data.qvel, data.qacc, data.qacc_warmstart, data.sensordata):
        nan_mask = nan_mask | torch.isnan(t).any(dim=-1) | torch.isinf(t).any(dim=-1)
    return nan_mask


@functools.cache
def _fall_states(path: str, joint_names: tuple[str, ...]) -> np.ndarray:
    data = np.load(path)
    if tuple(data["joint_names"]) != joint_names:
        raise ValueError("Fall-state joint order does not match the robot joint order.")
    return data["qpos"]


def reset_from_fall_states(
    env,
    env_ids: torch.Tensor | None,
    path: str,
    z_offset: float = 0.01,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Upstream's class-based reset as a function: the table its ``__init__`` loads
    becomes a constant of the graph, and the yaw draws through ``sample_uniform``, the
    same uniform on [-pi, pi) as upstream's ``torch.rand``."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    asset = env.scene[asset_cfg.name]
    states = _fall_states(path, tuple(asset.joint_names))
    states = torch.as_tensor(states, device=env.device, dtype=torch.float32)
    n = len(env_ids)
    q = states[torch.randint(0, len(states), (n,), device=env.device)]

    pos = q[:, 0:3].clone()
    pos[:, 2] += z_offset
    pos += env.scene.env_origins[env_ids]
    zeros = torch.zeros(n, device=env.device)
    yaw = quat_from_euler_xyz(
        zeros, zeros, sample_uniform(-torch.pi, torch.pi, (n,), env.device)
    )
    quat = quat_mul(yaw, q[:, 3:7])

    asset.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(
        torch.zeros(n, 6, device=env.device), env_ids=env_ids
    )
    asset.write_joint_state_to_sim(
        q[:, 7:], torch.zeros_like(q[:, 7:]), env_ids=env_ids
    )
