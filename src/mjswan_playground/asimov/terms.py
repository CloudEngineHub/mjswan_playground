"""Upstream's terms that mjswan cannot trace as written, restated so it can. ``main.py``
puts them in the env config, so the build and parity both run them."""

from __future__ import annotations

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
    data = np.load(path)
    if tuple(data["joint_names"]) != tuple(asset.joint_names):
        raise ValueError("Fall-state joint order does not match the robot joint order.")
    states = torch.as_tensor(data["qpos"], device=env.device, dtype=torch.float32)
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


def reset_standing_with_push(
    env,
    env_ids: torch.Tensor | None,
    joint_noise: float,
    push_speed_range: tuple[float, float],
    push_ang_vel: float,
    yaw_rate: float,
    z_offset: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Upstream's safe-fall reset with every draw through ``sample_uniform``, from the
    same ranges: the tracer cannot follow ``Tensor.uniform_``."""
    from asimov_policies import robot

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    asset = env.scene[asset_cfg.name]
    n = len(env_ids)
    dev = env.device

    pos = torch.zeros(n, 3, device=dev)
    pos[:, 2] = robot.standing_heights().pelvis + 0.005 + z_offset
    pos += env.scene.env_origins[env_ids]
    zeros = torch.zeros(n, device=dev)
    yaw = quat_from_euler_xyz(
        zeros, zeros, sample_uniform(-torch.pi, torch.pi, (n,), dev)
    )
    asset.write_root_link_pose_to_sim(torch.cat([pos, yaw], dim=-1), env_ids=env_ids)

    heading = sample_uniform(0.0, 2 * torch.pi, (n,), dev)
    speed = sample_uniform(*push_speed_range, (n,), dev)
    vel = torch.cat(
        [
            (speed * torch.cos(heading))[:, None],
            (speed * torch.sin(heading))[:, None],
            zeros[:, None],
            sample_uniform(-push_ang_vel, push_ang_vel, (n, 2), dev),
            sample_uniform(-yaw_rate, yaw_rate, (n, 1), dev),
        ],
        dim=-1,
    )
    asset.write_root_link_velocity_to_sim(vel, env_ids=env_ids)

    q = asset.data.default_joint_pos[env_ids]
    q = q + sample_uniform(-joint_noise, joint_noise, q.shape, dev)
    limits = asset.data.soft_joint_pos_limits[env_ids]
    q = torch.clamp(q, limits[..., 0], limits[..., 1])
    asset.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
