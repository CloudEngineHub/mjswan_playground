"""Upstream's reorient terms the browser cannot run as they are.

Upstream's EMA action keeps its smoothed target in Python; here each actuator's filter
holds it (``main._settle_filters``), so ``qpos_error`` reads the filter state. Upstream's
goal command and reset events draw with ``torch.rand``, which the tracer cannot replay;
their rewrites draw the same distributions with mjlab's ``sample_uniform``."""

from __future__ import annotations

import math
import types
from typing import Any

import mujoco
import numpy as np
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
    sample_uniform,
)
from mjswan import CommandBinding, register_command

_ROBOT = SceneEntityCfg("robot")
_OBJECT = SceneEntityCfg("object")


def _normalized(asset: Any, values: torch.Tensor, joint_ids: Any) -> torch.Tensor:
    soft = asset.data.soft_joint_pos_limits[:, joint_ids]
    center = 0.5 * (soft[..., 0] + soft[..., 1])
    half_range = 0.5 * (soft[..., 1] - soft[..., 0])
    return ((values - center) / (half_range + 1e-6)).clamp(-1.0, 1.0)


def joint_pos_target_error(
    env: Any, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
    """Upstream's ``joint_pos_target_error`` against the actuator filters' state, which
    is the hand's whole ``act``, in joint order."""
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    target = env.sim.data.act[:, asset_cfg.joint_ids]
    return _normalized(asset, joint_pos, asset_cfg.joint_ids) - _normalized(
        asset, target, asset_cfg.joint_ids
    )


def goal_rot_err_6d(
    env: Any,
    command_name: str,
    tag_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = _OBJECT,
) -> torch.Tensor:
    """Upstream's ``goal_rot_err_6d`` reading the goal as the command's value, which is
    what the browser serves of a traced command."""
    goal_quat_w = env.command_manager.get_command(command_name)
    cube_quat_w = env.scene[object_cfg.name].data.root_link_quat_w
    tag_quat_w = env.scene[tag_cfg.name].data.site_pose_w[:, tag_cfg.site_ids[0], 3:7]
    tag_inv = quat_inv(tag_quat_w)
    error = quat_mul(
        quat_mul(tag_inv, cube_quat_w), quat_inv(quat_mul(tag_inv, goal_quat_w))
    )
    rot = matrix_from_quat(error)
    return rot.reshape(*rot.shape[:-2], 9)[..., 3:]


def _random_quat(n: int, device: Any) -> torch.Tensor:
    """Upstream's ``random_quat_uniform`` (Shoemake), drawn with ``sample_uniform``."""
    u = sample_uniform(0.0, 1.0, (n, 3), device=device)
    a, b = 2.0 * math.pi * u[:, 1], 2.0 * math.pi * u[:, 2]
    outer, inner = torch.sqrt(1.0 - u[:, 0]), torch.sqrt(u[:, 0])
    return torch.stack(
        [
            outer * torch.sin(a),
            outer * torch.cos(a),
            inner * torch.sin(b),
            inner * torch.cos(b),
        ],
        dim=-1,
    )


def _sample_goal(self: Any) -> torch.Tensor:
    """A goal drawn uniformly on SO(3) in the wrist-tag frame, in world."""
    tag_quat_w = self._tag_quat_w.expand(self.num_envs, 4)
    return quat_mul(tag_quat_w, _random_quat(self.num_envs, self.device))


def _resample_goal(self: Any, env_ids: Any) -> None:
    """``InHandReorientCommand._resample_command`` for every env, on the state its
    update reads."""
    del env_ids
    self.goal_quat_w = _sample_goal(self)
    self.hold_counter = torch.zeros_like(self.hold_counter)
    self.goal_timer = torch.zeros_like(self.goal_timer)
    self.in_success_window = torch.zeros_like(self.in_success_window)
    self.window_timer = torch.zeros_like(self.window_timer)


def _update_goal(self: Any, env_ids: Any) -> None:
    """``InHandReorientCommand._update_command`` without its training-side counters.
    Each step draws a goal from upstream's distribution, taken only on a switch."""
    del env_ids
    cfg = self.cfg
    goal_timer = self.goal_timer + 1
    ori_err = quat_error_magnitude(self.object.data.root_link_quat_w, self.goal_quat_w)
    within = ori_err < cfg.success_threshold
    approaching = ~self.in_success_window
    hold = torch.where(
        approaching & within,
        self.hold_counter + 1,
        torch.where(
            approaching, torch.zeros_like(self.hold_counter), self.hold_counter
        ),
    )
    succeeded = approaching & (hold >= cfg.success_hold_steps)
    in_window = self.in_success_window | succeeded
    window_timer = torch.where(
        succeeded, torch.zeros_like(self.window_timer), self.window_timer
    )
    hold = torch.where(succeeded, torch.zeros_like(hold), hold)
    window_timer = torch.where(in_window, window_timer + 1, window_timer)
    switch = (
        in_window
        & (window_timer >= cfg.goal_switch_delay)
        & (goal_timer >= self._min_goal_steps())
    )
    self.goal_quat_w = torch.where(
        switch.unsqueeze(-1), _sample_goal(self), self.goal_quat_w
    )
    self.in_success_window = in_window & ~switch
    self.window_timer = torch.where(
        switch, torch.zeros_like(window_timer), window_timer
    )
    self.hold_counter = torch.where(switch, torch.zeros_like(hold), hold)
    self.goal_timer = torch.where(switch, torch.zeros_like(goal_timer), goal_timer)
    axes = matrix_from_quat(self.goal_quat_w) * _GOAL_ARROW_M
    self.goal_x, self.goal_y, self.goal_z = axes.unbind(dim=-1)


def _keyframe_site_quat(model: mujoco.MjModel, site: str) -> torch.Tensor:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    mujoco.mj_forward(model, data)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, data.site(site).xmat)
    return torch.tensor(quat, dtype=torch.float32)


def bind_goal_override(term: Any) -> None:
    """The rewrite reads the cube alone, since a traced command reads one entity. The
    mount has no reset event here, so the wrist tag holds its keyframe pose. The browser
    keeps a traced command's state in float32 or bool, so the counters count in float."""
    tag = term.robot.site_names[term.tag_site_id]
    model = term._env.sim.mj_model
    term._tag_quat_w = _keyframe_site_quat(model, f"{term.cfg.robot_entity_name}/{tag}")
    for counter in ("hold_counter", "goal_timer", "window_timer"):
        setattr(term, counter, getattr(term, counter).float())
    term.goal_x, term.goal_y, term.goal_z = torch.zeros(3, term.num_envs, 3).unbind()
    term._resample_command = types.MethodType(_resample_goal, term)
    term._update_command = types.MethodType(_update_goal, term)


#: Each goal axis is an arrow out of the cube in the color of the face that belongs
#: there: the +X, +Y and +Z textures' purple, red and green (cube.xml).
_GOAL_ARROW_M = 0.06
_GOAL_AXES = {
    "goal_x": [0.54, 0.47, 0.78, 1.0],
    "goal_y": [1.0, 0.0, 0.0, 1.0],
    "goal_z": [0.0, 0.63, 0.0, 1.0],
}

register_command(
    "InHandReorientCommandCfg",
    CommandBinding(
        state_fields=[
            "goal_quat_w",
            "hold_counter",
            "goal_timer",
            "in_success_window",
            "window_timer",
            *_GOAL_AXES,
        ],
        command_field="goal_quat_w",
        trace_override=bind_goal_override,
        viz=[
            {
                "shape": "arrow",
                "color": color,
                "width": 0.006,
                "origin": {"entity": "object", "field": "root_link_pos_w"},
                "vector": {"state": axis},
            }
            for axis, color in _GOAL_AXES.items()
        ],
    ),
)


def nan_detection(env: Any) -> torch.Tensor:
    """mjlab's ``nan_detection`` without the in-place ``|=`` the ONNX exporter refuses.
    An empty field (``sensordata`` without sensors) has nothing to flag."""
    data = env.sim.data
    found = torch.zeros(data.qpos.shape[0], dtype=torch.bool, device=data.qpos.device)
    for field in (
        data.qpos,
        data.qvel,
        data.qacc,
        data.qacc_warmstart,
        data.sensordata,
    ):
        if field.shape[-1]:
            found = (
                found | torch.isnan(field).any(dim=-1) | torch.isinf(field).any(dim=-1)
            )
    return found


def reset_joints_within_limits_range(
    env: Any,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    use_default_offset: bool = False,
    asset_cfg: SceneEntityCfg = _ROBOT,
    operation: str = "abs",
) -> None:
    """Upstream's ``reset_joints_within_limits_range`` as play runs it: absolute position
    ranges, about the default pose when ``use_default_offset``, clamped to the soft
    limits, and the default velocity."""
    if (
        velocity_range
        or operation != "abs"
        or any(None in r for r in position_range.values())
    ):
        raise ValueError(
            "The rewrite carries play's reset only: absolute position ranges."
        )
    asset = env.scene[asset_cfg.name]
    default = asset.data.default_joint_pos[env_ids]
    joint_pos = default.clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    soft = asset.data.soft_joint_pos_limits[env_ids]
    for pattern, (low, high) in position_range.items():
        joint_ids, _ = asset.find_joints(pattern)
        lower, upper = soft[:, joint_ids, 0], soft[:, joint_ids, 1]
        sampled = sample_uniform(low, high, lower.shape, device=env.device)
        if use_default_offset:
            sampled = sampled + default[:, joint_ids]
        joint_pos[:, joint_ids] = torch.clamp(sampled, min=lower, max=upper)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_object_orientation(
    env: Any,
    env_ids: torch.Tensor,
    pos_noise: float = 0.01,
    asset_cfg: SceneEntityCfg = _OBJECT,
) -> None:
    """Upstream's ``reset_object_orientation``: the cube at rest at its default position
    give or take ``pos_noise`` per axis, turned uniformly on SO(3)."""
    asset = env.scene[asset_cfg.name]
    n = len(env_ids)
    positions = (
        asset.data.default_root_state[env_ids, :3] + env.scene.env_origins[env_ids]
    )
    positions = positions + sample_uniform(
        -pos_noise, pos_noise, (n, 3), device=env.device
    )
    pose = torch.cat([positions, _random_quat(n, env.device)], dim=-1)
    asset.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(
        torch.zeros(n, 6, device=env.device), env_ids=env_ids
    )
