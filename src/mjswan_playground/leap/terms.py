"""The terms the hand-cube task cannot take from upstream as they are. See ``README.md``.

* :func:`joint_pos_commanded`: the action's target, without the ``target_ids`` lookup
  the tracer does not serve.
* :func:`nan_detection`: mjlab's, without the in-place ``|=`` the ONNX exporter refuses.
* :func:`reset_from_grasp_cache`: upstream's, with the cache in its params, loaded by
  :func:`load_grasp_cache`.
"""

from __future__ import annotations

import numpy as np
import torch
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

# `sample_uniform` must be a module global: mjswan's RNG spy patches these to record
# draws, and an unseen draw is baked in as a constant: the same grasp, every time.
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform


def joint_pos_commanded(
    env, action_name: str = "joint_pos", asset_cfg=None
) -> torch.Tensor:
    """Upstream's ``joint_pos_commanded`` with every joint in the action and in
    ``asset_cfg``, in order: the action's integrated target. Upstream reads
    ``target_ids`` through a ``getattr`` default the tracer refuses, which would leave
    its graph reading ``joint_pos_target``."""
    return env.action_manager.get_term(action_name)._target


def nan_detection(env) -> torch.Tensor:
    """mjlab's ``nan_detection`` without the in-place ``|=`` the ONNX exporter refuses."""
    data = env.sim.data
    found = torch.zeros(data.qpos.shape[0], dtype=torch.bool, device=data.qpos.device)
    for field in (
        data.qpos,
        data.qvel,
        data.qacc,
        data.qacc_warmstart,
        data.sensordata,
    ):
        found = found | torch.isnan(field).any(dim=-1) | torch.isinf(field).any(dim=-1)
    return found


def load_grasp_cache(path: str) -> dict[str, torch.Tensor]:
    """Upstream's grasp cache as :func:`reset_from_grasp_cache` takes it: the cube sizes
    sorted, as upstream sorts them on every reset, and the poses in their order."""
    with np.load(path) as data:
        sizes = torch.from_numpy(np.asarray(data["cube_size"], np.float32).reshape(-1))
        poses = torch.from_numpy(np.asarray(data["cube_pose_rel"], np.float32))
    sizes, order = torch.sort(sizes)
    return {"cube_sizes": sizes, "cube_poses": poses[order]}


@requires_model_fields("geom_size")
def reset_from_grasp_cache(
    env,
    env_ids: torch.Tensor | None,
    cube_sizes: torch.Tensor,
    cube_poses: torch.Tensor,
    scale_list: tuple[float, ...],
    scale_jitter: float = 0.025,
    pose_range: dict[str, tuple[float, float]] | None = None,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube", geom_names=("cube_geom",)),
) -> None:
    """Upstream's ``reset_from_grasp_cache``: the cube resized and set in the cached
    grasp whose size is nearest. Upstream keeps the cache on the env, which the tracer
    does not serve, and finds the nearest size with ``searchsorted``, which ONNX has no
    operator for: the count of smaller sizes is the same index."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    cube = env.scene[cube_cfg.name]
    geom_ids = cube.indexing.geom_ids[cube_cfg.geom_ids]

    scales = torch.tensor(scale_list, device=env.device, dtype=cube_sizes.dtype)
    base_scale = scales[env_ids.long() % len(scales)]
    sampled_scale = sample_uniform(
        base_scale - scale_jitter,
        base_scale + scale_jitter,
        (len(env_ids),),
        device=env.device,
    )
    default_size = env.sim.get_default_field("geom_size")[geom_ids, 0].mean()
    selected_sizes = sampled_scale * default_size
    insert = (cube_sizes < selected_sizes[:, None]).sum(dim=-1)
    left = torch.clamp(insert - 1, min=0)
    right = torch.clamp(insert, max=cube_sizes.numel() - 1)
    left_diff = torch.abs(cube_sizes[left] - selected_sizes)
    right_diff = torch.abs(cube_sizes[right] - selected_sizes)
    nearest = torch.where(left_diff <= right_diff, left, right)

    env.sim.model.geom_size[env_ids[:, None], geom_ids[None, :], :] = selected_sizes[
        :, None, None
    ].expand(-1, len(geom_ids), 3)

    pose = cube_poses[nearest].clone()
    pose[:, 0:3] += env.scene.env_origins[env_ids]
    ranges = pose_range or {}
    draws = [
        sample_uniform(*ranges.get(axis, (0.0, 0.0)), len(env_ids), device=env.device)
        for axis in ("x", "y", "z", "roll", "pitch", "yaw")
    ]
    pose[:, 0:3] += torch.stack(draws[:3], dim=-1)
    pose[:, 3:7] = quat_mul(pose[:, 3:7], quat_from_euler_xyz(*draws[3:]))
    cube.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    cube.write_root_link_velocity_to_sim(
        torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids
    )
