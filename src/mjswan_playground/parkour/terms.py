"""Depth cameras as raycasts, and the student's clip command, as the browser can run them.

mjswan serves no rendered frames, so each depth camera becomes a ``RayCastSensor`` that
casts the camera's own pixel rays, and each depth term restates upstream's processing
on the ray distances. Upstream trained on mujoco-warp 3.5.0, whose camera reports each
pixel's distance along its ray, as a ray does. Only the pixels upstream keeps are cast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CameraRaysCfg:
    """A camera's pixel-centre rays, as MuJoCo renders them, from its pose on the
    sensor's frame body: rows ``rows[0]:rows[1]`` and columns ``cols[0]:cols[1]``."""

    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    fovy: float
    width: int
    height: int
    rows: tuple[int, int]
    cols: tuple[int, int]

    def generate_rays(self, mj_model: Any, device: str) -> tuple[Any, Any]:
        from mjlab.utils.lab_api.math import quat_apply

        del mj_model
        focal = 0.5 * self.height / math.tan(math.radians(self.fovy) / 2)
        x = (torch.arange(*self.cols) + 0.5 - self.width / 2) / focal
        y = (torch.arange(*self.rows) + 0.5 - self.height / 2) / focal
        # MuJoCo cameras look along -z with +y up, rows running down the image.
        grid_y, grid_x = torch.meshgrid(-y, x, indexing="ij")
        rays = torch.stack([grid_x, grid_y, -torch.ones_like(grid_x)], -1).view(-1, 3)
        rays = rays / rays.norm(dim=-1, keepdim=True)
        quat = torch.tensor(self.quat).expand(len(rays), 4)
        directions = quat_apply(quat, rays).to(device)
        offsets = torch.tensor(self.pos, device=device).expand(len(rays), 3)
        return offsets.contiguous(), directions


def _blur(depth: torch.Tensor, gaussian_blur: tuple[int, float]) -> torch.Tensor:
    """Upstream's Gaussian blur of a ``[B, 1, H, W]`` image, reflect-padded."""
    size, sigma = gaussian_blur
    coords = torch.arange(size, dtype=depth.dtype) - (size - 1) / 2
    kernel = torch.exp(-0.5 * (coords / sigma).square())
    kernel = kernel / kernel.sum()
    padding = size // 2
    return F.conv2d(
        F.pad(depth, (padding,) * 4, mode="reflect"),
        (kernel[:, None] * kernel[None, :]).view(1, 1, size, size),
    )


def pie_depth(
    env: Any,
    sensor_name: str,
    shape: tuple[int, int],
    cutoff_distance: float = 3.0,
    min_depth: float = 0.05,
    gaussian_blur: tuple[int, float] = (3, 1.0),
) -> torch.Tensor:
    """PIE's ``camera_depth``, on the cropped rays: one frame, flattened."""
    depth = env.scene[sensor_name].data.distances.view(-1, 1, *shape)
    depth = torch.where(depth > 0, depth, torch.full_like(depth, cutoff_distance))
    depth = _blur(depth, gaussian_blur)
    return (depth.clamp(min_depth, cutoff_distance) / cutoff_distance).flatten(1)


def php_depth(
    env: Any,
    sensor_name: str,
    shape: tuple[int, int],
    ray_to_z: torch.Tensor,
    resize: torch.Tensor,
    gaussian_blur: tuple[int, float] = (3, 1.0),
) -> torch.Tensor:
    """``PhpDepthImage.process`` uncorrupted, on the cropped rays: ray distance to
    optical depth, blur, then the bicubic resize to 58 x 87 as the linear map it is."""
    ranges = env.scene[sensor_name].data.distances.view(-1, 1, *shape)
    depth = torch.where(ranges > 0, ranges * ray_to_z, torch.full_like(ranges, 3.0))
    depth = _blur(depth.clamp(0.3, 3.0), gaussian_blur)
    depth = (depth @ resize).clamp(max=3.0)
    depth = torch.where(depth < 0.15, torch.full_like(depth, 3.0), depth)
    return ((depth - 0.3) / 2.7 - 0.5).flatten(1)


def php_resize(width: int, new_width: int) -> torch.Tensor:
    """The ``[width, new_width]`` matrix of upstream's antialiased bicubic resize along
    a row. Its height does not change, which leaves every column as it was."""
    basis = torch.eye(width).view(width, 1, 1, width)
    out = F.interpolate(
        basis, (1, new_width), mode="bicubic", align_corners=False, antialias=True
    )
    return out.view(width, new_width)


def clip_velocity_command(
    env: Any, clock_name: str, table: torch.Tensor
) -> torch.Tensor:
    """The student's ``velocity_command``: the clip's one-hot ``vel_cmd`` at its frame,
    which a clock command counts, as mjswan's tracking command keeps no frame index."""
    step = env.command_manager.get_term(clock_name).step_count
    return table[step[:, 0].long()]
