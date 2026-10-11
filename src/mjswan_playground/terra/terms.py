"""TERRA's observation, its fall and its start, as torch terms mjswan can trace.

``merc-s/TERRA-4B`` reads the 1459 values myosuite's ``TerrainObservation`` builds in
numpy over ``mujoco.MjData``. :func:`observation` restates them in torch from the fields
the browser serves: the heightmap from a raycast sensor, which the browser casts itself,
and every reference quantity from tables baked from the clip and indexed by sim time.

Layout, in order (``TerrainObservation.__call__``):

=====================================================  =====
root height, and gravity in the root frame                 4
non-root qpos                                             82
root velocity in the heading frame                         6
non-root qvel                                             82
per actuator: ctrl, act                                  708
touch sensor sums (r_foot, r_toes, l_foot, l_toes)         4
heightmap, 11 x 11 at 0.1 m around the pelvis            121
tracking errors against the next reference frame         204
the reference 20, 40, 60 and 80 frames ahead          4 x 62
=====================================================  =====
"""

from __future__ import annotations

from typing import Any, Callable

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import matrix_from_quat
from scipy.spatial.transform import Rotation

from mjswan_playground.musclemimic.upstream import _mat_to_rotvec

from .upstream import CTRL_DT

ENTITY = "robot"
#: The raycast sensor the heightmap reads.
SCAN = "heightmap"
#: The tutorial's fall: the pelvis site below this height.
MIN_PELVIS_HEIGHT = 0.5

#: The touch sensors ``FullbodyStateAdapter`` sums, in its order.
_TOUCH = ("r_foot", "r_toes", "l_foot", "l_toes")


def _id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    found = mujoco.mj_name2id(model, kind, f"{ENTITY}/{name}")
    if found < 0:
        raise ValueError(f"{name!r} is missing from the scene")
    return found


def _heading(rotation: torch.Tensor) -> torch.Tensor:
    """The yaw-only rotation of root rotations ``(B, 3, 3)``, as ``_heading`` takes it."""
    yaw = torch.atan2(
        rotation[:, 1, 0] - rotation[:, 0, 1], rotation[:, 0, 0] + rotation[:, 1, 1]
    )
    c, s = torch.cos(yaw), torch.sin(yaw)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([c, -s, zero, s, c, zero, zero, zero, one], -1).unflatten(
        -1, (3, 3)
    )


def _local(heading: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """World vectors ``(B, ..., 3)`` in the heading frame: ``heading.T @ v``."""
    return torch.einsum("bij,b...i->b...j", heading, vectors)


def _rotation_error(
    current: torch.Tensor, target: torch.Tensor, heading: torch.Tensor
) -> torch.Tensor:
    """``_rotation_error``: ``target @ current.T`` as a rotation vector in the heading
    frame. ``current`` and ``target`` are ``(B, ..., 3, 3)``."""
    error = target @ current.transpose(-1, -2)
    return _mat_to_rotvec(
        torch.einsum("bji,b...jk,bkl->b...il", heading, error, heading)
    )


def observation(model: mujoco.MjModel, clip: Any, cfg: Any) -> Callable[[Any], Any]:
    """TERRA's observation in the composed scene ``model``, tracking ``clip``:
    ``func(env) -> (B, 1459)``.

    ``cfg`` is the checkpoint's ``TerrainObsCfg``. The heightmap reads the
    :data:`SCAN` sensor, which casts ``cfg.grid`` rays at ``cfg.resolution`` around the
    root body, yaw-aligned, onto geom group 2.
    """
    root = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    if model.jnt_qposadr[root] != 0 or model.jnt_dofadr[root] != 0:
        raise ValueError("TERRA's root must be the scene's first joint")
    touch = torch.as_tensor(
        [model.sensor_adr[_id(model, mujoco.mjtObj.mjOBJ_SENSOR, n)] for n in _TOUCH]
    )
    sites = [_id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in cfg.sites]
    site_body = model.site_bodyid[sites]
    site_ix = torch.as_tensor(sites)
    body_ix = torch.as_tensor(site_body)
    com_ix = torch.as_tensor(model.body_rootid[site_body])

    # TERRA flattens its grid rows-first (forward offsets outer), mjlab columns-first.
    rows, cols = cfg.grid
    grid_order = torch.as_tensor(
        [j * rows + i for i in range(rows) for j in range(cols)]
    )

    # The reference, per frame. A lookahead past the end reads the last frame.
    frames = clip.num_frames
    rotation = Rotation.from_quat(np.roll(clip.qpos[:, 3:7], -1, axis=-1)).as_matrix()
    tables = {
        "position": clip.qpos[:, :3],
        "rotation": rotation,
        "velocity": np.concatenate(
            [clip.qvel[:, :3], np.einsum("nij,nj->ni", rotation, clip.qvel[:, 3:6])], -1
        ),
        "sites": clip.site_xpos[:, 1:] - clip.site_xpos[:, :1],
        "site_rotation": clip.site_xmat[:, 1:],
        "site_velocity": clip.site_velocity[:, 1:] - clip.site_velocity[:, :1],
    }
    ref = {
        k: torch.as_tensor(np.asarray(v, dtype=np.float32)) for k, v in tables.items()
    }
    offsets = torch.as_tensor(cfg.lookahead_steps)
    ahead = torch.as_tensor(
        [step / clip.frequency_hz for step in cfg.lookahead_steps[1:]],
        dtype=torch.float32,
    )

    def terra_observation(env: Any) -> torch.Tensor:
        data = env.sim.data
        qpos, qvel = data.qpos, data.qvel
        batch = qpos.shape[0]
        rotation = matrix_from_quat(qpos[:, 3:7])
        heading = _heading(rotation)

        # FullbodyStateAdapter, egocentric root.
        velocity = torch.cat(
            [
                _local(heading, qvel[:, :3]),
                _local(heading, (rotation @ qvel[:, 3:6, None])[..., 0]),
            ],
            -1,
        )
        state = [
            qpos[:, 2:3],
            -rotation[:, 2, :],  # rotation.T @ (0, 0, -1)
            qpos[:, 7:],
            velocity,
            qvel[:, 6:],
            # `[:]` unwraps the `TorchArray` a live env serves; `stack` won't in a list.
            torch.stack([data.ctrl[:], data.act[:]], -1).flatten(1),
            data.sensordata[:, touch],
        ]

        # The heightmap: ground height under each ray, relative to the root body. A ray
        # that hits nothing reads the floor at z = 0, as TerrainHeights does.
        scan = env.scene[SCAN].data
        ground = torch.where(scan.distances >= 0, scan.hit_pos_w[..., 2], 0.0)
        heightmap = ground[:, grid_order] - scan.pos_w[:, 2:3]

        frame = torch.floor(data.time / CTRL_DT + 0.5).long()
        requested = frame[:, None] + offsets
        index = requested.clamp(0, frames - 1)
        reference_velocity = ref["velocity"][index]
        goal_root = torch.cat(
            [
                _local(heading, ref["position"][index] - qpos[:, None, :3]),
                _rotation_error(rotation[:, None], ref["rotation"][index], heading),
                _local(heading, reference_velocity[..., :3]),
                _local(heading, reference_velocity[..., 3:]),
            ],
            -1,
        )

        # Site velocities as mj_objectVelocity gives them: world frame, at the site.
        site_pos = data.site_xpos[:, site_ix]
        cvel = data.cvel[:, body_ix]
        angular = cvel[..., :3]
        linear = cvel[..., 3:] - torch.cross(
            site_pos - data.subtree_com[:, com_ix], angular, dim=-1
        )
        site_velocity = torch.cat([angular, linear], -1)
        now = index[:, 0]
        position_error = ref["sites"][now] - (site_pos[:, 1:] - site_pos[:, :1])
        velocity_error = ref["site_velocity"][now] - (
            site_velocity[:, 1:] - site_velocity[:, :1]
        )
        immediate = [
            goal_root[:, 0, :6],
            goal_root[:, 0, 6:] - velocity,
            _local(heading, position_error).flatten(1),
            _rotation_error(
                data.site_xmat[:, site_ix[1:]], ref["site_rotation"][now], heading
            ).flatten(1),
            torch.cat(
                [
                    _local(heading, velocity_error[..., :3]),
                    _local(heading, velocity_error[..., 3:]),
                ],
                -1,
            ).flatten(1),
        ]
        future = torch.cat(
            [
                goal_root[:, 1:],
                _local(heading, ref["sites"][index[:, 1:]]).flatten(2),
                ahead.expand(batch, -1)[..., None],
                (requested[:, 1:] < frames).float()[..., None],
            ],
            -1,
        )
        return torch.cat([*state, heightmap, *immediate, future.flatten(1)], -1).float()

    return terra_observation


def fall(model: mujoco.MjModel) -> Callable[[Any], Any]:
    """The tutorial's fall, ``func(env) -> (B,)`` bool: the pelvis site below
    :data:`MIN_PELVIS_HEIGHT`."""
    site = _id(model, mujoco.mjtObj.mjOBJ_SITE, "pelvis_mimic")

    def pelvis_below(env: Any) -> torch.Tensor:
        return env.sim.data.site_xpos[:, site, 2] < MIN_PELVIS_HEIGHT

    return pelvis_below


def reset_to_start(qpos: np.ndarray) -> Callable[[Any, Any], None]:
    """A reset event that puts the body at ``qpos``, at rest: the tutorial's spawn, in
    place of the registered task's reset with joint noise. No RNG, so it traces to a
    constant-output graph."""
    root_state = torch.cat(
        [torch.as_tensor(qpos[:7], dtype=torch.float32), torch.zeros(6)]
    )
    joint_pos = torch.as_tensor(qpos[7:], dtype=torch.float32)

    def reset_to_start(env: Any, env_ids: Any) -> None:
        n = env.num_envs if env_ids is None else len(env_ids)
        entity = env.scene[ENTITY]
        entity.write_root_state_to_sim(root_state.expand(n, 13), env_ids=env_ids)
        entity.write_joint_state_to_sim(
            joint_pos.expand(n, -1),
            torch.zeros_like(joint_pos).expand(n, -1),
            env_ids=env_ids,
        )

    return reset_to_start
