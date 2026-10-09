"""Upstream's dances and gestures, each policy on the clip it was trained to track.

Upstream's tracking tasks read mjlab's ``MotionCommand``: the frame index, as a phase;
the reference at that frame and 2, 5 and 10 control steps ahead; and the anchor's tilt.
mjswan plays a ``MotionCommand`` natively and serves neither the index nor frames
ahead, so the clip becomes a traced command term: its tables are constants of the
command graph, which writes the rows the policy observes as state fields. The
observation and termination terms are restated over those fields, and a reset event
writes the clip's first frame where ``MotionCommand`` would.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjswan import CommandBinding, register_command

from . import servos, upstream

SCENE_ID = "Jumper-Dance"
#: Each policy's upstream task, in the order the scene lists them: the short gestures
#: first, the default among them.
POLICIES = {
    "Hello": "gesture_hello",
    "Bow": "gesture_bow",
    "Paw": "gesture_paw",
    "Salute": "gesture_salute",
    "Dance: Maze": "dance_maze",
    "Dance: Waist": "dance_waist",
    "Dance: Dream Wings": "dance_dream_wings",
    "Dance: Brazilian": "dance_brazilian",
    "Dance: Demo": "dance",
}

# A resampling time no episode reaches, so the clip restarts on episode reset alone.
_NEVER = 1.0e9


@dataclass(kw_only=True)
class ClipCommandCfg(CommandTermCfg):
    """``MotionCommandCfg`` played from its first frame: the clip at ``motion_file``,
    and the rows its observation and termination terms read."""

    motion_file: str
    body_names: tuple[str, ...]
    anchor_body_name: str
    support_body_names: tuple[str, ...]
    horizons: tuple[int, ...]
    resampling_time_range: tuple[float, float] = (_NEVER, _NEVER)

    def build(self, env: Any) -> "ClipCommand":
        return ClipCommand(self, env)


class ClipCommand(CommandTerm):
    cfg: ClipCommandCfg

    def __init__(self, cfg: ClipCommandCfg, env: Any) -> None:
        super().__init__(cfg, env)
        clip = np.load(cfg.motion_file)
        names = [str(n) for n in clip["body_names"]]
        tracked = [names.index(n) for n in cfg.body_names]
        anchor = tracked[cfg.body_names.index(cfg.anchor_body_name)]
        support = [names.index(n) for n in cfg.support_body_names]

        def table(array: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(
                np.asarray(array, dtype=np.float32), device=self.device
            )

        self.total = int(clip["joint_pos"].shape[0])
        self._joint_pos = table(clip["joint_pos"])
        self._joint_vel = table(clip["joint_vel"])
        self._anchor_pos = table(clip["body_pos_w"][:, anchor])
        self._anchor_quat = table(clip["body_quat_w"][:, anchor])
        self._support_z = table(clip["body_pos_w"][:, support, 2])
        n = self.num_envs
        self.time_steps = torch.zeros(n, 1, device=self.device)
        self.phase = torch.zeros(n, 2, device=self.device)
        self.joint_pos = torch.zeros(n, self._joint_pos.shape[1], device=self.device)
        self.joint_vel = torch.zeros_like(self.joint_pos)
        self.future_joint_pos = self.joint_pos.repeat(1, len(cfg.horizons))
        self.anchor_pos_w = torch.zeros(n, 3, device=self.device)
        self.anchor_quat_w = torch.zeros(n, 4, device=self.device)
        self.support_body_z = torch.zeros(n, len(support), device=self.device)
        self._read_frame()

    @property
    def command(self) -> torch.Tensor:
        return self.joint_pos

    def _read_frame(self) -> None:
        """The rows at ``time_steps``: ``MotionCommand``'s properties, as fields."""
        last = self.total - 1
        index = self.time_steps[:, 0].long()
        angle = self.time_steps * (2.0 * math.pi / max(self.total, 1))
        self.phase = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
        self.joint_pos = self._joint_pos[index]
        self.joint_vel = self._joint_vel[index]
        self.future_joint_pos = torch.cat(
            [self._joint_pos[(index + h).clamp(max=last)] for h in self.cfg.horizons],
            dim=-1,
        )
        self.anchor_pos_w = self._anchor_pos[index]
        self.anchor_quat_w = self._anchor_quat[index]
        self.support_body_z = self._support_z[index]

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        del env_ids
        self.time_steps = torch.zeros_like(self.time_steps)
        self._read_frame()

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        del env_ids
        # `clip_end` ends the episode on the last frame, where `MotionCommand` wraps.
        self.time_steps = (self.time_steps + 1.0).clamp(max=float(self.total - 1))
        self._read_frame()

    def _update_metrics(self) -> None:
        pass


register_command(
    "ClipCommandCfg",
    CommandBinding(
        state_fields=[
            "time_steps",
            "phase",
            "joint_pos",
            "joint_vel",
            "future_joint_pos",
            "anchor_pos_w",
            "anchor_quat_w",
            "support_body_z",
        ],
        command_field="joint_pos",
    ),
)

_ROBOT = SceneEntityCfg("robot")


def _clip(env: Any, command_name: str) -> ClipCommand:
    return env.command_manager.get_term(command_name)


# --- Upstream's observations (tasks/jumper/common/dance/observations.py) ---


def clip_phase(env: Any, command_name: str) -> torch.Tensor:
    return _clip(env, command_name).phase


def ref_joint_pos(
    env: Any, command_name: str, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return _clip(env, command_name).joint_pos - asset.data.default_joint_pos


def ref_joint_vel(env: Any, command_name: str) -> torch.Tensor:
    return _clip(env, command_name).joint_vel


def ref_future(
    env: Any,
    command_name: str,
    horizons: tuple[int, ...],
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    default = asset.data.default_joint_pos.repeat(1, len(horizons))
    return _clip(env, command_name).future_joint_pos - default


def ref_tilt_error(
    env: Any, command_name: str, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
    """The anchor is the root body, so the robot's half is its projected gravity."""
    asset = env.scene[asset_cfg.name]
    quat = _clip(env, command_name).anchor_quat_w
    down = torch.zeros_like(quat[:, :3]) + torch.tensor(
        [0.0, 0.0, -1.0], device=quat.device
    )
    return quat_apply_inverse(quat, down) - asset.data.projected_gravity_b


# --- mjlab's tracking terminations (mjlab/tasks/tracking/mdp/terminations.py) ---


def bad_anchor_pos_z_only(
    env: Any, command_name: str, threshold: float, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ref_z = _clip(env, command_name).anchor_pos_w[:, -1]
    return torch.abs(ref_z - asset.data.root_link_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: Any, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    gravity = asset.data.gravity_vec_w
    ref = quat_apply_inverse(_clip(env, command_name).anchor_quat_w, gravity)
    robot = quat_apply_inverse(asset.data.root_link_quat_w, gravity)
    return (ref[:, 2] - robot[:, 2]).abs() > threshold


def bad_motion_body_pos_z_only(
    env: Any,
    command_name: str,
    threshold: float,
    body_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """``body_pos_relative_w``'s height is the reference body's own: its anchor alignment
    moves the anchor to the reference's height and turns about z only. ``asset_cfg``
    resolves ``body_names``, in their order."""
    del body_names
    asset = env.scene[asset_cfg.name]
    robot_z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    error = torch.abs(_clip(env, command_name).support_body_z - robot_z)
    return torch.any(error > threshold, dim=-1)


def clip_end(env: Any, command_name: str, last: int) -> torch.Tensor:
    return _clip(env, command_name).time_steps[:, 0] >= float(last)


# --- MotionCommand's reset, from the first frame without jitter ---


def reset_to_clip_start(
    env: Any,
    env_ids: Any,
    root_state: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> None:
    asset = env.scene[asset_cfg.name]
    # The tracer refuses an output with no data dependence on its inputs, and these are
    # the clip's constants: a zero times the state each one overwrites ties them.
    data = asset.data
    limits = data.soft_joint_pos_limits
    pos = torch.clip(joint_pos + 0.0 * data.joint_pos, limits[:, :, 0], limits[:, :, 1])
    vel = joint_vel + 0.0 * data.joint_vel
    asset.write_joint_state_to_sim(pos, vel, env_ids=env_ids)
    asset.write_root_state_to_sim(
        root_state + 0.0 * data.root_link_pos_w[:, :1], env_ids=env_ids
    )


def _play_clip(env_cfg: Any, support_body_names: tuple[str, ...]) -> None:
    """Swap the task's ``MotionCommandCfg`` for the clip command, in place."""
    motion = env_cfg.commands["motion"]
    terms = env_cfg.observations["actor"].terms
    env_cfg.commands["motion"] = ClipCommandCfg(
        motion_file=motion.motion_file,
        body_names=tuple(motion.body_names),
        anchor_body_name=motion.anchor_body_name,
        support_body_names=support_body_names,
        horizons=tuple(terms["ref_future"].params["horizons"]),
    )
    for name, func in {
        "clip_phase": clip_phase,
        "ref_joint_pos": ref_joint_pos,
        "ref_joint_vel": ref_joint_vel,
        "ref_future": ref_future,
        "ref_tilt_error": ref_tilt_error,
    }.items():
        terms[name].func = func
    terminations = env_cfg.terminations
    terminations["anchor_pos"].func = bad_anchor_pos_z_only
    terminations["anchor_ori"].func = bad_anchor_ori
    feet = terminations["ee_body_pos"]
    feet.func = bad_motion_body_pos_z_only
    if tuple(feet.params["body_names"]) != support_body_names:
        raise ValueError("ee_body_pos bounds other bodies than the clip command reads.")
    feet.params["asset_cfg"] = SceneEntityCfg(
        "robot", body_names=support_body_names, preserve_order=True
    )
    clip = np.load(motion.motion_file)
    terminations["clip_end"] = TerminationTermCfg(
        func=clip_end,
        params={"command_name": "motion", "last": int(clip["joint_pos"].shape[0]) - 1},
        time_out=True,
    )
    names = [str(n) for n in clip["body_names"]]
    root = names.index(motion.body_names[0])
    root_state = np.concatenate(
        [
            clip["body_pos_w"][0, root],
            clip["body_quat_w"][0, root],
            clip["body_lin_vel_w"][0, root],
            clip["body_ang_vel_w"][0, root],
        ]
    ).astype(np.float32)
    env_cfg.events["reset_to_clip"] = EventTermCfg(
        func=reset_to_clip_start,
        mode="reset",
        params={
            "root_state": torch.as_tensor(root_state)[None],
            "joint_pos": torch.as_tensor(clip["joint_pos"][:1].astype(np.float32)),
            "joint_vel": torch.as_tensor(clip["joint_vel"][:1].astype(np.float32)),
        },
    )


def play_env_cfg(root: Path, task: str) -> Any:
    """``jumper.<task>``'s play config as the browser runs it."""
    env_cfg = upstream.play_env_cfg(root, task)
    upstream.drop_training_terms(env_cfg)
    feet = upstream.import_module(root, "tasks.jumper.common.dance.env").SUPPORT_FEET
    _play_clip(env_cfg, tuple(feet))
    servos.observe_reset_force(env_cfg, "actuator_force")
    return env_cfg
