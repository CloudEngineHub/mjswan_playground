"""Upstream's five-leg walk, the left-front arm carried as a claw, as the browser runs it.

``jumper.five_foot`` walks on five legs and carries the left-front leg folded with its
claw open: an event writes that pose and its position target each reset, and in play an
operator's trigger drives the claw. The carried joints get biased position actuators
(:func:`servos.hold_unactuated`), and the claw stays open as untouched play leaves it.
The body-pose command is restated over ``sample_uniform`` and ``torch.where``.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import torch
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import sample_uniform
from mjswan import CommandBinding, register_command

from . import servos, upstream

SCENE_ID = "Jumper-Five-Foot"
TASK = "five_foot"


def _standing(self: Any) -> torch.Tensor:
    vel = self._env.command_manager.get_command(self.cfg.velocity_command_name)
    return torch.norm(vel[:, :3], dim=1, keepdim=True) < self.cfg.stand_threshold


def _resample_body_pose(self: Any, env_ids: Any) -> None:
    """``BodyPoseCommand._resample_command`` at ``N=1``, without ``Tensor.uniform_`` or
    per-env writes; it draws in upstream's order: pitch, roll, twist, then neutral."""
    del env_ids
    standing = _standing(self)
    n, device = self.num_envs, self.device
    axes = []
    for stand, move in (
        (self.cfg.stand_pitch, self.cfg.move_pitch),
        (self.cfg.stand_roll, self.cfg.move_roll),
        (self.cfg.stand_twist, self.cfg.move_twist),
    ):
        lo = torch.where(standing, stand[0], move[0])
        hi = torch.where(standing, stand[1], move[1])
        axes.append(lo + sample_uniform(0.0, 1.0, (n, 1), device=device) * (hi - lo))
    self.pose_target_b = torch.cat(axes, dim=-1)
    neutral = (
        sample_uniform(0.0, 1.0, (n, 1), device=device) <= self.cfg.rel_neutral_envs
    )
    self.is_neutral_env = neutral.float()


def _update_body_pose(self: Any, env_ids: Any = None) -> None:
    """``_update_command``: a neutral environment targets zero, and the observed command
    ramps toward the target at ``max_rate``."""
    del env_ids
    target = self.pose_target_b * (1.0 - self.is_neutral_env)
    self.pose_target_b = target
    step = self.cfg.max_rate * self._env.step_dt
    self.pose_command_b = self.pose_command_b + (target - self.pose_command_b).clamp(
        -step, step
    )


def bind_body_pose_override(term: Any) -> None:
    term.is_neutral_env = term.is_neutral_env.float().reshape(-1, 1)
    term._resample_command = types.MethodType(_resample_body_pose, term)
    term._update_command = types.MethodType(_update_body_pose, term)


def _body_pose_ui(cfg: Any) -> dict[str, Any]:
    """Upstream's operator axes, as sliders over the standing bands in the command's
    order. A slider sets the observed command itself, past the ramp."""
    inputs: list[dict[str, Any]] = [
        {"type": "checkbox", "name": "enabled", "label": "Enable", "default": False}
    ]
    for name, (lo, hi) in (
        ("pitch", cfg.stand_pitch),
        ("roll", cfg.stand_roll),
        ("twist", cfg.stand_twist),
    ):
        inputs.append(
            {
                "type": "slider",
                "name": name,
                "label": f"{name} (rad)",
                "min": lo,
                "max": hi,
                "step": 0.01,
                "default": 0.0,
                "enabled_when": "enabled",
            }
        )
    return {"inputs": inputs}


register_command(
    "BodyPoseCommandCfg",
    CommandBinding(
        state_fields=["pose_command_b", "pose_target_b", "is_neutral_env"],
        command_field="pose_command_b",
        trace_override=bind_body_pose_override,
        ui=_body_pose_ui,
    ),
)


def play_env_cfg(root: Path) -> tuple[Any, dict[str, tuple[int, ...]]]:
    """``jumper.five_foot``'s play config as the browser runs it."""
    pose = upstream.import_module(root, "tasks.jumper.five_foot.mdp.pose_command")
    env_cfg = upstream.play_env_cfg(root, TASK)
    upstream.drop_training_terms(env_cfg)
    # The claw's trigger and `play --hold` are step-mode events, and the arm's hold
    # writes a position target: the held actuators stand in.
    for name in ("gripper_teleop", "claw_hold", "hold_carried_arm"):
        env_cfg.events.pop(name, None)
    contract = upstream.contract(root, TASK)
    servos.hold_unactuated(env_cfg, contract)
    # Play's operator terms read a keyboard or a pad; the browser's panel does instead.
    env_cfg.commands["twist"] = upstream.as_base(
        UniformVelocityCommandCfg, env_cfg.commands["twist"]
    )
    env_cfg.commands["body_pose"] = upstream.as_base(
        pose.BodyPoseCommandCfg, env_cfg.commands["body_pose"]
    )
    offsets = upstream.unstride(env_cfg)
    servos.observe_reset_force(
        env_cfg, "actuator_force", hold=contract["unactuated_joints"]
    )
    return env_cfg, offsets
