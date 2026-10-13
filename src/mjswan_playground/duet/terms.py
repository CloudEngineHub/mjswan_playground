"""Upstream's ``BaseHeightCommand`` bound for the browser, and its arm generator recast.

Its update picks the walk or the squat target by reading the twist command, and a command
term cannot read another one in the browser. So the bound command holds the squat target
as its value and the ``height_command`` observation, which can read both commands, makes
that pick instead.

The arm generator, ``UpperBodyPoseAction``, is an action term mjswan does not run, so
``ArmPoseCommand`` restates it for ``CommandJointPositionActionCfg`` to apply."""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import sample_uniform
from mjlab.utils.lab_api.string import resolve_matching_names_values
from mjswan import CommandBinding, register_command


def _resample_height(self: Any, env_ids: Any) -> None:
    """``BaseHeightCommand._resample_command`` at ``N=1``, without the per-env writes
    and ``Tensor.uniform_`` the tracer cannot follow."""
    del env_ids
    cfg = self.cfg
    n, device = self.num_envs, self.device
    hi = cfg.height_range[1]
    self._squat_target = sample_uniform(self.current_floor, hi, (n, 1), device=device)
    self._walk_target = sample_uniform(cfg.walk_min_height, hi, (n, 1), device=device)


def _update_height(self: Any, env_ids: Any) -> None:
    del env_ids
    self._height = self._squat_target


def bind_height_override(term: Any) -> None:
    if not term.cfg.enabled or term.cfg.nominal_env_fraction > 0.0:
        raise ValueError(
            "The rewrite carries play's height command only: enabled, and no "
            "nominal_env_fraction mixture."
        )
    term._resample_command = types.MethodType(_resample_height, term)
    term._update_command = types.MethodType(_update_height, term)


def height_command(env: Any, command_name: str) -> torch.Tensor:
    """Upstream's ``BaseHeightCommand._update_command`` as the observation reads it: the
    walk target while the twist command moves, else the command's own value.

    The panel's slider holds while moving too, as upstream's does, but no lower than the
    walk band's floor: the checkpoint never walked deeper."""
    term = env.command_manager.get_term(command_name)
    cfg = term.cfg
    twist = env.command_manager.get_command(cfg.velocity_command_name)
    moving = torch.norm(twist[:, :3], dim=1, keepdim=True) > cfg.velocity_threshold
    command = env.command_manager.get_command(command_name)
    # ponytail: no graph input reads the Enable box, so an override is told by value: the
    # command parts from the squat target it otherwise equals. A slider landing exactly on
    # that target reads as none until the next resample.
    overridden = command != term._squat_target
    walk = torch.where(
        overridden, command.clamp(min=cfg.walk_min_height), term._walk_target
    )
    return torch.where(moving, walk, command)


def _height_ui(cfg: Any) -> dict[str, Any]:
    """``create_gui``'s controls. Recording it would read the slider as a "Max"
    companion, since its range does not straddle zero."""
    lo, hi = cfg.height_range
    return {
        "inputs": [
            {
                "type": "checkbox",
                "name": "enabled",
                "label": "Enable",
                "default": False,
            },
            {
                "type": "slider",
                "name": "height",
                "label": "height",
                "min": lo,
                "max": hi,
                "step": 0.01,
                "default": hi,
                "enabled_when": "enabled",
            },
        ]
    }


register_command(
    "BaseHeightCommandCfg",
    CommandBinding(
        state_fields=["_height", "_squat_target", "_walk_target"],
        command_field="_height",
        trace_override=bind_height_override,
        ui=_height_ui,
    ),
)


@dataclass(kw_only=True)
class ArmPoseCommandCfg(CommandTermCfg):
    """The arm generator model_15500 trained with: ``UpperBodyPoseAction`` at DUET
    5521a3f, at full ratio. The defaults are its ``exported/env.yaml`` values."""

    home: dict[str, float]
    """Each arm joint, in model order, at its default angle."""
    task_poses: tuple[dict[str, float], ...]
    task_pose_noise: dict[str, float]
    slider_ranges: dict[str, tuple[float, float]]
    resample_steps: int = 50
    sample_range_scale: float = 0.5
    task_pose_prob: float = 0.7
    velocity_command_name: str = "twist"
    velocity_threshold: float = 0.1
    walking_motion_scale: float = 0.15
    # resample_steps at the 0.02 s control step.
    resampling_time_range: tuple[float, float] = (1.0, 1.0)

    def build(self, env: Any) -> "ArmPoseCommand":
        return ArmPoseCommand(self, env)


class ArmPoseCommand(CommandTerm):
    """Arm joint targets: a new goal each ``resample_steps``, reached linearly."""

    cfg: ArmPoseCommandCfg

    def __init__(self, cfg: ArmPoseCommandCfg, env: Any) -> None:
        super().__init__(cfg, env)
        robot = env.scene["robot"]
        ids, names = robot.find_joints(tuple(cfg.home))
        default = robot.data.default_joint_pos[:, ids].clone()
        soft = robot.data.soft_joint_pos_limits[:, ids].clone()
        s = cfg.sample_range_scale
        self._default = default
        self._soft_lo, self._soft_hi = soft[..., 0], soft[..., 1]
        self._lo = default - s * (default - self._soft_lo)
        self._hi = default + s * (self._soft_hi - default)

        def by_joint(values: dict) -> list:
            return resolve_matching_names_values(dict(values), names)[2]

        self._anchors = torch.tensor(
            [by_joint(pose) for pose in cfg.task_poses], device=self.device
        )
        self._noise = torch.tensor(by_joint(cfg.task_pose_noise), device=self.device)
        self.current = default.clone()
        self.goal = default.clone()
        self.delta = torch.zeros_like(default)
        self.fresh = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.current

    def reset(self, env_ids: Any = None) -> dict[str, float]:
        # Upstream's reset: the target starts where the robot does.
        self.current = self._default.clone()
        return super().reset(env_ids)

    def _resample_command(self, env_ids: Any) -> None:
        del env_ids
        n, j = self.current.shape
        dev = self.device
        uniform = self._lo + sample_uniform(0.0, 1.0, (n, j), device=dev) * (
            self._hi - self._lo
        )
        k = torch.randint(len(self._anchors), (n,), device=dev)
        noise = (sample_uniform(0.0, 1.0, (n, j), device=dev) * 2.0 - 1.0) * self._noise
        task = torch.minimum(
            torch.maximum(self._anchors[k] + noise, self._soft_lo), self._soft_hi
        )
        use_task = (
            sample_uniform(0.0, 1.0, (n, 1), device=dev) < self.cfg.task_pose_prob
        )
        self.goal = torch.where(use_task, task, uniform)
        self.fresh = torch.ones_like(self.fresh)

    def _update_command(self, env_ids: Any) -> None:
        del env_ids
        cfg = self.cfg
        # Upstream fixes the increment when it draws the goal. It waits one call here,
        # since the tracer runs a reset's move home after `_resample_command`.
        step = (self.goal - self.current) / cfg.resample_steps
        self.delta = torch.where(self.fresh > 0.5, step, self.delta)
        self.fresh = torch.zeros_like(self.fresh)
        twist = self._env.command_manager.get_command(cfg.velocity_command_name)
        moving = torch.norm(twist[:, :3], dim=1, keepdim=True) > cfg.velocity_threshold
        scale = 1.0 + (cfg.walking_motion_scale - 1.0) * moving.float()
        self.current = self.current + self.delta * scale

    def _update_metrics(self) -> None:
        pass


def _arms_ui(cfg: ArmPoseCommandCfg) -> dict[str, Any]:
    """An Enable box and a slider per joint, in command column order."""
    _, names, ranges = resolve_matching_names_values(
        dict(cfg.slider_ranges), list(cfg.home)
    )
    sliders = [
        {
            "type": "slider",
            "name": name,
            "label": name.removesuffix("_joint"),
            "min": lo,
            "max": hi,
            "step": 0.01,
            "default": cfg.home[name],
            "enabled_when": "enabled",
        }
        for name, (lo, hi) in zip(names, ranges)
    ]
    enable = {
        "type": "checkbox",
        "name": "enabled",
        "label": "Enable",
        "default": False,
    }
    return {"inputs": [enable, *sliders]}


register_command(
    "ArmPoseCommandCfg",
    CommandBinding(
        state_fields=["current", "goal", "delta", "fresh"],
        command_field="current",
        ui=_arms_ui,
    ),
)
