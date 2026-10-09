"""Upstream's play-time state, recast as command terms the browser can keep.

Upstream's gait clock is an observation term that integrates its own phase, and its
posture command draws with ``Tensor.uniform_`` into per-env rows. A browser observation
holds no state and the tracer follows neither, so the clock becomes a command term and
the posture command is rebound below, each restating upstream's math at ``N=1``."""

from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Any

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import sample_uniform
from mjswan import CommandBinding, register_command

GAIT_CLOCK = "gait_clock"

# A resampling time no episode reaches, so the clock restarts on episode reset alone.
_NEVER = 1.0e9


@dataclass(kw_only=True)
class CadenceClockCommandCfg(CommandTermCfg):
    """Upstream's ``CadenceClock``: a phase advanced each control step at the cadence the
    twist command asked for on the step before."""

    command_name: str
    stride: float
    freq_min: float
    freq_max: float
    turn_radius: float
    resampling_time_range: tuple[float, float] = (_NEVER, _NEVER)

    def build(self, env: Any) -> "CadenceClockCommand":
        return CadenceClockCommand(self, env)


class CadenceClockCommand(CommandTerm):
    cfg: CadenceClockCommandCfg

    def __init__(self, cfg: CadenceClockCommandCfg, env: Any) -> None:
        super().__init__(cfg, env)
        self.phase = torch.zeros(self.num_envs, 1, device=self.device)
        self.rate = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.phase

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        del env_ids
        # mjlab updates right after every reset; a zero rate keeps that update at phase 0.
        self.phase = torch.zeros_like(self.phase)
        self.rate = torch.zeros_like(self.rate)

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        del env_ids
        cfg = self.cfg
        self.phase = torch.remainder(self.phase + self.rate * self._env.step_dt, 1.0)
        # upstream's gait_frequency
        cmd = self._env.command_manager.get_command(cfg.command_name)
        v_eff = (
            cmd[:, :2].norm(dim=1, keepdim=True) + cmd[:, 2:3].abs() * cfg.turn_radius
        )
        self.rate = (v_eff / (2.0 * cfg.stride)).clamp(cfg.freq_min, cfg.freq_max)

    def _update_metrics(self) -> None:
        pass


register_command(
    "CadenceClockCommandCfg",
    CommandBinding(state_fields=["phase", "rate"], command_field="phase"),
)


def gait_phase(
    env: Any, clock_name: str, command_name: str, command_threshold: float
) -> torch.Tensor:
    """Upstream's ``VariableGaitClock`` output: sin and cos of the phase, zero while the
    twist command asks to stand."""
    angle = env.command_manager.get_term(clock_name).phase * (2.0 * math.pi)
    clock = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
    cmd = env.command_manager.get_command(command_name)
    moving = torch.norm(cmd[:, :3], dim=1, keepdim=True) > command_threshold
    return clock * moving.float()


def clock_gait_phase(env_cfg: Any) -> None:
    """Point the ``gait_phase`` observation at a clock the browser can keep."""
    term = env_cfg.observations["actor"].terms["gait_phase"]
    params = term.params
    env_cfg.commands[GAIT_CLOCK] = CadenceClockCommandCfg(
        command_name=params["command_name"],
        stride=params["stride"],
        freq_min=params["freq_min"],
        freq_max=params["freq_max"],
        turn_radius=params["turn_radius"],
    )
    term.func = gait_phase
    term.params = {
        "clock_name": GAIT_CLOCK,
        "command_name": params["command_name"],
        "command_threshold": params["command_threshold"],
    }


def _resample_posture(self: Any, env_ids: Any) -> None:
    """``PostureCommand._resample_command`` at ``N=1``, without the per-env writes and
    ``Tensor.uniform_`` the tracer cannot follow."""
    del env_ids
    cfg = self.cfg
    n, device = self.num_envs, self.device
    twist = self._env.command_manager.get_command(cfg.velocity_command)
    in_band = torch.norm(twist[:, :3], dim=1, keepdim=True) < cfg.stand_threshold
    angles = []
    for name in ("twist", "pitch", "roll"):
        half = torch.where(
            in_band,
            torch.full_like(twist[:, :1], getattr(cfg.ranges, name)),
            torch.full_like(twist[:, :1], getattr(cfg.moving, name)),
        )
        angles.append(
            (2.0 * sample_uniform(0.0, 1.0, (n, 1), device=device) - 1.0) * half
        )
    height = sample_uniform(*cfg.ranges.height, (n, 1), device=device)
    standing = self._env.command_manager.get_term(cfg.velocity_command).is_standing_env
    rate = torch.where(
        standing.reshape(-1, 1),
        torch.full_like(height, cfg.rel_neutral_standing_envs),
        torch.full_like(height, cfg.rel_neutral_envs),
    )
    neutral = sample_uniform(0.0, 1.0, (n, 1), device=device) <= rate
    drawn = torch.cat([*angles, height], dim=-1)
    rest = torch.zeros_like(drawn) + torch.tensor(
        [0.0, 0.0, 0.0, cfg.neutral_height], device=device
    )
    self.posture_command = torch.where(neutral, rest, drawn)


def _update_posture(self: Any, env_ids: Any = None) -> None:
    """The anchors upstream's update moves feed only its metrics."""
    del env_ids


def bind_posture_override(term: Any) -> None:
    term._resample_command = types.MethodType(_resample_posture, term)
    term._update_command = types.MethodType(_update_posture, term)


def _posture_ui(cfg: Any) -> dict[str, Any]:
    """Upstream's operator axes, as sliders over the standing band. Recording would read
    the height slider as a "Max" companion, since its range does not straddle zero."""
    inputs: list[dict[str, Any]] = [
        {"type": "checkbox", "name": "enabled", "label": "Enable", "default": False}
    ]
    for name in ("twist", "pitch", "roll"):
        half = getattr(cfg.ranges, name)
        inputs.append(
            {
                "type": "slider",
                "name": name,
                "label": f"{name} (rad)",
                "min": -half,
                "max": half,
                "step": 0.01,
                "default": 0.0,
                "enabled_when": "enabled",
            }
        )
    lo, hi = cfg.ranges.height
    inputs.append(
        {
            "type": "slider",
            "name": "height",
            "label": "height (m)",
            "min": lo,
            "max": hi,
            "step": 0.005,
            "default": cfg.neutral_height,
            "enabled_when": "enabled",
        }
    )
    return {"inputs": inputs}


register_command(
    "PostureCommandCfg",
    CommandBinding(
        state_fields=["posture_command"],
        command_field="posture_command",
        trace_override=bind_posture_override,
        ui=_posture_ui,
    ),
)
