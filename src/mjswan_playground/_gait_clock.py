"""The ``phase`` observation unitree_rl_mjlab's descendants share (``bipedhrl``, ``duet``),
with its clock moved from ``env.episode_length_buf``, which the tracer does not serve, to
a command term that counts steps itself."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjswan import CommandBinding, register_command

GAIT_CLOCK = "gait_clock"

# A resampling time no episode reaches, so the clock restarts on episode reset alone.
_NEVER = 1.0e9


@dataclass(kw_only=True)
class GaitClockCommandCfg(CommandTermCfg):
    """Control steps since the episode began, modulo one gait period."""

    steps_per_period: int
    resampling_time_range: tuple[float, float] = (_NEVER, _NEVER)

    def build(self, env: Any) -> "GaitClockCommand":
        return GaitClockCommand(self, env)


class GaitClockCommand(CommandTerm):
    cfg: GaitClockCommandCfg

    def __init__(self, cfg: GaitClockCommandCfg, env: Any) -> None:
        super().__init__(cfg, env)
        self.step_count = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.step_count

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        del env_ids
        # mjlab updates right after every reset, so -1 reads as upstream's 0 on frame one.
        self.step_count = torch.full_like(self.step_count, -1.0)

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        del env_ids
        self.step_count = torch.remainder(
            self.step_count + 1.0, float(self.cfg.steps_per_period)
        )

    def _update_metrics(self) -> None:
        pass


register_command(
    "GaitClockCommandCfg",
    CommandBinding(state_fields=["step_count"], command_field="step_count"),
)


def phase(
    env: Any, clock_name: str, steps_per_period: int, command_name: str
) -> torch.Tensor:
    """Upstream's ``phase``: sin and cos of the gait clock, zero while standing."""
    angle = env.command_manager.get_term(clock_name).step_count * (
        2.0 * math.pi / steps_per_period
    )
    clock = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
    command = env.command_manager.get_term(command_name).vel_command_b
    standing = torch.linalg.norm(command, dim=1, keepdim=True) < 0.1
    return torch.where(standing, torch.zeros_like(clock), clock)


def clock_phase(env_cfg: Any) -> None:
    """Point the ``phase`` observation at a gait clock the browser can keep."""
    term = env_cfg.observations["actor"].terms["phase"]
    period_steps = term.params["period"] / (
        env_cfg.sim.mujoco.timestep * env_cfg.decimation
    )
    steps_per_period = round(period_steps)
    if abs(period_steps - steps_per_period) > 1e-6:
        raise ValueError(
            f"The gait period is not a whole number of steps ({period_steps})."
        )
    env_cfg.commands[GAIT_CLOCK] = GaitClockCommandCfg(
        steps_per_period=steps_per_period
    )
    term.func = phase
    term.params = {
        "clock_name": GAIT_CLOCK,
        "steps_per_period": steps_per_period,
        "command_name": term.params["command_name"],
    }
