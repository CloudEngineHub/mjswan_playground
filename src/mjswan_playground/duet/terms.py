"""Upstream's ``BaseHeightCommand`` bound for the browser.

Its update picks the walk or the squat target by reading the twist command, and a command
term cannot read another one in the browser. So the bound command holds the squat target
as its value and the ``height_command`` observation, which can read both commands, makes
that pick instead."""

from __future__ import annotations

import types
from typing import Any

import torch
from mjlab.utils.lab_api.math import sample_uniform
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
    walk target while the twist command moves, else the command's own value."""
    term = env.command_manager.get_term(command_name)
    twist = env.command_manager.get_command(term.cfg.velocity_command_name)
    moving = torch.norm(twist[:, :3], dim=1, keepdim=True) > term.cfg.velocity_threshold
    command = env.command_manager.get_command(command_name)
    return torch.where(moving, term._walk_target, command)


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
