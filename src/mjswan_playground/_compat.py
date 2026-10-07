"""Shims for upstream tasks written against mjlab before 1.6."""

from __future__ import annotations

from unittest import mock

from mjlab.utils.spec_config import CollisionCfg

_COLLISION_DEFAULTS = {"contype": 1, "conaffinity": 1, "condim": 3, "priority": 0}


def collision_defaults():
    """Patch ``CollisionCfg`` to default the fields mjlab 1.6 made required."""
    return mock.patch("mjlab.utils.spec_config.CollisionCfg", _collision_cfg)


def _collision_cfg(**kwargs) -> CollisionCfg:
    """A pattern dict without a catch-all gets the default as a last ``.*``: patterns
    match first-wins."""
    for name, default in _COLLISION_DEFAULTS.items():
        value = kwargs.setdefault(name, default)
        if isinstance(value, dict):
            kwargs[name] = {**value, ".*": value.get(".*", default)}
    return CollisionCfg(**kwargs)
