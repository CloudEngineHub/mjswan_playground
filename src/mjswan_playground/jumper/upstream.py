"""The pinned jumper checkout, its deploy contracts, and shared play-config edits."""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

from mjswan_playground._deps import ensure_repo

REPO_URL = "https://github.com/KingKongRobotics/jumper.git"
REPO_COMMIT = "61d065219fca767f3142c8f10aff59eae5a5a004"

TASK_ID = "Jumper-Posture"
#: model_74800 exported by upstream's scripts/export.py, with layout.json beside it.
POLICY_ONNX = "tasks/jumper/posture/out/example/actor.onnx"


def resolve_root() -> Path:
    return ensure_repo(
        name="jumper",
        url=REPO_URL,
        commit=REPO_COMMIT,
        marker=POLICY_ONNX,
        root_env_var="MJSWAN_JUMPER_ROOT",
    )


def export(root: Path, task: str) -> Path:
    """Where upstream's ``scripts/export.py`` put ``jumper.<task>``'s policy: its
    ``actor.onnx``, with the normalizer inside, and ``layout.json``."""
    return root / "tasks" / "jumper" / task / "out" / "example"


def contract(root: Path, task: str) -> dict:
    return json.loads((export(root, task) / "layout.json").read_text())


def import_module(root: Path, module: str) -> Any:
    """Import an upstream module, with the root first on ``sys.path`` for its top-level
    ``tasks`` and ``controller``, and ``rl/`` (``mjrl``) last, behind the installed
    mjlab and rsl_rl it also vendors."""
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(root / "rl") not in sys.path:
        sys.path.append(str(root / "rl"))
    return importlib.import_module(module)


def play_env_cfg(root: Path, task: str) -> Any:
    """``jumper.<task>``'s play config, without the ToF sensor only its viewer draws."""
    env_cfg = import_module(root, f"tasks.jumper.{task}.env_cfg").env_cfg(play=True)
    env_cfg.scene.sensors = tuple(s for s in env_cfg.scene.sensors if s.name != "tof")
    return env_cfg


def register_tasks(root: Path) -> None:
    """Register upstream's ``jumper.posture`` play config with mjlab as ``TASK_ID``."""
    import mjlab.tasks.registry as registry
    from mjlab.rl import RslRlOnPolicyRunnerCfg

    if TASK_ID in registry.list_tasks():
        return
    play = import_module(root, "tasks.jumper.posture.env_cfg").env_cfg(play=True)
    # Only the runner's obs_groups and clip_actions reach the browser, and upstream's
    # runner config sets neither; it also sets a field only its vendored rsl_rl has.
    registry.register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=play,
        play_env_cfg=play,
        rl_cfg=RslRlOnPolicyRunnerCfg(),
    )


def as_base(cls: type, cfg: Any) -> Any:
    """``cfg`` rebuilt as its base class ``cls``, dropping the subclass's fields."""
    return cls(**{f.name: getattr(cfg, f.name) for f in fields(cls)})


def unstride(env_cfg: Any) -> dict[str, tuple[int, ...]]:
    """Unwrap upstream's ``StridedHistory`` from each actor term, returning the frames it
    stacked as look-back offsets, oldest first."""
    from tasks.jumper.posture.mdp.history import StridedHistory

    prefix = StridedHistory.PREFIX
    offsets = {}
    for name, term in env_cfg.observations["actor"].terms.items():
        if term.func is not StridedHistory:
            continue
        params = term.params
        frames, stride = params[prefix + "frames"], params[prefix + "stride"]
        offsets[name] = tuple(stride * i for i in reversed(range(frames)))
        term.func = params[prefix + "func"]
        term.params = {k: v for k, v in params.items() if not k.startswith(prefix)}
    return offsets


def drop_training_terms(env_cfg: Any) -> None:
    """Drop training-only terms, some of which read state the browser does not keep."""
    env_cfg.observations = {"actor": env_cfg.observations["actor"]}
    env_cfg.rewards = {}
    env_cfg.metrics = {}
    env_cfg.curriculum = {}
