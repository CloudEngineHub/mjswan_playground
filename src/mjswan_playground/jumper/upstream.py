"""The jumper checkout: pinned clone, its tasks registered with mjlab, each policy's
deploy contract, and the edits every play config needs."""

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
CONTRACT = "tasks/jumper/posture/out/example/layout.json"


def resolve_root() -> Path:
    return ensure_repo(
        name="jumper",
        url=REPO_URL,
        commit=REPO_COMMIT,
        marker=POLICY_ONNX,
        root_env_var="MJSWAN_JUMPER_ROOT",
    )


def deployed_contract(root: Path) -> dict:
    return json.loads((root / CONTRACT).read_text())


def export(root: Path, task: str) -> Path:
    """Where upstream's ``scripts/export.py`` put ``jumper.<task>``'s policy: its
    ``actor.onnx``, with the normalizer inside, and ``layout.json``."""
    return root / "tasks" / "jumper" / task / "out" / "example"


def contract(root: Path, task: str) -> dict:
    return json.loads((export(root, task) / "layout.json").read_text())


def import_module(root: Path, module: str) -> Any:
    """Import one of upstream's modules.

    Upstream keeps its own registry, so nothing reaches mjlab's on import. Its tasks
    import themselves as top-level ``tasks`` and ``controller`` from the root, and
    ``mjrl`` from ``rl/``. That directory also vendors mjlab and rsl_rl, so it goes
    last on ``sys.path``, behind the installed copies.
    """
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(root / "rl") not in sys.path:
        sys.path.append(str(root / "rl"))
    return importlib.import_module(module)


def play_env_cfg(root: Path, task: str) -> Any:
    """``jumper.<task>``'s play config, without the ToF sensor play adds for its viewer
    alone; no term reads it."""
    env_cfg = import_module(root, f"tasks.jumper.{task}.env_cfg").env_cfg(play=True)
    env_cfg.scene.sensors = tuple(s for s in env_cfg.scene.sensors if s.name != "tof")
    return env_cfg


def register(task_id: str, env_cfg: Any) -> str:
    """Register ``env_cfg`` with mjlab as ``task_id``, for train and play alike."""
    import mjlab.tasks.registry as registry
    from mjlab.rl import RslRlOnPolicyRunnerCfg

    if task_id not in registry.list_tasks():
        # Only the runner's obs_groups and clip_actions reach the browser, and upstream's
        # runner config sets neither; it also sets a field only its vendored rsl_rl has.
        registry.register_mjlab_task(
            task_id=task_id,
            env_cfg=env_cfg,
            play_env_cfg=env_cfg,
            rl_cfg=RslRlOnPolicyRunnerCfg(),
        )
    return task_id


def register_tasks(root: Path) -> None:
    """Register upstream's ``jumper.posture`` play config with mjlab as ``TASK_ID``."""
    import mjlab.tasks.registry as registry

    if TASK_ID in registry.list_tasks():
        return
    module = import_module(root, "tasks.jumper.posture.env_cfg")
    register(TASK_ID, module.env_cfg(play=True))


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
    """Drop what only training reads: the critic, the rewards, the metrics and the
    curricula, several of which read state the browser does not keep."""
    env_cfg.observations = {"actor": env_cfg.observations["actor"]}
    env_cfg.rewards = {}
    env_cfg.metrics = {}
    env_cfg.curriculum = {}
