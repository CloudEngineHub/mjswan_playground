"""The jumper checkout: pinned clone, its posture task registered with mjlab, and the
deploy contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

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


def register_tasks(root: Path) -> None:
    """Register upstream's ``jumper.posture`` play config with mjlab as ``TASK_ID``.

    Upstream keeps its own registry, so nothing reaches mjlab's on import. Its tasks
    import themselves as top-level ``tasks`` and ``controller`` from the root, and
    ``mjrl`` from ``rl/``. That directory also vendors mjlab and rsl_rl, so it goes
    last on ``sys.path``, behind the installed copies.
    """
    import mjlab.tasks.registry as registry
    from mjlab.rl import RslRlOnPolicyRunnerCfg

    if TASK_ID in registry.list_tasks():
        return
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(root / "rl") not in sys.path:
        sys.path.append(str(root / "rl"))
    from tasks.jumper.posture.env_cfg import env_cfg

    play = env_cfg(play=True)
    # Only the runner's obs_groups and clip_actions reach the browser, and upstream's
    # runner config sets neither; it also sets a field only its vendored rsl_rl has.
    registry.register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=play,
        play_env_cfg=play,
        rl_cfg=RslRlOnPolicyRunnerCfg(),
    )
