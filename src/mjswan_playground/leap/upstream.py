"""The in-hand rotation checkout: pinned clone, task ids registered, policy as ONNX."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import mjlab.actuator
import mjlab.envs.mdp as envs_mdp
import mjlab.rl
import mjlab.terrains
import onnx
from mjlab.managers import event_manager

from mjswan_playground._compat import collision_defaults, legacy_update_assets
from mjswan_playground._deps import CACHE_DIR, ensure_repo

REPO_URL = "https://github.com/Msornerrrr/in-hand-rotation-mjlab.git"
REPO_COMMIT = "14004fe3bd5ee2ed51d8b7348ca5da34338c7f1e"

#: Importing it registers the three hand-cube task ids.
TASK_PACKAGE = "in_hand_rotation_mjlab.tasks"
#: The one checkpoint upstream publishes, for the custom left hand.
CHECKPOINT = "ckpts/leap_left_custom_model_4900.pt"
#: The functions mjlab 1.6 removed that upstream's domain randomization calls.
_REMOVED = ("randomize_field", "randomize_pd_gains", "sync_actuator_delays")

_EventTermCfg = event_manager.EventTermCfg
_RslRlModelCfg = mjlab.rl.RslRlModelCfg


def resolve_root() -> Path:
    return ensure_repo(
        name="in_hand_rotation_mjlab",
        url=REPO_URL,
        commit=REPO_COMMIT,
        marker=CHECKPOINT,
        root_env_var="MJSWAN_LEAP_ROOT",
    )


def register_tasks(root: Path) -> None:
    """Import upstream's task package so mjlab's registry knows its task ids."""
    if TASK_PACKAGE in sys.modules:
        return
    source = str(root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    # ponytail: upstream predates mjlab 1.6; drop the shims once REPO_COMMIT is on 1.6.
    with (
        collision_defaults(),
        legacy_update_assets(),
        mock.patch.object(
            mjlab.actuator, "DelayedActuatorCfg", _delayed_actuator_cfg, create=True
        ),
        mock.patch.object(
            mjlab.terrains,
            "TerrainImporterCfg",
            mjlab.terrains.TerrainEntityCfg,
            create=True,
        ),
        mock.patch.object(event_manager, "EventTermCfg", _event_term_cfg),
        mock.patch.object(mjlab.rl, "RslRlModelCfg", _rsl_rl_model_cfg),
        mock.patch.multiple(
            envs_mdp, create=True, **{name: _removed(name) for name in _REMOVED}
        ),
    ):
        importlib.import_module(TASK_PACKAGE)


def ensure_policy(root: Path, task_id: str, env_cfg: Any) -> tuple[Path, dict]:
    """``CHECKPOINT`` as ONNX, exported once into the cache by mjlab's own runner, and
    the joints its actions drive with their rest pose, as mjlab's env has them: the
    export carries no metadata, since mjlab's needs a ``JointPositionAction``."""
    from mjswan.mjlab.runner import create_pt_onnx_export_context, export_checkpoint

    onnx_path = CACHE_DIR / "leap" / f"{Path(CHECKPOINT).stem}.onnx"
    meta_path = onnx_path.with_suffix(".json")
    if not (onnx_path.exists() and meta_path.exists()):
        context = create_pt_onnx_export_context(task_id, env_cfg=env_cfg)
        try:
            model = export_checkpoint(context, root / CHECKPOINT)
            robot = context.env.unwrapped.scene["robot"]
            rest = dict(
                zip(robot.joint_names, robot.data.default_joint_pos[0].tolist())
            )
            meta = {
                "policy_joint_names": context.joint_names,
                "default_joint_pos": [
                    round(rest[name.split("/")[-1]], 3) for name in context.joint_names
                ],
            }
        finally:
            context.close()
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(onnx_path))
        meta_path.write_text(json.dumps(meta, indent=1))
    return onnx_path, json.loads(meta_path.read_text())


def _delayed_actuator_cfg(*, base_cfg: Any, delay_target: str, **delay: Any) -> Any:
    """1.6 puts the delay fields on every actuator config, and delays every target."""
    for field, value in delay.items():
        setattr(base_cfg, field, value)
    return base_cfg


def _event_term_cfg(*args: Any, domain_randomization: bool = False, **kwargs: Any):
    return _EventTermCfg(*args, **kwargs)


def _rsl_rl_model_cfg(
    *, stochastic: bool = False, init_noise_std: float = 1.0, **kwargs: Any
):
    if stochastic:
        kwargs["distribution_cfg"] = {
            "class_name": "GaussianDistribution",
            "init_std": init_noise_std,
            "std_type": "scalar",
        }
    return _RslRlModelCfg(**kwargs)


def _removed(name: str):
    def removed(*args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(f"mjlab 1.6 has no {name}")

    removed.__name__ = name
    return removed
