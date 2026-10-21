"""The parkour_mjlab checkout with PHP's motion bundles in place, its task ids
registered, the teachers' checkpoints as ONNX and PHP's clips in mjlab's layout."""

from __future__ import annotations

import hashlib
import importlib
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from mjswan_playground._compat import collision_defaults, legacy_update_assets
from mjswan_playground._deps import CACHE_DIR, ensure_repo

REPO_URL = "https://github.com/yhx1203/parkour_mjlab"
REPO_COMMIT = "0e2cb60f4af9645ea9fc0a9645ae752129f80645"
#: amazon-far/php_parkour's release training-motion-examples-v1, each bundle with the
#: sha256 its SHA256SUMS lists, in upstream's ``BUNDLES`` order.
BUNDLE_URL = (
    "https://github.com/amazon-far/php_parkour/releases/download/"
    "training-motion-examples-v1/{}.tar.gz"
)
BUNDLES = {
    "Locomotion": (
        "php-motions-locomotion-v1",
        "2397220209b5ce32cd886340f5b12f62821f17e40bff0d2958b1a70295a08dcf",
    ),
    "Low-Step": (
        "php-motions-low-step-v1",
        "1f63ee82004d6e705d64b293a9ff6988304ea757394d309209eeabb1b5ee8809",
    ),
    "High-Step": (
        "php-motions-high-step-v1",
        "c4cf035cf2a81336a42277584ef65ee16044f9c5f9d921b08e84c1adaab7381d",
    ),
    "Low-Climb": (
        "php-motions-low-climb-76-v1",
        "1d94450645e5f6ae457c65425bd06b56851ca3793b1e2d9fdd80bba4821e533a",
    ),
    "High-Climb": (
        "php-motions-high-climb-76-v1",
        "045b7ef2ce2231fc8a27aaba16e86d67bb43435d40754df4e0f1ad02ad6e6e05",
    ),
}
#: Where upstream's ``dataset.DATA_ROOT`` looks for them.
MOTION_DIR = "src/assets/motions/php_motion"

PIE = "Unitree-Go2-PIE"
STUDENT = "Unitree-G1-PHP-Student"
PIE_POLICY = "logs/rsl_rl/go2_pie/policy.onnx"
STUDENT_POLICY = "logs/rsl_rl/g1_php_student/model_23000.onnx"
_CACHE = CACHE_DIR / "parkour"


def teacher(skill: str) -> str:
    return f"Unitree-G1-PHP-{skill}-Teacher"


def teacher_checkpoint(skill: str) -> str:
    folder = "g1_php_" + skill.lower().replace("-", "_") + "_teacher"
    return f"logs/rsl_rl/{folder}/model_20000.pt"


def resolve_root() -> Path:
    root = ensure_repo(
        name="parkour_mjlab",
        url=REPO_URL,
        commit=REPO_COMMIT,
        marker=PIE_POLICY,
        root_env_var="MJSWAN_PARKOUR_ROOT",
    )
    for name, sha256 in BUNDLES.values():
        _ensure_bundle(root / MOTION_DIR / name, name, sha256)
    return root


def _ensure_bundle(target: Path, name: str, sha256: str) -> None:
    """Extract one release bundle where upstream's README puts it, checked first."""
    if any(target.glob("*_motion.npz")):
        return
    url = BUNDLE_URL.format(name)
    with tempfile.TemporaryFile() as archive:
        digest = hashlib.sha256()
        with urllib.request.urlopen(url) as response:
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                archive.write(chunk)
        if digest.hexdigest() != sha256:
            raise RuntimeError(f"{url} has sha256 {digest.hexdigest()}, not {sha256}.")
        archive.seek(0)
        with tarfile.open(fileobj=archive) as bundle:
            bundle.extractall(target, filter="data")


def register_tasks(root: Path) -> None:
    """Import upstream's task packages, which registers all seven task ids.

    Upstream imports itself as the top-level ``src``, so its root goes first on
    ``sys.path``, as ``pacman`` does."""
    if "src.tasks" in sys.modules:
        return
    imported = sys.modules.get("src")
    if imported is not None and not any(
        str(path).startswith(str(root)) for path in getattr(imported, "__path__", ())
    ):
        raise RuntimeError(
            "A different top-level 'src' package is already imported, so upstream's "
            "'src.*' cannot be reached. Build this task in a fresh interpreter."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # ponytail: upstream is on mjlab 1.2.0; drop both shims once REPO_COMMIT is on 1.6.
    with collision_defaults(), legacy_update_assets():
        importlib.import_module("src.tasks")


def clips(root: Path, skill: str) -> list[Path]:
    """A bundle's clips, in the order upstream's sequential play draws them."""
    from src.tasks.php.dataset import motion_paths

    return list(motion_paths(str(root / MOTION_DIR / BUNDLES[skill][0])))


def clip_npz(clip: Path, joint_names: list[str], body_names: list[str]) -> Path:
    """A PHP clip in mjlab's ``MotionLoader`` layout: the robot's joints without the
    floating base, and every robot body, in the robot's order, as float32 as upstream's
    ``MotionLibrary`` reads it."""
    out = _CACHE / f"{clip.parent.name}-{clip.stem}.npz"
    if out.exists():
        return out
    with np.load(clip, allow_pickle=False) as src:
        joints = [list(src["joint_names"]).index(name) for name in joint_names]
        bodies = [list(src["body_names"]).index(name) for name in body_names]
        data = {
            # Upstream's MotionLibrary: 7 base columns before the joints, 6 for velocity.
            "joint_pos": src["joint_pos"][:, [7 + j for j in joints]],
            "joint_vel": src["joint_vel"][:, [6 + j for j in joints]],
            **{
                key: src[key][:, bodies]
                for key in (
                    "body_pos_w",
                    "body_quat_w",
                    "body_lin_vel_w",
                    "body_ang_vel_w",
                )
            },
        }
        # C order: the browser cannot read the Fortran order column indexing leaves.
        data = {k: np.ascontiguousarray(v, np.float32) for k, v in data.items()}
        data["fps"] = src["fps"]
    _CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(out, **data)
    return out


def teacher_onnx(root: Path, skill: str, env_cfg: Any) -> Path:
    """A teacher's ``model_20000.pt`` exported once by mjlab's own runner, on the env
    the browser runs, with the metadata mjlab's own exports carry."""
    out = _CACHE / f"{teacher(skill)}-{REPO_COMMIT[:7]}.onnx"
    if out.exists():
        return out
    import onnx
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
    from mjswan.mjlab.runner import create_pt_onnx_export_context, export_checkpoint

    checkpoint = teacher_checkpoint(skill)
    context = create_pt_onnx_export_context(teacher(skill), env_cfg=env_cfg)
    try:
        model = export_checkpoint(context, root / checkpoint)
        metadata = get_base_metadata(context.env.unwrapped, checkpoint)
    finally:
        context.close()
    _CACHE.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out)
    attach_metadata_to_onnx(str(out), metadata)
    return out
