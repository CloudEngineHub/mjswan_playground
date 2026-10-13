"""The wuji-mjlab checkout and its release bundle: pinned, fetched into ``.cache/``."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

from mjswan_playground._deps import CACHE_DIR, ensure_repo

REPO_URL = "https://github.com/wuji-technology/wuji-mjlab.git"
REPO_COMMIT = "26b99c6338641e8edc17caf87922e6e1767121fa"

RELEASE = "v2026.9.27"
BUNDLE_URL = (
    "https://github.com/wuji-technology/wuji-mjlab/releases/download/"
    f"{RELEASE}/wuji-mjlab-{RELEASE}-assets.zip"
)
BUNDLE_SHA256 = "de01b29921b69ea38100a29416c324b384905e1ae4ebbfa40d769be948fc2d39"
#: What the build reads out of the bundle, which is mostly CAD and pen-spin data.
BUNDLE_FILES = (
    "reorient/checkpoints/hand2/policy.onnx",
    "reorient/checkpoints/hand2/config.json",
)

#: Upstream's ``mjlab.tasks`` entry point, which registers every task id it ships.
TASK_PACKAGE = "wuji_mjlab.tasks"


def resolve_root() -> Path:
    return ensure_repo(
        name="wuji-mjlab",
        url=REPO_URL,
        commit=REPO_COMMIT,
        marker="src/wuji_mjlab/tasks/reorient/reorient_env_cfg.py",
        root_env_var="MJSWAN_WUJI_ROOT",
    )


def resolve_bundle() -> Path:
    """The release bundle's files the build reads, extracted once into the cache."""
    root = CACHE_DIR / "wuji" / RELEASE
    if all((root / name).exists() for name in BUNDLE_FILES):
        return root
    archive = root.with_suffix(".zip")
    archive.parent.mkdir(parents=True, exist_ok=True)
    sha256 = hashlib.sha256()
    with urllib.request.urlopen(BUNDLE_URL) as response, archive.open("wb") as out:
        while chunk := response.read(1 << 20):
            sha256.update(chunk)
            out.write(chunk)
    digest = sha256.hexdigest()
    if digest != BUNDLE_SHA256:
        archive.unlink()
        raise RuntimeError(
            f"{BUNDLE_URL} has sha256 {digest}, not the pinned {BUNDLE_SHA256}."
        )
    prefix = f"wuji-mjlab-{RELEASE}-assets/"
    with zipfile.ZipFile(archive) as bundle:
        for name in BUNDLE_FILES:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(prefix + name))
    archive.unlink()
    return root


def checkpoint(bundle: Path, hand: str) -> tuple[Path, dict]:
    """A hand's ``policy.onnx`` and the deploy config beside it."""
    directory = bundle / "reorient" / "checkpoints" / hand
    return directory / "policy.onnx", json.loads(
        (directory / "config.json").read_text()
    )


def register_tasks(root: Path) -> None:
    """Import upstream's task package from ``src/`` so mjlab's registry knows it."""
    if TASK_PACKAGE in sys.modules:
        return
    source = str(root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    importlib.import_module(TASK_PACKAGE)
