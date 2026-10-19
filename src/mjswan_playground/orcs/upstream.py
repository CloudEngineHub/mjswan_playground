"""The ORCS checkout with its mocke and rsl_rl dependencies, the data its tasks read,
and the released checkpoints as ONNX.

Run as a script (``python upstream.py <checkpoint> <out.onnx> <actor json>``), this file
is the conversion step, with the ``rsl_rl`` fork :func:`ensure_policy` puts first on
``PYTHONPATH``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_URL = "https://github.com/lok-i/orcs"
REPO_COMMIT = "49886d16d4b39e89442ce82ecadc7ae4288fae17"
#: As upstream's deps.lock pins both; release.json's differ only in README and pyproject.
MOCKE_URL = "https://github.com/lok-i/mocke"
MOCKE_COMMIT = "b50fb6d11e7616598dec8b444e88c55b559ec87b"
RSL_RL_URL = "https://github.com/lok-i/rsl_rl"
RSL_RL_COMMIT = "215b6f4ceda569d2ca39df64dd679b9af15375cb"
HF_REPO = "lkrajan/orcs"
#: Tag v0.1.0, the revision upstream's release.json names.
HF_REVISION = "ebc6834a559eb63b55f48f5b43dedb2c0dbc0280"
OMNI_REPO = "omniretarget/OmniRetarget_Dataset"
OMNI_REVISION = "135259130158f1cb1799e4ab9b6566c938285fc9"
GRAIL_REPO = "nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL"
GRAIL_REVISION = "40e795761302e611c1e7e3a6caefdd010d56c199"

DODGE = "Orcs-Dodge-AdaptSonic"
PERLOCO_OMRE = "Orcs-PerLoco-OmRe-AdaptSonic"
PERLOCO_GRAIL = "Orcs-PerLoco-Grail-AdaptSonic"
#: Each PerLoco task's staging source, the name of its roster.
PERLOCO = {PERLOCO_OMRE: "omni", PERLOCO_GRAIL: "grail"}


def _cache() -> Path:
    from mjswan_playground._deps import CACHE_DIR

    return CACHE_DIR / "orcs"


def resolve_roots() -> dict[str, Path]:
    from mjswan_playground._deps import ensure_repo

    return {
        "orcs": ensure_repo(
            name="orcs/orcs",
            url=REPO_URL,
            commit=REPO_COMMIT,
            marker="src/orcs/release.json",
            root_env_var="MJSWAN_ORCS_ROOT",
        ),
        "mocke": ensure_repo(
            name="orcs/mocke",
            url=MOCKE_URL,
            commit=MOCKE_COMMIT,
            marker="src/mocke/sonic/profile.py",
            root_env_var="MJSWAN_MOCKE_ROOT",
        ),
        "rsl_rl": ensure_repo(
            name="orcs/rsl_rl",
            url=RSL_RL_URL,
            commit=RSL_RL_COMMIT,
            marker="rsl_rl/models/sonic_adapter_model.py",
            root_env_var="MJSWAN_ORCS_RSL_RL_ROOT",
        ),
    }


def register_tasks(roots: dict[str, Path]) -> None:
    """Make the data ``DODGE`` and ``PERLOCO`` read, then register them with mjlab."""
    if "orcs.tasks.perloco" in sys.modules:
        return
    from unittest import mock

    for source in (roots["orcs"] / "src", roots["mocke"] / "src"):
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
    # Upstream's switches: its data root, and no git check of mocke and rsl_rl on import.
    os.environ.setdefault("ORCS_DATA_ROOT", str(_cache() / "data"))
    os.environ.setdefault("ORCS_SKIP_DEP_CHECK", "1")
    import mjlab.scripts.gcs

    # ponytail: importing mocke registers two sandbox tasks on mjlab's demo clip (LAFAN1,
    # CC BY-NC-ND), downloading it; without it they skip, as mocke does offline.
    with mock.patch.object(mjlab.scripts.gcs, "ensure_default_motion", _offline):
        import mocke  # noqa: F401
    import orcs.assets.g1 as g1

    g1._replace_hand_with_box = _replace_hand_collision
    _ensure_nominal_stand()
    import orcs.tasks.dodge  # noqa: F401

    for source in PERLOCO.values():
        _ensure_tiles(roots["orcs"], source)
    # Staging imported this package before the tiles existed, skipping its tasks.
    import orcs.tasks.perloco as perloco
    from mjlab.tasks.registry import list_tasks
    from orcs.core.registry import register_all

    rows = [r for r in perloco._TASKS if r[0] in PERLOCO and r[0] not in list_tasks()]
    if skipped := register_all(rows):
        raise RuntimeError(f"PerLoco did not register: {skipped}")


def _offline() -> str:
    raise OSError("mjlab's demo clip is not fetched")


def _replace_hand_collision(spec: Any) -> None:
    """``orcs.assets.g1._replace_hand_with_box`` without its flat-hand visual mesh, from
    drcl-deploy/assets, whose G1 meshes record no license. Visual geoms collide with
    nothing, so physics is unchanged."""
    import mujoco
    import numpy as np
    from orcs.assets.g1 import _SIDES, _Y_OFFSET, find_body

    for side in _SIDES:
        body = find_body(spec.worldbody, f"{side}_wrist_yaw_link")
        for geom in body.geoms:
            if geom.name == f"{side}_hand_collision":
                geom.fromto[:] = np.nan
                geom.type = mujoco.mjtGeom.mjGEOM_BOX
                geom.size[:] = [0.07, 0.005, 0.045]
                geom.pos[:] = [0.12, _Y_OFFSET[side], 0.0]
                break


def _ensure_nominal_stand() -> None:
    """``orcs-make-nominal``'s stand, made once on the CPU where Dodge reads it. Not via
    ``nominal_root``: importing it registers Dodge before the clip exists."""
    from unittest import mock

    from orcs.core.paths import DATA_ROOT

    root = DATA_ROOT / "nominal_motions"
    if any(root.rglob("motion.npz")):
        return
    from orcs.cli import make_nominal_motion

    argv = ["orcs-make-nominal", "--out", str(root), "--device", "cpu"]
    with mock.patch.object(sys, "argv", argv):
        make_nominal_motion.main()


def _roster(root: Path, source: str) -> dict[str, list]:
    import tomllib

    return tomllib.loads(
        (root / f"src/orcs/tasks/perloco/rosters/{source}.toml").read_text()
    )


def _ensure_tiles(root: Path, source: str) -> None:
    """``source``'s roster tiles, as ``orcs-stage-terrain`` stages them from the Hub."""
    from unittest import mock

    from huggingface_hub import snapshot_download
    from orcs.core.paths import DATA_ROOT

    if any((DATA_ROOT / "terrain_motions" / source).rglob("motion.npz")):
        return
    roster = _roster(root, source)
    families = roster["families"]
    if source == "omni":
        # The clips' zip, the terrain URDFs, and the robot URDF that names the joints.
        dataset = snapshot_download(
            OMNI_REPO,
            repo_type="dataset",
            revision=OMNI_REVISION,
            allow_patterns=[
                "robot-terrain.zip",
                "models/g1/g1_29dof.urdf",
                *(f"models/terrain/{family}/*" for family in families),
            ],
        )
    else:
        # Each take's robot clip and curb geometry; the roster's curbs only.
        snapshot = snapshot_download(
            GRAIL_REPO,
            repo_type="dataset",
            revision=GRAIL_REVISION,
            allow_patterns=[
                f"data/curb/{kind}/*__{family}__*"
                for family in families
                for kind in ("robot", "object_usd")
            ],
        )
        dataset = str(Path(snapshot) / "data")
    from orcs.cli import stage_terrain_motions

    argv = ["orcs-stage-terrain", "--source", source, "--root", dataset]
    argv += ["--device", "cpu", "--families", *families]
    argv += ["--levels", *map(str, roster["levels"])]
    with mock.patch.object(sys, "argv", argv), mock.patch("joblib.load", _load_arrays):
        stage_terrain_motions.main()


#: All a GRAIL clip's pickle holds: numpy arrays, as joblib wraps them.
_ARRAY_CLASSES = {
    ("joblib.numpy_pickle", "NumpyArrayWrapper"),
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
}


def _load_arrays(path: Path) -> Any:
    """``joblib.load`` allowing only ``_ARRAY_CLASSES``, so a download runs no code."""
    import pickle

    from joblib.numpy_pickle import NumpyUnpickler

    class Arrays(NumpyUnpickler):
        def find_class(self, module: str, name: str) -> Any:
            if (module, name) not in _ARRAY_CLASSES:
                raise pickle.UnpicklingError(f"{path} holds {module}.{name}")
            return super().find_class(module, name)

    with open(path, "rb") as file:
        return Arrays(str(path), file, ensure_native_byte_order=True).load()


def play_env_cfg(roots: dict[str, Path], task_id: str) -> Any:
    """``task_id``'s play config, a PerLoco one on its roster's first tile and clip alone:
    the browser runs one env, where upstream's play gives each env a tile."""
    from mjlab.tasks.registry import load_env_cfg

    if task_id not in PERLOCO:
        return load_env_cfg(task_id, play=True)
    from orcs.core.paths import DATA_ROOT
    from orcs.tasks.perloco import env_cfg as perloco_cfg

    source = PERLOCO[task_id]
    roster = _roster(roots["orcs"], source)
    family, level = roster["families"][0], roster["levels"][0]
    path = DATA_ROOT / f"{source}_tile.toml"
    path.write_text(
        f'families = ["{family}"]\nlevels = [{level}]\n\n'
        f'[clips]\n"{family}/level_{level:.2f}" = ["sample0"]\n'
    )
    factory = getattr(perloco_cfg, f"{source}_env_cfg")
    return factory(play=True, roster=str(path))


def ensure_policy(
    roots: dict[str, Path], task_id: str, env_cfg: Any
) -> tuple[Path, dict[str, Any]]:
    """``task_id``'s released checkpoint as ONNX, converted once into the cache, and the
    joints its actions drive with their rest pose, as mjlab's env has them."""
    from mjlab.tasks.registry import load_rl_cfg

    onnx_path = _cache() / "onnx" / f"{task_id}.onnx"
    meta_path = onnx_path.with_suffix(".json")
    if not (onnx_path.exists() and meta_path.exists()):
        checkpoint = _fetch_checkpoint(roots["orcs"], task_id)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        rl_cfg = load_rl_cfg(task_id)
        # The checkpoint holds the frozen base too, so mocke's is not loaded.
        actor = {
            k: v
            for k, v in rl_cfg.actor.items()
            if k not in ("class_name", "base_checkpoint")
        }
        spec = {"obs_groups": {"actor": list(rl_cfg.obs_groups["actor"])}, **actor}
        subprocess.run(
            [
                sys.executable,
                __file__,
                str(checkpoint),
                str(onnx_path),
                json.dumps(spec),
            ],
            env={**os.environ, "PYTHONPATH": str(roots["rsl_rl"])},
            check=True,
        )
        meta_path.write_text(json.dumps(_action_contract(env_cfg), indent=1))
    return onnx_path, json.loads(meta_path.read_text())


def checkpoint_license() -> Path:
    """The release's license text for its checkpoints, under an SPDX tag: untagged,
    mjswan reads it as the Apache-2.0 its opening grants SONIC's code."""
    from huggingface_hub import hf_hub_download

    text = Path(
        hf_hub_download(
            HF_REPO, "LICENSES/NVIDIA-OPEN-MODEL-LICENSE.txt", revision=HF_REVISION
        )
    ).read_text()
    path = _cache() / "LICENSE.checkpoints"
    path.write_text(
        f"SPDX-License-Identifier: LicenseRef-NVIDIA-Open-Model-License\n\n{text}"
    )
    return path


def _fetch_checkpoint(root: Path, task_id: str) -> Path:
    """The release's checkpoint, checked against the size and hash release.json pins."""
    from huggingface_hub import hf_hub_download

    release = json.loads((root / "src/orcs/release.json").read_text())
    spec = release["models"][task_id]["checkpoint"]
    path = Path(hf_hub_download(HF_REPO, spec["path"], revision=HF_REVISION))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.stat().st_size != spec["size_bytes"] or digest != spec["sha256"]:
        raise RuntimeError(f"{spec['path']} does not match release.json ({digest})")
    return path


def _action_contract(env_cfg: Any) -> dict[str, Any]:
    """``joint_pos``'s joints and offsets, read off the env mjlab builds from
    ``env_cfg``: the checkpoint carries no deploy contract."""
    import copy

    from mjswan.mjlab.env import build_mjlab_env

    cfg = copy.deepcopy(env_cfg)
    cfg.scene.num_envs = 1
    env = build_mjlab_env(cfg)
    try:
        term = env.action_manager.get_term("joint_pos")
        return {
            "policy_joint_names": [f"robot/{name}" for name in term.target_names],
            "default_joint_pos": term.offset[0].tolist(),
        }
    finally:
        env.close()


def _export(checkpoint: str, out: str, spec_json: str) -> None:
    """Rebuild the actor as upstream's runner config builds it, with the fork's
    ``SonicWithAdapterModel``, export it through the fork's own ONNX wrapper (LoRA folded
    into the decoder), and check it."""
    import numpy as np
    import onnxruntime as ort
    import torch
    from rsl_rl.models import SonicWithAdapterModel
    from tensordict import TensorDict

    spec = json.loads(spec_json)
    (proprio,) = spec["obs_groups"]["actor"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    actor = state["actor_state_dict"]
    widths = {
        "tokenizer": actor["encoder.0.weight"].shape[1],
        proprio: actor["decoder.base.0.weight"].shape[1]
        - actor["encoder.8.bias"].numel(),
        spec["adapter_obs_group"]: actor["adapter_normalizer._mean"].shape[1],
    }
    obs = TensorDict(
        {name: torch.zeros(1, width) for name, width in widths.items()}, batch_size=[1]
    )
    model = SonicWithAdapterModel(
        obs=obs,
        obs_set="actor",
        output_dim=actor["decoder.base.12.bias"].numel(),
        **spec,
    )
    model.load_state_dict(actor, strict=True)
    exported = model.as_onnx().eval()
    torch.onnx.export(
        exported,
        exported.get_dummy_inputs(),
        out,
        opset_version=18,
        input_names=exported.input_names,
        output_names=exported.output_names,
        dynamo=False,
    )

    generator = torch.Generator().manual_seed(0)
    sample = {
        name: torch.randn(64, width, generator=generator)
        for name, width in widths.items()
    }
    with torch.no_grad():
        reference = model.eval()(TensorDict(sample, batch_size=[64])).numpy()

    def actions() -> np.ndarray:
        session = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        feeds = [
            {k: v[i : i + 1].numpy() for k, v in sample.items()} for i in range(64)
        ]
        return np.concatenate([session.run(None, feed)[0] for feed in feeds])

    error = float(np.abs(actions() - reference).max())
    if error > 1e-4:
        Path(out).unlink()
        raise RuntimeError(f"The ONNX actor disagrees with the fork's by {error:.2e}")
    _half_decoder(out)
    error = float(np.abs(actions() - reference).max())
    print(f"float16 decoder weights move the actions by {error:.2e}")


def _half_decoder(path: str) -> None:
    """Store the decoder's weights as float16, cast back to float32 in the graph, so the
    file fits mjswan Cloud's 50 MB a file. The encoder stays float32: FSQ rounds its
    output, and a weight off by float16's precision could flip a token."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(path)
    graph = model.graph
    for init in list(graph.initializer):
        if not (init.name.startswith("decoder.") and init.name.endswith(".weight")):
            continue
        half = numpy_helper.from_array(
            numpy_helper.to_array(init).astype(np.float16), f"{init.name}.fp16"
        )
        graph.initializer.remove(init)
        graph.initializer.append(half)
        cast = helper.make_node(
            "Cast", [half.name], [init.name], to=onnx.TensorProto.FLOAT
        )
        graph.node.insert(0, cast)
    onnx.save(model, path)


if __name__ == "__main__":
    _export(*sys.argv[1:])
