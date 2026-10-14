"""The booster_mjlab checkout, its task ids, and the project page's walking policy."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from mjswan_playground._deps import ensure_repo

REPO_URL = "https://github.com/IntelligentRoboticsLab/booster_mjlab.git"
REPO_COMMIT = "d92e11395d961096d639446c0a8a176f12bb51da"
#: The project page's demo, whose `scene.json` lays out `policy.bin`'s float32 weights.
DEMO = "website/static/demo"
TASK_PACKAGE = "booster_mjlab.tasks"


def resolve_root() -> Path:
    return ensure_repo(
        name="booster_mjlab",
        url=REPO_URL,
        commit=REPO_COMMIT,
        marker=f"{DEMO}/policy.bin",
        root_env_var="MJSWAN_BOOSTER_ROOT",
    )


def register_tasks(root: Path) -> None:
    """Import upstream's task package from ``src/`` so mjlab's registry knows it."""
    if TASK_PACKAGE in sys.modules:
        return
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    importlib.import_module(TASK_PACKAGE)


def demo_policy(root: Path) -> tuple[onnx.ModelProto, dict]:
    """The demo's MLP as ONNX, rebuilt from ``policy.bin`` by inverting
    ``export_web_demo.export_policy``, and the ``scene.json`` that lays it out."""
    scene = json.loads((root / DEMO / "scene.json").read_text())
    spec = scene["policy"]
    blob = np.fromfile(root / DEMO / spec["file"], dtype="<f4")

    def weight(name: str, offset: int, *shape: int) -> str:
        data = blob[offset : offset + int(np.prod(shape))].reshape(shape)
        weights.append(numpy_helper.from_array(data, name))
        return name

    nodes, weights = [], []
    x = "obs"
    for i, layer in enumerate(spec["layers"]):
        out = f"x{i}"
        if layer["op"] == "gemm":
            w = weight(f"w{i}", layer["w_offset"], layer["out"], layer["in"])
            b = weight(f"b{i}", layer["b_offset"], layer["out"])
            nodes.append(helper.make_node("Gemm", [x, w, b], [out], transB=1))
        elif layer["op"] == "elu":
            nodes.append(helper.make_node("Elu", [x], [out], alpha=layer["alpha"]))
        else:  # The normalizer's `sub` and `div`, the observation first.
            c = weight(f"c{i}", layer["offset"], layer["size"])
            nodes.append(helper.make_node(layer["op"].capitalize(), [x, c], [out]))
        x = out
    nodes[-1].output[0] = "actions"

    def io(name: str, width: int) -> onnx.ValueInfoProto:
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, width])

    graph = helper.make_graph(
        nodes,
        "booster_k1_velocity",
        [io("obs", spec["input_dim"])],
        [io("actions", spec["output_dim"])],
        weights,
    )
    # IR 8 and opset 17, as torch exports them: onnxruntime reads no IR past 10.
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8
    )
    return model, scene
