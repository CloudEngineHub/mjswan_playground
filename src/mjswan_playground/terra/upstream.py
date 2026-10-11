"""TERRA-4B and the two routes it walks: the checkpoint as ONNX, myosuite tutorial 4.4's
course and the registered task's flat route, and the reference planned for each.

TERRA tracks a reference; it does not plan a route. Each route's reference comes from
myosuite's ``plan_waypoint_reference`` with the tutorial's gait, planned once into
``.cache/terra/`` since planning takes about 40 s.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from mjswan_playground._deps import CACHE_DIR

#: The policy's and the reference's control step: 100 Hz, five 2 ms physics steps.
CTRL_DT = 0.01
_CACHE = CACHE_DIR / "terra"


def ensure_policy() -> tuple[Path, Any]:
    """TERRA-4B's actor as ONNX, behind its frozen observation normalizer, and the
    checkpoint's ``TerrainObsCfg``. The ONNX runs the mean action, where the tutorial
    samples one.

    Converted once into the cache, the way ``musclemimic.upstream`` converts mm-10m-2,
    and checked against myosuite's numpy forward pass. At 24 MiB it fits a 25 MiB file
    cap, so the weights stay float32.
    """
    from myosuite.integrations.musclemimic.terrain_observation import TerrainObsCfg
    from myosuite.integrations.musclemimic.terrain_policy import TERRA_CHECKPOINT

    path = _CACHE / f"terra_4b_{TERRA_CHECKPOINT}.onnx"
    cfg_path = path.with_suffix(".json")
    if not (path.exists() and cfg_path.exists()):
        _convert(path, cfg_path)
    record = json.loads(cfg_path.read_text())
    return path, TerrainObsCfg(
        **{k: tuple(v) if isinstance(v, list) else v for k, v in record.items()}
    )


def _convert(path: Path, cfg_path: Path) -> None:
    import onnxruntime as ort
    import torch
    from myosuite.integrations.musclemimic.actor_onnx import export_to_onnx
    from myosuite.integrations.musclemimic.actor_torch import MimicActorModule
    from myosuite.integrations.musclemimic.fullbody_local_policy import (
        _actor_forward,
        load_local_policy_artifacts,
        read_checkpoint_config_metadata,
    )
    from myosuite.integrations.musclemimic.terrain_observation import TerrainObsCfg
    from myosuite.integrations.musclemimic.terrain_policy import (
        download_terrain_checkpoint,
    )

    root = download_terrain_checkpoint()
    # Raises on a layout other than the one `terms.observation` restates.
    cfg = TerrainObsCfg.from_config(read_checkpoint_config_metadata(root))
    artifacts = load_local_policy_artifacts(root)
    module = MimicActorModule.from_artifacts(artifacts)
    export_to_onnx(module, path)

    sample = (
        np.random.default_rng(0)
        .normal(
            artifacts.obs_mean,
            np.sqrt(artifacts.obs_var) + 1e-3,
            (16, artifacts.obs_dim),
        )
        .astype(np.float32)
    )
    reference = _actor_forward(
        artifacts.params,
        module.normalize(torch.from_numpy(sample)).detach().numpy(),
    )
    exported = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(
        ["actions"], {"obs": sample}
    )[0]
    error = float(np.abs(exported - reference).max())
    if error > 1e-4:
        path.unlink()
        raise RuntimeError(f"Exported actor disagrees with myosuite's by {error:.2e}")
    cfg_path.write_text(json.dumps(dataclasses.asdict(cfg), indent=1))


# --- The scenes: tutorial 4.4's, verbatim but for the split ------------------------


def _ground(spec: mujoco.MjSpec) -> None:
    """The tutorial's floor and lighting: the floor in geom group 2 with the terrain's
    friction, where TERRA's heightmap looks."""
    for texture in spec.textures:
        if texture.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            texture.rgb1 = [0.80, 0.86, 0.92]
            texture.rgb2 = [0.94, 0.95, 0.96]
    spec.visual.headlight.ambient = [0.5, 0.5, 0.5]
    spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]
    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            geom.material = ""
            geom.rgba = [0.86, 0.86, 0.83, 1]
            geom.group = 2
            geom.friction = [0.8, 0.005, 0.0001]


def course_scene(
    spec: mujoco.MjSpec, rise: float = 0.06, beam_width: float = 0.36, gap: float = 0.20
) -> None:
    """Stairs up, a beam with a right-angle turn, a jump, stairs down and a second jump."""
    _ground(spec)

    def box(name: str, x: float, y: float, width: float, length: float, height: float):
        spec.worldbody.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            group=2,
            pos=[x, y, height / 2],
            size=[width / 2, length / 2, height / 2],
            contype=1,
            conaffinity=1,
            condim=3,
            friction=[0.8, 0.005, 0.0001],
            rgba=[0.36, 0.49, 0.56, 1],
        )

    for i in range(4):
        box(f"course_up_{i}", 0, -0.8 - (i + 0.5) * 0.3, 1, 0.3, (i + 1) * rise)
    top = 4 * rise
    box("course_top", 0, -2.3, 0.8, 0.6, top)
    box("course_beam_east", 1.1, -2.3, 1.4, beam_width, top)
    box("course_turn", 2.1, -2.3, 0.6, 0.8, top)
    box("course_beam_south", 2.1, -3.0, beam_width, 0.6, top)
    box("course_jump_landing", 2.1, -3.6 - gap, 0.8, 0.6, top)
    for i in range(3):
        box(f"course_down_{i}", 2.1, -4.05 - gap - i * 0.3, 1, 0.3, (3 - i) * rise)
    box("course_jump2_landing", 2.1, -5.1 - 2 * gap, 0.8, 0.6, rise)


_GAP = 0.20
_COURSE_WAYPOINTS = (
    (0, -1.1),
    (0, -2.3),
    (1.05, -2.3),
    (2.1, -2.3),
    (2.1, -3.1),
    (2.1, -3.6 - _GAP),
    (2.1, -4.7 - _GAP),
    (2.1, -5.1 - 2 * _GAP),
    (2.1, -6.0 - 2 * _GAP),
)


def _flat_route() -> tuple[np.ndarray, float]:
    """``myoFullBodyWaypoint-v0``'s route at reset seed 0, start first, and its arrival
    radius: the registered task as it is, on its own model. Cached with the references."""
    path = _CACHE / "flat_route.json"
    if not path.exists():
        from myosuite import make_env

        env = make_env("myoFullBodyWaypoint-v0")
        env.reset(seed=0)
        base = env.unwrapped
        record = {
            "route": np.vstack([base._start, base.waypoints]).tolist(),
            "arrival_radius": float(base.task.arrival_radius),
        }
        env.close()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1))
    record = json.loads(path.read_text())
    return np.asarray(record["route"]), record["arrival_radius"]


@dataclasses.dataclass(frozen=True)
class Route:
    """One scene: its terrain, the waypoints TERRA walks through, and its reference.

    Attributes:
        name: Scene name.
        scene_fn: ``MjSpec`` edit adding the terrain, as myosuite's ``edit_fn``.
        jump_segments: Route segments the reference jumps, counting from the start.
        route: Start then waypoints, ``(W + 1, 2)``.
        arrival_radius: The waypoint radius the scene marks.
    """

    name: str
    scene_fn: Callable[[mujoco.MjSpec], None]
    jump_segments: tuple[int, ...]
    route: np.ndarray
    arrival_radius: float

    def actor_spec(self) -> mujoco.MjSpec:
        """TERRA's actor in the scene, the waypoints marked: myosuite's
        ``build_terrain_fullbody_spec``, which rebuilds the ``musclemimic_models``
        1.0.5 body TERRA trained on with its wrap fixes and solver settings."""
        from myosuite.integrations.musclemimic import build_terrain_fullbody_spec

        spec = build_terrain_fullbody_spec(self.scene_fn)
        model = spec.compile()
        # mjlab attaches this spec into a scene of its own, whose compiler does not raise
        # the smallest bodies to TERRA's 1 g and 1e-4 kg m^2: pin what this one compiled.
        for i, body in enumerate(spec.bodies[1:], start=1):
            body.explicitinertial = True
            body.mass = model.body_mass[i]
            body.ipos = model.body_ipos[i]
            body.iquat = model.body_iquat[i]
            body.inertia = model.body_inertia[i]
            body.fullinertia = np.full(6, np.nan)
        heights = _heights(model)
        for i, (x, y) in enumerate(self.route[1:]):
            # Visual only: no contact, and out of group 2, where the heightmap looks.
            spec.worldbody.add_geom(
                name=f"waypoint_{i + 1}",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                group=1,
                pos=[x, y, heights((x, y)) + 0.003],
                size=[self.arrival_radius, 0.002, 0],
                contype=0,
                conaffinity=0,
                rgba=[1.0, 0.55, 0.0, 0.5],
            )
        return spec

    def initial_qpos(self, model: mujoco.MjModel) -> np.ndarray:
        """``qpos0`` moved onto the route's start, as the tutorial spawns."""
        qpos = model.qpos0.copy()
        qpos[:2] = self.route[0]
        return qpos


def _heights(model: mujoco.MjModel) -> Callable[[tuple[float, float]], float]:
    from myosuite.integrations.musclemimic import TerrainHeights

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    heights = TerrainHeights(model, top=10.0)
    return lambda xy: float(heights(data, [xy])[0])


def routes() -> tuple[Route, ...]:
    """The course, then the registered task's flat route, in the order the scenes are
    listed."""
    flat_route, flat_radius = _flat_route()
    return (
        Route(
            name="TERRA-Course",
            scene_fn=course_scene,
            jump_segments=(5, 7),
            route=np.vstack([np.zeros(2), np.asarray(_COURSE_WAYPOINTS)]),
            arrival_radius=0.12,
        ),
        Route(
            name="TERRA-Flat",
            scene_fn=_ground,
            jump_segments=(),
            route=flat_route,
            arrival_radius=flat_radius,
        ),
    )


def reference(route: Route, model: mujoco.MjModel) -> Any:
    """The route's reference as a ``MotionClip`` at 100 Hz, planned with the tutorial's
    gait (0.45 s flight, 16 cm tuck on both jumps).

    The planner's poses are cached; the clip is rebuilt from them with the same call
    the planner ends on, so its velocities and site kinematics come out identical.
    """
    from myosuite.core.trajectory_io import motion_clip_from_states
    from myosuite.integrations.musclemimic import MIMIC_SITES

    path = _CACHE / f"{route.name.lower()}_reference.npz"
    if not path.exists():
        _plan(route, path)
    qpos = np.load(path)["qpos"]
    return motion_clip_from_states(model, qpos, CTRL_DT, MIMIC_SITES)


def _plan(route: Route, path: Path) -> None:
    from myosuite.integrations.musclemimic import (
        MIMIC_SITES,
        GaitParameters,
        build_terrain_fullbody_spec,
        plan_waypoint_reference,
    )

    model = build_terrain_fullbody_spec(route.scene_fn).compile()
    clip, _ = plan_waypoint_reference(
        model,
        route.route,
        _heights(model),
        route.initial_qpos(model),
        sites=MIMIC_SITES,
        jump_segments=route.jump_segments,
        parameters=GaitParameters(
            jump_flight_s=0.45, jump_tuck=0.16, second_jump_tuck=0.16
        ),
    )
    if not math.isclose(1.0 / clip.frequency_hz, CTRL_DT):
        raise RuntimeError(f"The planner's reference runs at {clip.frequency_hz} Hz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, qpos=clip.qpos)
