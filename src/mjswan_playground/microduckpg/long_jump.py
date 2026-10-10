"""Long jump: microduck jumps a 30 cm gap from one block down onto another 25 cm lower,
lands and stands."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

import mjswan
import mujoco
import torch
from huggingface_hub import hf_hub_download
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjswan.managers.termination_manager import TerminationTermCfg
from mjswan.mjlab import apply_mjlab_sim_options, build_single_entity_trace_env

from mjswan_playground.microduck.main import CONTROL_DT, ENTITY, VELOCITY_ENV, _padded

from . import _common

POLICY_REPO_ID = "HannesVonEssen/microduck-long-jump"
POLICY_REVISION = "fdcdc787e72b8784bb842159ee70607859cd17ed"

ROBOT_PY = "src/mjlab_microduck/robot/long_jump_robot.py"
STAGE_PY = "src/mjlab_microduck/robot/platform_stage.py"

#: ``platform_jump_mdp``'s ``GAP_SCALE`` and ``DROP_SCALE``.
COURSE_SCALE = 0.3
#: ``reset_platform_jump_state``'s standing spawn, at the middle of its 3 to 10 cm edge
#: draw plus the 2 cm it adds: the trunk 8.5 cm behind A's front edge, 11.8 cm above it.
SPAWN_X = -(0.065 + 0.02)
SPAWN_HEIGHT = 0.118
#: The ankle body sits 2.3 cm above its sole, so a foot on the floor reads about 0.02
#: and one on B at least 0.11.
PIT_HEIGHT = 0.06


def course(env: Any, *, gap: float, drop: float, **_) -> torch.Tensor:
    """``pj_command_obs``, ``[gap, drop, 0, 0, 0, 0] / 0.3``: all the blind policy knows
    of the course."""
    out = torch.zeros(env.num_envs, 6, device=env.device)
    out[:, 0] = gap / COURSE_SCALE
    out[:, 1] = drop / COURSE_SCALE
    return out


def feet_below(
    env: Any, *, height: float, asset_cfg: SceneEntityCfg, **_
) -> torch.Tensor:
    """A foot in the pit: any of ``asset_cfg``'s bodies below ``height``.

    ``pj_fell`` reads a sole touching the floor; the trace env carries no contact
    sensors, so this reads the ankle's height instead."""
    z = env.scene[asset_cfg.name].data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    return (z < height).any(dim=-1)


def _scene_spec(
    root: Path,
    home: dict[str, float],
    gap: float,
    drop: float,
    *,
    tracing: bool = False,
) -> mujoco.MjSpec:
    """``scene.xml`` with ``long_jump_robot``'s hulls and leg-fold pairs under
    ``FULL_COLLISION``, ``platform_stage``'s blocks as static world geoms at this course,
    mjlab's velocity sim settings and the servo filters; ``tracing`` leaves out the last
    two and the mesh copies, as nothing traced reads them."""
    robot = _common.load_upstream(root, ROBOT_PY)
    stage = _common.load_upstream(root, STAGE_PY)
    spec = mujoco.MjSpec.from_file(str(root / _common.ALLCOLLISIONS_SCENE_XML))
    robot.add_full_collision_geoms(spec)
    robot.add_leg_fold_pairs(spec)
    _common.full_collision(spec)
    for block, center in (
        (stage.platform_a_spec(), stage.platform_a_center(drop)),
        (stage.platform_b_spec(), stage.platform_b_center(gap)),
    ):
        (box,) = block.geoms
        spec.worldbody.add_geom(
            name=box.name,
            type=box.type,
            size=box.size,
            pos=center,
            rgba=box.rgba,
            friction=box.friction,
            condim=box.condim,
            priority=box.priority,
            solref=box.solref,
            solimp=box.solimp,
        )
    if not tracing:
        apply_mjlab_sim_options(spec, VELOCITY_ENV.sim)
        _common.filter_servos(spec)
        _common.own_collision_meshes(spec)
    root_pose = (SPAWN_X, 0.0, stage.a_top(drop) + SPAWN_HEIGHT, 1.0, 0.0, 0.0, 0.0)
    _common.set_only_keyframe(spec, home, root_pose)
    return spec


def add_scenes(project: mjswan.ProjectHandle, root: Path) -> None:
    # The release's deploy contract: joint order and offsets, and the course it jumps.
    release = json.loads(
        Path(
            hf_hub_download(POLICY_REPO_ID, "config.json", revision=POLICY_REVISION)
        ).read_text()
    )
    home = dict(zip(release["joint_names"], release["joint_offset_rad"], strict=True))
    gap = float(release["environment"]["MICRODUCK_PJ_GAP"])
    drop = float(release["environment"]["MICRODUCK_PJ_DROP"])
    spec = _scene_spec(root, home, gap, drop)
    joint_names = _common.servo_joints(spec)

    scene = project.add_scene(name="Long Jump", spec=spec, control_dt=CONTROL_DT)
    scene.add_attribution("3d-models", license=root / "LICENSE-HARDWARE")
    # Side-on and still, so the gap shows between the blocks through the whole jump.
    scene.set_viewer(
        mjswan.ViewerConfig(
            origin_type=mjswan.ViewerConfig.OriginType.WORLD,
            lookat=(0.2, 0.0, 0.36),
            distance=1.3,
            elevation=-12.0,
            azimuth=95.0,
        )
    )
    scene.set_trace_env(
        build_single_entity_trace_env(
            partial(_scene_spec, root, home, gap, drop, tracing=True),
            entity_name=ENTITY,
        )
    )
    scene.add_policy(
        name="Jump",
        policy=_common.hub_policy(POLICY_REPO_ID, "policy.onnx", POLICY_REVISION),
        commands={},
        # 48 of proprioception, then twist(3), head(4), body(6) as config.json lays out.
        observations=ObservationGroupCfg(
            terms={
                **_common.proprioception(_common.joints_cfg(joint_names)),
                "twist": _padded(3),
                "head_command": _padded(4),
                "course": ObservationTermCfg(
                    func=course, params={"gap": gap, "drop": drop}
                ),
            }
        ),
        actions=_common.servo_action(),
        terminations={
            "fell_over": VELOCITY_ENV.terminations["fell_over"],
            "in_pit": TerminationTermCfg(
                func=feet_below,
                params={
                    "height": PIT_HEIGHT,
                    "asset_cfg": SceneEntityCfg(
                        ENTITY, body_names=("ankle_left", "ankle_right")
                    ),
                },
            ),
        },
        policy_joint_names=joint_names,
        default_joint_pos=[home[name] for name in joint_names],
    )
