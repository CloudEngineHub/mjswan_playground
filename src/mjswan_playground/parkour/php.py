"""PHP's G1: five teachers, each tracking a clip over its boxes."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from mjlab.entity import EntityCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg

from mjswan_playground._trace import nan_detection

from . import upstream


def env_cfg(task_id: str, clip: Path) -> Any:
    """Upstream's play config as the browser runs it: mjlab's ``MotionCommand`` over the
    clip in its layout, from the first frame, and boxes where the clip's terrain variant
    0 puts them, as play's ``fixed_variant_id=0`` does."""
    from src.tasks.php.dataset import read_terrain
    from src.tasks.php.mdp.terrain import box_spec

    cfg = load_env_cfg(task_id, play=True)
    php = cfg.commands["motion"]
    robot = cfg.scene.entities["robot"].spec_fn().compile()
    npz = upstream.clip_npz(
        clip,
        [robot.joint(i).name for i in range(robot.njnt) if robot.jnt_type[i] != 0],
        [robot.body(i).name for i in range(1, robot.nbody)],
    )
    cfg.commands["motion"] = MotionCommandCfg(
        entity_name=php.entity_name,
        motion_file=str(npz),
        anchor_body_name=php.anchor_body_name,
        body_names=php.body_names,
        resampling_time_range=php.resampling_time_range,
        debug_vis=php.debug_vis,
        pose_range=php.pose_range,
        velocity_range=php.velocity_range,
        joint_position_range=php.joint_position_range,
        sampling_mode="start",
        viz=php.viz,
    )
    for name in [n for n in cfg.scene.entities if n.startswith("php_box_")]:
        del cfg.scene.entities[name]
    for i, box in enumerate(read_terrain(clip)[:, 0]):
        cfg.scene.entities[f"php_box_{i}"] = EntityCfg(
            spec_fn=functools.partial(_box, box_spec, tuple(box[7:] / 2)),
            init_state=EntityCfg.InitialStateCfg(
                pos=tuple(box[:3]), rot=tuple(box[3:7])
            ),
        )
    # Declares the box fields `place_boxes` resizes; static boxes resize nothing.
    cfg.events.pop("php_box_fields")
    # Upstream's clip end, a time-out: mjswan's tracking command restarts the episode
    # when its clip loops.
    cfg.terminations.pop("motion_end")
    cfg.terminations["nan"].func = nan_detection
    return cfg


def _box(box_spec: Any, half_size: tuple[float, ...]) -> Any:
    spec = box_spec()
    spec.geoms[0].size = half_size
    return spec
