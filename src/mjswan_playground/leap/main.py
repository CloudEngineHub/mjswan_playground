"""A LEAP hand turning a cube in its palm. See ``README.md``."""

from __future__ import annotations

from typing import Any

import mjswan
import onnx
from mjlab.tasks.registry import load_env_cfg

from . import terms, upstream

TASK_ID = "Mjlab-Leap-Left-Custom-HandCube-Rotate"
#: Physics steps each command waits, on every joint: the middle of the 5 to 15 that
#: upstream's `dr_action_delay` draws per episode, an event mjlab 1.6 cannot run.
COMMAND_DELAY = 10
#: Newton iterations per physics step, where upstream has 10 for mujoco_warp: MuJoCo's C
#: build, which the browser runs, needs 20 to converge on the hand's contacts. See README.
SOLVER_ITERATIONS = 20
#: Upstream's frame gizmo, drawn in mjlab's viewer only.
DROPPED_COMMANDS = ("frame_viz",)
#: Ends every episode on its first step under mjlab 1.6; see README.
DROPPED_TERMINATIONS = ("cube_pose_deviation",)


def _browser_cfg(env_cfg: Any) -> None:
    """Upstream's play config as the browser runs it, in place: the nominal hand and
    cube, and the terms in ``terms.py``."""
    for name in [name for name in env_cfg.events if name.startswith("dr_")]:
        del env_cfg.events[name]
    for name in DROPPED_COMMANDS:
        env_cfg.commands.pop(name, None)
    for name in DROPPED_TERMINATIONS:
        env_cfg.terminations.pop(name, None)
    for actuator in env_cfg.scene.entities["robot"].articulation.actuators:
        actuator.delay_min_lag = actuator.delay_max_lag = COMMAND_DELAY
        # Upstream's 1.0 would hold the lag a reset zeroes, never drawing it.
        actuator.delay_hold_prob = 0.0
    env_cfg.sim.mujoco.iterations = SOLVER_ITERATIONS
    actor = env_cfg.observations["actor"].terms
    actor["prev_commanded_joint_pos"].func = terms.joint_pos_commanded
    env_cfg.terminations["nan"].func = terms.nan_detection
    reset = env_cfg.events["reset_from_grasp_cache"]
    reset.func = terms.reset_from_grasp_cache
    reset.params.update(terms.load_grasp_cache(reset.params.pop("cache_file")))


def setup_builder() -> mjswan.Builder:
    root = upstream.resolve_root()
    upstream.register_tasks(root)
    mjswan.register_action(
        "JointPositionDeltaActionCfg",
        mjswan.ActionBinding(
            apply="joint_position", interpolate=lambda cfg: cfg.interpolate_decimation
        ),
    )
    env_cfg = load_env_cfg(TASK_ID, play=True)
    _browser_cfg(env_cfg)
    policy_path, contract = upstream.ensure_policy(root, TASK_ID, env_cfg)

    builder = mjswan.Builder()
    project = builder.add_project(name="LEAP Hand", license=root / "LICENSE")
    scene = project.add_scene_mjlab(TASK_ID, env_cfg=env_cfg)
    scene.add_policy(name="Rotate", policy=onnx.load(str(policy_path)), **contract)
    return builder
