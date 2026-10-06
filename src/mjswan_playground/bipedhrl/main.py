"""Unitree H1-2 walking to velocity commands. See ``README.md``."""

from __future__ import annotations

import mjswan
import onnx
from mjlab.tasks.registry import load_env_cfg
from mjswan.mjlab.onnx_meta import read_mjlab_metadata

from . import terms, upstream

TASK_ID = "Unitree-H1_2-Flat"
GAIT_CLOCK = "gait_clock"


def _clock_phase(env_cfg) -> None:
    """Point the ``phase`` observation at a gait clock the browser can keep."""
    term = env_cfg.observations["actor"].terms["phase"]
    period_steps = term.params["period"] / (
        env_cfg.sim.mujoco.timestep * env_cfg.decimation
    )
    steps_per_period = round(period_steps)
    if abs(period_steps - steps_per_period) > 1e-6:
        raise ValueError(
            f"The gait period is not a whole number of steps ({period_steps})."
        )
    env_cfg.commands[GAIT_CLOCK] = terms.GaitClockCommandCfg(
        steps_per_period=steps_per_period
    )
    term.func = terms.phase
    term.params = {
        "clock_name": GAIT_CLOCK,
        "steps_per_period": steps_per_period,
        "command_name": term.params["command_name"],
    }


def setup_builder() -> mjswan.Builder:
    root = upstream.resolve_root()
    upstream.register_tasks(root)
    policy = onnx.load(str(root / upstream.POLICY_ONNX))
    contract = read_mjlab_metadata(policy)
    env_cfg = load_env_cfg(TASK_ID, play=True)
    _clock_phase(env_cfg)

    builder = mjswan.Builder()
    project = builder.add_project(name="H1-2 Velocity")
    scene = project.add_scene_mjlab(TASK_ID, env_cfg=env_cfg)
    scene.add_policy(
        name="A0",
        policy=policy,
        policy_joint_names=[f"robot/{name}" for name in contract.joint_names],
        default_joint_pos=contract.default_joint_pos,
    )
    return builder
