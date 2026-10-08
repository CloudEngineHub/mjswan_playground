"""KingKong Robotics' jumper walking and holding a commanded body posture. See
``README.md``."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

import mjswan
import onnx
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjswan.mjlab.observation import adapt_observations

from . import terms, upstream

TASK_ID = upstream.TASK_ID


def _as(cls: type, cfg: Any) -> Any:
    """``cfg`` rebuilt as its base class ``cls``, dropping the subclass's fields."""
    return cls(**{f.name: getattr(cfg, f.name) for f in fields(cls)})


def _unstride(env_cfg: Any) -> dict[str, tuple[int, ...]]:
    """Unwrap upstream's ``StridedHistory`` from each actor term, returning the frames it
    stacked as look-back offsets, oldest first."""
    from tasks.jumper.posture.mdp.history import StridedHistory

    prefix = StridedHistory.PREFIX
    offsets = {}
    for name, term in env_cfg.observations["actor"].terms.items():
        if term.func is not StridedHistory:
            continue
        params = term.params
        frames, stride = params[prefix + "frames"], params[prefix + "stride"]
        offsets[name] = tuple(stride * i for i in reversed(range(frames)))
        term.func = params[prefix + "func"]
        term.params = {k: v for k, v in params.items() if not k.startswith(prefix)}
    return offsets


def _hold_claws(env_cfg: Any, contract: dict) -> None:
    """Hold the claws with MuJoCo position actuators of the servos' gains.

    Upstream's servo PD holds them at their target, which no action moves. The browser
    applies PD to the action's joints alone, so the claws would hang unpowered.
    """
    claws = contract["unactuated_joints"]
    if any(claws.values()):
        raise ValueError(f"A claw is held off zero, the target ctrl starts at: {claws}")
    articulation = env_cfg.scene.entities["robot"].articulation
    (servo,) = articulation.actuators
    articulation.actuators = (
        replace(servo, target_names_expr=tuple(contract["action_joint_order"])),
        BuiltinPositionActuatorCfg(
            target_names_expr=tuple(claws),
            stiffness=servo.stiffness,
            damping=servo.damping,
            effort_limit=servo.effort_limit,
            armature=servo.armature,
        ),
    )


def play_env_cfg() -> tuple[Any, dict[str, tuple[int, ...]]]:
    """Upstream's play config as the browser runs it, and each strided term's offsets."""
    root = upstream.resolve_root()
    upstream.register_tasks(root)
    from tasks.jumper.posture.mdp.commands import PostureCommandCfg

    env_cfg = load_env_cfg(TASK_ID, play=True)
    _hold_claws(env_cfg, upstream.deployed_contract(root))
    # Play's operator terms read a keyboard or a pad; the browser's panel does instead.
    env_cfg.commands["twist"] = _as(
        UniformVelocityCommandCfg, env_cfg.commands["twist"]
    )
    env_cfg.commands["posture"] = _as(PostureCommandCfg, env_cfg.commands["posture"])
    # Play adds the ToF sensor for its viewer alone; no term reads it.
    env_cfg.scene.sensors = tuple(s for s in env_cfg.scene.sensors if s.name != "tof")
    offsets = _unstride(env_cfg)
    terms.clock_gait_phase(env_cfg)
    return env_cfg, offsets


def setup_builder() -> mjswan.Builder:
    root = upstream.resolve_root()
    contract = upstream.deployed_contract(root)
    env_cfg, offsets = play_env_cfg()

    builder = mjswan.Builder()
    project = builder.add_project(name="jumper", license=root / "LICENSE")
    project.set_notice(root / "NOTICE")
    scene = project.add_scene_mjlab(TASK_ID, env_cfg=env_cfg)
    observations = adapt_observations(env_cfg.observations["actor"])
    for name, steps in offsets.items():
        observations["actor"].terms[name].history_steps = steps
    joints = contract["action_joint_order"]
    scene.add_policy(
        name="model_74800",
        policy=onnx.load(str(root / upstream.POLICY_ONNX)),
        mdp=mjswan.MdpConfig(observations=observations),
        policy_joint_names=[f"robot/{name}" for name in joints],
        default_joint_pos=[contract["default_joint_pos"][name] for name in joints],
    )
    return builder
