"""The jumper's servos as the browser runs them.

The browser applies PD to the action's joints alone and zeroes every other ``ctrl``, so
the joints upstream's one servo also holds get position actuators of its gains, biased to
hold their pose at ``ctrl = 0``. mjlab's reset clears every position target, so an
episode's first ``actuator_force`` is restated as that reset leaves it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import mujoco
import torch
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.actuator.builtin_actuator import BuiltinPositionActuator
from mjlab.managers.scene_entity_config import SceneEntityCfg


@dataclass(kw_only=True)
class HeldPositionActuatorCfg(BuiltinPositionActuatorCfg):
    """A ``<position>`` actuator per joint whose bias holds ``hold[joint]`` at
    ``ctrl = 0``: force ``kp * (ctrl + hold - q) - kd * qdot``."""

    hold: dict[str, float]

    def build(self, entity: Any, target_ids: list[int], target_names: list[str]) -> Any:
        # mjlab batches builtin actuators by their exact type, so this stays one and
        # biases the elements it adds.
        actuator = BuiltinPositionActuator(self, entity, target_ids, target_names)
        add = actuator.edit_spec

        def edit_spec(spec: mujoco.MjSpec, names: list[str]) -> None:
            start = len(actuator._mjs_actuators)
            add(spec, names)
            for element, name in zip(actuator._mjs_actuators[start:], names):
                element.biasprm[0] = self.stiffness * self.hold[name]

        actuator.edit_spec = edit_spec
        return actuator


def _split_servo(env_cfg: Any, contract: dict) -> dict[str, float]:
    """Keep the servo on the action's joints, and hold the rest where the deploy contract
    says they stay with actuators of its gains and limit. Returns the holds."""
    hold = {name: float(value) for name, value in contract["unactuated_joints"].items()}
    articulation = env_cfg.scene.entities["robot"].articulation
    (servo,) = articulation.actuators
    articulation.actuators = (
        replace(servo, target_names_expr=tuple(contract["action_joint_order"])),
        HeldPositionActuatorCfg(
            target_names_expr=tuple(hold),
            stiffness=servo.stiffness,
            damping=servo.damping,
            effort_limit=servo.effort_limit,
            armature=servo.armature,
            hold=hold,
        ),
    )
    return hold


def hold_claws(env_cfg: Any, contract: dict) -> None:
    """Hold the claws at 0 rad, where their position target starts."""
    if any(_split_servo(env_cfg, contract).values()):
        raise ValueError("A claw holds away from 0, where a position target starts.")


def hold_unactuated(env_cfg: Any, contract: dict) -> None:
    """Hold the joints the action leaves out, and reset them there."""
    hold = _split_servo(env_cfg, contract)
    default = contract["default_joint_pos"]
    if any(default[name] != value for name, value in hold.items()):
        raise ValueError("A held joint's default pose is not its hold.")
    # `ctrl` in joint order, as upstream's one actuator has it.
    env_cfg.scene.entities["robot"].sort_actuators = True
    # The browser resets every joint to the keyframe; widened to every joint, upstream's
    # zero-width reset does the same in mjlab.
    reset = env_cfg.events["reset_robot_joints"]
    if any(reset.params[k] != (0.0, 0.0) for k in ("position_range", "velocity_range")):
        raise ValueError("The joint reset randomizes, and would move the held joints.")
    reset.params["asset_cfg"] = SceneEntityCfg("robot")


def servo_force(
    env: Any,
    asset_cfg: SceneEntityCfg,
    joint_ids: list[int],
    reset_target: torch.Tensor,
    stiffness: float,
    damping: float,
    effort_limit: float,
) -> torch.Tensor:
    """mjlab's ``actuator_force``, with an episode's first observation as mjlab's reset
    leaves it: each servo's PD toward its cleared target, ``reset_target`` (0, or a held
    joint's hold), within the effort limit."""
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, joint_ids]
    qd = asset.data.joint_vel[:, joint_ids]
    reset = (stiffness * (reset_target - q) - damping * qd).clamp(
        -effort_limit, effort_limit
    )
    first = (env.episode_length_buf == 0).unsqueeze(-1)
    return torch.where(
        first, reset, asset.data.actuator_force[:, asset_cfg.actuator_ids]
    )


def observe_reset_force(env_cfg: Any, hold: dict[str, float] | None = None) -> None:
    """Point the ``actuator_force`` term at :func:`servo_force`, over the same actuators
    in the same order and with the servos' gains. Each actuator drives the joint it is
    named after, in the joints' model order; ``hold`` names the joints whose reset target
    is not 0."""
    hold = hold or {}
    term = env_cfg.observations["actor"].terms["actuator_force"]
    servo = env_cfg.scene.entities["robot"].articulation.actuators[0]
    spec = env_cfg.scene.entities["robot"].spec_fn()
    joints = [j.name for j in spec.joints if j.type != mujoco.mjtJoint.mjJNT_FREE]
    asset_cfg = (term.params or {}).get("asset_cfg") or SceneEntityCfg("robot")
    names = list(asset_cfg.actuator_names or joints)
    if not asset_cfg.preserve_order:
        names.sort(key=joints.index)
    term.func = servo_force
    term.params = {
        "asset_cfg": SceneEntityCfg("robot", actuator_names=names, preserve_order=True),
        "joint_ids": [joints.index(name) for name in names],
        "reset_target": torch.tensor([[hold.get(name, 0.0) for name in names]]),
        "stiffness": float(servo.stiffness),
        "damping": float(servo.damping),
        "effort_limit": float(servo.effort_limit),
    }
