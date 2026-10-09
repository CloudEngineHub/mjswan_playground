"""The jumper's servos as the browser runs them.

Upstream drives every joint through one ``ServoCurveActuatorCfg``, an explicit PD that
holds the joints its action leaves out at the position target an event writes. The
browser applies PD to the action's joints alone and zeroes every other ``ctrl``, so those
joints get MuJoCo position actuators of the same gains here, biased to hold their pose at
``ctrl = 0``. And mjlab's reset clears every position target, so the first observation of
an episode carries the servos pushing toward it; the browser resets to the keyframe
instead, and its torque is restated here for that one observation.
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


def hold_unactuated(env_cfg: Any, contract: dict) -> None:
    """Hold the joints the action leaves out where the deploy contract says they stay,
    with actuators of the servos' gains and limit, and reset them there."""
    hold = {name: float(value) for name, value in contract["unactuated_joints"].items()}
    default = contract["default_joint_pos"]
    if any(default[name] != value for name, value in hold.items()):
        raise ValueError("A held joint's default pose is not its hold.")
    entity = env_cfg.scene.entities["robot"]
    (servo,) = entity.articulation.actuators
    entity.articulation.actuators = (
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
    # Upstream's one actuator lists every joint in model order, and the observed forces
    # are read in actuator order: keep it.
    entity.sort_actuators = True
    # mjlab's reset leaves a joint outside every event where it was; the browser's
    # puts it back at the keyframe. Upstream's joint reset is zero-width, so widened to
    # every joint it writes the default pose, the hold, in both.
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


def observe_reset_force(
    env_cfg: Any, term_name: str, *, hold: dict[str, float] | None = None
) -> None:
    """Point the ``actuator_force`` term ``term_name`` at :func:`servo_force`, over the
    same actuators in the same order and with the servos' gains. Each actuator drives the
    joint it is named after, in the joints' model order; ``hold`` names the joints whose
    reset target is not 0."""
    hold = hold or {}
    term = env_cfg.observations["actor"].terms[term_name]
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


def hold_claws(env_cfg: Any, contract: dict) -> None:
    """Hold the claws with MuJoCo position actuators of the servos' gains.

    Upstream's servo PD holds them at their target, which no action moves. The browser
    applies PD to the action's joints alone, so the claws would hang unpowered.
    """
    claws = contract["unactuated_joints"]
    if any(claws.values()):
        raise ValueError(f"The claws hold {claws}, but a position target starts at 0.")
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
