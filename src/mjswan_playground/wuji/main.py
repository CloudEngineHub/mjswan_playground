"""Wuji Hand 1 and Hand 2 turning a cube to commanded orientations, and Hand 2
spinning a pen. See ``README.md``."""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import Any

import mjswan
import mujoco
import onnx
from mjlab.envs.mdp import observations as obs_fns
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg
from mjswan.mjlab.observation import adapt_observations

from . import pen_spin, terms, upstream

#: Each scene's upstream task, and the bundle's checkpoint trained on it.
HANDS = {"WujiHand2_Reorient_50Hz": "hand2", "WujiHand_Reorient": "hand1"}


def _add_hands(project: Any, root: Path, bundle: Path) -> None:
    for task_id, hand in HANDS.items():
        policy, config = upstream.checkpoint(bundle, hand)
        scene = _add_hand(project, task_id, hand, policy, config)
        upstream.add_attributions(scene, bundle)


#: Parts built on their own, in the order the whole lists their scenes.
PARTS = {"wuji-reorient": _add_hands, "wuji-penspin": pen_spin.add_scene}


def setup_builder(part: str | None = None) -> mjswan.Builder:
    root = upstream.resolve_root()
    upstream.register_tasks(root)
    bundle = upstream.resolve_bundle()
    builder = mjswan.Builder()
    project = builder.add_project(name="wuji-mjlab", license=root / "LICENSE")
    project.set_notice(root / "NOTICE")
    for add_scenes in [PARTS[part]] if part else PARTS.values():
        add_scenes(project, root, bundle)
    return builder


def _add_hand(project: Any, task_id: str, hand: str, policy: Path, config: dict) -> Any:
    env_cfg = play_env_cfg(task_id, config)
    scene = project.add_scene_mjlab(task_id, env_cfg=env_cfg)
    observations = adapt_observations(env_cfg.observations["policy"])
    # Upstream observes the action manager's `prev_action`, the action before the last.
    observations["actor"].terms["action_history"].history_steps = (3, 2, 1)
    scene.add_policy(
        name=hand,
        policy=onnx.load(str(policy)),
        observations=observations,
        policy_joint_names=[f"robot/{joint}" for joint in config["joint_order"]],
        default_joint_pos=config["default_joint_pos"],
    )
    return scene


def play_env_cfg(task_id: str, config: dict) -> Any:
    """Upstream's play config, with its EMA action moved into the actuators."""
    env_cfg = load_env_cfg(task_id, play=True)
    ema = env_cfg.actions["joint_pos"]
    control_dt = env_cfg.decimation * env_cfg.sim.mujoco.timestep
    deployed = (config["action_scale"], config["ema_alpha"], config["ctrl_dt"])
    if not all(
        map(math.isclose, deployed, (ema.action_scale, ema.ema_alpha, control_dt))
    ):
        raise ValueError(
            f"{task_id}'s action does not match its checkpoint's config.json."
        )
    # Upstream clamps the action to [-1, 1] and the target to the soft limits, both in
    # one clip on the target; the EMA becomes each actuator's filter, whose state equals
    # it at the end of every control step.
    env_cfg.actions["joint_pos"] = JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=ema.action_scale,
        use_default_offset=True,
        clip=_target_clip(env_cfg, ema.action_scale, config),
    )
    time_constant = -control_dt / math.log(1.0 - ema.ema_alpha)
    env_cfg.scene.spec_fn = partial(_settle_filters, time_constant=time_constant)

    policy = env_cfg.observations["policy"].terms
    policy["qpos_error"].func = terms.joint_pos_target_error
    policy["cube_ori_error"].func = terms.goal_rot_err_6d
    policy["cube_ori_error"].params.pop("injection_prob")
    policy["action_history"].func = obs_fns.last_action
    # As a param the manager resolves the wrist tag's site up front; upstream's default
    # looks it up on every call, which a traced graph cannot.
    for name in ("cube_pos_in_tag", "cube_ori_error"):
        policy[name].params["tag_cfg"] = SceneEntityCfg(
            "robot", site_names=(".*_wrist_tag",)
        )
    # Training-only, as are the contact sensors only they read.
    env_cfg.observations.pop("critic")
    env_cfg.rewards.clear()
    env_cfg.metrics.clear()
    env_cfg.scene.sensors = ()
    env_cfg.terminations.pop("cage_drop")
    env_cfg.terminations["nan_guard"].func = terms.nan_detection
    env_cfg.events.pop("reset_joint_acc_cache")
    # It tilts the mount through a mocap write, which mjswan's event tracer lacks.
    env_cfg.events.pop("reset_robot_pose")
    env_cfg.events["reset_robot_joints"].func = terms.reset_joints_within_limits_range
    env_cfg.events["reset_object_pose"].func = terms.reset_object_orientation
    return env_cfg


def _target_clip(
    env_cfg: Any, scale: float, config: dict
) -> dict[str, tuple[float, float]]:
    """Per joint, the soft limits narrowed to ``scale`` either side of the default pose.

    Keyed as upstream keys joints, ``.*_finger1_joint1``: mjlab matches the pattern
    against ``right_finger1_joint1``, the browser against ``robot/right_finger1_joint1``.
    """
    model = Scene(env_cfg.scene, device="cpu").spec.compile()
    robot = env_cfg.scene.entities["robot"]
    factor = robot.articulation.soft_joint_pos_limit_factor
    default = dict(zip(config["joint_order"], config["default_joint_pos"]))
    clip = {}
    for name in config["joint_order"]:
        joint = model.joint(f"robot/{name}")
        if abs(model.key("init_state").qpos[joint.qposadr[0]] - default[name]) > 1e-6:
            raise ValueError(f"{name}'s default pose differs from its checkpoint's.")
        low, high = map(float, joint.range)
        center, half = 0.5 * (low + high), 0.5 * (high - low) * factor
        clip[".*" + name.removeprefix("right")] = (
            max(center - half, default[name] - scale),
            min(center + half, default[name] + scale),
        )
    return clip


def _settle_filters(spec: mujoco.MjSpec, time_constant: float) -> None:
    """Give each of the hand's actuators a first-order filter, settled at the keyframe."""
    for actuator in spec.actuators:
        actuator.dyntype = mujoco.mjtDyn.mjDYN_FILTEREXACT
        actuator.dynprm[0] = time_constant
    key = spec.keys[0]
    key.act = key.ctrl
