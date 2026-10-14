"""Menlo Research's walking policy, and the MDP its ``env.yaml`` records, on upstream's
get-up scene. The policy was trained in Isaac Lab and has no mjlab task, so its
observations, command and terminations are restated here from that file."""

from __future__ import annotations

import math
from typing import Any

import onnx
from huggingface_hub import hf_hub_download
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from . import upstream

POLICY_REPO_ID = "Menlo/asimov1-locomotion-0818"
POLICY_REVISION = "18114a730668a983423e6361f0522eb62c76cb74"

#: ``env.yaml``'s torque limits where they exceed upstream's, which takes the motors'
#: rated torques. N·m.
EFFORT_LIMITS = {
    r".*_hip_pitch_joint": 45.0,
    r".*_hip_roll_joint": 45.0,
    r".*_hip_yaw_joint": 28.0,
    r".*_knee_joint": 45.0,
}


def policy() -> onnx.ModelProto:
    return onnx.load(
        hf_hub_download(POLICY_REPO_ID, "policy.onnx", revision=POLICY_REVISION)
    )


def raise_effort_limits(env_cfg: Any) -> None:
    """Give upstream's actuators ``env.yaml``'s torque limits."""
    for actuator in env_cfg.scene.entities["robot"].articulation.actuators:
        for pattern in actuator.target_names_expr:
            actuator.effort_limit = EFFORT_LIMITS.get(pattern, actuator.effort_limit)


def env_cfg() -> Any:
    """``env.yaml``'s observations, command and terminations as play runs them: no noise,
    no delays, no randomization, a random heading at reset."""
    from asimov_policies import robot

    cfg = upstream.flat_play_cfg()
    raise_effort_limits(cfg)
    # Joint terms read the motors in their CAN slots, the order upstream's get-up reads.
    order = robot.OBS_JOINT_ORDER
    slots = {"01": order[:9], "23": order[9:17], "45": order[17:]}

    def joints(names: tuple[str, ...]) -> SceneEntityCfg:
        return SceneEntityCfg("robot", joint_names=names, preserve_order=True)

    terms = {
        "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel, scale=0.25),
        "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
        "command": ObservationTermCfg(
            func=vel_mdp.generated_commands, params={"command_name": "twist"}
        ),
    }
    for slot, names in slots.items():
        terms[f"joint_pos_slot{slot}"] = ObservationTermCfg(
            func=envs_mdp.joint_pos_rel, params={"asset_cfg": joints(names)}
        )
    for slot, names in slots.items():
        terms[f"joint_vel_slot{slot}"] = ObservationTermCfg(
            func=envs_mdp.joint_vel_rel, params={"asset_cfg": joints(names)}, scale=0.1
        )
    terms["actions"] = ObservationTermCfg(func=envs_mdp.last_action)
    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms=terms, concatenate_terms=True, enable_corruption=False
        )
    }
    cfg.commands = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(3.0, 8.0),
            rel_standing_envs=0.2,
            rel_heading_envs=0.3,
            heading_command=True,
            heading_control_stiffness=0.5,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.6, 0.8),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.8, 0.8),
                heading=(0.0, 0.0),
            ),
        )
    }
    cfg.events = {
        "reset_base": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {"yaw": (-math.pi, math.pi)}, "velocity_range": {}},
        ),
        "reset_robot_joints": EventTermCfg(
            func=envs_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
    }
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=envs_mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}
        ),
    }
    # The get-up's training terms, which this MDP has no use for.
    cfg.rewards = {}
    cfg.curriculum = {}
    cfg.metrics = {}
    return cfg
