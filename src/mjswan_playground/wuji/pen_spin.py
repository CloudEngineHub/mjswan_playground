"""Wuji Hand 2 spinning a pen along the clips of one of upstream's five recipes.

Upstream's action term smooths the policy's residual alone and adds it to the reference
its command plays, which no actuator filter can stand in for. Here a wrapper around the
checkpoint does it: the EMA and the raw action ride the recurrent carry mjswan feeds
back, and the target goes to a plain joint position action. The command is restated as
a traced clip player: its tables are graph constants, the clip and the frame its state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import combine_frame_transforms, sample_uniform
from mjswan import CommandBinding, register_command
from onnx import TensorProto, helper, numpy_helper

from . import terms, upstream

TASK_ID = "WujiHand2_PenSpin"
#: Each policy's recipe, the bundle's checkpoint trained on it, in the scene's order.
GROUPS = {
    "Single axis": "01_single_axis",
    "Continuous single axis": "02_continuous_single_axis",
    "Continuous multi-axis": "03_continuous_multi_axis",
    "Stable grasp rotation": "04_stable_grasp_rotation",
    "Multi-axis, two rotations": "05_multi_axis_two_rotations",
}

# A resampling time no episode reaches, so a clip is picked on episode reset alone.
_NEVER = 1.0e9
_ROBOT = SceneEntityCfg("robot")
_OBJECT = SceneEntityCfg("object")


@dataclass(kw_only=True)
class PenClipCommandCfg(CommandTermCfg):
    """Play's ``SingleHandTrackingCommandCfg``: one of the recipe's clips at a time, its
    first frame held for ``hold_steps`` before it plays."""

    recipe_path: str
    clips_root: str
    fixed_wrist_pose_w: tuple[float, ...]
    hold_steps: int
    axis_point_radius_m: float
    object_position_tanh_scale_m: float
    entity_name: str = "robot"
    resampling_time_range: tuple[float, float] = (_NEVER, _NEVER)

    def build(self, env: Any) -> PenClipCommand:
        return PenClipCommand(self, env)


class PenClipCommand(CommandTerm):
    """``SingleHandTrackingCommand`` as play runs it, on float state the browser keeps.

    The reset event picks the clip and writes its first frame; this command, reset after
    the events as in mjlab, finds that clip again by the joint positions it was handed.
    """

    cfg: PenClipCommandCfg

    def __init__(self, cfg: PenClipCommandCfg, env: Any) -> None:
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.entity_name]
        clips = load_clips(cfg)
        if abs(clips["frame_dt"] - env.step_dt) > 1e-6:
            raise ValueError("The clips must advance one frame per control step.")
        self._frames = clips["frames"].to(self.device)
        self._starts = clips["starts"].to(self.device)
        self._lengths = clips["lengths"].to(self.device)
        self._first_joint_pos = clips["first_joint_pos"].to(self.device)
        self.clip = torch.zeros(self.num_envs, 1, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, 1, device=self.device)
        self._read_frame()

    @property
    def command(self) -> torch.Tensor:
        return self.reference

    def _read_frame(self) -> None:
        """``SingleHandTrackingCommand.command`` at the clip's frame, which holds at the
        first while ``time_steps`` is negative and at the last past the end."""
        clip = self.clip[:, 0].long()
        self.length = self._lengths[clip].unsqueeze(-1)
        elapsed = torch.minimum(self.time_steps.clamp(min=0.0), self.length - 1.0)
        self.reference = self._frames[
            (self._starts[clip].unsqueeze(-1) + elapsed)[:, 0].long()
        ]

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        joint_pos = self.robot.data.joint_pos[env_ids]
        distance = (joint_pos.unsqueeze(1) - self._first_joint_pos).square().sum(-1)
        self.clip[env_ids] = distance.argmin(dim=-1, keepdim=True).float()
        self.time_steps[env_ids] = -float(self.cfg.hold_steps)
        self._read_frame()

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        # Upstream's: on the reset path, only the envs just reset advance.
        if env_ids is None:
            self.time_steps = self.time_steps + 1.0
        else:
            self.time_steps[env_ids] += 1.0
        self._read_frame()

    def _update_metrics(self) -> None:
        pass


register_command(
    "PenClipCommandCfg",
    CommandBinding(
        state_fields=["clip", "time_steps", "length", "reference"],
        command_field="reference",
    ),
)


def load_clips(cfg: PenClipCommandCfg) -> dict[str, Any]:
    """The recipe's clips through upstream's own resolver and reader: each frame's
    reference, the pen's first pose in world on the fixed mount, and where each clip
    starts."""
    from wuji_mjlab.tasks.pen_spin.mdp.commands import SingleHandTrackingReader
    from wuji_mjlab.tasks.pen_spin.mdp.observations import (
        bounded_object_pose_to_points_wrist_local,
        make_axis_points,
    )
    from wuji_mjlab.tasks.pen_spin.tooling.recipe import resolve_recipe

    (clip_set,) = resolve_recipe(
        cfg.recipe_path, clips_root=cfg.clips_root, verify_clip_hashes=True
    ).clip_sets
    data = SingleHandTrackingReader(clip_set).read()
    lengths = data["trajectory_lengths"]
    starts = torch.cumsum(lengths, 0) - lengths
    points = bounded_object_pose_to_points_wrist_local(
        data["object_pose"],
        make_axis_points(cfg.axis_point_radius_m),
        position_tanh_scale_m=cfg.object_position_tanh_scale_m,
    )
    frames = torch.cat(
        (data["joint_pos"], points.flatten(1), data["link_pos"].flatten(1)), dim=-1
    )
    wrist = torch.tensor(cfg.fixed_wrist_pose_w, dtype=torch.float32)
    first = data["object_pose"][starts]
    pos, quat = combine_frame_transforms(
        wrist[:3].expand(len(starts), 3),
        wrist[3:].expand(len(starts), 4),
        first[:, :3],
        first[:, 3:],
    )
    return {
        "frame_dt": data["frame_dt"],
        "frames": frames,
        "starts": starts.float(),
        "lengths": lengths.float(),
        "first_joint_pos": data["joint_pos"][starts],
        "first_pen_pose_w": torch.cat((pos, quat), dim=-1),
    }


# --- Upstream's terms (tasks/pen_spin/mdp), restated where the tracer needs it ---


def object_axis_points_bounded(
    env: Any,
    radius_m: float,
    tanh_scale_m: float,
    robot_cfg: SceneEntityCfg = _ROBOT,
    object_cfg: SceneEntityCfg = _OBJECT,
) -> torch.Tensor:
    """Upstream's ``object_axis_points_wrist_local_bounded`` with the axis points as
    params, where upstream reads them off the command: they are its constants."""
    from wuji_mjlab.tasks.pen_spin.mdp.observations import (
        bounded_object_pose_to_points_wrist_local,
        make_axis_points,
        object_pose_wrist_local,
    )

    pose = object_pose_wrist_local(env, robot_cfg, object_cfg)
    points = make_axis_points(radius_m, device=pose.device, dtype=pose.dtype)
    return bounded_object_pose_to_points_wrist_local(
        pose, points, position_tanh_scale_m=tanh_scale_m
    ).reshape(env.num_envs, -1)


def reference_end(env: Any, command_name: str = "motion") -> torch.Tensor:
    motion = env.command_manager.get_term(command_name)
    return (motion.time_steps + 1.0 >= motion.length)[:, 0]


def numerical_instability(
    env: Any,
    object_cfg: SceneEntityCfg = _OBJECT,
    max_pos: float = 3.0,
    max_vel: float = 80.0,
) -> torch.Tensor:
    """Upstream's ``numerical_instability`` with ``terms.nan_detection`` for mjlab's
    ``NanGuard.detect_nans``, whose in-place ``|=`` the exporter refuses. The task spaces
    its envs 0 m apart, so the pen's position needs no env origin taken off."""
    joint_vel = env.scene["robot"].data.joint_vel
    pen = env.scene[object_cfg.name].data
    return (
        terms.nan_detection(env)
        | (pen.root_link_pos_w.abs() > max_pos).any(-1)
        | (pen.root_link_lin_vel_w.abs() > max_vel).any(-1)
        | (joint_vel.abs() > max_vel).any(-1)
    )


def reset_to_clip_start(
    env: Any,
    env_ids: torch.Tensor,
    joint_pos: torch.Tensor,
    pen_pose_w: torch.Tensor,
    robot_cfg: SceneEntityCfg = _ROBOT,
    object_cfg: SceneEntityCfg = _OBJECT,
) -> None:
    """Upstream's ``reset_to_reference_state`` as play runs it: a clip drawn uniformly,
    the hand and the pen at rest on its first frame. ``sample_uniform`` floored stands in
    for the command's ``torch.randint``, which the tracer cannot replay; the mount's pose
    is the keyframe's, so its write is left out."""
    n = len(env_ids)
    clip = sample_uniform(0.0, float(len(joint_pos)), (n,), device=env.device).long()
    clip = clip.clamp(max=len(joint_pos) - 1)
    robot = env.scene[robot_cfg.name]
    robot.write_joint_state_to_sim(
        joint_pos[clip], torch.zeros_like(joint_pos[clip]), env_ids=env_ids
    )
    pose = pen_pose_w[clip].clone()
    pose[:, :3] += env.scene.env_origins[env_ids]
    pen = env.scene[object_cfg.name]
    pen.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    pen.write_root_link_velocity_to_sim(
        torch.zeros(n, 6, device=env.device), env_ids=env_ids
    )


def play_env_cfg(root: Path, clips_root: Path, group: str, config: dict) -> Any:
    """Upstream's play config on one recipe, as the browser runs it."""
    env_cfg = load_env_cfg(TASK_ID, play=True)
    action = env_cfg.actions["joint_pos"]
    control_dt = env_cfg.decimation * env_cfg.sim.mujoco.timestep
    deployed = (
        config["action_scale"],
        tuple(config["action_residual_clip"]),
        config["action_ema_alpha"],
        config["ctrl_dt"],
        config["control_mode"],
    )
    if deployed != (
        action.scale,
        tuple(action.residual_clip),
        action.alpha,
        round(control_dt, 6),
        "reference_residual",
    ):
        raise ValueError(
            f"{group}'s action does not match its checkpoint's config.json."
        )
    motion = env_cfg.commands["motion"]
    if (
        motion.first_frame_hold_prob != 1.0
        or motion.trajectory_sampling_mode != "uniform"
    ):
        raise ValueError(
            "Play's command changed: it no longer holds every first frame."
        )
    hold_s = motion.first_frame_hold_s
    if hold_s[0] != hold_s[1]:
        raise ValueError("Play's first-frame hold is no longer one fixed duration.")
    command = PenClipCommandCfg(
        recipe_path=str(
            root / "src/wuji_mjlab/tasks/pen_spin/data/recipe" / f"{group}.json"
        ),
        clips_root=str(clips_root),
        fixed_wrist_pose_w=tuple(motion.fixed_wrist_pose_w),
        hold_steps=round(hold_s[0] / control_dt),
        axis_point_radius_m=motion.axis_point_radius_m,
        object_position_tanh_scale_m=motion.object_position_tanh_scale_m,
    )
    env_cfg.commands["motion"] = command
    clips = load_clips(command)

    policy = env_cfg.observations["policy"].terms
    policy["object_pose"] = ObservationTermCfg(
        func=object_axis_points_bounded,
        params={
            "radius_m": command.axis_point_radius_m,
            "tanh_scale_m": command.object_position_tanh_scale_m,
            "robot_cfg": policy["object_pose"].params["robot_cfg"],
        },
    )
    if list(policy) != [
        "motion",
        "joint_pos",
        "joint_vel",
        "object_pose",
        "last_action",
    ]:
        raise ValueError(
            "The policy's observation order changed; the wrapper slices it."
        )
    # The wrapper hands the action its absolute targets; see `wrap_policy`.
    env_cfg.actions["joint_pos"] = JointPositionActionCfg(
        entity_name="robot",
        actuator_names=("right_finger.*_joint.*",),
        scale=1.0,
        use_default_offset=False,
    )
    env_cfg.events = {
        "reset_to_clip_start": EventTermCfg(
            func=reset_to_clip_start,
            mode="reset",
            params={
                "joint_pos": clips["first_joint_pos"],
                "pen_pose_w": clips["first_pen_pose_w"],
            },
        )
    }
    if env_cfg.scene.env_spacing != 0.0:
        raise ValueError(
            "The task spaces its envs apart now; see `numerical_instability`."
        )
    env_cfg.terminations["unstable"].func = numerical_instability
    env_cfg.terminations["reference_end"].func = reference_end
    # Training-only, as are the contact sensors only the rewards read.
    env_cfg.observations.pop("critic")
    env_cfg.rewards.clear()
    env_cfg.metrics.clear()
    env_cfg.scene.sensors = ()
    return env_cfg


def wrap_policy(policy: onnx.ModelProto, config: dict) -> onnx.ModelProto:
    """The checkpoint with upstream's ``SingleHandResidualEMAAction`` around it.

    The carry holds the smoothed residual, then the raw action. mjswan opens every
    episode with ``is_init`` and a ``[1, 128]`` zero carry, so the carry is cut to width
    and zeroed while ``is_init`` holds, as the action term's reset zeroes both. The
    observation's last action, its final slice, becomes the carried raw action: the
    browser's ``last_action`` is the target this graph outputs. Out comes the reference
    (the observation's first slice, the command's joint positions) plus the smoothed
    residual, ``clip(scale * action) * alpha + previous * (1 - alpha)``.
    """
    graph = policy.graph
    (obs,) = graph.input
    (actions,) = graph.output
    n_obs = obs.type.tensor_type.shape.dim[-1].dim_value
    n = actions.type.tensor_type.shape.dim[-1].dim_value
    low, high = config["action_residual_clip"]
    alpha = config["action_ema_alpha"]
    constants = {
        "carry_zeros": np.zeros((1, 2 * n), np.float32),
        "axis_1": np.array([1]),
        "at_0": np.array([0]),
        "at_n": np.array([n]),
        "at_2n": np.array([2 * n]),
        "obs_head_end": np.array([n_obs - n]),
        "residual_scale": np.array(config["action_scale"], np.float32),
        "residual_low": np.array(low, np.float32),
        "residual_high": np.array(high, np.float32),
        "ema_alpha": np.array(alpha, np.float32),
        "ema_keep": np.array(1.0 - alpha, np.float32),
    }
    policy_obs = "policy_obs"

    def renamed(node: onnx.NodeProto) -> onnx.NodeProto:
        copy = onnx.NodeProto()
        copy.CopyFrom(node)
        copy.input[:] = [policy_obs if i == obs.name else i for i in node.input]
        return copy

    nodes = [
        helper.make_node(
            "Concat", ["adapt_hx", "carry_zeros"], ["carry_padded"], axis=1
        ),
        helper.make_node(
            "Slice", ["carry_padded", "at_0", "at_2n", "axis_1"], ["carry_sized"]
        ),
        helper.make_node("Not", ["is_init"], ["carry_kept"]),
        helper.make_node("Cast", ["carry_kept"], ["carry_keep"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["carry_sized", "carry_keep"], ["carry"]),
        helper.make_node("Slice", ["carry", "at_0", "at_n", "axis_1"], ["ema_prev"]),
        helper.make_node(
            "Slice", ["carry", "at_n", "at_2n", "axis_1"], ["action_prev"]
        ),
        helper.make_node(
            "Slice", [obs.name, "at_0", "obs_head_end", "axis_1"], ["obs_head"]
        ),
        helper.make_node("Concat", ["obs_head", "action_prev"], [policy_obs], axis=1),
        *(renamed(node) for node in graph.node),
        helper.make_node("Mul", [actions.name, "residual_scale"], ["residual_raw"]),
        helper.make_node(
            "Clip", ["residual_raw", "residual_low", "residual_high"], ["residual"]
        ),
        helper.make_node("Mul", ["residual", "ema_alpha"], ["ema_new"]),
        helper.make_node("Mul", ["ema_prev", "ema_keep"], ["ema_old"]),
        helper.make_node("Add", ["ema_new", "ema_old"], ["ema"]),
        helper.make_node("Slice", [obs.name, "at_0", "at_n", "axis_1"], ["reference"]),
        helper.make_node("Add", ["reference", "ema"], ["target"]),
        helper.make_node("Concat", ["ema", actions.name], ["next_adapt_hx"], axis=1),
    ]
    wrapped = helper.make_model(
        helper.make_graph(
            nodes,
            graph.name,
            [
                obs,
                helper.make_tensor_value_info("is_init", TensorProto.BOOL, [1]),
                helper.make_tensor_value_info(
                    "adapt_hx", TensorProto.FLOAT, [1, "carry"]
                ),
            ],
            [
                helper.make_tensor_value_info("target", TensorProto.FLOAT, [1, n]),
                helper.make_tensor_value_info(
                    "next_adapt_hx", TensorProto.FLOAT, [1, 2 * n]
                ),
            ],
            initializer=[
                *graph.initializer,
                *(numpy_helper.from_array(v, k) for k, v in constants.items()),
            ],
        ),
        opset_imports=policy.opset_import,
        ir_version=policy.ir_version,
    )
    onnx.checker.check_model(wrapped, full_check=True)
    return wrapped


def add_scene(project: Any, root: Path, bundle: Path) -> None:
    """One scene, the hand on its pen mount; each recipe's checkpoint a policy on it."""
    from mjswan.mjlab.observation import adapt_observations
    from wuji_mjlab.tasks.pen_spin.config.wuji_hand2.hand2_constants import (
        RIGHT_FINGER_JOINT_NAMES,
    )

    clips_root = bundle / upstream.PEN_SPIN_CLIPS
    for i, (name, group) in enumerate(GROUPS.items()):
        policy, config = upstream.pen_spin_checkpoint(bundle, group)
        env_cfg = play_env_cfg(root, clips_root, group, config)
        if i == 0:
            scene = project.add_scene_mjlab(TASK_ID, env_cfg=env_cfg)
        scene.add_policy(
            name=name,
            policy=wrap_policy(onnx.load(str(policy)), config),
            env_cfg=env_cfg,
            observations=adapt_observations(env_cfg.observations["policy"]),
            in_keys=["actor", "is_init", "adapt_hx"],
            out_keys=["action", ["next", "adapt_hx"]],
            policy_joint_names=[f"robot/{joint}" for joint in RIGHT_FINGER_JOINT_NAMES],
            default_joint_pos=[0.0] * len(RIGHT_FINGER_JOINT_NAMES),
            default=i == 0,
        )
