"""The reference clip and Dodge's ball throw, as traced commands the browser can keep.

Upstream's ``MultiClipMotionCommand`` indexes its clips by an integer frame, which the
browser feeds a graph as float32, so ``ClipCommand`` restates it over the one clip each
scene plays, with the frame as a float, and with the anchor rows the twist observations
read as fields. SONIC's tokenizer reads it unchanged, through ``future_frames`` and the
clip tables, which the graph bakes in.

mjswan traces an event as a stateless function and lets a command write the sim only when
it resamples, so upstream's ``ThrowBall``, an interval event that keeps a countdown and a
flight timer and pins the ball between throws, runs as ``ThrowBallCommand``: a command
resampled every control step, its resample being the event's per-step tick.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_from_euler_xyz,
    quat_mul,
    sample_gaussian,
    sample_uniform,
    yaw_quat,
)
from mjswan import CommandBinding, register_command

# A resampling time no episode reaches, so the clip restarts on episode reset alone.
_NEVER = 1.0e9
_POSE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


@dataclass(kw_only=True)
class ClipCommandCfg(CommandTermCfg):
    """``MultiClipMotionCommandCfg`` over one clip, played from its first frame."""

    motion_file: str
    body_names: tuple[str, ...]
    anchor_body_name: str = "pelvis"
    entity_name: str = "robot"
    pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    joint_position_range: tuple[float, float] = (0.0, 0.0)
    resampling_time_range: tuple[float, float] = (_NEVER, _NEVER)

    def build(self, env: Any) -> ClipCommand:
        return ClipCommand(self, env)


class ClipCommand(CommandTerm):
    cfg: ClipCommandCfg

    def __init__(self, cfg: ClipCommandCfg, env: Any) -> None:
        from orcs.core.data.loader import ConcatMotionLoader

        super().__init__(cfg, env)
        self.robot = env.scene[cfg.entity_name]
        file = Path(cfg.motion_file)
        self.motion = ConcatMotionLoader(str(file.parent), self.device, [str(file)])
        self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
        self.time_steps = torch.zeros(self.num_envs, device=self.device)
        # Steps held on the last frame, which upstream's `exceeded_motion_by_eps` reads.
        self._steps_past_end = torch.zeros(self.num_envs, device=self.device)
        self._read_frame()

    @property
    def command(self) -> torch.Tensor:
        return self.time_steps

    def future_frames(self, steps: int, skip: int = 1) -> torch.Tensor:
        """``MultiClipMotionCommand.future_frames``: the frames ahead, held at the end."""
        offsets = torch.arange(steps, device=self.device) * skip
        index = self.time_steps.long()[:, None] + offsets[None, :]
        return index.clamp(max=self.motion.time_step_total - 1)

    def _read_frame(self) -> None:
        """``MotionCommand``'s anchor properties at ``time_steps``, as fields."""
        index = self.time_steps.long()
        anchor = self.motion_anchor_body_index
        self.anchor_quat_w = self.motion.body_quat_w[index, anchor]
        self.anchor_lin_vel_w = self.motion.body_lin_vel_w[index, anchor]
        self.anchor_ang_vel_w = self.motion.body_ang_vel_w[index, anchor]

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Frame zero, and the robot written onto it with upstream's jitter: the
        ``start_from_zero`` path of ``MultiClipMotionCommand._resample_command``, then
        mjlab's ``_write_reference_state_to_sim``."""
        cfg, n = self.cfg, len(env_ids)
        self.time_steps = torch.zeros_like(self.time_steps)
        self._steps_past_end = torch.zeros_like(self._steps_past_end)
        root_pos = self.motion.body_pos_w[0:1, 0] + self._env.scene.env_origins[env_ids]
        root_quat = self.motion.body_quat_w[0:1, 0].expand(n, 4)
        root_lin_vel = self.motion.body_lin_vel_w[0:1, 0].expand(n, 3)
        root_ang_vel = self.motion.body_ang_vel_w[0:1, 0].expand(n, 3)
        joint_pos = self.motion.joint_pos[0:1].expand(n, -1)
        joint_vel = self.motion.joint_vel[0:1].expand(n, -1)

        pose = self._sample_se3(cfg.pose_range, n)
        root_pos = root_pos + pose[:, 0:3]
        root_quat = quat_mul(
            quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5]), root_quat
        )
        twist = self._sample_se3(cfg.velocity_range, n)
        root_lin_vel = root_lin_vel + twist[:, 0:3]
        root_ang_vel = root_ang_vel + twist[:, 3:6]
        lo, hi = cfg.joint_position_range
        joint_pos = joint_pos + sample_uniform(lo, hi, joint_pos.shape, self.device)
        limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        root_state = torch.cat([root_pos, root_quat, root_lin_vel, root_ang_vel], -1)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self._read_frame()

    def _sample_se3(self, ranges: dict, n: int) -> torch.Tensor:
        """``orcs.core.mdp.commands.sample_se3``, drawing here so the tracer sees it."""
        lims = torch.tensor(
            [ranges.get(key, (0.0, 0.0)) for key in _POSE_KEYS], device=self.device
        )
        return sample_uniform(lims[:, 0], lims[:, 1], (n, 6), self.device)

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        del env_ids
        last = float(self.motion.time_step_total - 1)
        steps = self.time_steps + 1.0
        self._steps_past_end = self._steps_past_end + (steps > last).float()
        self.time_steps = steps.clamp(max=last)
        self._read_frame()

    def _update_metrics(self) -> None:
        pass


register_command(
    "ClipCommandCfg",
    CommandBinding(
        state_fields=[
            "time_steps",
            "_steps_past_end",
            "anchor_quat_w",
            "anchor_lin_vel_w",
            "anchor_ang_vel_w",
        ],
        command_field="time_steps",
    ),
)


@dataclass(kw_only=True)
class ThrowBallCommandCfg(CommandTermCfg):
    """``ThrowBall``'s params, every one, for its tick each control step."""

    params: dict[str, Any]
    #: The robot; the tracer reads a command's entity by this field.
    entity_name: str = "robot"
    #: The clip command, whose frame restarts with each episode.
    clock_name: str = "motion"

    def build(self, env: Any) -> ThrowBallCommand:
        return ThrowBallCommand(self, env)


class ThrowBallCommand(CommandTerm):
    """``ThrowBall.__call__`` as the resample, ``ThrowBall.reset`` on an episode's first
    tick: play's cone throw only, with no camera and no anchors."""

    cfg: ThrowBallCommandCfg

    def __init__(self, cfg: ThrowBallCommandCfg, env: Any) -> None:
        super().__init__(cfg, env)
        params = cfg.params
        if params["launch_camera_name"] is not None or params["stand_fraction"] != 0.0:
            raise ValueError("Only play's throw, with no camera and no anchors, ports.")
        self.robot = env.scene[cfg.entity_name]
        self._g = -float(env.sim.mj_model.opt.gravity[2])
        self._step_dt = env.step_dt
        self._window = float(max(int(params["flight_window_s"] / env.step_dt), 1))
        self.countdown = torch.zeros(self.num_envs, device=self.device)
        self.flight = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.flight

    def _sample_countdown(self) -> torch.Tensor:
        lo, hi = self.cfg.params["interval_range_s"]
        secs = sample_uniform(lo, hi, self.num_envs, self.device)
        return (secs / self._step_dt).long().clamp(min=1).float()

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        del env_ids
        p, env, n, dev = self.cfg.params, self._env, self.num_envs, self.device
        robot = self.robot

        clock = env.command_manager.get_term(self.cfg.clock_name).time_steps
        fresh = clock <= 1.0
        countdown = torch.where(fresh, self._sample_countdown(), self.countdown)
        flight = torch.where(fresh, torch.zeros_like(self.flight), self.flight)

        # 1. Pin every ball that is not mid-flight.
        flight = (flight - 1.0).clamp(min=0.0)
        parked = flight == 0.0

        # 2. Tick the throw timer.
        countdown = countdown - 1.0
        fire = countdown <= 0.0
        countdown = torch.where(fire, self._sample_countdown(), countdown)
        flight = torch.where(fire, torch.full_like(flight, self._window), flight)

        # 3. Launch, from the robot-relative frontal cone.
        def u(lo: float, hi: float) -> torch.Tensor:
            return sample_uniform(lo, hi, n, dev)

        root_pos = robot.data.root_link_pos_w
        dist = u(*p["dist_range"])
        bearing = u(-p["angle_deg"], p["angle_deg"]) * (math.pi / 180.0)
        offset_b = torch.stack(
            [dist, dist * torch.tan(bearing), torch.zeros_like(dist)], dim=-1
        )
        yq = yaw_quat(robot.data.root_link_quat_w)
        low = u(0.0, 1.0) < p["low_arc_fraction"]
        start_xy = root_pos[:, 0:2] + quat_apply(yq, offset_b)[:, 0:2]
        start_z = torch.where(
            low, u(*p["low_launch_height_range"]), u(*p["launch_height_range"])
        )
        # A descending ball falls the whole flight, so cap its time at the ground.
        t_req = u(*p["flight_time_range"])
        t_max = torch.sqrt((2.0 * (start_z - 0.05).clamp(min=1e-3)) / self._g)
        t = torch.where(low, t_req, torch.minimum(t_req, t_max))
        aim = root_pos[:, 0:2]
        if p["lead_target"]:
            aim = aim + robot.data.root_link_lin_vel_w[:, 0:2] * t[:, None]
        if p["aim_noise"] > 0.0:
            aim = aim + sample_gaussian(0.0, p["aim_noise"], (n, 2), dev)
        vel_xy = (aim - start_xy) / t[:, None]
        vel_z = torch.where(
            low,
            (u(*p["low_target_z_range"]) - start_z) / t + 0.5 * self._g * t,
            torch.zeros_like(t),
        )

        # The pin, then the throw over it, as one write each to the envs either touches.
        park = env.scene.env_origins + torch.tensor([0.0, 0.0, p["park_z"]], device=dev)
        pos = torch.where(
            fire[:, None], torch.cat([start_xy, start_z[:, None]], -1), park
        )
        lin_vel = torch.where(
            fire[:, None],
            torch.cat([vel_xy, vel_z[:, None]], -1),
            torch.zeros_like(park),
        )
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).expand(n, 4)
        ids = (fire | parked).nonzero().flatten()
        if ids.numel():
            ball = env.scene[p["ball_name"]]
            ball.write_root_link_pose_to_sim(
                torch.cat([pos, quat], -1)[ids], env_ids=ids
            )
            twist = torch.cat([lin_vel, torch.zeros_like(lin_vel)], -1)
            ball.write_root_link_velocity_to_sim(twist[ids], env_ids=ids)
        self.countdown, self.flight = countdown, flight

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        del env_ids

    def _update_metrics(self) -> None:
        pass


register_command(
    "ThrowBallCommandCfg",
    CommandBinding(state_fields=["countdown", "flight"], command_field="flight"),
)


def robot_root_pos_env(
    env: Any, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """``orcs.core.mdp.robot_root_pos_env`` for the browser's one env, whose origin is
    the world's: the tracer serves an observation no ``env.scene.env_origins``."""
    return env.scene[robot_cfg.name].data.root_link_pos_w


def _only_clip(motion: Any) -> str:
    """The one clip upstream's command samples from, on its roster's tiles and clips if
    it has them; ``ClipCommand`` plays no other."""
    root = Path(motion.dataset_dir)
    keep = getattr(motion, "clips", {})
    files = sorted(
        file
        for key in getattr(motion, "tile_keys", ("",))
        for file in (root / key).rglob("motion.npz")
        if file.parent.name in keep.get(key, (file.parent.name,))
    )
    if files != [Path(motion.motion_file)] or not motion.start_from_zero:
        raise ValueError(f"ClipCommand plays one clip from its start, not {files}.")
    return motion.motion_file


def _throw_params(event: Any) -> dict[str, Any]:
    """The event's params over ``ThrowBall.__call__``'s defaults: every knob it reads."""
    signature = inspect.signature(event.func.__call__)
    defaults = {
        name: param.default
        for name, param in signature.parameters.items()
        if param.default is not inspect.Parameter.empty
    }
    return {**defaults, **event.params}


def browser_env_cfg(env_cfg: Any) -> Any:
    """Upstream's play config as the browser runs it, in place: the commands above in
    place of the clip command and the throw, and none of what only training reads."""
    motion = env_cfg.commands["motion"]
    env_cfg.commands["motion"] = ClipCommandCfg(
        motion_file=_only_clip(motion),
        body_names=tuple(motion.body_names),
        anchor_body_name=motion.anchor_body_name,
        entity_name=motion.entity_name,
        pose_range=dict(motion.pose_range),
        velocity_range=dict(motion.velocity_range),
        joint_position_range=tuple(motion.joint_position_range),
    )
    if "throw_ball" in env_cfg.events:
        throw = _throw_params(env_cfg.events.pop("throw_ball"))
        step_dt = env_cfg.sim.mujoco.timestep * env_cfg.decimation
        env_cfg.commands["throw_ball"] = ThrowBallCommandCfg(
            params=throw,
            entity_name=throw["robot_name"],
            resampling_time_range=(step_dt, step_dt),
        )
    env_cfg.observations["augmentation"].terms[
        "robot_root_pos_env"
    ].func = robot_root_pos_env
    # Feeds the training anneals and curricula only.
    env_cfg.events.pop("policy_update_counter", None)
    del env_cfg.observations["critic"]
    env_cfg.rewards = {}
    return env_cfg
