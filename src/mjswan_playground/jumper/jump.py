"""Upstream's two high jumps, on terms the browser can trace.

``jumper.jump``'s command counts its go instant off ``episode_length_buf`` and passes
its spawn phase on an env side channel; its residual action reads the recording. Here
the command keeps its own counter and spawn and publishes the residual's baseline for
mjswan's ``joint_position_reference`` action; the terms read its ``go_step``.

``jumper.ref_free_jump``'s ``jump`` command feeds play only ``back_home``, which ends an
episode after the landing; both go, so each episode ends on its time-out.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform
from mjswan import CommandBinding, register_command
from mjswan.envs.mdp.actions import ReferenceJointPositionActionCfg

from . import servos, upstream

SCENE_ID = "Jumper-Jump"
#: Each policy's upstream task, in the order the scene lists them.
POLICIES = {"Jump": "jump", "Jump (no reference)": "ref_free_jump"}


@functools.cache
def jump_clock_cfg(root: Path) -> type:
    """``JumpMotionCommandCfg`` restated with its own counter and spawn. Made once
    upstream is importable: it subclasses upstream's command to keep its recording."""
    commands = upstream.import_module(root, "tasks.jumper.jump.mdp.commands")
    obs = upstream.import_module(root, "tasks.jumper.jump.mdp.observations")
    gait = list(obs._GAIT_COL)

    class JumpClockCommand(commands.JumpMotionCommand):
        def __init__(self, cfg: Any, env: Any) -> None:
            super().__init__(cfg, env)
            self.elapsed = torch.zeros(self.num_envs, device=self.device)
            self.tracked_joint_pos = torch.zeros(
                self.num_envs, len(gait), device=self.device
            )

        def _resample_command(self, env_ids: Any) -> None:
            del env_ids
            ref, dt = self.reference, self._env.step_dt
            phase = sample_uniform(
                *self.cfg.spawn_phase, (self.num_envs,), device=self.device
            )
            self.go_step = -phase * ref.span / dt
            # mjlab updates every command right after a reset; that update reaches 0.
            self.elapsed = torch.full_like(self.elapsed, -1.0)
            s = ref.sample(ref.index_of(phase * ref.span))
            pos = self._env.scene.env_origins + s["base_pos"]
            lin_vel = s["base_lin_vel"]
            root_state = torch.cat(
                [pos, s["base_quat"], lin_vel, torch.zeros_like(lin_vel)], -1
            )
            self.robot.write_root_state_to_sim(root_state)
            self.robot.write_joint_state_to_sim(s["q"], s["qd"])

        def _update_command(self, env_ids: Any = None) -> None:
            del env_ids
            ref, dt = self.reference, self._env.step_dt
            self.elapsed = self.elapsed + 1.0
            t_since_go = (self.elapsed - self.go_step) * dt
            # Upstream's `ReferenceResidualJointPositionAction` reads one step ahead.
            self.tracked_joint_pos = ref.sample_cmd(t_since_go + ref.t_go + dt)[:, gait]

        def _update_metrics(self) -> None:
            pass

    @dataclass(kw_only=True)
    class JumpClockCommandCfg(commands.JumpMotionCommandCfg):
        spawn_phase: tuple[float, float]

        def build(self, env: Any) -> Any:
            return JumpClockCommand(self, env)

    return JumpClockCommandCfg


register_command(
    "JumpClockCommandCfg",
    CommandBinding(
        state_fields=["go_step", "elapsed", "tracked_joint_pos"],
        command_field="go_step",
    ),
)


# --- Upstream's terms (tasks/jumper/jump/mdp/), over the command's go_step ---


def _since_go(env: Any, command_name: str) -> tuple[Any, torch.Tensor]:
    """The command's recording and its ``time_since_go``, over the ``go_step`` field: the
    browser serves a command's state fields, not its properties."""
    command = env.command_manager.get_term(command_name)
    return command.reference, (env.episode_length_buf - command.go_step) * env.step_dt


def jump_phase(env: Any, command_name: str = "jump") -> torch.Tensor:
    ref, t_since_go = _since_go(env, command_name)
    return ref.phase(t_since_go).unsqueeze(1)


def ref_future(
    env: Any, command_name: str = "jump", offsets: tuple[float, ...] = ()
) -> torch.Tensor:
    from tasks.jumper.jump.mdp.observations import _GAIT_COL, _HOME_VEC

    ref, t_since_go = _since_go(env, command_name)
    home = _HOME_VEC.to(t_since_go.device)
    cols = [
        ref.sample(ref.index_of(t_since_go + dt))["q"][:, _GAIT_COL] - home
        for dt in offsets
    ]
    return torch.cat(cols, dim=1)


def reference_diverged(
    env: Any,
    command_name: str = "jump",
    threshold: float = 6.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    ref, t_since_go = _since_go(env, command_name)
    q = ref.sample(ref.index_of(t_since_go))["q"]
    robot = env.scene[asset_cfg.name]
    return ((robot.data.joint_pos - q) ** 2).sum(dim=1) > threshold


def time_out(env: Any, steps: int) -> torch.Tensor:
    """mjlab's ``time_out`` at ``steps``: the scene traces every policy's terms against
    the first policy's env, whose episode length mjlab's would read."""
    return env.episode_length_buf >= steps


def play_env_cfg(root: Path, task: str) -> tuple[Any, dict[str, tuple[int, ...]]]:
    """A jump's play config as the browser runs it, and its strided terms' offsets."""
    env_cfg = upstream.play_env_cfg(root, task)
    upstream.drop_training_terms(env_cfg)
    spawn = env_cfg.events.pop("reset_from_reference")
    servos.hold_claws(env_cfg, upstream.contract(root, task))
    step_dt = env_cfg.sim.mujoco.timestep * env_cfg.decimation
    timeout = env_cfg.terminations["time_out"]
    timeout.func = time_out
    timeout.params = {"steps": math.ceil(env_cfg.episode_length_s / step_dt)}
    if task == "jump":
        old = env_cfg.commands["jump"]
        env_cfg.commands["jump"] = jump_clock_cfg(root)(
            entity_name=old.entity_name,
            motion_file=old.motion_file,
            spawn_phase=spawn.params["phase_range"],
        )
        terms = env_cfg.observations["actor"].terms
        terms["jump_phase"].func = jump_phase
        terms["ref_future"].func = ref_future
        env_cfg.terminations["reference_diverged"].func = reference_diverged
        return env_cfg, {}
    # The action's prior feedforward is zero at this checkpoint's step count.
    env_cfg.actions["joint_pos"] = upstream.as_base(
        JointPositionActionCfg, env_cfg.actions["joint_pos"]
    )
    env_cfg.commands.pop("jump")
    env_cfg.terminations.pop("back_home")
    offsets = upstream.unstride(env_cfg)
    servos.observe_reset_force(env_cfg)
    return env_cfg, offsets


def actions(task: str, env_cfg: Any) -> dict[str, Any] | None:
    """The browser's action terms where mjlab's differ: the jump's residual baseline is
    the command's ``tracked_joint_pos``."""
    if task != "jump":
        return None
    action = env_cfg.actions["joint_pos"]
    # Built rather than adapted, so prefixed as the adapter would.
    return {
        "joint_pos": ReferenceJointPositionActionCfg(
            entity_name="robot",
            actuator_names=tuple(f"robot/{name}" for name in action.actuator_names),
            scale=action.scale,
            command_name="jump",
        )
    }
