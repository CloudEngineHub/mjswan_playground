"""Upstream's two high jumps, on terms the browser can trace.

``jumper.jump`` adds the policy's residual to a recorded jump. Its command keeps the go
instant in a counter mjlab reads off ``episode_length_buf``, its spawn rides on an env
side channel, and its action reads the recording itself. The command is restated with
its own counter and spawn, and publishes the residual's baseline for mjswan's
``joint_position_reference`` action; its terms are restated over its ``go_step``.

``jumper.ref_free_jump`` observes only the robot. Its ``jump`` command feeds play nothing
but ``back_home``, which ends an episode once the robot has stood at home for a while
after landing, judged from the feet's contact forces; both go, so each episode is one
jump that ends on its 2.5 s time-out.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.utils.lab_api.math import sample_uniform
from mjswan import CommandBinding, register_command
from mjswan.envs.mdp.actions import ReferenceJointPositionActionCfg

from . import servos, upstream

SCENE_ID = "Jumper-Jump"
#: Each policy's upstream task, in the order the scene lists them.
POLICIES = {"Jump": "jump", "Jump (no reference)": "ref_free_jump"}
#: Play's spawn phase: the jump starts at the reset.
SPAWN_PHASE = (0.0, 0.01)

_UPSTREAM: dict[str, Any] = {}


def _jump_modules(root: Path) -> tuple[Any, Any, Any]:
    """Upstream's jump command, observations and recording, imported once."""
    if not _UPSTREAM:
        _UPSTREAM["commands"] = upstream.import_module(
            root, "tasks.jumper.jump.mdp.commands"
        )
        _UPSTREAM["obs"] = upstream.import_module(
            root, "tasks.jumper.jump.mdp.observations"
        )
        reference = upstream.import_module(root, "tasks.jumper.jump.mdp.reference")
        home = upstream.import_module(root, "tasks.jumper.common.constants").HOME
        _UPSTREAM["ref"] = reference.JumpReference(
            reference.JUMP_REF_NPZ, "cpu", list(home)
        )
    return _UPSTREAM["commands"], _UPSTREAM["obs"], _UPSTREAM["ref"]


def jump_clock_cfg(root: Path) -> type:
    """``JumpMotionCommandCfg`` restated with its own counter and spawn. Made once
    upstream is importable: it subclasses upstream's command to keep its recording."""
    if "cfg" in _UPSTREAM:
        return _UPSTREAM["cfg"]
    commands, obs, _ = _jump_modules(root)
    gait = list(obs._GAIT_COL)

    class JumpClockCommand(commands.JumpMotionCommand):
        """``go_step`` as upstream counts it, against a counter of its own; the spawn
        ``reset_from_reference_phase`` writes; and the residual's baseline."""

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
        spawn_phase: tuple[float, float] = SPAWN_PHASE

        def __post_init__(self) -> None:
            self.class_type = JumpClockCommand

        def build(self, env: Any) -> Any:
            return JumpClockCommand(self, env)

    _UPSTREAM["cfg"] = JumpClockCommandCfg
    return JumpClockCommandCfg


register_command(
    "JumpClockCommandCfg",
    CommandBinding(
        state_fields=["go_step", "elapsed", "tracked_joint_pos"],
        command_field="go_step",
    ),
)


def _t_since_go(env: Any, command_name: str) -> torch.Tensor:
    """``time_since_go``: ``episode_length_buf`` against the command's ``go_step``."""
    go_step = env.command_manager.get_term(command_name).go_step
    return (env.episode_length_buf - go_step) * env.step_dt


# --- Upstream's terms (tasks/jumper/jump/mdp/), over the command's go_step ---


def jump_phase(env: Any, command_name: str = "jump") -> torch.Tensor:
    return _UPSTREAM["ref"].phase(_t_since_go(env, command_name)).unsqueeze(1)


def ref_future(
    env: Any, command_name: str = "jump", offsets: tuple[float, ...] = ()
) -> torch.Tensor:
    obs, ref = _UPSTREAM["obs"], _UPSTREAM["ref"]
    t_since_go = _t_since_go(env, command_name)
    home = obs._HOME_VEC.to(t_since_go.device)
    cols = [
        ref.sample(ref.index_of(t_since_go + dt))["q"][:, obs._GAIT_COL] - home
        for dt in offsets
    ]
    return torch.cat(cols, dim=1)


def reference_diverged(
    env: Any, command_name: str = "jump", threshold: float = 6.0, asset_cfg: Any = None
) -> torch.Tensor:
    ref = _UPSTREAM["ref"]
    q = ref.sample(ref.index_of(_t_since_go(env, command_name)))["q"]
    robot = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    return ((robot.data.joint_pos - q) ** 2).sum(dim=1) > threshold


def play_env_cfg(root: Path, task: str) -> tuple[Any, dict[str, tuple[int, ...]]]:
    """A jump's play config as the browser runs it, and its strided terms' offsets."""
    env_cfg = upstream.play_env_cfg(root, task)
    upstream.drop_training_terms(env_cfg)
    env_cfg.events.pop("reset_from_reference", None)
    servos.hold_claws(env_cfg, upstream.contract(root, task))
    if task == "jump":
        _, obs, _ = _jump_modules(root)
        old = env_cfg.commands["jump"]
        env_cfg.commands["jump"] = jump_clock_cfg(root)(
            entity_name=old.entity_name, motion_file=old.motion_file
        )
        terms = env_cfg.observations["actor"].terms
        terms["jump_phase"].func = jump_phase
        terms["ref_future"].func = ref_future
        terms["ref_future"].params.setdefault("offsets", obs.FUTURE_OFFSETS)
        env_cfg.terminations["reference_diverged"].func = reference_diverged
        return env_cfg, {}
    # The action's prior feedforward is zero at this checkpoint's step count.
    env_cfg.actions["joint_pos"] = upstream.as_base(
        JointPositionActionCfg, env_cfg.actions["joint_pos"]
    )
    env_cfg.commands.pop("jump")
    env_cfg.terminations.pop("back_home")
    offsets = upstream.unstride(env_cfg)
    servos.observe_reset_force(env_cfg, "actuator_force")
    return env_cfg, offsets


def actions(task: str, env_cfg: Any) -> dict[str, Any] | None:
    """The browser's action terms where mjlab's differ: the jump's residual baseline is
    the command's ``tracked_joint_pos``."""
    if task != "jump":
        return None
    action = env_cfg.actions["joint_pos"]
    # Built here rather than adapted from mjlab's term, so prefixed here as the adapter
    # would, to match the policy's joint names.
    return {
        "joint_pos": ReferenceJointPositionActionCfg(
            entity_name="robot",
            actuator_names=tuple(f"robot/{name}" for name in action.actuator_names),
            scale=action.scale,
            command_name="jump",
        )
    }
