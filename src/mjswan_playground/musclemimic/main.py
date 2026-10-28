"""MuscleMimic: a 354-muscle full body tracking a walking clip, driven by the public
checkpoint. See ``README.md``."""

from __future__ import annotations

import os

import mjswan
import onnx
from mjlab.managers.event_manager import EventTermCfg
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjswan.managers.termination_manager import TerminationTermCfg

from . import terms, upstream

TASK_ID = "myoMimicFullbody-v0"


def setup_builder() -> mjswan.Builder:
    from myosuite.core.trajectory_io import load_motion_clip
    from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
        bootstrap_myosuite_mjlab_registry,
    )
    from myosuite.integrations.musclemimic.model_versions import (
        MODELS_VERSION_ENV_VAR,
        checkpoint_models_version,
    )

    # The checkpoint falls within 2 s on the musclemimic_models release myosuite builds
    # by default; myosuite's mjlab env takes the one it was trained on from this variable.
    os.environ[MODELS_VERSION_ENV_VAR] = checkpoint_models_version(
        upstream.POLICY_REPO_ID
    )
    clip_path = upstream.fetch_clip()
    onnx_path, params = upstream.ensure_policy()
    # myosuite registers the clip-tracking variant of the task only when handed a clip.
    bootstrap_myosuite_mjlab_registry(clip_path=clip_path)

    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.events["rsi"] = EventTermCfg(func=terms.make_reset(clip_path), mode="reset")
    # Upstream writes the policy output straight to `ctrl`; ~190 of the 354 outputs are
    # negative every step, and the default sigmoid mode would squash them.
    env_cfg.actions["muscles"].action_mode = "direct"

    model = Scene(env_cfg.scene, device="cpu").spec.compile()
    clip = load_motion_clip(clip_path, expected_nq=model.nq, expected_nv=model.nv)
    ctrl_dt = env_cfg.sim.mujoco.timestep * env_cfg.decimation
    # A time-out on the clip's last frame, before the reference wraps to its start and
    # the deviation term would end the episode instead. mjlab rounds up to whole steps.
    env_cfg.episode_length_s = (clip.qpos.shape[0] - 1.5) * ctrl_dt
    contract = upstream.build_terms(model, clip, params, ctrl_dt)

    builder = mjswan.Builder()
    project = builder.add_project(name="MuscleMimic")
    scene = project.add_scene_mjlab(TASK_ID, env_cfg=env_cfg)

    scene.add_policy(
        "mm-10m-2",
        onnx.load(onnx_path),
        observations=ObservationGroupCfg(
            terms={"upstream": ObservationTermCfg(func=contract.observation)}
        ),
        terminations={
            "time_out": env_cfg.terminations["time_out"],
            "deviation": TerminationTermCfg(func=contract.termination),
        },
    )
    return builder
