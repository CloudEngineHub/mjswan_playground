"""OpenHorizon Labs' safe-fall policy, with its fall-triggered action folded into a
wrapper around the checkpoint. See ``README.md``."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import onnx
from mjlab.envs import mdp as envs_mdp
from onnx import TensorProto, helper, numpy_helper

from . import terms, walk


def env_cfg() -> Any:
    """Upstream's play config on flat ground, with a plain joint-position action: the
    wrapper gates the policy in place of ``TriggeredJointPositionActionCfg``."""
    from asimov_policies.common.env_parts import TerrainParams, joint_action_cfg
    from asimov_policies.tasks.safefall.env_cfg import (
        SafeFallParams,
        make_safefall_env_cfg,
    )
    from asimov_policies.tasks.safefall.mdp import ACTION_NAME

    cfg = make_safefall_env_cfg(
        SafeFallParams(terrain=TerrainParams(rough=False)), play=True
    )
    walk.raise_effort_limits(cfg)
    cfg.actions = {ACTION_NAME: joint_action_cfg()}
    actor = cfg.observations["actor"]
    # The wrapper outputs the action applied, which is what upstream observes.
    actor.terms["actions"].func = envs_mdp.last_action
    # The critic reads the trigger off upstream's action term; only the actor runs here.
    cfg.observations = {"actor": actor}
    cfg.events["reset_push"].func = terms.reset_standing_with_push
    cfg.terminations["nan"].func = terms.nan_detection
    # Training terms; a metric among them reads upstream's action term too.
    cfg.rewards = {}
    cfg.curriculum = {}
    cfg.metrics = {}
    return cfg


def wrap_policy(policy: onnx.ModelProto, spec: dict) -> onnx.ModelProto:
    """The checkpoint behind upstream's fall trigger, at its deployed thresholds.

    The action stays 0, the default pose, until the newest gyro or gravity reading in
    the observation crosses a threshold, then passes through until the episode ends.
    ``fired`` rides in the recurrent carry, which mjswan zeroes while ``is_init`` holds.
    """
    from asimov_policies.tasks.safefall.mdp import (
        ACTION_NAME,
        TriggeredJointPositionActionCfg,
    )

    newest = {}
    start = 0
    for term in spec["observation"]["terms"]:
        end = start + term["dim"] * spec["observation"]["history_length"]
        newest[term["name"]] = (end - term["dim"], end, term.get("scale", 1.0))
        start = end
    gyro_start, gyro_end, gyro_scale = newest["base_ang_vel"]
    _, gravity_end, _ = newest["projected_gravity"]
    trigger = spec["fall_trigger"]["deploy_defaults"]

    (obs,) = policy.graph.input
    (actions,) = policy.graph.output
    n = actions.type.tensor_type.shape.dim[-1].dim_value
    constants = {
        "trigger_axis": np.array([1]),
        "trigger_gyro_start": np.array([gyro_start]),
        "trigger_gyro_end": np.array([gyro_end]),
        "trigger_gz_start": np.array([gravity_end - 1]),
        "trigger_gz_end": np.array([gravity_end]),
        "trigger_carry_start": np.array([0]),
        "trigger_carry_end": np.array([1]),
        # Squared, in the observation's scaled units.
        "trigger_spin_sq": np.array((trigger["ang_vel"] * gyro_scale) ** 2, np.float32),
        # Upright gravity is (0, 0, -1): tilted past the angle once g_z rises above this.
        "trigger_tilt_gz": np.array(
            -math.cos(math.radians(trigger["tilt_deg"])), np.float32
        ),
        "trigger_half": np.array(0.5, np.float32),
    }

    def node(op: str, inputs: list[str], output: str, **attrs: Any) -> onnx.NodeProto:
        return helper.make_node(op, inputs, [output], **attrs)

    nodes = [
        node(
            "Slice",
            [obs.name, "trigger_gyro_start", "trigger_gyro_end", "trigger_axis"],
            "trigger_gyro",
        ),
        node("Mul", ["trigger_gyro", "trigger_gyro"], "trigger_gyro_sq"),
        node("ReduceSum", ["trigger_gyro_sq", "trigger_axis"], "trigger_spin"),
        node("Greater", ["trigger_spin", "trigger_spin_sq"], "trigger_spinning"),
        node(
            "Slice",
            [obs.name, "trigger_gz_start", "trigger_gz_end", "trigger_axis"],
            "trigger_gz",
        ),
        node("Greater", ["trigger_gz", "trigger_tilt_gz"], "trigger_tilted"),
        node("Or", ["trigger_spinning", "trigger_tilted"], "trigger_now"),
        node(
            "Slice",
            ["adapt_hx", "trigger_carry_start", "trigger_carry_end", "trigger_axis"],
            "trigger_carried",
        ),
        node("Not", ["is_init"], "trigger_kept"),
        node("Cast", ["trigger_kept"], "trigger_keep", to=TensorProto.FLOAT),
        node("Mul", ["trigger_carried", "trigger_keep"], "trigger_before"),
        node("Greater", ["trigger_before", "trigger_half"], "trigger_was"),
        node("Or", ["trigger_was", "trigger_now"], "trigger_fired"),
        node("Cast", ["trigger_fired"], "next_adapt_hx", to=TensorProto.FLOAT),
        *policy.graph.node,
        node("Mul", [actions.name, "next_adapt_hx"], "action"),
    ]
    wrapped = helper.make_model(
        helper.make_graph(
            nodes,
            policy.graph.name,
            [
                obs,
                helper.make_tensor_value_info("is_init", TensorProto.BOOL, [1]),
                helper.make_tensor_value_info(
                    "adapt_hx", TensorProto.FLOAT, [1, "carry"]
                ),
            ],
            [
                helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, n]),
                helper.make_tensor_value_info(
                    "next_adapt_hx", TensorProto.FLOAT, [1, 1]
                ),
            ],
            initializer=[
                *policy.graph.initializer,
                *(numpy_helper.from_array(v, k) for k, v in constants.items()),
            ],
        ),
        opset_imports=policy.opset_import,
        ir_version=policy.ir_version,
        # Apache-2.0 asks a changed file to say so.
        doc_string=(
            "OpenHorizon Labs' safe-fall checkpoint (asimov1-getup-safefall, Apache-2.0), "
            "changed by mjswan_playground: wrapped in upstream's fall trigger and gate."
        ),
    )
    # The keys mjswan stamps on the graphs it traces, naming the term this one replaces.
    term = TriggeredJointPositionActionCfg
    helper.set_model_props(
        wrapped,
        {
            "mjswan.kind": "action",
            "mjswan.term": ACTION_NAME,
            "mjswan.func": f"{term.__module__}:{term.__qualname__}",
        },
    )
    onnx.checker.check_model(wrapped, full_check=True)
    return wrapped
