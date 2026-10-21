"""Build-time shims for mjswan's ONNX tracer, shared by the tasks that need them."""

from __future__ import annotations

import torch


class CommandValues:
    """Width-only stand-in for a command term: ``generated_commands`` traces against an
    env with no command manager, and the runtime serves the live command browser-side.

    Pass one per command a traced term reads to
    ``build_single_entity_trace_env(commands=...)``.
    """

    def __init__(self, width: int) -> None:
        self.command = torch.zeros(1, width)


def nan_detection(env) -> torch.Tensor:
    """mjlab's ``nan_detection`` without the in-place ``|=`` the ONNX exporter refuses
    (``aten::__ior_``). An empty field (``sensordata`` without sensors) flags nothing."""
    data = env.sim.data
    found = torch.zeros(data.qpos.shape[0], dtype=torch.bool, device=data.qpos.device)
    for field in (
        data.qpos,
        data.qvel,
        data.qacc,
        data.qacc_warmstart,
        data.sensordata,
    ):
        if field.shape[-1]:
            found = (
                found | torch.isnan(field).any(dim=-1) | torch.isinf(field).any(dim=-1)
            )
    return found
