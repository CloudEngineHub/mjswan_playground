"""The preview's run checks catch what a broken port does in the browser."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "record_preview.py"
_SPEC = importlib.util.spec_from_file_location("record_preview", _PATH)
record_preview = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = record_preview
_SPEC.loader.exec_module(record_preview)

DT = 0.02
RATE = 50.0


def _segment(n: int = 200, filmed: int = 100) -> "record_preview.Segment":
    """A healthy run: n steps, every graph run each step but the reset event."""
    return record_preview.Segment(
        next_index=60,
        seconds=n * DT,
        filmed=filmed,
        steps=[
            {"nan": None, "zero": [], "same": False, "z": 1.0, "tilt": 2.0}
            for _ in range(n)
        ],
        runs={
            "mdp/a0/obs/actor.onnx": n,
            "mdp/a0/command/twist.onnx": n,
            "mdp/a0/term/fell_over.onnx": n,
            "mdp/a0/event/reset_base.onnx": 0,
        },
        rate=RATE,
        fps=30.0,
    )


#: The graphs `_segment` runs that have to run every step: not the reset event.
PER_STEP = {
    "mdp/a0/obs/actor.onnx",
    "mdp/a0/command/twist.onnx",
    "mdp/a0/term/fell_over.onnx",
}


def _check(segment, preview=None) -> list:
    preview = preview or record_preview.Preview()
    return record_preview.check_run(segment, preview, DT, RATE, PER_STEP)


def test_a_healthy_run_passes() -> None:
    segment = _segment()
    segment.terms = [{"k": 150, "reasons": ["time_out"]}]
    assert _check(segment) == []


def test_the_first_bipedhrl_port_fails_on_everything_it_did() -> None:
    # Its observation graph never ran, so the policy read zeros and fell every 1.4 s.
    segment = _segment()
    for k, step in enumerate(segment.steps):
        step.update(zero=["actor"], same=k > 0, tilt=80.0 if k % 70 >= 65 else 2.0)
    segment.terms = [{"k": k, "reasons": ["fell_over"]} for k in (69, 139)]
    segment.runs["mdp/a0/obs/actor.onnx"] = 0
    found = dict((message, steps) for steps, message in _check(segment))
    assert found == {
        "terminated 2x (fell_over), first at 1.4 s": [69, 139],
        "root tipped past 75° 2x, first at 1.3 s, to 80°": [65, 135],
        "observation 'actor' all zeros on every step": [],
        "actions never changed over 200 steps": [],
        "mdp/a0/obs/actor.onnx ran on 0 of 200 steps": [],
    }


def test_a_clip_can_call_for_a_flip_or_a_termination() -> None:
    segment = _segment()
    segment.steps[120]["tilt"] = 170.0
    segment.terms = [{"k": 130, "reasons": ["out_of_bounds"]}]
    assert [m for _, m in _check(segment)] == [
        "terminated 1x (out_of_bounds), first at 2.6 s (after the clip)",
        "root tipped past 75° 1x, first at 2.4 s (after the clip), to 170°",
    ]
    preview = record_preview.Preview(upright=False, ok_terminations=("out_of_bounds",))
    assert _check(segment, preview) == []


def test_nan_names_what_went_and_when() -> None:
    segment = _segment()
    for step in segment.steps[40:]:
        step["nan"] = "state"
    segment.steps[40]["nan"] = "actions"
    assert _check(segment) == [([40], "actions went NaN at 0.8 s")]


def test_a_stalled_or_slow_run_fails() -> None:
    segment = _segment()
    segment.steps = segment.steps[:100]
    segment.rate, segment.fps = 30.0, 9.0
    assert [m for _, m in _check(segment)] == [
        "slow motion: 30.0 of 50 steps/s",
        "9.0 frames/s captured, under 15: frames repeat",
        "the step loop ran 100 of 200 steps: it stopped or stalled",
    ]


def test_a_graph_that_runs_on_most_steps_passes_and_one_that_stops_fails() -> None:
    segment = _segment()
    segment.runs["mdp/a0/command/twist.onnx"] = 185
    segment.runs["mdp/a0/term/fell_over.onnx"] = 120
    segment.errors = ["TypeError: x is undefined"]
    assert [m for _, m in _check(segment)] == [
        "page error: TypeError: x is undefined",
        "mdp/a0/term/fell_over.onnx ran on 120 of 200 steps",
    ]


def test_per_step_graphs_leave_out_reset_and_event_graphs(tmp_path) -> None:
    mdp = {
        "observations": {
            "actor": {"fused": "mdp/a0/obs/actor.onnx"},
            "extra": [{"name": "h", "onnx": "mdp/a0/obs/h.onnx"}],
        },
        "terminations": {
            "time_out": {"native": "elapsed_s >= episode_length_s"},
            "__fused__": {"fused": "mdp/a0/term/terminations.onnx"},
        },
        "commands": {
            "twist": {"name": "OnnxCommand", "onnx": "mdp/a0/command/twist.onnx"},
            "motion": {
                "name": "TrackingCommand",
                "reset_graph": {"onnx": "mdp/a0/command/motion_reset.onnx"},
            },
        },
        "events": [{"onnx": "mdp/a0/event/reset_base.onnx"}],
    }
    manifest = {"projects": [{"scenes": [{"mdps": [mdp]}]}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert record_preview.per_step_graphs(tmp_path) == {
        "mdp/a0/obs/actor.onnx",
        "mdp/a0/obs/h.onnx",
        "mdp/a0/term/terminations.onnx",
        "mdp/a0/command/twist.onnx",
    }


def test_a_task_that_will_not_build_leaves_the_rest_to_run(monkeypatch) -> None:
    built = []

    def ensure_built(task_id, dist):
        built.append(task_id)
        raise RuntimeError("gated")

    monkeypatch.setattr(record_preview, "ensure_built", ensure_built)
    monkeypatch.setattr(sys, "argv", ["record_preview.py", "musclemimic", "wbc"])
    with pytest.raises(SystemExit, match="failed: musclemimic, wbc"):
        record_preview.main()
    assert built == ["musclemimic", "wbc"]
