"""The build refuses a graph input the browser cannot serve."""

import importlib.util
import json
import re
from pathlib import Path

from typer.testing import CliRunner

from mjswan_playground import _cli


def _manifest(slots: list[dict]) -> dict:
    commands = {
        "clock": {
            "name": "OnnxCommand",
            "state_fields": [{"name": "step_count"}],
            "command_field": "step_count",
        },
        "motion": {"name": "TrackingCommand"},
        "pad": {"name": "UiCommand"},
    }
    mdp = {
        "id": "a0",
        "commands": commands,
        "observations": {"actor": {"input_slots": slots}},
    }
    return {"projects": [{"scenes": [{"id": "s", "mdps": [mdp]}]}]}


def _refused(*slots: tuple[str, str]) -> list[str]:
    manifest = _manifest([{"command": c, "field": f} for c, f in slots])
    return _cli._unservable_command_slots(manifest)


def test_served_slots_pass() -> None:
    assert not _refused(
        ("clock", "step_count"),
        ("motion", "anchor_quat_w"),
        ("motion", "command"),
        ("pad", "command"),
    )


def test_unserved_slots_are_refused() -> None:
    for slot in [
        ("clock", "command"),
        ("motion", "x"),
        ("pad", "x"),
        ("gone", "x"),
    ]:
        assert len(_refused(slot)) == 1, slot


def test_native_fields_match_the_installed_mjswan() -> None:
    package = Path(importlib.util.find_spec("mjswan").submodule_search_locations[0])
    source = (package / "template/src/core/command/TrackingCommand.ts").read_text()
    body = source.split("getStateField(field: string)", 1)[1].split("\n  }\n", 1)[0]
    served = set(re.findall(r"case '(\w+)'", body))
    assert served == set(_cli._NATIVE_COMMAND_FIELDS["TrackingCommand"])


def test_build_exits_on_an_unservable_slot(tmp_path, monkeypatch) -> None:
    class Builder:
        def build(self, output_dir: Path) -> None:
            output_dir.mkdir(parents=True)
            manifest = _manifest([{"command": "clock", "field": "command"}])
            (output_dir / "manifest.json").write_text(json.dumps(manifest))

    monkeypatch.setattr(_cli, "load", lambda task_id: Builder())
    result = CliRunner().invoke(
        _cli.app, ["build", "x", "--output-dir", str(tmp_path / "d")]
    )
    assert result.exit_code == 1
    assert "clock.command" in result.output
