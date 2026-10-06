"""The build refuses a graph input that reads a traced command by a field the browser
does not serve, the way ``get_command()`` inside a traced term does."""

from mjswan_playground._cli import _unservable_command_slots


def _manifest(slots: list[dict]) -> dict:
    commands = {
        "clock": {
            "state_fields": [{"name": "step_count"}],
            "command_field": "step_count",
        },
        "motion": {"name": "TrackingCommand"},
    }
    mdp = {
        "id": "a0",
        "commands": commands,
        "observations": {"actor": {"input_slots": slots}},
    }
    return {"projects": [{"scenes": [{"mdps": [mdp]}]}]}


def test_get_command_slot_is_refused() -> None:
    problems = _unservable_command_slots(
        _manifest([{"command": "clock", "field": "command"}])
    )
    assert len(problems) == 1
    assert "clock.command" in problems[0]


def test_state_field_and_native_command_slots_pass() -> None:
    slots = [
        {"command": "clock", "field": "step_count"},
        {"command": "motion", "field": "anchor_quat_w"},
    ]
    assert _unservable_command_slots(_manifest(slots)) == []
