"""CLI entry point for the playground: list, run and build the tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from mjswan_playground.registry import ALL_TASKS, load

app = typer.Typer(
    name="mjswan-playground",
    help="Browser-ready tasks built on mjswan.",
    no_args_is_help=True,
)

TaskId = Annotated[str, typer.Argument(help=f"One of: {', '.join(ALL_TASKS)}.")]


def _build(task_id: str, output_dir: Optional[Path]):
    """Always an absolute path: ``Builder.build`` resolves a relative one against its
    *caller's* directory, which from here is wherever this package is installed."""
    try:
        builder = load(task_id)
    except KeyError as exc:
        typer.echo(str(exc.args[0]), err=True)
        raise typer.Exit(1) from exc
    path = (output_dir or Path("dist") / task_id).resolve()
    built = builder.build(output_dir=path)
    problems = _unservable_command_slots(
        json.loads((path / "manifest.json").read_text())
    )
    if problems:
        typer.echo(
            "These graph inputs read a command field the browser does not serve, so "
            "their graphs would never run:",
            err=True,
        )
        for problem in problems:
            typer.echo(f"  {problem}", err=True)
        typer.echo(
            "Read a traced command as env.command_manager.get_term(name).<state field> "
            "and a ui_command as get_command(name); a command the MDP lacks goes in the "
            "policy's commands=.",
            err=True,
        )
        raise typer.Exit(1)
    return built, path


#: What the browser serves for a ``{command, field}`` slot on mjswan's own command
#: classes (``getStateField`` under its template's core/command/); a traced
#: ``OnnxCommand`` serves its state fields. tests/test_cli.py checks this against the
#: installed mjswan.
_NATIVE_COMMAND_FIELDS = {
    "UiCommand": ("command",),
    "TrackingCommand": (
        "is_ready",
        "ref_root_pos_w",
        "ref_root_quat_w",
        "ref_joint_pos",
        "anchor_pos_w",
        "anchor_quat_w",
        "anchor_lin_vel_w",
        "anchor_ang_vel_w",
        "ref_base_height",
        "ref_base_lin_vel_b",
        "ref_base_ang_vel_b",
        "ref_gravity_b",
        "joint_pos",
        "tracked_joint_pos",
        "body_pos_w",
        "robot_anchor_pos_w",
        "robot_anchor_quat_w",
        "robot_body_pos_w",
        "body_pos_relative_w",
    ),
}


def _unservable_command_slots(manifest: dict) -> list[str]:
    """``{command, field}`` slots the browser cannot serve. It never runs a graph with
    such an input, while parity, which reads the attribute in mjlab, still passes."""
    problems = []
    for project in manifest.get("projects", []):
        for scene in project.get("scenes", []):
            for mdp in scene.get("mdps", []):
                commands = mdp.get("commands") or {}
                for where, slot in _command_slots(mdp):
                    read = (
                        f"{scene.get('id')}/{mdp.get('id')}{where}: "
                        f"{slot['command']}.{slot['field']}"
                    )
                    command = commands.get(slot["command"])
                    if command is None:
                        problems.append(f"{read} (no such command in this MDP)")
                        continue
                    note = ""
                    if command.get("name") == "OnnxCommand":
                        served = [
                            field["name"] for field in command.get("state_fields", [])
                        ]
                        note = f"; get_command() is {command.get('command_field')}"
                    else:
                        served = _NATIVE_COMMAND_FIELDS.get(command.get("name"))
                    if served is not None and slot["field"] not in served:
                        problems.append(f"{read} (serves {', '.join(served)}{note})")
    return problems


def _command_slots(node, where: str = ""):
    if isinstance(node, dict):
        if "command" in node and "field" in node:
            yield where, node
        for key, value in node.items():
            yield from _command_slots(value, f"{where}/{key}")
    elif isinstance(node, list):
        for value in node:
            yield from _command_slots(value, where)


@app.command("list")
def list_cmd() -> None:
    """List the available tasks."""
    for task_id in ALL_TASKS:
        typer.echo(task_id)


@app.command("run")
def run_cmd(
    task_id: TaskId,
    port: Annotated[int, typer.Option(help="HTTP server port.")] = 8080,
    host: Annotated[str, typer.Option(help="HTTP server host.")] = "localhost",
    no_open: Annotated[
        bool, typer.Option("--no-open", help="Do not open the browser automatically.")
    ] = False,
    output_dir: Annotated[
        Optional[Path], typer.Option(help="Where to write the built app.")
    ] = None,
) -> None:
    """Build a task and serve it in the browser."""
    built, _ = _build(task_id, output_dir)
    built.launch(host=host, port=port, open_browser=not no_open)


@app.command("build")
def build_cmd(
    task_id: TaskId,
    output_dir: Annotated[
        Optional[Path], typer.Option(help="Where to write the built app.")
    ] = None,
) -> None:
    """Build a task into a dist directory without launching it."""
    _, path = _build(task_id, output_dir)
    typer.echo(str(path))
