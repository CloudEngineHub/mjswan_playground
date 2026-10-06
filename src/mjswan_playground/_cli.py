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
            "These graph inputs read a command field the browser cannot serve, so the "
            "graph would never run. In a traced term, read "
            "env.command_manager.get_term(name).<state field>, not get_command(name):",
            err=True,
        )
        for problem in problems:
            typer.echo(f"  {problem}", err=True)
        raise typer.Exit(1)
    return built, path


def _unservable_command_slots(manifest: dict) -> list[str]:
    """Each `{command, field}` input slot on a traced command that has no such state
    field. The browser serves those by state-field name only, and an input it cannot
    serve leaves the graph's output at its initial zeros, with no warning."""
    problems = []
    for project in manifest.get("projects", []):
        for scene in project.get("scenes", []):
            for mdp in scene.get("mdps", []):
                commands = mdp.get("commands") or {}
                for where, slot in _command_slots(mdp):
                    command = commands.get(slot["command"]) or {}
                    fields = [
                        field["name"] for field in command.get("state_fields", [])
                    ]
                    if fields and slot["field"] not in fields:
                        problems.append(
                            f"{mdp.get('id')}{where}: {slot['command']}.{slot['field']} "
                            f"(state fields: {', '.join(fields)})"
                        )
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
