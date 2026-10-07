"""The site's task order, and the merge of the tasks' builds into one app."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mjswan_playground import _cli, _site
from mjswan_playground.registry import ALL_TASKS

HEADER = {"format": 3, "version": "0.0.0", "uses_custom_js": False}


def _build(root: Path, name: str, *project_ids: str, **header) -> Path:
    """A task's build as the site reads it: a manifest and a directory per project."""
    build = root / name
    for project_id in project_ids:
        (build / project_id / "scene").mkdir(parents=True)
        (build / project_id / "scene" / "scene.mjz").write_bytes(b"\0" * 10)
    projects = [{"id": pid, "name": pid.title(), "scenes": []} for pid in project_ids]
    manifest = {**HEADER, **header, "projects": projects}
    (build / "manifest.json").write_text(json.dumps(manifest))
    return build


def test_readme_lists_every_task_once() -> None:
    on_site = set(ALL_TASKS) - set(_site.NOT_ON_SITE)
    assert sorted(_site.task_order()) == sorted(on_site)


def test_the_site_leaves_out_registered_tasks_only() -> None:
    assert set(_site.NOT_ON_SITE) <= set(ALL_TASKS)


def test_a_task_without_a_readme_row_is_refused(tmp_path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("| [`husky`](src/mjswan_playground/husky/README.md) | G1 |\n")
    with pytest.raises(_site.SiteError, match="no row for"):
        _site.task_order(readme)


def test_merge_keeps_the_order_and_opens_on_the_first(tmp_path) -> None:
    first = _build(tmp_path, "first", "alpha")
    second = _build(tmp_path, "second", "beta", "gamma")
    manifest = json.loads((second / "manifest.json").read_text())
    manifest["projects"][1]["default"] = True
    (second / "manifest.json").write_text(json.dumps(manifest))

    _site.merge([first, second], tmp_path / "site")

    merged = json.loads((tmp_path / "site" / "manifest.json").read_text())
    assert {k: v for k, v in merged.items() if k != "projects"} == HEADER
    assert [p["id"] for p in merged["projects"]] == ["alpha", "beta", "gamma"]
    assert [p.get("default") for p in merged["projects"]] == [True, None, None]
    assert (tmp_path / "site" / "gamma" / "scene" / "scene.mjz").exists()


@pytest.mark.parametrize(
    ("second", "match"),
    [
        ({"project_ids": ("beta",), "version": "0.0.1"}, "differs .* in version"),
        ({"project_ids": ("alpha",)}, "earlier build has too"),
        ({"project_ids": ("beta",), "uses_custom_js": True}, "custom-TS"),
    ],
)
def test_merge_refuses_builds_that_cannot_share_an_app(tmp_path, second, match) -> None:
    first = _build(tmp_path, "first", "alpha")
    other = _build(tmp_path, "second", *second.pop("project_ids"), **second)
    with pytest.raises(_site.SiteError, match=match):
        _site.merge([first, other], tmp_path / "site")


def test_merge_refuses_a_task_that_was_not_built(tmp_path) -> None:
    first = _build(tmp_path, "first", "alpha")
    with pytest.raises(_site.SiteError, match="build that task first"):
        _site.merge([first, tmp_path / "missing"], tmp_path / "site")


def test_site_merges_without_building(tmp_path, monkeypatch) -> None:
    _build(tmp_path, "one", "first")
    _build(tmp_path, "two", "second")
    engines = []
    monkeypatch.setattr(_site, "task_order", lambda: ["two", "one"])
    monkeypatch.setattr(
        _site, "install_engine", lambda site, base: engines.append(base)
    )

    result = CliRunner().invoke(
        _cli.app,
        ["site", "--no-build", "--dist-dir", str(tmp_path), "--base-path", "/p/"],
    )

    assert result.exit_code == 0, result.output
    merged = json.loads((tmp_path / "_site" / "manifest.json").read_text())
    assert [p["id"] for p in merged["projects"]] == ["second", "first"]
    assert engines == ["/p/"]


def test_site_says_why_it_leaves_a_task_out(monkeypatch) -> None:
    monkeypatch.setattr(_site, "task_order", lambda: ["one"])
    monkeypatch.setattr(_site, "NOT_ON_SITE", {"two": "a reason"})

    result = CliRunner().invoke(_cli.app, ["site", "two"])

    assert result.exit_code == 1
    assert "two is not on the site: a reason." in result.output
