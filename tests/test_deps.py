"""``ensure_repo`` fetches only the pinned commit, and leaves a checkout already parked
on it alone, so a second build needs no network."""

import subprocess

from mjswan_playground import _deps


def _run(args, cwd):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True)


def _out(args, cwd):
    return subprocess.run(
        args, cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(message, cwd):
    _run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q"]
        + ["--allow-empty", "-m", message],
        cwd,
    )


def test_pinned_checkout_is_not_refetched(tmp_path, monkeypatch):
    origin = tmp_path / "origin"
    origin.mkdir()
    _run(["git", "init", "-q", "-b", "main"], origin)
    _commit("one", origin)
    commit = _out(["git", "rev-parse", "HEAD"], origin)
    # The pin is behind the branch tip, as most pins end up.
    _commit("two", origin)

    cache = tmp_path / "cache"
    monkeypatch.setattr(_deps, "CACHE_DIR", cache)
    kwargs = dict(
        name="origin", commit=commit, marker=".git", root_env_var="UNSET_ROOT"
    )

    # file:// so git honors --depth, as it does over https.
    repo = _deps.ensure_repo(url=origin.as_uri(), **kwargs)
    assert repo == cache / "origin"
    assert _out(["git", "rev-parse", "HEAD"], repo) == commit
    assert _out(["git", "rev-parse", "--is-shallow-repository"], repo) == "true"
    # A url that cannot be fetched: only reachable if the second call skips the fetch.
    assert (
        _deps.ensure_repo(url="file:///nonexistent.git", **kwargs) == cache / "origin"
    )
