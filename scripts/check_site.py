"""Open every project of a site `msp site` made, the way GitHub Pages serves it: under
its base path, with no COOP/COEP headers.

    uv run --group previews python scripts/check_site.py dist/_site \\
        --base-path /mjswan_playground/ --chrome

A project fails when the app never reports ready, reports an error, opens another
project than its link names, or logs a page or console error within `SETTLE_SECONDS`
of ready. In a GitHub Actions job the results also go to the job summary. Exits 1
when any project fails.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import socketserver
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from mjswan_playground._site import size

READY_TIMEOUT_MS = 120_000
#: How long the app runs on after ready, so an error in its first steps shows up.
SETTLE_SECONDS = 3


@dataclass
class Result:
    project: dict
    state: str = "timed out"
    seconds: float = 0.0
    opened: str | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def problem(self) -> str | None:
        if self.state != "ready":
            return f"app {self.state}"
        if self.opened != self.project["id"]:
            return f"opened {self.opened!r}"
        return self.errors[0] if self.errors else None


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass


def check(browser, url: str, project: dict) -> Result:
    result = Result(project)
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    page.on("pageerror", lambda exc: result.errors.append(str(exc)))
    page.on(
        "console",
        lambda msg: result.errors.append(msg.text) if msg.type == "error" else None,
    )
    start = time.monotonic()
    page.goto(f"{url}?project={project['id']}")
    try:
        page.wait_for_function(
            "() => window.__mjswanReady || window.__mjswanError",
            timeout=READY_TIMEOUT_MS,
        )
        error = page.evaluate("() => !!window.__mjswanError")
        result.state = "reported an error" if error else "ready"
    except PlaywrightError:
        pass
    result.seconds = time.monotonic() - start
    page.wait_for_timeout(SETTLE_SECONDS * 1000)
    # The app rewrites the query to the project it opened: the first for an unknown id.
    result.opened = page.evaluate(
        "() => new URLSearchParams(location.search).get('project')"
    )
    page.close()
    return result


def report(results: list[Result], site: Path) -> None:
    rows = ["| Project | Result | Load | Size |", "|---|---|---|---|"]
    for result in results:
        project_id = result.project["id"]
        megabytes = size(site / project_id) / 1e6
        verdict = result.problem or "ready"
        print(
            f"{'FAIL' if result.problem else 'ok':<5}{project_id:<28}"
            f"{result.seconds:5.1f} s{megabytes:8.1f} MB  {verdict}"
        )
        for error in result.errors:
            print(f"     {error}")
        rows.append(
            f"| {result.project['name']} (`{project_id}`) | {verdict} "
            f"| {result.seconds:.1f} s | {megabytes:.1f} MB |"
        )
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as out:
            out.write("\n".join(rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("site", type=Path, help="a directory `msp site` wrote")
    parser.add_argument(
        "--base-path", default="/", help="the path the site was built for (default: /)"
    )
    browsers = parser.add_mutually_exclusive_group()
    browsers.add_argument(
        "--chrome", action="store_true", help="launch the installed Google Chrome"
    )
    browsers.add_argument("--chromium", help="launch this Chromium executable")
    args = parser.parse_args()

    site = args.site.resolve()
    manifest = json.loads((site / "manifest.json").read_text())
    with tempfile.TemporaryDirectory() as root:
        # Served at its base path, where the engine was built to fetch from.
        served = Path(root)
        if args.base_path.strip("/"):
            mount = served / args.base_path.strip("/")
            mount.parent.mkdir(parents=True, exist_ok=True)
            mount.symlink_to(site, target_is_directory=True)
        else:
            served = site
        handler = functools.partial(_Quiet, directory=str(served))
        with socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler) as server:
            threading.Thread(target=server.serve_forever, daemon=True).start()
            url = f"http://127.0.0.1:{server.server_address[1]}{args.base_path}"
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    channel="chrome" if args.chrome else None,
                    executable_path=args.chromium,
                    # GPU-less WebGL: Chrome runs SwiftShader only with these flags.
                    args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
                )
                results = [
                    check(browser, url, project) for project in manifest["projects"]
                ]
                browser.close()
            server.shutdown()

    report(results, site)
    sys.exit(1 if any(result.problem for result in results) else 0)


if __name__ == "__main__":
    main()
