# Daily run

What the "mjswan playground daily" routine does each morning. Its prompt only points here, so changing the routine is a pull request against this file.

Each run is a fresh cloud session with ttktjmt/mjswan_playground and ttktjmt/mjswan checked out. It asks nothing: what this file and the backlog leave open is a stop. The parts run in order.

## A. Tasks waiting on mjswan

For each open pull request labelled `needs-mjswan`, which links one ttktjmt/mjswan pull request:

- The mjswan PR closed without merging: close the task's PR with a comment saying so, and open a `daily-task-skipped` issue for its id.
- The mjswan PR has new commits: move the pin to its new head, `uv lock --upgrade-package mjswan`, build, rerun parity and push. The PR stays a draft.
- `main` locks an mjswan release that contains the mjswan PR's merge commit: finish the task as step 9 of msp:add-new-task says, then label it as in C4. If the mjswan PR changed `src/mjswan/template/`, add `publish` only once that PR carries `cloud-ready`.
- Anything else: leave it.

## B. mjswan release bump

PyPI has a stable mjswan newer than the one `main` locks, and no open PR is labelled `mjswan-bump`: on `claude/mjswan-bump-<version>`, raise the floor in `pyproject.toml` (and the `mjlab` pin if mjswan's `mjlab` extra moved), `uv lock`, build every task, run `make test`, and open the PR labelled `mjswan-bump`.

## C. One new task

Skipped while three or more `needs-mjswan` PRs are open.

1. Pick the first entry of `daily/backlog.yaml` on `main` whose id is not in `mjswan_playground.registry.ALL_TASKS`, has no PR from `claude/daily-<id>` in any state, and has no open `daily-task-skipped` issue. None left: open one issue titled "Daily backlog is empty", unless one is open.
2. Run msp:add-new-task in unattended mode on branch `claude/daily-<id>`, with the entry as every answer: `repo` the target, `id` the playground id, `task` the mjlab task, `license` the license, `policy` and `notes` what it says about them. When the skill is not in your skill list, follow `.claude/skills/msp/skills/add-new-task/SKILL.md` directly. Its step 9 may open or reuse an mjswan PR, on branch `claude/playground-<id>` of ttktjmt/mjswan.
3. Label the PR `daily-task`.
4. Also label it `publish` when there is no step 9 pin, the entry says `publish: true`, and its license is MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause or CC-BY-4.0. The `publish` workflow then publishes it and puts the link in its README row.
5. On a stop: open, or update, an issue labelled `daily-task-skipped` and titled `<id>: <reason>`, with the step, the error verbatim and what would unblock it.

## Never

Merge anything, push to `main` of either repository, run mjswan's release workflow, edit `daily/backlog.yaml`, or run `mjswan login` or `mjswan publish`.

## Report

End with the waiting PRs touched, the new task's id with its PR or issue URL, the mjswan PR URL if one was opened or reused, and whether `publish` went on.
