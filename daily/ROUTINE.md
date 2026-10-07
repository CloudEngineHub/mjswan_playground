# Daily run

What the "mjswan playground daily" routine does each morning. Its prompt only points here, so changing the routine is a pull request against this file.

Each run is a fresh cloud session with ttktjmt/mjswan_playground and ttktjmt/mjswan checked out. It asks nothing: what this file and the backlog leave open is a stop. The parts run in order, and the run ends only after part E has checked its work and the report is published.

## A. Scout for trending tasks

A task that has just caught on is worth porting before the rest of the queue. Look for repositories built on mjlab that are drawing attention now, and put them at the front of the backlog.

1. Search what appeared or spread in the last 30 days: the `mjlab` topic on GitHub (github.com/topics/mjlab, sorted by recently updated and by stars), mjlab's "Show and tell" discussions, and the web for posts, project pages and papers about mjlab releases (X, Reddit, Hacker News, YouTube, arXiv). Read GitHub through its web pages and plain `git` on public repositories: the session's GitHub tools reach only the attached ones.
2. A repository is trending when it was created or first released in the last 30 days and has 100 or more stars, or when a post about it spread widely (a front page, or thousands of views).
3. Vet each one as the backlog's entries were: it registers mjlab tasks; a trained policy is public at a pinned source (git, the Hub, a public W&B run), and so is a tracking task's clip unless the ONNX carries it; its actuators are mjlab's own; and the license of everything the build fetches is known. Any license will do, as long as the entry states it: an NC or SA one is the author's to weigh when publishing. Drop what is already in the backlog, in `mjswan_playground.registry.ALL_TASKS`, in a PR, or in a `daily-task-skipped` issue.
4. Put each one that passes at the top of `daily/backlog.yaml`, the one drawing the most attention first, with every field filled and a `found` line giving the date and the evidence (stars, the post). While fewer than three entries are left untried, also append the best other candidates the search turned up.
5. Commit on `claude/daily-backlog`, cut from `main` or with `main` merged in, and open or update its PR, labelled `daily-backlog`, listing each new entry with its evidence. Nothing found: change nothing.

Until a person merges that PR, the backlog for part D is the one on its branch.

## B. Tasks waiting on mjswan

For each open pull request labelled `needs-mjswan`, which links one ttktjmt/mjswan pull request:

- The mjswan PR closed without merging: close the task's PR with a comment saying so, and open a `daily-task-skipped` issue for its id.
- The mjswan PR has new commits: move the pin to its new head, `uv lock --upgrade-package mjswan`, build, rerun parity and push. The PR stays a draft.
- `main` locks an mjswan release that contains the mjswan PR's merge commit: finish the task as step 9 of msp:add-new-task says.
- Anything else: leave it.

## C. mjswan release bump

PyPI has a stable mjswan newer than the one `main` locks, and no open PR is labelled `mjswan-bump`: on `claude/mjswan-bump-<version>`, raise the floor in `pyproject.toml` (and the `mjlab` pin if mjswan's `mjlab` extra moved), `uv lock`, build every task, run `make test`, check every task's run with `scripts/record_preview.py --all --out-dir` a scratch directory (msp:add-new-task step 7 has the cloud flags), and open the PR labelled `mjswan-bump` with each task's check result.

## D. One new task

Skipped while three or more `needs-mjswan` PRs are open.

1. Pick the first backlog entry whose id is not in `mjswan_playground.registry.ALL_TASKS`, has no PR from `claude/daily-<id>` in any state, and has no open `daily-task-skipped` issue. None left: open one issue titled "Daily backlog is empty", unless one is open.
2. Run msp:add-new-task in unattended mode on branch `claude/daily-<id>`, with the entry as every answer: `repo` the target and `commit` the commit to pin, `id` the playground id, `task` the mjlab task, `license` the license, and `policy` and `notes` what they say. When the skill is not in your skill list, follow `.claude/skills/msp/skills/add-new-task/SKILL.md` directly. Its step 9 may open or reuse an mjswan PR, on branch `claude/playground-<id>` of ttktjmt/mjswan.
3. Label the PR `daily-task`.
4. On a stop: open, or update, an issue labelled `daily-task-skipped` and titled `<id>: <reason>`, with the step, the error verbatim and what would unblock it.

## E. Check the run's work

Before the report, check every result of this run against GitHub and the files, never against memory. Fix what is off and check it again: a result is done only once its check passes. One that cannot be fixed this run leads the report as a failure, with what is wrong.

- Every branch this run pushed: its remote head is the local commit, and the working tree is clean.
- Every pull request this run opened or updated: it targets `main` and carries its labels (`daily-backlog`, `daily-task`, `mjswan-bump`, or `needs-mjswan` on a draft). Its checks have finished green on its current head: wait for them, up to 30 minutes. Root-cause a red check, fix it, push and wait again; one that is red on `main` too is noted, not fixed here. A `needs-mjswan` draft's `released-mjswan` check is red by design.
- The new task's pull request holds the whole task:
  - `src/mjswan_playground/<id>/`, with a README that opens with the GIF;
  - the registry line, the README row, `assets/<id>.gif` and its `PREVIEWS` entry;
  - nothing from `.cache/` or `dist/`.

  Its last preview round passed (`"passed": true` in `dist/preview/<id>.json`), and the committed GIF is that round's. A fresh worktree of the pushed head passes `make sync`, `make test` and `uv run msp build <id>`, so nothing the build needs was left uncommitted.
- `daily/backlog.yaml` on the backlog branch parses, and every new entry has every field.
- A stop's `daily-task-skipped` issue exists, with the step, the error verbatim and what would unblock it.

## Never

Merge anything; push to `main` of either repository; edit `daily/backlog.yaml` anywhere but on `claude/daily-backlog`; run mjswan's release workflow; run `mjswan login` or `mjswan publish`. Publishing stays with a person.

## Report

End every run, a stopped one included, by publishing one Artifact titled `Daily run <YYYY-MM-DD>`, written in Japanese for the owner, then close with its link and a two-line summary. Build it to be read at a glance, pictures first:

- At the top, one card per part, A to D: what it did, or why it did nothing. A stop leads, with the step, the error verbatim and what would unblock it.
- The new task, whether it reached a pull request or stopped:
  - msp:add-new-task's steps 00 to 11 as a strip, each marked done, skipped, or where the run stopped;
  - the preview GIF and its contact sheet, `dist/preview/<id>.png`, published as the page's own files;
  - a chart of the checked run from `dist/preview/<id>.json`: root height and tilt over the control steps, the filmed part shaded, every termination marked;
  - the preview's rounds as a table, each failure beside its fix;
  - parity, with the terms traced, dropped and skipped; its sources and licenses; and its pull request or issue, with the mjswan PR if one was opened or reused.
- The scout's new backlog entries as a table: repository, the evidence it trends, license.
- The waiting PRs touched, each with what changed. After a bump, every task's check as a pass-or-fail grid, naming each first failure.

Build and publish it the way the Artifact tool's own instructions say, with the images as supporting files and the charts drawn from the data, and put nothing secret on the page. Then read the published page back with the Artifact tool, and republish until every card, image and chart is on it. Without the Artifact tool, end with the same content in Markdown and send the contact sheet as a file.
