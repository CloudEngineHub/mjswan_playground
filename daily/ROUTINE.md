# Daily run

What the "mjswan playground daily" routine does each morning. Its prompt only points here, so changing the routine is a pull request against this file.

Each run is a fresh cloud session with ttktjmt/mjswan_playground and ttktjmt/mjswan checked out. It asks nothing: what this file and the backlog leave open is a stop. The parts run in order.

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

## Never

Merge anything; push to `main` of either repository; edit `daily/backlog.yaml` anywhere but on `claude/daily-backlog`; run mjswan's release workflow; run `mjswan login` or `mjswan publish`. Publishing stays with a person.

## Report

End with the entries the scout added and why, the waiting PRs touched, the new task's id with its PR or issue URL, and the mjswan PR URL if one was opened or reused.
