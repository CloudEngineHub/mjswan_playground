# Daily tasks

A routine adds one task a day from [`backlog.yaml`](backlog.yaml) and opens a pull request for it, following [`ROUTINE.md`](ROUTINE.md). Merging and publishing stay with people.

- `backlog.yaml`: the queue. A person fills in each entry's license and whether the demo may be published.
- `ROUTINE.md`: what each run does, and what it never does.
- [`publish.yml`](../.github/workflows/publish.yml): publishes a task when its pull request gets the `publish` label, then links it from the README row.
- [`released-mjswan.yml`](../.github/workflows/released-mjswan.yml): keeps a task that waits on an unreleased mjswan off `main`.
- Labels: `daily-task`, `daily-task-skipped`, `needs-mjswan`, `mjswan-bump` and `publish` here; `from-playground` and `cloud-ready` on ttktjmt/mjswan.

## One-time setup

1. A cloud environment named `mjswan-daily`:
   - Network: Custom, with the default list included, plus:

     ```
     huggingface.co
     *.huggingface.co
     *.hf.co
     api.wandb.ai
     ```

   - Environment variables:

     ```
     CLAUDE_CODE_PLUGIN_DIRS=/home/user/mjswan_playground/.claude/skills/msp
     BASH_DEFAULT_TIMEOUT_MS=600000
     MUJOCO_GL=disable
     ```

2. Two repository secrets for `publish.yml`:
   - `MJSWAN_CREDENTIALS`: the contents of `~/.config/mjswan/credentials.json` after `mjswan login` with the bot's GitHub account.
   - `SECRETS_WRITE_PAT`: a fine-grained token for this repository with "Secrets: Read and write", so each run can store the session it refreshed.
3. A ruleset on `main` in both repositories that blocks direct pushes, and here also requires `released-mjswan`.
4. The routine:
   - Repositories: ttktjmt/mjswan_playground and ttktjmt/mjswan
   - Environment: `mjswan-daily`
   - Schedule: every day at 04:47 JST
   - Prompt:

     ```
     Do today's daily run for ttktjmt/mjswan_playground exactly as daily/ROUTINE.md on its main branch says. You may push to claude/ branches of ttktjmt/mjswan_playground and ttktjmt/mjswan, open pull requests and issues in both, and add the labels that file names. Never merge, and never push to main.
     ```

5. Run it once by hand and read the run before leaving the schedule on. A finished run only means the session ended: check the pull request, the issue it opened, or both.
