# Shapes, by precedent

The five tasks, grouped by how they reach upstream. Read the precedent's files before writing a task of the same shape.

## extra: upstream is a package

The playground resolves upstream as an optional dependency, and importing it registers the tasks.

- `wbc`: `wbc = ["wbc-mjlab>=0.0.6"]`, from PyPI. `wbc-mjlab` registers `Wbc-*` through mjlab's `mjlab.tasks` entry point, so `add_scene_mjlab("Wbc-G1", env_cfg=...)` defaults every term off the task's own config. The policy and the clip library come from a separate deploy repo through `ensure_repo` (`MJSWAN_WBC_DEPLOY_ROOT`), whose `config.yaml` gives the joint order and default pose and whose manifest lists the clips. Two events the deploy runtime does not apply are popped from `env_cfg`.
- `musclemimic`: myosuite's `ms3` branch, a git source under `[tool.uv.sources]` that names the branch and leaves the commit to `uv.lock`; a new task pins `rev` instead. Registration is a call that needs a clip (`bootstrap_myosuite_mjlab_registry(clip_path=...)`). The policy is a Hub checkpoint converted once into `.cache/musclemimic/` (`upstream.ensure_policy`), the clip a gated Hub dataset (`hf auth login`), and `upstream.py` rebuilds in torch the observation the checkpoint was trained on.

## checkout: upstream runs only from source

The playground already resolves what upstream imports, but upstream is not a package it can install.

- `pacman`: `upstream.py` holds `resolve_root()`, an `ensure_repo` checkout (`MJSWAN_PACMAN_ROOT`), and `register_tasks(root)`, which puts the root first on `sys.path` (upstream imports itself as the top-level `src`) and imports its task package. Upstream is on mjlab 1.5.3; the shims for 1.6 sit under one `# ponytail:` comment that says when to drop them. The ONNX checkpoints and the deployed contract (`deploy/common/g1_deploy_constants.py`, loaded by path) come from the same checkout, and the terms that do not trace are in `terms.py`.

## data-only: only after a precedent

The package is never imported: the scene compiles from XML in a pinned checkout, the terms are picked from `mjlab.envs.mdp` by hand, and `build_single_entity_trace_env` traces them. This skill builds one only for a robot a task here already runs this way, by following that task's files, as a separate task with its own id; otherwise it stops and proposes the shape. There is no mjlab env to run parity against, so the preview is the check.

- `husky`: upstream ships the scene as XML generated from its own `scene_cfg` (`test_scene/mjlab_scene.xml`), so everything the demo needs is data.
- `microduck`: every env config drives the robot through BAM (`FrictionDRBamActuatorCfg`), which the browser has no counterpart for. The MJCF already carries the position actuators the servos run, which upstream's own `infer_policy.py` drives.
