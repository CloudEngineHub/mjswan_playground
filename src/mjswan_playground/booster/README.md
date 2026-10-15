# Booster K1 (`booster`)

<img src="../../../assets/booster.gif" width="480" alt="Booster K1 preview"/>

Source: https://github.com/IntelligentRoboticsLab/booster_mjlab, its project page's walk
(**no license stated**, see [License](#license)) ·
https://github.com/BoosterRobotics/booster_deploy, Booster's own walk, dance and fight
(Apache-2.0: the policies, the clips, whose origin it does not state, and the deploy
settings) · their robot model: https://github.com/BoosterRobotics/booster_assets
(BSD-3-Clause) · project page: https://intelligentroboticslab.github.io/booster_mjlab/

A Booster K1 humanoid walking to velocity commands with two policies, the one
booster_mjlab's project page runs live in the browser and Booster Robotics' own, and
dancing and fighting along two clips. Booster's three are the ones booster_deploy's
`scripts/deploy.py --task <name>` runs on the robot and in MuJoCo.

| Scene | What it does |
|---|---|
| `Mjlab-Velocity-Flat-Booster-K1` | Walks on flat ground to a command drawn anew every 1 to 4 s, a fifth of the time standing still, as upstream's play config draws it; mjlab's joystick panel can take over. Play also keeps upstream's random pushes, one every 1.5 to 4 s. |
| `K1 Walk` | Walks to a command drawn anew every 4 to 8 s within booster_deploy's limits, -0.3 to 1.6 m/s forward, ±0.3 m/s sideways and ±1.8 rad/s, a fifth of the time standing still; the joystick panel can take over. |
| `K1 MJ Dance` | Dances along `k1_mj2_seg1.npz`: 25.2 s, the last 3.5 s standing still, then starts over. |
| `K1 Fight` | Punches and kicks along `k1_fight_final_deploy.npz`: 12.0 s, the last 3 s standing still, then starts over. |

booster_mjlab's other tasks, `Mjlab-Tracking-Flat-Booster-K1` and the velocity tasks on
rough ground or with the parallel-linkage ankle (`-Parallel`), publish no checkpoint, so
they are left out. The project page's kick, side flip and Thomas flare come from the
tracking task, and neither their checkpoints nor their clips are published: not in the
repository, on its Hub organization (`whirlwind-ams`, datasets only) or on the team's
site. booster_deploy's other policies run on other robots, each with its own model and
gains: the T1 and T2 dances and walks.

## Run

```sh
uv run msp run booster
```

The build clones booster_mjlab, booster_deploy and booster_assets into `.cache/` at pinned
commits; set `MJSWAN_BOOSTER_ROOT`, `MJSWAN_BOOSTER_DEPLOY_ROOT` or
`MJSWAN_BOOSTER_ASSETS_ROOT` to point at a checkout you already have. It replays each clip
through mjlab once, into `.cache/booster/`. `uv run msp build booster-open` builds every
scene but booster_mjlab's walk, to publish without it (see [License](#license)).

| Scene | Policy | Upstream task | Checkpoint |
|---|---|---|---|
| `Mjlab-Velocity-Flat-Booster-K1` | `walk` | booster_mjlab `Mjlab-Velocity-Flat-Booster-K1` | the project page's `policy.bin`, run `2026-09-18_18-14-22` |
| `K1 Walk` | `k1_walk` | booster_deploy `k1_walk` | `k1_walk.pt` |
| `K1 MJ Dance` | `k1_mj2` | booster_deploy `k1_mj2` | `k1_mj_dance_002_2025-12-03_00-10-28.pt` |
| `K1 Fight` | `k1_fight` | booster_deploy `k1_fight` | `k1_fight_001.pt` |

booster_mjlab's walk runs its own K1 model and actuators, which its policy was trained
on; booster_deploy's walk, dance and fight run booster_assets' K1, as booster_deploy does.

### booster_mjlab's walk

booster_mjlab is a package that requires Python 3.13, which the playground cannot
install, so [`upstream.py`](upstream.py) puts its `src/` on `sys.path` and imports
`booster_mjlab.tasks`, which registers the tasks. The code imports under the playground's
3.12 as it is.

| From `booster_mjlab` | Used as |
|---|---|
| `Mjlab-Velocity-Flat-Booster-K1` (`play=True`) | the scene, the observations, the action term, the command, the terminations and the startup randomization |
| `website/static/demo/policy.bin`, `scene.json` | the policy (75 → 22), rebuilt as ONNX; `scene.json` gives the joint order and the rest pose |

The project page's exporter, `scripts/export_web_demo.py`, flattened the run's ONNX into
raw float32 weights and a list of layers: the observation normalizer (`Sub`, `Div`) and an
MLP of 512, 256 and 128 ELU units. `upstream.py` writes those layers back as ONNX, which
agrees with upstream's own evaluator to float32 rounding. `scene.json`'s observation order,
joint order, PD gains and rest pose are the flat config's, and its action scales are the
config's rounded to three decimals. The project page calls the run its AMP velocity
policy without naming its task. The flat `-Amp` tasks share this policy's observation,
action and command with the plain flat task, which the scene runs, and their play differs
from it only in its resets (see [What differs from upstream](#what-differs-from-upstream)).

### booster_deploy's walk, dance and fight

booster_deploy is a deploy framework, not an mjlab repository: [`deploy.py`](deploy.py)
imports its K1 tasks from the checkout and converts their policies and clips, and
[`envs.py`](envs.py) builds mjlab envs around booster_assets' K1.

| From | Used as |
|---|---|
| booster_deploy `tasks/locomotion/robots/k1/__init__.py`, `tasks/beyond_mimic/robots/k1/__init__.py` | each scene's PD gains, effort limits, rest pose and timing, and the walk's command limits, imported from the checkout |
| booster_deploy `booster_deploy/robots/k1.py` | the model's joint order, and Isaac Lab's that the clip policies were trained in |
| booster_deploy `tasks/locomotion/robots/k1/models/k1_walk.pt` | the walk policy (TorchScript, ten 69-wide frames → 20), rebuilt as ONNX with `LocomotionPolicy` folded in |
| booster_deploy `tasks/beyond_mimic/robots/k1/models/*.pt` | the clip policies (TorchScript, 119 → 22), rebuilt as ONNX |
| booster_deploy `tasks/beyond_mimic/robots/k1/motions/*.npz` | the clips, replayed through mjlab into the model's joint and body order |
| booster_assets `robots/K1/K1_22dof.xml` | the robot, without its floor, lights and motors |
| mjlab's tracking task | the clips' observations, motion command and terminations |

Each scene runs what booster_deploy's MuJoCo controller runs: booster_assets' MJCF with
its own solver options, physics at 500 Hz and the policy at 50 Hz, and a PD torque every
physics step, clipped to the task's effort limit.

**The walk.** Its policy reads the last ten observations, oldest first, and booster_deploy's
`LocomotionPolicy` keeps them, the last action and a filtered joint target between steps.
The ONNX keeps the same three in mjswan's recurrent carry, `adapt_hx`. It takes one
observation without the last action from mjswan, scales the joint velocities by 0.1,
fills the history with the first observation after a reset, takes 0.2 off the right
elbow's action, and moves each target 80% of the way to the policy's new one. Stepped
beside booster_deploy's own controller for 17 s of changing commands, it gives the same
joint targets to float32 rounding (1.4e-5 rad). Closed loop over 26 s of seven commands,
from standing to 1.6 m/s forward and a 1.8 rad/s turn, it walks as booster_deploy's
MuJoCo run does:

| | booster_deploy / here |
|---|---|
| Falls or safety stops | none / none |
| Mean velocity error, forward and sideways | 0.278 and 0.100 / 0.284 and 0.099 m/s |
| Mean yaw rate error | 0.146 / 0.127 rad/s |
| Largest tilt | 5.8 / 6.6° |

**The dance and the fight.** The ONNX permutes the observation into Isaac Lab's joint
order and the action back out of it, and matches the TorchScript checkpoint to float32
rounding. Over each whole clip in mjlab, the port tracks as booster_deploy's own MuJoCo
run does:

| | MJ dance: booster_deploy / here | Fight: booster_deploy / here |
|---|---|---|
| Mean anchor position error | 0.164 / 0.147 m | 0.295 / 0.297 m |
| Mean joint position error (norm over 22 joints) | 0.553 / 0.549 rad | 1.507 / 1.503 rad |
| Largest tilt | 24.1 / 24.1° | 34.9 / 35.0° |

## What the policies read

`walk`: six terms, one frame, no history.

| Term | Width | Source |
|---|---|---|
| `base_ang_vel` | 3 | the `imu_ang_vel` sensor |
| `projected_gravity` | 3 | `projected_gravity_b` |
| `joint_pos` | 22 | `joint_pos_rel`, read through the encoder bias |
| `joint_vel` | 22 | `joint_vel_rel` |
| `actions` | 22 | the previous action |
| `command` | 3 | the `twist` command |

`k1_walk`: one frame from mjswan, in the policy's joint order, which leaves out the head;
the ONNX adds the last action and keeps ten frames.

| Term | Width | Source |
|---|---|---|
| `base_ang_vel` | 3 | `root_link_ang_vel_b` |
| `projected_gravity` | 3 | `projected_gravity_b` |
| `command` | 3 | the `twist` command |
| `joint_pos` | 20 | `joint_pos_rel` |
| `joint_vel` | 20 | `joint_vel_rel`, scaled by 0.1 in the ONNX |
| last action | 20 | the ONNX's carry |

`k1_mj2` and `k1_fight`: mjlab's tracking observation without state estimation, one
frame, no history, in the model's joint order.

| Term | Width | Source |
|---|---|---|
| `command` | 44 | the clip's joint positions and velocities |
| `motion_anchor_ori_b` | 6 | the clip's trunk orientation in the robot's trunk frame, two columns of its matrix |
| `base_ang_vel` | 3 | the trunk's gyro |
| `joint_pos` | 22 | `joint_pos_rel` |
| `joint_vel` | 22 | `joint_vel_rel` |
| `actions` | 22 | the previous action |

## What differs from upstream

booster_mjlab's walk:

- **The plain flat task's resets.** The `-Amp` tasks' play resets the robot to a pose
  drawn from the clips of the
  [lafan_locomotion_k1](https://huggingface.co/datasets/whirlwind-ams/lafan_locomotion_k1)
  dataset. Those clips are retargeted from LAFAN1, whose CC BY-NC-ND 4.0 license forbids
  sharing adaptations, so the build fetches none: each reset places the home pose within
  ±0.5 m at any heading and offsets every joint by up to ±0.1 rad.
- **The command ranges are the curriculum's first stage.** Play's config widens them to
  ±2.5 m/s forward, ±2.0 m/s sideways and ±3.7 rad/s, but it keeps the `command_vel`
  curriculum, which sets them back to its first stage at every reset: -1.0 to 1.2 m/s,
  ±1.0 m/s and ±1.0 rad/s, until 40 minutes in. mjswan runs no curriculum, so `main.py`
  sets that stage itself, and the joystick's limits follow it.
- **No inertia randomization.** Play draws the mass, center of mass and inertia of the
  trunk and of every other body once at startup, with mjlab's `pseudo_inertia`. It does
  not trace: torch exports neither `diag_embed` nor `mT` to ONNX at opset 17, and the
  eigenvector sign fix after mjlab's Jacobi solver branches on the data, which a trace
  would freeze. `main.py` drops both events, and the browser runs the nominal model.
- **No actuator delay.** The actuators take each command 2 to 8 physics steps late
  (10 to 40 ms) in training and play; mjswan applies it at once, as upstream's own web
  demo does.
- **No encoder bias.** mjlab's play config draws a ±0.015 rad bias per joint at startup;
  the browser applies the policy config's `encoder_bias` instead, and the demo ships none.
- **One randomized robot, not a population.** The foot friction, the PD gains and the
  ground's contact softness are drawn once from the browser's seeded PRNG, where mjlab
  draws them per env.
- **mjlab 1.6.0.** Upstream locks mjlab at `b517e0c`, six commits past 1.6.0: rsl-rl 5.5.0
  in the runners and a `sample_gaussian` fix, neither of which this play config reaches.

booster_deploy's walk:

- **The command is drawn at random.** booster_deploy reads it off the keyboard; here a
  new one comes every 4 to 8 s, within the same limits, until the joystick takes over.
- **A fall starts over.** booster_deploy's safety stop, once the trunk tilts 60°, ends
  the episode here, and the next starts at the rest pose, 0.6 m up, as booster_deploy
  starts.

The dance and the fight:

- **The clip starts over.** At its last frame the episode resets to its first, as mjlab's
  play does; booster_deploy holds the last frame.
- **Each episode starts on the clip's first frame.** booster_deploy starts the robot at
  its rest pose, 0.57 m up, wherever the clip begins.
- **mjlab's tracking terminations.** The episode resets when the trunk's or an ankle's or
  hand's height is 0.25 m off the clip's, or the trunk's tilt is off it by 0.8 in
  projected gravity. booster_deploy has only a safety stop, once the tilt is 60° off the
  clip's.

## License

booster_mjlab states no license: there is no LICENSE file, and neither its README, its
code nor its K1 model carries a notice. Without one, all rights stay with the authors, so
the project ships no `LICENSE`, and publishing its walk is its author's call. The model's
meshes are Booster Robotics' own: 23 of the 25 its MJCF loads are byte-identical to the
K1 in booster_assets, and the trunk's two are not there.

booster_deploy's three scenes are permissive: each carries booster_deploy's Apache-2.0
and booster_assets' BSD-3-Clause, and the ONNX files say what changed.
`uv run msp build booster-open` builds them without booster_mjlab's walk.
