# Parkour MJLab (`parkour`)

<img src="../../../assets/parkour.gif" width="480" alt="Parkour MJLab preview"/>

Source: https://github.com/yhx1203/parkour_mjlab, a port of PHP to mjlab (no license
stated, see [License](#license); its G1 model is Unitree's, BSD-3-Clause) · PHP's motion
clips and their boxes from https://github.com/amazon-far/php_parkour, release
`training-motion-examples-v1` (Apache-2.0)

> Zhen Wu, Xiaoyu Huang, Lujie Yang, Yuanhang Zhang, Xi Chen, Pieter Abbeel, Rocky Duan,
> Angjoo Kanazawa, Carmelo Sferrazza, Guanya Shi, C. Karen Liu.
> [Perceptive Humanoid Parkour: Chaining Dynamic Human Skills via Motion Matching](https://arxiv.org/abs/2602.15827).
> RSS, 2026. arXiv:2602.15827.

parkour_mjlab trains perceptive locomotion on mjlab: a Unitree Go2 on stairs with PIE,
and a Unitree G1 doing Amazon FAR's PHP parkour skills. Here PHP's five teachers run, one
scene each: **Unitree-G1-PHP-Locomotion-Teacher**, **-Low-Step-**, **-High-Step-**,
**-Low-Climb-** and **-High-Climb-Teacher** each track the first clip of their skill's
PHP bundle, over that clip's boxes, reading a height scan: a walk with turns, steps onto
0.35 m boxes at a walk and at a run, and climbs onto 0.76 m boxes at a walk and at a run.

Waiting on mjswan: **Unitree-Go2-PIE**, the Go2 on stairs, and **Unitree-G1-PHP-Student**,
the depth student the teachers distill into, each read a depth camera, which the browser
casts as rays. mjswan 0.12.1 casts each ray with its own call, and both scenes fall to 40
to 43 of their 50 control steps a second;
[mjswan#172](https://github.com/ttktjmt/mjswan/pull/172) casts a camera's rays in one. They
wait for it in a draft pull request of their own, from `claude/daily-parkour-mjswan`.

Left out: the G1 stair policy (`logs/rsl_rl/g1_stair/stair_test/policy.onnx`), whose task
is not in the repository ("Release the G1 Goal Stair code" is unchecked), and
`deploy/parkour`'s two ONNX files: `amp_policy.onnx` is a run of another repository,
humanoid_amp_mjlab, and `student_policy_v6_model_10000.onnx` trains on
`Unitree-G1-Direction-Mixed-Student-V6`, which the repository does not have either.

## Run

```sh
uv run msp run parkour
```

The first build clones parkour_mjlab into `.cache/` at a pinned commit (399 MB with its
checkpoints); set `MJSWAN_PARKOUR_ROOT` to use a checkout you have. It then downloads
PHP's five motion bundles (306 MB), checks each against the sha256 the release lists, and
extracts them where upstream's README puts them, `src/assets/motions/php_motion/`:
importing upstream's tasks reads every bundle. Last, under `.cache/parkour/`, it exports
each teacher's `model_20000.pt` once with mjlab's own runner, and writes each scene's
clip in the layout mjlab's motion command reads.

| From `parkour_mjlab` | Used as |
|---|---|
| the five `Unitree-G1-PHP-<skill>-Teacher` tasks (`play=True`) | each scene: the robot, its boxes, observations, action, command and terminations |
| `logs/rsl_rl/g1_php_<skill>_teacher/model_20000.pt` | each teacher, converted |
| `src/assets/robots/unitree_g1/LICENSE` | the scenes' attribution |

| From `php_parkour`, release `training-motion-examples-v1` | Used as |
|---|---|
| `php-motions-<skill>-v1.tar.gz`, five bundles | the tasks' clips: each teacher plays its bundle's first, `*_motion.npz`, over its `*_terrain.npy` boxes, variant 0 |

| Scene | Policy | Checkpoint | Input → action | Rate |
|---|---|---|---|---|
| Unitree-G1-PHP-<skill>-Teacher | Locomotion, Low-Step, High-Step, Low-Climb, High-Climb | `g1_php_<skill>_teacher/model_20000.pt` | 449 → 29 | 50 Hz |

## What the teachers read

mjlab's tracking actor: the reference's joint positions and velocities (58), its anchor
position and orientation in the robot's torso frame (3 + 6), `base_lin_vel`,
`base_ang_vel`, `joint_pos`, `joint_vel` and the last action, then a height scan: 17 × 17
rays 0.1 m apart under `torso_link`, yaw-aligned, onto group 0 (289).

## How it behaves

In the browser, each teacher steps or climbs onto its boxes with its clip. The Locomotion
teacher loses its clip in the fast turns, 11.3 s in, and its `ee_body_pos` termination
starts the clip over: the turns are marginal for it, as in mjlab, where 0.02 rad of
action noise ends the same clip at 14.5 s.

## What differs from upstream

- **mjlab 1.6.0.** Upstream pins mjlab 1.2.0 and mujoco 3.5.0. Importing it on 1.6 needs
  two shims ([`_compat.py`](../_compat.py)): `update_assets`, which 1.6 removed, and the
  `CollisionCfg` fields 1.6 made required. mjlab's tracking env config is the same in
  both, and its G1 constants differ only in how they load assets and spell out the
  collision defaults. On 1.6 an explicit `env.reset()` also advances each command once,
  which 1.2 did not.
- **PHP's clips run on mjlab's own motion command.** Upstream's `PhpMotionCommand` plays
  a bundle's clips one after another, resizing and placing its mocap boxes for each. Here
  mjlab's `MotionCommand` plays the bundle's first clip from its first frame, as play's
  sequential draw begins, over that clip's boxes, baked into the scene where variant 0
  puts them; the clip is written in mjlab's layout, as float32 as upstream reads it. At
  the clip's end, mjswan's tracking command restarts the episode, where upstream's
  `motion_end` time-out does, and then plays the next clip. In mjlab, beside upstream's
  own env, the High-Climb teacher's observations agree to 1e-7 until upstream's
  `motion_end` fires on the clip's last frame.
- **`nan_detection` without `|=`** ([`_trace.py`](../_trace.py)): mjlab's own builds its
  mask in place, which `torch.onnx.export` refuses (`aten::__ior_`).
- **The teachers are converted here,** with mjlab's own runner and its export metadata,
  from the `.pt` files upstream publishes.
- **Dropped:** `php_box_fields`, a startup event that declares the box fields
  `place_boxes` resizes, which static boxes do not need; and what only training reads:
  the critic group, rewards and curricula.
- **No encoder bias.** Play randomizes nothing, and the teachers' metadata carries none.

## License

parkour_mjlab is a port of PHP to mjlab, which PHP's README lists among its community
projects. Only some of what the scenes carry is PHP's:

- **PHP's clips and boxes** are Amazon's, under Apache-2.0 like PHP's repository, and the
  release ships them unchanged. Each scene carries the license and Amazon's notice, which
  also says what the port changes: the clip is rewritten in mjlab's layout, and its boxes
  are built into the scene.
- **parkour_mjlab's task code and checkpoints** are its own. The code is written for
  mjlab, not copied from PHP or the Holosoma framework it trains with, and the teachers
  are its own training runs, since PHP releases no teacher checkpoints. parkour_mjlab
  states no license, so they carry no grant, and the project ships no license file, only
  a notice saying so.
- **The G1 model** is Unitree's under BSD-3-Clause (upstream's
  `src/assets/robots/unitree_g1/LICENSE`, which every scene carries).

The demo is published only by its author.
