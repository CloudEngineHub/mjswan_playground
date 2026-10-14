# Asimov 1 (`asimov`)

<img src="../../../assets/asimov.gif" width="480" alt="Asimov 1 preview"/>

Source: https://github.com/openhorizon-labs/asimov-v1-policies (Apache-2.0) · walking
policy from https://huggingface.co/Menlo/asimov1-locomotion-0818 (BSD-3-Clause) · get-up
policy from https://huggingface.co/openhorizon-labs/asimov1-getup-safefall (Apache-2.0) ·
robot model from https://github.com/menloresearch/asimov-1 (**CERN-OHL-S-2.0 or GPL-2.0**,
see [License](#license)) · robot: [Asimov](https://asimov.inc) by Menlo Research

Asimov 1, Menlo Research's open-source 1.2 m humanoid, with two policies on one scene:

- **`walk`**, Menlo's own walking policy, follows a velocity command: mjlab's, drawn
  anew every 3 to 8 s, or the panel's joystick. Each episode starts standing at a random
  heading and lasts 20 s, unless the robot tips past 70°.
- **`getup`**, OpenHorizon Labs' get-up policy, starts each episode from one of the 512
  settled fall states upstream ships (prone, supine, on a side or in between) at a random
  heading; the robot stands within about two seconds and holds the stance until the
  20 s episode ends and drops it into the next fall.

Switching policies starts the chosen one from its own reset, so the get-up does not pick
up a walk that fell. OpenHorizon says its policies have only run in simulation.

Upstream's other policy, `Asimov1-SafeFall`, is left out: its action term,
`TriggeredJointPositionActionCfg`, holds the action at zero until a fall trigger of its
own fires, and mjswan has no counterpart for it.

## Run

```sh
uv run msp run asimov
```

The first build clones both repositories into `.cache/` at pinned commits and downloads
the two policies, and the get-up's spec, from the Hub at pinned revisions. Set
`MJSWAN_ASIMOV_ROOT` and `MJSWAN_ASIMOV_MODEL_ROOT` to point at checkouts you already
have. Upstream is a package registered through mjlab's `mjlab.tasks` entry point, but
the checkout is not installed: [`upstream.py`](upstream.py) points `ASIMOV_1_MODEL_DIR`
at the asimov-1 checkout, which stands in for the submodule the clone leaves empty, puts
`src/` on `sys.path` and imports `asimov_policies.tasks`.

| From upstream | Used as |
|---|---|
| `make_getup_env_cfg(..., play=True)` at `695ceba`, the commit the get-up was trained at, with flat ground | the scene (the robot with upstream's collision overlay), and the get-up's observations, action term, terminations and reset |
| `data/fall_states.v1-local/fall_states.npz` at `695ceba` | the get-up's 512 fall states; later commits delete the file |
| `policies/getup.onnx` on the Hub at `f657abe` | the get-up policy (375 → 23), its normalizer inside the graph |
| `policies/getup_policy_spec.json` on the Hub at `f657abe` | the action's joint order and the default pose, for both policies: the ONNX carries no mjlab metadata |
| `Menlo/asimov1-locomotion-0818` on the Hub at `18114a7`: `policy.onnx` | the walking policy (78 → 23), with no normalizer, as its training config sets |
| the same revision's `env.yaml`, its Isaac Lab training config | the walking MDP, restated in [`walk.py`](walk.py), and the scene's torque limits |
| asimov-1 `sim-model/` at `35ae7b3`, the submodule's commit | the robot's MJCF and meshes |
| `LICENSE`, `NOTICE`; asimov-1's `HARDWARE-LICENSE.txt`, `SOFTWARE-LICENSE.txt` | the project's license and notice; two of the scene's attributions |

## What the policies read

`walk` reads 78 values at 50 Hz:

| Term | Width | Source |
|---|---|---|
| `base_ang_vel` | 3 | the pelvis's angular velocity, × 0.25 |
| `projected_gravity` | 3 | gravity in the pelvis frame |
| `command` | 3 | the velocity command: forward, sideways, turning |
| `joint_pos_slot01`, `_slot23`, `_slot45` | 9, 8, 6 | joint positions minus the default pose, in upstream's CAN slot order |
| `joint_vel_slot01`, `_slot23`, `_slot45` | 9, 8, 6 | joint velocities × 0.1, same order |
| `actions` | 23 | the previous action |

`getup` reads five terms over the last five control steps, each term's five values
oldest first:

| Term | Width | Source |
|---|---|---|
| `base_ang_vel` | 3 × 5 | the IMU gyro on `imu_in_pelvis`, × 0.25 |
| `projected_gravity` | 3 × 5 | gravity in the pelvis frame |
| `joint_pos` | 23 × 5 | joint positions minus the default pose, in upstream's CAN slot order |
| `joint_vel` | 23 × 5 | joint velocities × 0.1, same order |
| `actions` | 23 × 5 | the previous action |

Both set each of the 23 joints' targets to the default pose plus 0.25 × the action, in
the model's joint order, which differs from the observations'.

## What differs from upstream

### Both policies

- **Flat ground.** Upstream's play config keeps training's ground: flat patches, and
  heightfields with 2 to 5 cm bumps on the other 70%. mujoco_warp, which mjlab trains
  and plays on, gives a box or capsule touching a heightfield at most one contact (it
  warns so when it builds the scene), where the browser's MuJoCo gives several, and a
  robot lying down touches the ground with exactly those shapes. On a heightfield the
  browser's get-up fell back once before it stood in two of three filmed rounds from
  the same start; in mjlab, with upstream's motors, the policy falls back in 5 of 128
  starts there, and in none of 128 on flat ground. [`upstream.py`](upstream.py) builds
  the config with upstream's own `rough=False`: a plane, and the reset's spawn height
  for flat ground, 1 cm where the bumps take 6.
- **The actuators run as PD within the effort limit.** Upstream's `DcMotorActuatorCfg`
  also caps each motor's torque on a torque-speed curve, from its peak torque at rest
  to zero at its no-load speed, and delays each command by 0 to 5 physics steps (0 to
  25 ms). mjswan runs the `IdealPdActuatorCfg` it subclasses, dropping both. In mjlab on
  this flat ground, from the same 128 fall states, the get-up stands up in all 128
  either way: in a median 1.23 s with upstream's motors, 1.12 s without the curve and
  the delay.
- **The limits are `env.yaml`'s.** The walking policy trained with 45 N·m at the hips'
  pitch and roll and at the knees and 28 N·m at the hips' yaw, where upstream takes the
  motors' rated 40, 30, 25 and 20 N·m; the other joints' limits agree.
  [`walk.py`](walk.py) raises them for the scene, which both policies share. In MuJoCo's
  C build, which the browser runs, with the browser's PD, the walking policy fell in 5 of
  100 20-second runs of random commands at the rated limits and in none at
  `env.yaml`'s. In mjlab, with the same PD, the get-up stood from all of 64 fall states
  at either.
- **One robot, not a population.** The resets' draws come from the browser's seeded
  PRNG, where mjlab draws them per env.

### `walk`

- **The MDP is restated from `env.yaml`.** Menlo trained the policy in Isaac Lab, so
  there is no mjlab task. [`walk.py`](walk.py) writes its observations, velocity command
  and terminations from the Hub's `env.yaml`, on upstream's get-up scene, whose
  `robot.py` copies Menlo's joint order, default pose, PD gains, armature, joint friction
  and action scale from Menlo's training code. Menlo's own mjlab ports do not fit:
  [menloresearch/mjlab](https://github.com/menloresearch/mjlab)'s Asimov 1 task trains a
  different recipe (straight knees, stiffer ankles, a filtered action, a gait clock in
  the observation) and ships no policy, and
  [asimov-mjlab](https://github.com/menloresearch/asimov-mjlab) is the legs-only Asimov.
- **Play's conditions.** Training adds noise and delays to the observations (the gyro
  0 to 1 control steps late, gravity 0 to 2), randomizes the feet's friction, the
  joints' zeros, the torso's centre of mass and the PD gains, pushes the robot every 1
  to 3 s and resets it up to 0.5 m and 0.5 rad off the default pose. Here none of that
  happens, as in mjlab's play: each episode starts at the default pose, facing a random
  heading.
- **MuJoCo, not PhysX.** The policy never trained on MuJoCo's contacts, and its
  `env.yaml` gives joint friction as a PhysX coefficient, which upstream's entity takes
  as a frictionloss in N·m. In mjlab, over 32 robots for 20 s of `env.yaml`'s random
  commands, it kept within 0.13 to 0.15 m/s of the commanded velocity, and none fell at
  `env.yaml`'s limits.

### `getup`

- **No observation delays.** Upstream trains `base_ang_vel` behind a delay of 0 to 1
  control steps and `projected_gravity` behind 0 to 2. mjswan reads the current value,
  the low end of both ranges.
- **The reset is a function.** Upstream's `reset_from_fall_states` is a class that loads
  its table in `__init__`, and mjswan traces a term by calling it. [`terms.py`](terms.py)
  restates it as a function with the same table and draws: from the same seed it writes
  the same root pose and joint positions as upstream's class.
- **`nan_detection` without `|=`.** mjlab's own term builds its mask in place, which
  `torch.onnx.export` cannot export (`aten::__ior_`); [`terms.py`](terms.py) restates
  it with `|`.
- **No `assist_force`.** The training curriculum's upward pull on the torso, whose scale
  play already holds at zero. Dropped, with the curriculum, the rewards and the metrics,
  which only train.
- **No encoder bias.** Play randomizes nothing, and the policy's spec carries none.

## License

The tasks, the code and the get-up policy are Apache-2.0. Menlo's walking policy is
BSD-3-Clause by its model card, which names no copyright holder; the build ships the
standard text as `LICENSE.asimov1-locomotion`, under Menlo Research, which its training
code names. The robot model is neither: menloresearch/asimov-1 licenses its hardware
under CERN-OHL-S-2.0 and its software under GPL-2.0 without saying which covers
`sim-model/`, the MJCF and STL meshes this task compiles into `scene.mjz` (upstream's
open question Q1; Menlo says a statement is coming). Both are reciprocal. The build
ships both texts beside the scene, as `LICENSE.asimov-1-hardware` and
`LICENSE.asimov-1-software`, and upstream's `NOTICE` with the project, so
`mjswan publish` warns on the GPL-2.0 file. Publishing the demo is its author's call,
under those terms.
