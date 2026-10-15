# Booster K1 (`booster`)

<img src="../../../assets/booster.gif" width="480" alt="Booster K1 preview"/>

Source: https://github.com/BoosterRobotics/booster_deploy (Apache-2.0: the policies, the
clips, whose origin it does not state, and the deploy settings) · robot model:
https://github.com/BoosterRobotics/booster_assets (BSD-3-Clause)

A Booster K1 humanoid walking to velocity commands, and dancing and fighting along two
clips, with Booster Robotics' own policies: the ones booster_deploy runs on the robot.

| Scene | What it does |
|---|---|
| `K1 Walk` | Walks to a command drawn anew every 4 to 8 s within booster_deploy's limits, -0.3 to 1.6 m/s forward, ±0.3 m/s sideways and ±1.8 rad/s, a fifth of the time standing still; the joystick panel can take over. |
| `K1 MJ Dance` | Dances along `k1_mj2_seg1.npz`: 25.2 s, the last 3.5 s standing still, then starts over. |
| `K1 Fight` | Punches and kicks along `k1_fight_final_deploy.npz`: 12.0 s, the last 3 s standing still, then starts over. |

booster_deploy's other policies, the T1 and T2 dances and walks, run on other robots.
[booster_mjlab](https://github.com/IntelligentRoboticsLab/booster_mjlab), whose project
page runs another K1 walk in the browser, is left out: it states no license.

## Run

```sh
uv run msp run booster
```

The build clones booster_deploy and booster_assets into `.cache/` at pinned commits; set
`MJSWAN_BOOSTER_ROOT` or `MJSWAN_BOOSTER_ASSETS_ROOT` to point at a checkout you already
have. It replays each clip through mjlab once, into `.cache/booster/`. booster_deploy is a
deploy framework, not an mjlab repository: [`upstream.py`](upstream.py) imports its K1
tasks from the checkout and converts their policies and clips, and [`envs.py`](envs.py)
builds mjlab envs around booster_assets' K1.

| Scene | Policy | booster_deploy task | Checkpoint |
|---|---|---|---|
| `K1 Walk` | `k1_walk` | `k1_walk` | `k1_walk.pt` |
| `K1 MJ Dance` | `k1_mj2` | `k1_mj2` | `k1_mj_dance_002_2025-12-03_00-10-28.pt` |
| `K1 Fight` | `k1_fight` | `k1_fight` | `k1_fight_001.pt` |

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
physics step, clipped to the task's effort limit. Each converted ONNX says in its
`doc_string` what changed, as Apache-2.0 asks.

**The walk.** Its policy reads the last ten observations, oldest first. booster_deploy's
`LocomotionPolicy` keeps them, the last action and a filtered joint target between steps;
the ONNX keeps the same three in mjswan's recurrent carry, `adapt_hx`. From one
observation without the last action, it scales the joint velocities by 0.1, fills the
history with the first observation after a reset, takes 0.2 off the right elbow's action,
and moves each target 80% of the way to the policy's new one. Beside booster_deploy's own
controller over 17 s of changing commands, it gives the same joint targets to float32
rounding (1.4e-5 rad); closed loop over 26 s of seven commands, from standing to 1.6 m/s
and a 1.8 rad/s turn, it walks as booster_deploy's MuJoCo run does:

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

The walk:

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
