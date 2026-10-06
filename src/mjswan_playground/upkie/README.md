# Upkie Velocity (`upkie`)

Source: https://github.com/MarcDcls/mjlab_upkie (Apache-2.0; code, model and checkpoint) ·
robot design after [Upkie](https://github.com/upkie/upkie) by Stéphane Caron

Upkie, a wheeled biped, balancing on two wheels and driving to velocity commands with the
checkpoint its author ships. The command (forward speed and turn rate) is resampled every
3 to 8 s, as upstream's play config draws it, and mjlab's joystick panel can take over.

## Run

```sh
uv run msp run upkie
```

The build clones the repository into `.cache/` at a pinned commit; set `MJSWAN_UPKIE_ROOT`
to point at a checkout you already have. Upstream is a package registered through mjlab's
`mjlab.tasks` entry point, but it pins `mjlab==1.3.0`, so the playground cannot install
it: [`upstream.py`](upstream.py) puts the checkout's `src/` on `sys.path` and imports
`mjlab_upkie.tasks`, which registers the task.

| From `mjlab_upkie` | Used as |
|---|---|
| `Mjlab-Velocity-Upkie` (`play=True`) | the scene, the observations, both action terms, the command, the terminations and the events |
| `logs/rsl_rl/upkie_velocity/bests/default.onnx` | the policy (22 → 6), already exported; its mjlab metadata gives the joints and the rest pose |
| `LICENSE` | the project's `LICENSE` |

## What the policy reads

Six terms, one frame, no history. Upstream's own lambdas read `qpos` / `qvel` directly;
mjswan traces them as they are.

| Term | Width | Source |
|---|---|---|
| `joint_pos` | 4 | hip and knee positions |
| `joint_vel` | 2 | wheel velocities |
| `trunk_imu` | 4 | the trunk quaternion, sign-flipped to keep `w` non-negative |
| `trunk_gyro` | 3 | the trunk angular velocity |
| `actions` | 6 | the previous action |
| `command` | 3 | the `twist` command |

The action drives the hips and knees by position (scale 1) and the wheels by velocity
(scale 100) through the robot's own `<velocity>` actuators.

## What differs from upstream

- **The action is in term order.** mjlab concatenates the leg term and then the wheel
  term, so the action vector is `left_hip, left_knee, right_hip, right_knee, left_wheel,
  right_wheel`, not the model's joint order that the checkpoint's metadata lists.
  `main.py` derives that order from the action terms and hands it to the browser as the
  policy's joint names.
- **mjlab 1.6.0.** Upstream pins 1.3.0. Its robot constants build `CollisionCfg` without
  the fields 1.6.0 made required, so `upstream.py` fills in 1.3.0's defaults while
  importing it. Against 1.3.0, the code this task runs differs only where nothing changes
  in play: interval events now fire before the command update and the auto-reset (the
  `push_robot` intensity is 0 in play), `bad_orientation` clamps its `acos` argument, and
  the terrain no longer adds origin marker sites.
- **No encoder bias.** Neither the task nor the checkpoint carries one.
- **One randomized robot, not a population.** The foot friction (0.8 to 1.2) is drawn once
  from the browser's seeded PRNG, where mjlab draws it per env.
- **No `noslip_iterations`.** The robot's MJCF asks for 4, which mjlab drops when it
  attaches the robot to the scene, in 1.3.0 as in 1.6.0, so the policy never trained with
  it either.
- **No rewards, no curriculum.** Both are training-only; the play config already drops
  the curriculum that widens the command ranges and push intensity during training.
