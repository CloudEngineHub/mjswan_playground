# Jumper Posture (`jumper`)

<img src="../../../assets/jumper.gif" width="480" alt="Jumper Posture preview"/>

Source: https://github.com/KingKongRobotics/jumper (Apache-2.0; code, robot model and
checkpoint; its `NOTICE` names the third-party files it vendors)

KingKong Robotics' Jumper, a 22-joint crab robot, walking on a tripod gait while it holds
a commanded body posture: a twist over its planted feet, a pitch, a roll and a body
height from 0.07 m to 0.15 m. The policy is upstream's posture checkpoint, model_74800.
Both commands resample every 3 to 8 s as upstream's play config draws them, and the
control panel can take over either: mjlab's joystick for the twist, and an Enable box
with four sliders for the posture.

## Run

```sh
uv run msp run jumper
```

The build clones the repository into `.cache/` at a pinned commit; set
`MJSWAN_JUMPER_ROOT` to point at a checkout you already have. Upstream keeps a registry
of its own, so [`upstream.py`](upstream.py) puts the checkout first on `sys.path`, builds
`tasks.jumper.posture.env_cfg.env_cfg(play=True)` and registers it with mjlab as
`Jumper-Posture`. Its `rl/` directory, which vendors mjlab and rsl_rl beside upstream's
own `mjrl`, goes last on `sys.path`, so the installed mjlab 1.6.0 is the one imported.

| From `jumper` | Used as |
|---|---|
| `jumper.posture` (`play=True`) | the scene, the observations, the action term, the commands, the terminations and the events |
| `tasks/jumper/posture/out/example/actor.onnx` | the policy (415 → 20), model_74800, exported by upstream with its normalizer inside |
| `tasks/jumper/posture/out/example/layout.json` | the action joints, their default pose and the claws' held position |
| `LICENSE`, `NOTICE` | the project's license and notice |

## What the policy reads

Nine terms. Four of them stack five frames 4 control steps (20 ms) apart, oldest first:

| Term | Width | Source |
|---|---|---|
| `base_ang_vel` | 3 | the `imu_ang_vel` sensor |
| `projected_gravity` | 3 | `projected_gravity_b` |
| `joint_pos` | 5 × 20 | `joint_pos_rel` with the encoder bias, 16, 12, 8, 4 and 0 steps back |
| `joint_vel` | 5 × 20 | `joint_vel_rel`, same frames |
| `actions` | 5 × 20 | the previous action, same frames |
| `command` | 3 | the `twist` command |
| `actuator_force` | 5 × 20 | the servos' output, same frames |
| `gait_phase` | 2 | sin and cos of a gait clock whose cadence follows the twist, zero while standing |
| `posture_command` | 4 | twist, pitch, roll, and the height less the 0.10647 m standing height |

## What differs from upstream

- **The strided history is mjswan's own.** Upstream wraps each stacked term in
  `StridedHistory`, which keeps a ring of 17 frames and emits every fourth. Here the
  wrapper is unwrapped and each term carries the same frames as `history_steps`
  `(16, 12, 8, 4, 0)`.
- **The gait clock is a command term.** Upstream's `VariableGaitClock` is an observation
  that integrates its own phase at a cadence set by the twist command. A browser
  observation holds no state, so [`terms.py`](terms.py) moves the integration into a
  `gait_clock` command that advances each step at the cadence the twist asked for on the
  step before, and restarts on reset. Over 800 steps and 216 twist changes in mjlab it
  matches upstream's clock exactly.
- **The posture command's draw is restated as one graph.** Upstream's
  `PostureCommand._resample_command` writes per-env rows with `Tensor.uniform_`, which
  the tracer cannot follow; `terms.py` draws the same bands at one env. Its update moves
  only anchors that feed metrics, so it is dropped with them. The panel's sliders span
  the standing band; upstream's operator also clamps the angles to the narrower walking
  band while the twist moves, which the browser does not.
- **The operator terms are mjlab's and the panel's.** Play's `OperatorVelocityCommandCfg`
  and `TeleopPostureCommandCfg` read a keyboard or a gamepad; they become their bases,
  `UniformVelocityCommandCfg` and `PostureCommandCfg`, and the control panel takes their
  place. Play's ToF sensor, which only its viewer draws, is dropped.
- **The servos run as plain PD.** Upstream's `ServoCurveActuatorCfg` subclasses
  `IdealPdActuatorCfg`, and mjswan runs it as that: kp 10, kd 0.5, within the 1.746 N·m
  plateau. Dropped are its torque decay above 30.7 rad/s and its thermal derating toward
  1.2 N·m. The decay never engages here: with the policy driving in mjlab for 20 s, the
  fastest joint reaches 20.3 rad/s.
- **The claws are held by position actuators.** Upstream's servo PD holds the two claw
  joints, which no action moves, at 0 rad. The browser applies PD to the action's joints
  only, so [`main.py`](main.py) gives the claws MuJoCo position actuators with the same
  gains and limit, which hold them at 0 in mjlab and in the browser alike.
- **Startup randomization.** Foot friction and the base's center of mass are drawn once
  at startup by the browser's native model-field randomization, without a graph, so
  parity leaves them unchecked. Play also draws an encoder bias within ±0.015 rad, which
  mjswan takes from the policy config instead; the checkpoint ships none, so it is zero.
