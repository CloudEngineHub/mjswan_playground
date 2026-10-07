# G1 Double Spin Kick (`spinkick`)

<img src="../../../assets/spinkick.gif" width="480" alt="G1 Double Spin Kick preview"/>

Source: https://github.com/mujocolab/g1_spinkick_example (Apache-2.0) ·
reference motion from https://github.com/xbpeng/MimicKit (Apache-2.0)

A Unitree G1 doing a double spin kick with the checkpoint upstream ships for the real
robot: from a standing pose, through the kick, and back to standing, with the reference
drawn as a ghost. The clip is 4.6 s and starts over when it ends.

## Run

```sh
uv sync --extra spinkick        # spinkick-example registers Mjlab-Spinkick-Unitree-G1
uv run msp run spinkick
```

The extra installs upstream's package at a pinned commit, and it registers its task
through mjlab's `mjlab.tasks` entry point, so `add_scene_mjlab` builds the scene and
**every term set defaults off its env config**. The first build also clones the repository
into `.cache/` at the same commit for the checkpoint, and converts the clip it carries into
`.cache/spinkick/`; set `MJSWAN_SPINKICK_ROOT` to use a checkout you already have.

| From `g1_spinkick_example` | Used as |
|---|---|
| `Mjlab-Spinkick-Unitree-G1` (`play=True`) | the scene, the observations, the action term, the terminations (upstream's own `base_ang_vel_exceed` among them) and the startup randomization |
| `spinkick_safe.onnx` | the policy (154 → 29), already exported; its mjlab metadata gives the joint order and the rest pose |
| the clip inside `spinkick_safe.onnx` | the reference motion, 232 frames at 50 Hz |

## The clip, from the checkpoint

Upstream publishes the clip only to a W&B registry. mjlab's tracking exporter also bakes
it into the ONNX, which is where `motion_tracking_controller` reads it on the robot: a
second input, `time_step`, selects the frame, and besides the action the network returns
that frame's reference joint state and the poses and velocities of the 14 tracked bodies.
The browser feeds `time_step` its own step counter; the action does not read it.

[`upstream.ensure_clip`](upstream.py) reads those tables back into an mjlab motion `.npz`.
`MotionLoader` indexes the robot's whole body list, so the other 16 bodies are rebuilt the
way mjlab's `csv_to_npz` builds them: each frame's root and joint state is written into
mjlab's sim and every body is logged after `forward()`. The rebuilt tracked bodies match
the baked ones to 2.4e-7 (poses) and 4.8e-6 (velocities), and the build refuses a clip
that is off by more than 1e-4.

## What differs from upstream

- **The clip is rebuilt from the checkpoint**, not downloaded from W&B; see above. Every
  value a term reads is the exported one, so the 16 rebuilt bodies change nothing the
  policy sees.
- **mjlab 1.6.0, not 1.1.0.** Upstream locks mjlab 1.1.0 and the playground pins 1.6.0.
  The tracking config, its observation and termination functions and the G1 model agree
  between the two, apart from renamed randomization helpers and collision defaults 1.6.0
  states explicitly.
- **No encoder bias.** mjlab's play config draws a ±0.01 rad bias per joint at startup;
  the browser applies the policy config's `encoder_bias` instead, and the checkpoint's
  deploy contract carries none, as on the robot.
- **One randomized robot, not a population.** The CoM offset on `torso_link` and the foot
  friction (0.3–1.2, shared across the foot geoms) are drawn once from the browser's
  seeded PRNG, where mjlab draws them per env.

## How it behaves

Fed the bundled clip in a live mjlab env (the play config, under `mujoco_warp`), the
checkpoint ran the clip through twice without a termination in each of three rollouts,
with the tracked bodies about 5 cm off the reference on average (11 cm at worst) and the
pelvis turning at up to 6.9 rad/s. That is not a benchmark, and the browser's WASM MuJoCo
is not checked here.
