# ORCS (`orcs`)

<img src="../../../assets/orcs.gif" width="480" alt="ORCS preview"/>

Source: https://github.com/lok-i/orcs (BSD-3-Clause), with its dependencies
https://github.com/lok-i/mocke (BSD-3-Clause; Apache-2.0 for the SONIC code it adapts) and
lok-i's fork of rsl_rl, https://github.com/lok-i/rsl_rl (BSD-3-Clause) · checkpoints from
https://huggingface.co/lkrajan/orcs (NVIDIA Open Model License: they build on SONIC's
weights) · clips and terrain from OmniRetarget (MIT) and NVIDIA's GRAIL (**CC BY-NC 4.0**,
see [License](#license)) · the G1 is mjlab's (Unitree, BSD-3-Clause)

> Lokesh Krishna, Sarvesh Venkatesan, An Zhang, Quan Nguyen.
> [ViBe: Visual Behavior Adaptation for Perceptive Humanoid Whole-Body Control](https://arxiv.org/abs/2609.09918).
> 2026. arXiv:2609.09918.

ORCS trains the privileged teachers behind ViBe: a Unitree G1 on NVIDIA's frozen SONIC
whole-body controller, with a LoRA adapter on its decoder that reads what the task needs.
Three of the four policies upstream releases run here, one scene each:

- **Orcs-Dodge-AdaptSonic** holds a stand while a ball is thrown at it every 1 to 4 s,
  and ducks or steps aside. Its adapter reads the ball's position and velocity.
- **Orcs-PerLoco-OmRe-AdaptSonic** walks up onto a box and over it, along an
  OmniRetarget clip. Its adapter reads a 187-ray height scan.
- **Orcs-PerLoco-Grail-AdaptSonic** walks over a row of curbs, along a GRAIL clip, with
  the same scan.

Left out:

- **Orcs-Uolm-AdaptSonic**, loco-manipulation over an object roster: its clips come from
  the `l-k-r/retargeted_motions` dataset, whose card states no license, and its objects
  from `drcl-deploy/assets`, whose meshes record none.
- The three `-TaRa` tasks, PerLoco's two and Uolm's, and the four `-Smpl` tasks, PerLoco
  GRAIL's and Uolm's three: no released checkpoint, and the `-Smpl` ones need licensed
  SMPL-X body models.
- `Mocke-Tracking-Textop-G1` and `Mocke-Tracking-Sonic-G1`, which the mocke dependency
  registers, not ORCS.

## Run

```sh
uv sync --extra orcs   # onnxruntime and usd-core, for the conversion and GRAIL's curbs
uv run msp run orcs
```

The first build clones the three repositories into `.cache/orcs/` at pinned commits (mocke
carries its pretrained weights in git, 210 MB, which the port does not read); set
`MJSWAN_ORCS_ROOT`, `MJSWAN_MOCKE_ROOT` and `MJSWAN_ORCS_RSL_RL_ROOT` to use checkouts you
have. It then makes the data upstream's setup scripts make, under `.cache/orcs/data/`
(upstream's own `ORCS_DATA_ROOT` moves it): Dodge's nominal stand through
`orcs-make-nominal`, and each PerLoco roster's tiles and clips through
`orcs-stage-terrain`, from the dataset files it reads, fetched from the Hub at pinned
revisions. Last, it converts each checkpoint into `.cache/orcs/onnx/`, in a process with
the rsl_rl fork first on `PYTHONPATH`, since mjlab's own rsl_rl lacks its model.

| From | Used as |
|---|---|
| `orcs`: `Orcs-Dodge-AdaptSonic` (`play=True`), and the PerLoco play configs from `omni_env_cfg` and `grail_env_cfg` | each scene: the robot, its observations, action, terminations, sensors and terrain |
| `orcs`: `orcs-make-nominal`, `orcs-stage-terrain`, `rosters/omni.toml`, `rosters/grail.toml` | Dodge's stand, and the PerLoco tiles and clips |
| `orcs`: `src/orcs/release.json` | each checkpoint's path, size and hash |
| `orcs`: `LICENSE`, `NOTICE` | the project's license and notice |
| `mocke`: `sonic.profile`, `sonic_g1_tokenizer` | the frozen base's streams, its action scale and actuator grouping, and its tokenizer |
| `rsl_rl`: `SonicWithAdapterModel` and its ONNX wrapper | each checkpoint's conversion, LoRA folded into the decoder |
| `lkrajan/orcs` at v0.1.0: `<task>/checkpoint.pt`, `LICENSES/NVIDIA-OPEN-MODEL-LICENSE.txt` | the three policies, and their license |
| `omniretarget/OmniRetarget_Dataset`: `robot-terrain.zip`, `models/terrain/climb_{16,17,18}`, `models/g1/g1_29dof.urdf` | the OmniRetarget roster's nine tiles and clips |
| `nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL`: `data/curb/{robot,object_usd}` for the roster's eight curbs | the GRAIL roster's tiles and 63 clips |

| Scene | Policy | Checkpoint | Input → action | Rate |
|---|---|---|---|---|
| Orcs-Dodge-AdaptSonic | Dodge | `Orcs-Dodge-AdaptSonic/checkpoint.pt` (model_7500) | 640 + 930 + 21 → 29 | 50 Hz |
| Orcs-PerLoco-OmRe-AdaptSonic | Climb | `Orcs-PerLoco-OmRe-AdaptSonic/checkpoint.pt` (model_10500) | 640 + 930 + 202 → 29 | 50 Hz |
| Orcs-PerLoco-Grail-AdaptSonic | Curb | `Orcs-PerLoco-Grail-AdaptSonic/checkpoint.pt` (model_14999) | 640 + 930 + 202 → 29 | 50 Hz |

No encoder bias: play strips upstream's `rand_encoder_bias`, so mjlab runs with none.

## What the policies read

Three inputs, in the actor's own order:

| Input | Width | Source |
|---|---|---|
| `tokenizer` | 10 × 64 | SONIC's G1 encoder: the reference at 10 frames 0.1 s apart, each its joint positions and velocities (29 each) and its pelvis orientation relative to the robot's (6) |
| `actor` | 10 × 93 | SONIC's decoder, 10 frames of `base_ang_vel`, `joint_pos`, `joint_vel`, `actions` and `gravity_dir` (3, 29, 29, 29, 3) |
| `augmentation` | 21 or 202 | the adapter's: Dodge's ball position and velocity in the robot's yaw frame (6); PerLoco's height scan, 17 × 11 rays 0.1 m apart under the pelvis, yaw-aligned, over 5 m (187); then the root's position and body-frame twist (9) and the reference's root twist (6, zero for Dodge's stand) |

## What differs from upstream

- **mjlab 1.6.0.** Upstream pins mjlab 1.4.0 and mujoco 3.8.1, and imports unchanged on
  1.6.0: between the two, the G1 constants only spell out `FULL_COLLISION`'s defaults,
  and the G1's built-in position actuators are the same. The port imports the task
  packages rather than `orcs.registration`, whose `_mjlab_compat` patches mjlab's train
  and play scripts and caps mujoco_warp's collision workspace, neither of which the build
  runs.
- **The hands look like mjlab's.** Upstream's flat-hand G1 swaps each hand's capsule
  collider for a box plate, and its visual mesh for a flat hand from
  `drcl-deploy/assets`, whose G1 meshes record no license. The plates are here; the
  visual stays mjlab's hand, and visual geoms collide with nothing.
- **The reference clip is a traced command.** Upstream's `MultiClipMotionCommand`, and
  PerLoco's `TerrainMotionCommand` over it, keep an integer frame, which the browser feeds
  a graph as a float, so [`terms.py`](terms.py)'s `ClipCommand` restates them over the one
  clip each scene plays: frame zero with upstream's reset jitter, the frame advanced and
  held at the clip's end, and the anchor rows the twist observations read. SONIC's
  tokenizer reads it unchanged, its clip tables baked into the graph. Against upstream's
  command in mjlab, the reset's writes, the tokenizer, the twist observations and
  `exceeded_motion` agree exactly, over 20 reset draws and full rollouts. No reference
  ghost is drawn.
- **Dodge's throw is a command.** Upstream's `ThrowBall` is an interval event with a
  countdown and a flight timer that pins the ball overhead between throws; mjswan traces
  an event as a stateless function and lets a command write the sim only when it
  resamples. So `ThrowBallCommand` is resampled every control step, and its resample is
  the event's tick: play's own throw (from the robot's front cone, no camera, no
  ball-free anchors), its pin and launch merged into one write, and `ThrowBall.reset` on
  the first tick of each episode, which it reads off the clip command's frame. With every
  draw pinned, its writes match `ThrowBall`'s over 400 ticks. It runs after the step's
  forward, as mjlab 1.4.0 ran interval events; 1.6.0 runs them before.
- **One tile per PerLoco scene.** The browser runs one env, where upstream's play spreads
  envs over its roster's tiles: nine OmniRetarget climbs, and eight GRAIL curbs with 5 to
  10 clips each. Each scene stages upstream's whole roster and plays the roster's first
  tile and that tile's first clip, through a roster file of upstream's own format:
  `climb_16` at z-scale 1.0, its box drawn at upstream's 0.75, and `curb_000` take 000.
- **The env origin is the world's.** `robot_root_pos_env` subtracts
  `env.scene.env_origins`, which mjswan serves no observation; for the browser's one env,
  at the world's origin on a plane or a one-tile grid, the rewrite reads the root's
  position alone.
- **Training-only terms are dropped:** `PolicyUpdateCounter`, an event that counts PPO
  updates for upstream's anneals and curricula, the critic group and the rewards.
- **The checkpoints are converted here.** The release ships `.pt` only. Each is rebuilt
  with the fork's `SonicWithAdapterModel` and exported through its own ONNX wrapper,
  then checked against the fork's forward pass. The decoder's weights are stored as
  float16 and cast back in the graph, so each file fits mjswan Cloud's 50 MB; along a
  60 s Dodge rollout in mjlab, they move the actions by at most 9.6e-4. The encoder stays
  float32: its output is rounded into tokens, which a float16 weight could flip.
- **mocke's sandbox tasks are skipped.** Importing mocke registers two play-only tasks
  on mjlab's demo clip (LAFAN1, CC BY-NC-ND 4.0), downloading it; the port fails that
  fetch, so they skip as mocke does offline.
- **GRAIL's clips are read with numpy arrays allowed only.** They are joblib pickles,
  which upstream's staging loads whole; here any class but joblib's array wrapper, numpy's
  dtype and ndarray is refused, so a download cannot run code.

## License

The curb scene plays GRAIL's `curb_000`, its geometry and its clip, under the Creative
Commons Attribution-NonCommercial 4.0 license ([`LICENSE.grail`](LICENSE.grail)), and the
checkpoints are under the NVIDIA Open Model License, whose text and attribution each scene
carries. The demo is published only by its author.
