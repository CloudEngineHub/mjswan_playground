# Wuji Hand (`wuji`)

<img src="../../../assets/wuji.gif" width="480" alt="Wuji Hand preview"/>

Source: https://github.com/wuji-technology/wuji-mjlab (Apache-2.0) · policies and pen-spin
clips from its
[v2026.9.27 release bundle](https://github.com/wuji-technology/wuji-mjlab/releases/tag/v2026.9.27)
(Apache-2.0) · the Wuji Hand 1 and Hand 2 models from
[wuji-description](https://github.com/wuji-technology/wuji-description) (MIT)

Wuji Hand 1 and Hand 2, each a 20-joint right hand on its mounting jig, turning a 54 mm
cube in its palm to orientations drawn uniformly at random. Three arrows out of the cube
show the goal: each points where the face of its color has to face (purple +X, red +Y,
green +Z). Once the cube has held within 0.2 rad of the goal for about a quarter second
(13 control steps on Hand 2, 5 on Hand 1), the goal stays 1 s more and then switches.
Episodes last 60 s.

Hand 2 also spins a 250 mm pen along clips recorded on the real hand, one policy per
upstream recipe. Each episode plays one of the recipe's clips, drawn at random: the hand
holds the clip's first frame for 1 s, then follows it to its end, and the next episode
draws again.

Upstream ships four simulations; this task ports three:

| Upstream task | Here |
|---|---|
| `WujiHand2_Reorient_50Hz`, Hand 2 turning the cube at 50 Hz | a scene, with the bundle's Hand 2 checkpoint |
| `WujiHand_Reorient`, Hand 1 turning the cube at 20 Hz | a scene, with the bundle's Hand 1 checkpoint |
| `WujiHand2_Reorient`, Hand 2 at 20 Hz | left out: no checkpoint of its own, the bundle's Hand 2 checkpoint is the 50 Hz recipe's |
| `WujiHand2_PenSpin`, Hand 2 spinning a pen | a scene, with the bundle's five pen-spin checkpoints, one policy per recipe |

## Run

```sh
uv run msp run wuji
```

The first build clones the repository into `.cache/wuji-mjlab/` at a pinned commit and
downloads the 246 MB release bundle once, checks its sha256 and keeps the checkpoints,
their `config.json` and the pen-spin clips in `.cache/wuji/v2026.9.27/`; set
`MJSWAN_WUJI_ROOT` to point at a checkout you already have. Installed, upstream registers
its tasks through mjlab's entry point; [`upstream.py`](upstream.py) imports
`wuji_mjlab.tasks` from the checkout's `src/` instead.

| From `wuji-mjlab` | Used as |
|---|---|
| `WujiHand2_Reorient_50Hz`, `WujiHand_Reorient` (`play=True`) | the scenes, the observations, the command, the terminations and the reset events, changed as below |
| bundle `reorient/checkpoints/hand2/policy.onnx` | Hand 2's policy (207 → 20): run `RewardContactFix_Full_s42`'s `model_4999` |
| bundle `reorient/checkpoints/hand1/policy.onnx` | Hand 1's policy (207 → 20) |
| bundle `reorient/checkpoints/<hand>/config.json` | each hand's joint order and default pose; its action scale, EMA factor and control period are checked against the config |
| `WujiHand2_PenSpin` (`play=True`) | the pen-spin scene, its observations and terminations, changed as in [Pen spin](#pen-spin) |
| bundle `pen_spin/checkpoints/<recipe>/policy.onnx` | the five pen-spin policies (155 → 20), each wrapped as in [Pen spin](#pen-spin) |
| bundle `pen_spin/checkpoints/<recipe>/config.json` | each one's action scale, residual clip, EMA factor and control period, checked against the config |
| bundle `pen_spin/data/rawdata/clips/`, `data/recipe/` | the 40 clips and which recipe plays which, read through upstream's own resolver, which checks each clip's sha256 |
| `LICENSE`, `NOTICE` | the project's license and notice |

| Scene | Policy | Checkpoint | Input → action | Rate |
|---|---|---|---|---|
| `WujiHand2_Reorient_50Hz` | `hand2` | bundle `reorient/checkpoints/hand2` | 207 → 20 | 50 Hz, 100 Hz physics |
| `WujiHand_Reorient` | `hand1` | bundle `reorient/checkpoints/hand1` | 207 → 20 | 20 Hz, 100 Hz physics |
| `WujiHand2_PenSpin` | `Single axis`, `Continuous single axis`, `Continuous multi-axis`, `Stable grasp rotation`, `Multi-axis, two rotations` | bundle `pen_spin/checkpoints/<recipe>`, 10, 4, 9, 10 and 7 clips | 155 → 20 | 50 Hz, 200 Hz physics |

## Cube reorientation

### What the policies read

Five terms, three frames each, oldest first:

| Term | Width | Source |
|---|---|---|
| `noisy_joint_angles` | 20 × 3 | joint positions, normalized to the soft limits (play adds no noise) |
| `qpos_error` | 20 × 3 | the same, less the normalized smoothed target |
| `cube_pos_in_tag` | 3 × 3 | the cube's position in the wrist tag's frame |
| `cube_ori_error` | 6 × 3 | the last two rows of the rotation from the goal to the cube, in the tag's frame |
| `action_history` | 20 × 3 | the actions one to three steps back |

### What differs from upstream

- **The EMA action is the actuators' filter.** Upstream's
  `JointPositionOffsetEMAActionCfg`, a term of its own that mjswan has no counterpart for,
  clamps the action to [-1, 1], scales it by 0.5 about the default pose, clamps that
  target to the soft limits and smooths it with an EMA: 0.3 per 20 ms step for Hand 2,
  0.5 per 50 ms step for Hand 1. Here mjlab's `JointPositionActionCfg` applies the same
  scale and one clip per joint that equals both clamps, and each `<position>` actuator
  gets a `filterexact` filter with time constant `-dt / ln(1 - alpha)`, 56 ms for Hand 2
  and 72 ms for Hand 1, settled at the default pose in the keyframe. The filter equals the
  EMA at the end of every control step; within one its target moves smoothly rather than
  jumping, about half a step later. Parity cannot check this side: rolled out in mjlab
  over 4 envs for 60 s, Hand 2's checkpoint reached 28.0 goals a minute this way against
  29.0 with upstream's term, Hand 1's 24.0 against 25.5, and neither dropped the cube.
- **`qpos_error` reads the filter state**, `act`, where upstream reads its action term's
  smoothed target ([`terms.py`](terms.py)).
- **`action_history` is mjlab's `last_action` one to three steps back.** Upstream's
  `previous_raw_action` reads the action manager's `prev_action`, the action before the
  last.
- **No warmup.** Upstream's term holds the default pose for the first 0.4 s of an
  episode; the filter starts there, and the policy acts from the first step.
- **The goal command is restated** in [`terms.py`](terms.py). Upstream draws its goals
  with `torch.rand`, which the tracer cannot replay; the rewrite draws the same uniform
  rotations with mjlab's `sample_uniform`, draws the next goal every step and takes it
  only on a switch, counts in float, and reads the wrist tag's world orientation once from
  the keyframe, since a traced command reads one entity and the mount does not move here.
  Its training counters are dropped, and so is upstream's status panel, a line of text
  with the orientation error.
- **The goal is drawn as arrows.** Upstream's viewer draws a ghost of the cube above it;
  mjswan's command drawings are spheres and arrows.
- **The mount does not tilt on reset.** `reset_robot_pose` turns the mocap mount about
  world axes by -0.4 to 0.1 rad, and mjswan's event tracer captures no mocap write, so the
  hand stays at its nominal pose, inside that range.
- **The other two reset events are restated** with `sample_uniform` where upstream calls
  `torch.rand` and `torch.rand_like`, over the same ranges: `reset_robot_joints` and
  `reset_object_pose`.
- **`nan_guard` is mjlab's `nan_detection`** with `|` in place of the in-place `|=` the
  ONNX exporter refuses.
- **The wrist tag's site is a term param**, which mjlab resolves once, where upstream's
  `cube_pos_in_tag` looks it up on every call.
- **Training-only terms are dropped:** the critic group, the rewards, the metrics, the
  contact sensors only they read, `reset_joint_acc_cache`, and `cage_drop`, which reads a
  reward's counter.

## Pen spin

### What the policies read

| Term | Width | Source |
|---|---|---|
| `motion` | 89 | the clip's frame: its 20 joint positions, its pen's two axis points 10 cm either side of the centre, in the palm's frame and squashed by `0.15 tanh(x / 0.15)` (6), and its 21 links' positions in the palm's frame (63) |
| `joint_pos` | 20 | joint positions minus the default pose |
| `joint_vel` | 20 | joint velocities × 0.25 |
| `object_pose` | 6 | the pen's own two axis points, the same way |
| `last_action` | 20 | the previous action |

Upstream's action, `SingleHandResidualEMAAction`, turns the 20 outputs into joint
targets: the clip's joint positions plus a residual, each output × 0.5, clamped to
±0.6 rad and smoothed by an EMA of factor 0.5.

### What differs from upstream

- **A wrapper carries the residual's EMA.** mjswan has no action term that smooths a
  residual and adds it to a reference its command plays, and an actuator filter would
  smooth the reference too. [`pen_spin.py`](pen_spin.py)'s `wrap_policy` puts the term
  around the checkpoint instead: its ONNX keeps the smoothed residual and the raw
  action in the recurrent carry mjswan feeds back and zeroes on a reset, as the term
  zeroes its own, swaps the observation's last action for the carried raw one (the
  browser's is the target the wrapper outputs) and outputs the target itself, the
  observation's first 20 values (the clip's joint positions) plus the smoothed
  residual, to a plain joint position action. Fed the same observations as upstream's
  term over 4,487 control steps of upstream's play on the first recipe, nine episodes,
  its targets match exactly.
- **The command is restated** in [`pen_spin.py`](pen_spin.py): the same clips through
  upstream's own resolver and reader, its tables graph constants and the clip and the
  frame float state. Upstream's reset event picks the clip, through its command's
  `prepare_reset`, and the command then skips its own pick; a traced event cannot reach
  a command. Here the reset event draws the clip with `sample_uniform` floored, where
  upstream's command draws `torch.randint`, writes its first frame, and the command,
  reset after the events as in mjlab, finds the clip again by the joint positions it
  was handed: the first frames of a recipe's clips differ by at least 0.17 rad.
- **The reset leaves the mount alone.** Upstream writes the wrist's mocap pose on every
  reset; it is the keyframe's here, which the fixed mount never leaves.
- **`object_pose` takes its axis points as params**, where upstream reads them off its
  command; they are constants.
- **`unstable` uses `terms.nan_detection`** for mjlab's `NanGuard.detect_nans`, whose
  in-place `|=` the exporter refuses, and reads the pen's position in world: the task
  spaces its envs 0 m apart, so no env origin needs taking off.
- **The pen keeps its asset's size.** Play pins the pen to the hardware's, 250 mm by
  14.9 mm, with a startup event; the asset already has that size, and the event's
  rewrite of the inertia in its principal frame gives the same tensor to 2e-11.
- **Training-only terms are dropped:** the critic group, the rewards, the metrics and
  the contact sensor only a reward reads.

### How it behaves

Played once from its first frame to its end, every clip of every recipe keeps the pen
within upstream's two play limits, 5 cm off the reference and 60° of shaft, which play
leaves switched off:

| Run | Clips kept | Mean shaft error per recipe |
|---|---|---|
| mjlab, upstream's env and checkpoints | 40 of 40 | 2.9 to 4.8° |
| mjlab, this port's env and wrapped checkpoints | 40 of 40 | upstream's, to within 0.03° |
| MuJoCo's C build, which the browser runs, with the wrapped checkpoints | 40 of 40 | 2.9 to 4.9° |

In the browser, over three minutes of the first recipe, the pen stayed in hand through
all 20 episodes. That takes mjswan 0.11.8
([ttktjmt/mjswan#166](https://github.com/ttktjmt/mjswan/pull/166)): before it, the
browser turned every mesh collider 90° about its own x, the fingertips' among them, and
the pen fell in 9 of 21 episodes.
