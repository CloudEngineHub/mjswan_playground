# MuscleMimic full body (`musclemimic`)

<img src="../../../assets/musclemimic.gif" width="480" alt="MuscleMimic full body preview"/>

Source: https://github.com/MyoHub/myosuite/tree/992bd58928e61d3f7ce27a033be02d0be18a2c18 (Apache-2.0) ·
policy and clips from https://github.com/amathislab/musclemimic

> Chengkun Li, Cheryl Wang, Bianca Ziliotto, Merkourios Simos, Jozsef Kovecses,
> Guillaume Durandau, Alexander Mathis.
> [Towards Embodied AI with MuscleMimic: Unlocking full-body musculoskeletal motor learning at scale](https://arxiv.org/abs/2603.25544).
> arXiv:2603.25544, 2026.

A 354-muscle, 89-DoF musculoskeletal body tracking a retargeted walking clip, driven by
the public checkpoint [`amathislab/mm-10m-2`](https://huggingface.co/amathislab/mm-10m-2)
(2.05e9 steps). The scene is myosuite's mjlab task `myoMimicFullbody-v0`, its body built
as `musclemimic_models` 1.0.5, the release the checkpoint was trained on: on 1.0.6,
myosuite's default, which fixes the left knee's coupling, it falls within 2 s. The policy
reads upstream MuscleMimic's own 2418-wide observation, which `upstream.py` reproduces in
torch from myosuite's numpy builder.

## Run

`--extra musclemimic` takes myosuite from GitHub, at the commit the `terra` task pins.

```sh
uv sync --extra musclemimic
hf auth login          # the clip dataset is gated, auto-approved
uv run msp run musclemimic
```

The first build downloads the checkpoint (140 MB) and the clip, exports the actor to ONNX,
checks it against myosuite's forward pass, and stores it with float16 weights (19 MB) in
`.cache/`.

## What the policy reads

`obs[1, 2418] -> actions[1, 354]`. Five blocks are raw sim fields read every step; the
sixth is a 484-row table baked from the clip and indexed by sim time.

| Block | Width | Source |
|---|---:|---|
| joint positions (root z + quat, then the rest) | 87 | `qpos` |
| joint velocities | 88 | `qvel` |
| per muscle: length, velocity, force, ctrl, activation | 1770 | `actuator_*`, `ctrl`, `act` |
| foot touch sensor sums | 4 | `sensordata` |
| mimic sites relative to the pelvis: pos, angles, vel | 192 | `site_xpos`, `site_xmat`, `cvel`, `subtree_com` |
| clip lookahead (5 steps, stride 20) + motion phase | 277 | baked table, indexed by `time` |

Actions are written to `ctrl` as they come, clipped to the actuator's `[-1, 1]`
(`action_mode="direct"`). The 17 mimic sites and the four touch sensors exist in the
model under mjlab's `entity/` prefixes.

Each raw field reaches the graph indexed, so mjswan narrows the slot to the rows the
term actually reads: the browser ships 17 of the model's 2038 sites and 4 of its 322
sensor values per step, not the whole arrays.

| Compared | max \|Δ\| |
|---|---:|
| exported actor vs myosuite's numpy forward pass | 6.7e-06 |
| float16 vs float32 weights: actions / the root over a 483-step rollout | 1.3e-03 / 2.3 cm |
| torch observation vs myosuite's numpy builder, 44 states | 1.0e-06 |
| traced graph vs the torch observation, 24 steps | 4.5e-07 |

## What differs from upstream

- **Every episode starts on clip frame 0.** Training draws a random start frame and the
  browser resets `mjData.time` to 0, so `terms.py` pins the clip phase to sim time and
  replaces the random reset with a frame-0 one. It also rotates the clip's body-frame root
  angular velocity into the world frame mjlab's `write_root_state_to_sim` expects, which
  myosuite's own reset does not.
- **The termination is upstream's**, mean relative-site deviation or root deviation over
  1.0 m, not myosuite's `mimic_deviation`, which compares the root against the *mean of the
  target sites* with a 0.3 m limit and would end the episode at 1.8 s. The episode is the
  484-frame clip: it times out on the last frame, at 4.83 s, before the reference wraps
  to its start.
- **The actor's weights are float16**, cast back to float32 as the graph loads: 19 MB,
  not 38 MB, for hosts such as Cloudflare Pages that cap a file at 25 MiB. A float16 graph
  would run at under half the policy's 100 Hz in the browser. Int8 would give 9.6 MB but
  moves the actions by 0.22 and worsens tracking.
- **Browser behaviour is not asserted.** Parity proves each graph matches mjlab under
  `mujoco_warp`; the browser runs MuJoCo's WASM build.
