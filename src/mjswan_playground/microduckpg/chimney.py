"""Chimney climb: microduck side-steps into a 12.5 cm slot, braces between its walls and
climbs 3 m, works its way out over a platform under the overhanging walls, and stands up
on it."""

from __future__ import annotations

import hashlib
import io
import json
import math
from functools import partial
from pathlib import Path
from typing import Any

import mjswan
import mujoco
import onnx
import torch
from huggingface_hub import hf_hub_download
from mjlab.envs.mdp import observations as obs_fns
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjswan.mdp import MdpConfig
from mjswan.mjlab import apply_mjlab_sim_options, build_single_entity_trace_env
from onnx import numpy_helper
from torch import nn

from mjswan_playground.microduck.main import (
    CONTROL_DT,
    ENTITY,
    POLICY_DIR,
    TRACKED_BODY,
    VELOCITY_ENV,
    _resolve_deploy_root,
)

from . import _common

POLICY_REPO_ID = "HannesVonEssen/microduck-chimney-climb"
POLICY_REVISION = "1dea21224039f74fb2557456b9bcdec870d96fd7"
#: Enter, climb and exit, by Hub directory, in the order the chain runs them. The release
#: pins the fourth, Pollen's official get-up, by hash: the file the microduck task's
#: deploy checkout ships.
STAGES = ("enter/", "", "exit/")
GETUP_ONNX = "alpha_stand.onnx"

ROBOT_PY = "src/mjlab_microduck/robot/long_jump_robot.py"
CORRIDOR_PY = "src/mjlab_microduck/robot/corridor_stage.py"
EXIT_STAGE_PY = "src/mjlab_microduck/robot/exit_stage.py"
SYMMETRY_PY = "src/mjlab_microduck/tasks/symmetry.py"

#: ``reproduce_chimney_chain.py``'s stage: a 12.5 cm slot up to a platform 3 m high and as
#: thick as the walls, which end 40 cm above it, all in one colour.
WIDTH = 0.125
PLATFORM_TOP = 3.0
WALL_ABOVE = 0.40
STAGE_RGBA = "0.60 0.55 0.50 1"
#: 1 m wide and 60 cm longer at the far end than the release's 50 cm by 85 cm, where the
#: get-up wanders off the edge; the exit still aims where it did.
PLATFORM_WIDTH = 1.0
PLATFORM_EXTRA = 0.6
PLATFORM_GEOM = "exit_platform_collision"
#: ``place_outside``: the enter policy's start, 45 cm out from the slot and standing.
START = (0.0, -0.45, 0.115)
#: The middle of ``reset_exit_state``'s wedged spawn, the Exit entry's start: back on the
#: -x wall, 5 cm over standing height on the platform, pitched 0.2 rad into the wall.
EXIT_START = (-0.5 * WIDTH + 0.055, 0.04, PLATFORM_TOP + 0.115 + 0.05)
EXIT_PITCH = -0.2

#: ``approach_mdp.in_handover_pose``, enter to climb: inside the slot's throat, at
#: standing height, level, stopped, in the climb's stance and facing across the slot,
#: under the chain's own settle of 0.35 rad/s mean joint speed.
STAND_Z, STAND_BAND, THROAT_Y, LEVEL_DEG, STOPPED = 0.115, 0.025, 0.06, 35.0, 0.06
SETTLED, STANCE_TOL, ACROSS_DEG = 0.35, 0.20, 20.0
STANCE = {2: -0.4579, 3: -0.0049, 4: 0.4529, 11: 0.4579, 12: 0.0049, 13: -0.4529}
#: The chain's climb to exit: once the feet are 5 cm over the platform, the first moment
#: under 22 degrees of tilt and 0.10 m/s of climb, from 0.6 s on 26 degrees and
#: 0.30 m/s, and at 1.2 s regardless.
FOOT_OPEN, TILT_WAIT, TILT_HARD = 0.05, 0.6, 1.2
SETTLE_EARLY, SETTLE_LATE = (22.0, 0.10), (26.0, 0.30)
#: The chain's exit to get-up: two steps touching the platform past the walls' end, with
#: the jaw under 13 cm above the feet or the trunk past cos(tilt) 0.85.
LANDED_STEPS, LOW_CARRIAGE, TIPPED = 2, 0.13, 0.85

#: The two ankles, then the jaw: the feet's height and the head's carriage over them.
HEIGHT_BODIES = ("ankle_left", "ankle_right", "jaw_soft")
ON_PLATFORM = "on_platform"
#: The ``chain`` group: root position, orientation and COM velocity, joint speeds, the
#: ankle and jaw heights, and the platform contact.
CHAIN_SIZES = (3, 4, 3, 14, 3, 1)
#: mjswan's recurrent carry, ``adapt_hx``.
CARRY = 128


def _root(env: Any, *, field: str, **_) -> torch.Tensor:
    """A root ``EntityData`` field the browser serves as it is."""
    return getattr(env.scene[ENTITY].data, field)


def _heights(env: Any, *, asset_cfg: SceneEntityCfg, **_) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.body_link_pos_w[:, asset_cfg.body_ids, 2]


def _sensor(env: Any, *, row: int, **_) -> torch.Tensor:
    return env.sim.data.sensordata[:, row : row + 1]


class _Net(nn.Module):
    """A published actor rebuilt node for node from its graph, weights as published: the
    normalizer, the ELU MLP and, where baked in, the joint-travel clamp."""

    OPS = {
        "Sub": torch.sub,
        "Div": torch.div,
        "Gemm": nn.functional.linear,
        "Elu": nn.functional.elu,
        "Max": torch.maximum,
        "Min": torch.minimum,
    }

    def __init__(self, model: onnx.ModelProto):
        super().__init__()
        weights = {
            t.name: torch.from_numpy(numpy_helper.to_array(t).copy())
            for t in model.graph.initializer
        }
        self.nodes = []
        for k, node in enumerate(model.graph.node):
            for j, name in enumerate(node.input[1:]):
                self.register_buffer(f"w{k}_{j}", weights[name])
            self.nodes.append((node.op_type, len(node.input) - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for k, (op, n) in enumerate(self.nodes):
            x = self.OPS[op](x, *(getattr(self, f"w{k}_{j}") for j in range(n)))
        return x


class _Chain(nn.Module):
    """``reproduce_chimney_chain.py``'s controller as one recurrent graph, since picking a
    policy in the browser restarts the sim: enter, climb, exit reflected to -y, then the
    get-up, each handed over to on the chain's own tests.

    It reads ``actor`` (the 34 proprioceptive values), ``chain`` (root position,
    orientation and velocity, joint speeds, the ankle and jaw heights, the platform
    contact) and the carry: the phase, the step count and the step the feet first
    cleared the platform, the landed steps, the acting policy's previous output, and last
    step's target, which it sends on one step late as BAM's command delay. Every policy
    from ``start`` on runs each step and the phase picks one.
    """

    def __init__(
        self,
        nets: list[onnx.ModelProto],
        start: int,
        start_pose: list[float],
        offsets: list[list[float]],
        joint_range: list[list[float]],
        reflect: tuple[list[int], list[float], list[float]],
        goal_y: float,
        recovery_y: float,
    ):
        super().__init__()
        self.start, self.goal_y, self.recovery_y = start, goal_y, recovery_y
        self.nets = nn.ModuleList(_Net(net) for net in nets[start:])
        perm, sign, imu_sign = reflect
        rows = {
            "home": offsets[0],
            "brace": offsets[1],
            "getup_home": offsets[3],
            "start_pose": start_pose,
            "lo": [lo for lo, _ in joint_range],
            "hi": [hi for _, hi in joint_range],
            "sign": sign,
            "imu_sign": imu_sign,
            "stance": list(STANCE.values()),
        }
        for name, row in rows.items():
            self.register_buffer(name, torch.tensor([row], dtype=torch.float32))
        self.register_buffer("perm", torch.tensor(perm))
        self.register_buffer("stance_ids", torch.tensor(list(STANCE)))

    def forward(self, actor, chain, carry):
        ang, grav, joint_rel, joint_vel = actor.split((3, 3, 14, 14), dim=1)
        pos, quat, lin_vel, speeds, heights, contact = chain.split(CHAIN_SIZES, dim=1)
        slots = carry[:, :33].split((1, 1, 1, 1, 1, 14, 14), dim=1)
        phase, step, reached, reach_step, landed, prev, held = slots
        phase = phase + self.start
        joint_pos = joint_rel + self.home
        x, y, z = pos.split(1, dim=1)
        qw, qx, qy, qz = quat.split(1, dim=1)
        upright = 1.0 - 2.0 * (qx * qx + qy * qy)
        yaw_sin, yaw_cos = 2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)
        norm = torch.sqrt(yaw_sin * yaw_sin + yaw_cos * yaw_cos)
        yaw_sin, yaw_cos = yaw_sin / norm, yaw_cos / norm
        feet = torch.minimum(heights[:, 0:1], heights[:, 1:2])

        stance = (joint_pos[:, self.stance_ids] - self.stance).abs() < STANCE_TOL
        handover = (
            (x.abs() < 0.5 * WIDTH)
            & (y.abs() < THROAT_Y)
            & ((z - STAND_Z).abs() < STAND_BAND)
            & (upright > math.cos(math.radians(LEVEL_DEG)))
            & (lin_vel[:, 0:1] ** 2 + lin_vel[:, 1:2] ** 2 < STOPPED**2)
            & (speeds.abs().mean(dim=1, keepdim=True) < SETTLED)
            & stance.all(dim=1, keepdim=True)
            & (yaw_cos > math.cos(math.radians(ACROSS_DEG)))
        )
        to_climb = (phase == 0) & handover

        climbing = (phase == 1) & (feet > PLATFORM_TOP + FOOT_OPEN)
        reach_step = torch.where(climbing & (reached < 0.5), step, reach_step)
        reached = torch.where(climbing, torch.ones_like(reached), reached)
        # Whole steps, not a summed float clock, which drifts across the 1.2 s line.
        waited = step - reach_step
        early = waited < round(TILT_WAIT / CONTROL_DT)
        (tilt_early, vz_early), (tilt_late, vz_late) = SETTLE_EARLY, SETTLE_LATE
        level = upright > torch.where(
            early,
            torch.full_like(waited, math.cos(math.radians(tilt_early))),
            torch.full_like(waited, math.cos(math.radians(tilt_late))),
        )
        slow = lin_vel[:, 2:].abs() < torch.where(
            early, torch.full_like(waited, vz_early), torch.full_like(waited, vz_late)
        )
        to_exit = climbing & ((level & slow) | (waited > round(TILT_HARD / CONTROL_DT)))

        on_platform = (phase == 2) & (contact > 0.5) & (y < -self.recovery_y)
        landed = torch.where(on_platform, landed + 1.0, torch.zeros_like(landed))
        down = (heights[:, 2:] - feet < LOW_CARRIAGE) | (upright < TIPPED)
        to_getup = (phase == 2) & (landed >= LANDED_STEPS) & down

        phase = phase + to_climb.float() + to_exit.float() + to_getup.float()
        # A handover is no reset: the receiver's action history starts at zero, the
        # get-up's at its pose, as the chain sets them.
        prev = torch.where(to_climb | to_exit, torch.zeros_like(prev), prev)
        prev = torch.where(to_getup, joint_pos - self.getup_home, prev)

        def reflected(v):
            return v[:, self.perm] * self.sign

        width = torch.full_like(x, (WIDTH - 0.13) / 0.02)
        zero = torch.zeros_like(x)
        dx, dy = -x, -y
        bodies = [
            [yaw_cos * dx + yaw_sin * dy, yaw_cos * dy - yaw_sin * dx, yaw_sin, yaw_cos]
            + [width, zero],
            [width] + [zero] * 5,
            [self.goal_y + y, PLATFORM_TOP + STAND_Z - z, -x, zero, width, zero],
            [zero] * 6,
        ]
        imu = torch.cat([ang, grav], dim=1)
        observed = [
            (imu, joint_pos - self.home, joint_vel),
            (imu, joint_pos - self.brace, joint_vel),
            (
                imu * self.imu_sign,
                reflected(joint_pos) - self.brace,
                reflected(joint_vel),
            ),
            (imu, joint_pos - self.getup_home, joint_vel),
        ]
        outputs, targets = [], []
        for net, k in zip(self.nets, range(self.start, 4)):
            obs = torch.cat([*observed[k], prev, zero.repeat(1, 7), *bodies[k]], dim=1)
            outputs.append(net(obs))
            offset = (self.home, self.brace, self.brace, self.getup_home)[k]
            targets.append(offset + outputs[-1])
        if self.start <= 2:
            targets[2 - self.start] = reflected(targets[2 - self.start])

        def pick(options):
            out = options[-1]
            for k in range(len(options) - 2, -1, -1):
                out = torch.where(phase == self.start + k, options[k], out)
            return out

        output = pick(outputs)
        target = torch.minimum(torch.maximum(pick(targets), self.lo), self.hi)
        held = torch.where(step > 0.5, held, self.start_pose)
        carry = torch.cat(
            [phase - self.start, step + 1.0, reached, reach_step, landed]
            + [output, target, carry[:, 33:]],
            dim=1,
        )
        return held - self.home, carry


def _export(chain: _Chain) -> onnx.ModelProto:
    buffer = io.BytesIO()
    torch.onnx.export(
        chain.eval(),
        (torch.zeros(1, 34), torch.zeros(1, sum(CHAIN_SIZES)), torch.zeros(1, CARRY)),
        buffer,
        input_names=["actor", "chain", "adapt_hx"],
        output_names=["action", "next_adapt_hx"],
        opset_version=18,
        dynamo=False,
    )
    return onnx.load_from_string(buffer.getvalue())


def _stage(root: Path) -> tuple[list[dict], float, float]:
    """``exit_stage``'s stepped walls and platform as ``reproduce_chimney_chain.py`` sets
    them up, reflected to -y, as world geoms; and the exit's goal and the walls' end."""
    cs = _common.load_upstream(root, CORRIDOR_PY)
    ex = _common.load_upstream(
        root, EXIT_STAGE_PY, stubs={"mjlab_microduck.robot": {"corridor_stage": cs}}
    )
    ex.PLATFORM_TOP_M, ex.WALL_ABOVE_M = PLATFORM_TOP, WALL_ABOVE
    ex.PLATFORM_THICK_M, ex._RGBA = cs.WALL_THICKNESS_M, STAGE_RGBA

    def reflected(body: mujoco.MjSpec, at: tuple[float, float, float]) -> list[dict]:
        return [
            dict(
                name=geom.name,
                type=geom.type,
                size=geom.size.tolist(),
                pos=(at[0] + geom.pos[0], -(at[1] + geom.pos[1]), at[2] + geom.pos[2]),
                rgba=geom.rgba.tolist(),
                friction=geom.friction.tolist(),
                condim=geom.condim,
                priority=geom.priority,
                solref=geom.solref.tolist(),
                solimp=geom.solimp.tolist(),
            )
            for geom in body.geoms
        ]

    geoms = []
    for side, name in ((-1, cs.WALL_LEFT), (1, cs.WALL_RIGHT)):
        wall = ex._stepped_wall_spec(name, ex._RGBA)
        geoms += reflected(
            wall, (cs.wall_x(WIDTH, side), 0.0, ex.stepped_wall_mocap_z())
        )
    platform = ex._platform_spec()
    platform.geoms[0].size[0] = 0.5 * PLATFORM_WIDTH
    platform.geoms[0].size[1] += 0.5 * PLATFORM_EXTRA
    x, y, z = ex.platform_mocap_pos()
    geoms += reflected(platform, (x, y + 0.5 * PLATFORM_EXTRA, z))
    return geoms, ex.goal_y(), ex.wall_end_y()


def _scene_spec(
    root: Path, home: dict[str, float], stage: list[dict], *, tracing: bool = False
) -> mujoco.MjSpec:
    """``scene.xml`` with ``long_jump_robot``'s hulls and leg-fold pairs under
    ``FULL_COLLISION``, the robot every policy here trains on; the stage as static world
    geoms, a contact sensor on the platform, mjlab's velocity sim settings, and the
    enter policy's start as the one keyframe. ``tracing`` leaves out the sim settings and
    the mesh copies, as nothing traced reads them."""
    robot = _common.load_upstream(root, ROBOT_PY)
    spec = mujoco.MjSpec.from_file(str(root / _common.ALLCOLLISIONS_SCENE_XML))
    robot.add_full_collision_geoms(spec)
    robot.add_leg_fold_pairs(spec)
    _common.full_collision(spec)
    for geom in stage:
        spec.worldbody.add_geom(**geom)
    spec.add_sensor(
        name=ON_PLATFORM,
        type=mujoco.mjtSensor.mjSENS_CONTACT,
        objtype=mujoco.mjtObj.mjOBJ_XBODY,  # the robot's whole subtree
        objname=TRACKED_BODY,
        reftype=mujoco.mjtObj.mjOBJ_GEOM,
        refname=PLATFORM_GEOM,
        intprm=[1, 0, 1],  # found, unreduced, one contact
    )
    if not tracing:
        apply_mjlab_sim_options(spec, VELOCITY_ENV.sim)
        _common.own_collision_meshes(spec)
    _common.set_only_keyframe(spec, home, (*START, 1.0, 0.0, 0.0, 0.0))
    return spec


def _qpos(
    model: mujoco.MjModel, root: tuple[float, ...], pose: dict[str, float]
) -> list[float]:
    """The keyframe with the root moved to ``root`` and the joints set to ``pose``."""
    qpos = model.key_qpos[0].copy()
    qpos[:7] = root
    for name, value in pose.items():
        qpos[model.jnt_qposadr[model.joint(name).id]] = value
    return qpos.tolist()


def add_scenes(project: mjswan.ProjectHandle, root: Path) -> None:
    def hub(path: str) -> Path:
        return Path(hf_hub_download(POLICY_REPO_ID, path, revision=POLICY_REVISION))

    # Each release config carries its policy's joint order and offset: HOME for enter,
    # BRACE for climb and exit.
    configs = [json.loads(hub(f"{d}config.json").read_text()) for d in STAGES]
    nets = [onnx.load(hub(f"{d}{c['policy_file']}")) for d, c in zip(STAGES, configs)]
    getup_path = _resolve_deploy_root() / POLICY_DIR / GETUP_ONNX
    pinned = json.loads(hub("manifest.json").read_text())["official_getup"]["sha256"]
    if hashlib.sha256(getup_path.read_bytes()).hexdigest() != pinned:
        raise ValueError(f"{getup_path} is not the get-up the release pins ({pinned}).")
    nets.append(onnx.load(str(getup_path)))
    getup_meta = {p.key: p.value for p in nets[3].metadata_props}
    offsets = [c["joint_offset_rad"] for c in configs]
    offsets.append([float(v) for v in getup_meta["default_joint_pos"].split(",")])
    joint_names = configs[0]["joint_names"]
    home = dict(zip(joint_names, offsets[0], strict=True))
    brace = dict(zip(joint_names, offsets[1], strict=True))

    stage, goal_y, wall_end_y = _stage(root)
    spec = _scene_spec(root, home, stage)
    model = spec.compile()
    if _common.servo_joints(spec) != joint_names:
        raise ValueError("The scene's servos are not in the release's joint order.")
    sym = _common.load_upstream(root, SYMMETRY_PY)
    chain_policy = partial(
        _Chain,
        nets,
        offsets=offsets,
        joint_range=[model.jnt_range[model.joint(n).id].tolist() for n in joint_names],
        reflect=(sym._JOINT_PERM, sym._JOINT_SIGN, sym._OBS_SIGN[:6]),
        goal_y=goal_y,
        recovery_y=wall_end_y,
    )

    scene = project.add_scene(name="Chimney Climb", spec=spec, control_dt=CONTROL_DT)
    scene.add_attribution("3d-models", license=root / "LICENSE-HARDWARE")
    # From behind the slot and above: the platform never comes between, and the walls
    # running down to the floor show the height.
    scene.set_viewer(_common.follow_cam(distance=1.3, elevation=-35.0, azimuth=-90.0))
    scene.set_trace_env(
        build_single_entity_trace_env(
            partial(_scene_spec, root, home, stage, tracing=True), entity_name=ENTITY
        )
    )
    joints = _common.joints_cfg(joint_names)
    proprio = _common.proprioception(joints)
    chain = {
        "root_pos": ObservationTermCfg(func=_root, params={"field": "root_link_pos_w"}),
        "root_quat": ObservationTermCfg(
            func=_root, params={"field": "root_link_quat_w"}
        ),
        "root_vel": ObservationTermCfg(
            func=_root, params={"field": "root_com_lin_vel_w"}
        ),
        "joint_vel": ObservationTermCfg(
            func=obs_fns.joint_vel_rel, params={"asset_cfg": joints}
        ),
        "heights": ObservationTermCfg(
            func=_heights,
            params={
                "asset_cfg": SceneEntityCfg(
                    ENTITY, body_names=HEIGHT_BODIES, preserve_order=True
                )
            },
        ),
        ON_PLATFORM: ObservationTermCfg(
            func=_sensor,
            params={"row": int(model.sensor_adr[model.sensor(ON_PLATFORM).id])},
        ),
    }
    mdp = MdpConfig(
        observations={
            # The 48 of proprioception less the last action, which each policy's
            # history in the carry replaces.
            "actor": ObservationGroupCfg(
                terms={k: v for k, v in proprio.items() if k != "actions"}
            ),
            "chain": ObservationGroupCfg(terms=chain),
        },
        actions=_common.servo_action(),
        commands={},
        terminations={},
        events={},
    )
    a = 0.5 * EXIT_PITCH
    entries = (
        ("Enter, climb, exit", 0, offsets[0], None),
        # Standing in the slot: the state enter hands over.
        ("Climb", 1, offsets[0], _qpos(model, (0, 0, STAND_Z, 1, 0, 0, 0), {})),
        (
            "Exit",
            2,
            offsets[1],
            _qpos(model, (*EXIT_START, math.cos(a), 0, math.sin(a), 0), brace),
        ),
    )
    for name, start, start_pose, initial_qpos in entries:
        scene.add_policy(
            name=name,
            policy=_export(chain_policy(start, start_pose)),
            mdp=mdp,
            in_keys=["actor", "chain", "adapt_hx"],
            out_keys=["action", ["next", "adapt_hx"]],
            policy_joint_names=joint_names,
            default_joint_pos=offsets[0],
            initial_qpos=initial_qpos,
            default=start == 0,
        )
