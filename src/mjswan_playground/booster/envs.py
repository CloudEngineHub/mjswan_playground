"""mjlab envs for booster_deploy's K1 tasks: booster_assets' K1 as booster_deploy's
MuJoCo controller runs it, with its own MJCF, timestep and PD gains."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mujoco
from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sim import MujocoCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

ANCHOR = "trunk"
#: The root first: mjlab resets the robot to the first tracked body's pose.
BODY_NAMES = (
    "trunk",
    "left_hip_roll_link",
    "left_knee_pitch_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_pitch_link",
    "right_ankle_roll_link",
    "left_shoulder_roll_link",
    "left_elbow_pitch_link",
    "left_elbow_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_pitch_link",
    "right_elbow_yaw_link",
)
END_EFFECTORS = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_elbow_yaw_link",
    "right_elbow_yaw_link",
)


def _robot_spec(xml: Path) -> mujoco.MjSpec:
    """The robot and its sky: no floor, lights or motors, and its gyro named as mjlab's
    tracking observation reads it."""
    spec = mujoco.MjSpec.from_file(str(xml))
    floor = [t for t in spec.textures if t.type != mujoco.mjtTexture.mjTEXTURE_SKYBOX]
    for element in [
        *spec.worldbody.geoms,
        *spec.worldbody.lights,
        *spec.actuators,
        *spec.materials,
        *floor,
    ]:
        spec.delete(element)
    for sensor in list(spec.sensors):
        if sensor.type == mujoco.mjtSensor.mjSENS_GYRO:
            sensor.name = "imu_ang_vel"
            sensor.noise = 0.0
        else:
            spec.delete(sensor)
    return spec


def _robot_cfg(xml: Path, task: Any) -> EntityCfg:
    robot = task.robot
    gains = zip(
        robot.joint_names,
        robot.joint_stiffness,
        robot.joint_damping,
        robot.effort_limit,
    )
    return EntityCfg(
        spec_fn=lambda: _robot_spec(xml),
        init_state=EntityCfg.InitialStateCfg(
            pos=tuple(task.mujoco.init_pos),
            joint_pos=dict(zip(robot.joint_names, robot.default_joint_pos)),
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            # booster_deploy's MuJoCo controller: the PD torque each physics step,
            # clipped to the task's effort limit.
            actuators=tuple(
                IdealPdActuatorCfg(
                    target_names_expr=(name,),
                    stiffness=kp,
                    damping=kd,
                    effort_limit=limit,
                )
                for name, kp, kd, limit in gains
            ),
            soft_joint_pos_limit_factor=1.0,
        ),
    )


def _controller_cfg(task: Any, assets: Path) -> ManagerBasedRlEnvCfg:
    """mjlab's tracking env as booster_deploy's MuJoCo controller runs `task`."""
    xml = Path(task.robot.mjcf_path.replace("{BOOSTER_ASSETS_DIR}", str(assets)))
    cfg = make_tracking_env_cfg()
    cfg.scene.entities = {"robot": _robot_cfg(xml, task)}
    # booster_deploy runs the nominal robot: no pushes or randomization.
    cfg.events = {}
    cfg.rewards = {}
    cfg.viewer.body_name = ANCHOR
    cfg.decimation = task.mujoco.decimation
    cfg.sim.mujoco = MujocoCfg(
        timestep=task.mujoco.physics_dt,
        # The MJCF's own solver options, which booster_deploy keeps.
        integrator="euler",
        impratio=10.0,
        tolerance=1e-6,
    )
    cfg.episode_length_s = int(1e9)
    return cfg


def tracking_cfg(task: Any, assets: Path) -> ManagerBasedRlEnvCfg:
    """The tracking observation `BeyondMimicPolicy` reads: no state estimation."""
    cfg = _controller_cfg(task, assets)
    robot = task.robot
    actor = cfg.observations["actor"].terms
    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms={
                k: v
                for k, v in actor.items()
                if k not in ("motion_anchor_pos_b", "base_lin_vel")
            },
            concatenate_terms=True,
            enable_corruption=False,
        )
    }
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    # `BeyondMimicPolicy`'s default, which both K1 tasks keep.
    assert task.policy.action_scale_factors is None
    assert task.policy.fixed_action_scale is None
    action.scale = {
        name: 0.25 * limit / kp
        for name, limit, kp in zip(
            robot.joint_names, robot.effort_limit, robot.joint_stiffness
        )
    }

    motion = cfg.commands["motion"]
    assert isinstance(motion, MotionCommandCfg)
    motion.anchor_body_name = ANCHOR
    motion.body_names = BODY_NAMES
    motion.sampling_mode = "start"
    motion.pose_range = {}
    motion.velocity_range = {}
    motion.joint_position_range = (0.0, 0.0)

    cfg.terminations["ee_body_pos"].params["body_names"] = END_EFFECTORS
    return cfg


def walk_cfg(task: Any, assets: Path) -> ManagerBasedRlEnvCfg:
    """One frame of what `LocomotionPolicy` reads, without its last action: the policy
    keeps that and the history itself (`upstream.walk_onnx`)."""
    cfg = _controller_cfg(task, assets)
    joints = SceneEntityCfg(
        "robot", joint_names=tuple(task.policy.policy_joint_names), preserve_order=True
    )
    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms={
                "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
                "projected_gravity": ObservationTermCfg(
                    func=envs_mdp.projected_gravity
                ),
                "command": ObservationTermCfg(
                    func=envs_mdp.generated_commands, params={"command_name": "twist"}
                ),
                "joint_pos": ObservationTermCfg(
                    func=envs_mdp.joint_pos_rel, params={"asset_cfg": joints}
                ),
                "joint_vel": ObservationTermCfg(
                    func=envs_mdp.joint_vel_rel, params={"asset_cfg": joints}
                ),
            },
            concatenate_terms=True,
            enable_corruption=False,
        )
    }
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = float(task.policy.action_scale)

    limits = task.vel_command
    cfg.commands = {
        # booster_deploy reads the command off a keyboard, within these limits.
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(4.0, 8.0),
            rel_standing_envs=0.2,
            heading_command=False,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-limits.vx_backward_max, limits.vx_forward_max),
                lin_vel_y=(-limits.vy_max, limits.vy_max),
                ang_vel_z=(-limits.vyaw_max, limits.vyaw_max),
            ),
        )
    }
    cfg.events = {
        "reset_base": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {}, "velocity_range": {}},
        ),
        "reset_robot_joints": EventTermCfg(
            func=envs_mdp.reset_joints_by_offset,
            mode="reset",
            params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
        ),
    }
    cfg.terminations = {
        # booster_deploy's safety stop.
        "fell_over": TerminationTermCfg(
            func=envs_mdp.bad_orientation, params={"limit_angle": math.radians(60.0)}
        ),
    }
    return cfg
