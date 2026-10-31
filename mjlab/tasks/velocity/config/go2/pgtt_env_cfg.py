"""PGTT-specific Go2 rough-terrain task configuration."""

import math

from mjlab.envs import mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import GridPatternCfg
from mjlab.utils.noise import UniformNoiseCfg

from mjlab.tasks.velocity.mdp.pgtt import (
  gait_frequency,
  gait_phase_features,
  pgtt_angular_velocity_xy_cost,
  pgtt_default_pose_cost,
  pgtt_foot_phase_reward,
  pgtt_joint_limits_cost,
  pgtt_joint_power_cost,
  pgtt_joint_torque_cost,
  pgtt_linear_velocity_z_cost,
  pgtt_track_angular_velocity,
  pgtt_track_linear_velocity,
  pgtt_heightmap,
  pgtt_swing_contact_penalty,
)
from mjlab.tasks.velocity.mdp.phase_command import GaitPhaseCommandCfg

from .env_cfgs import unitree_go2_rough_env_cfg
from .rl_cfg import unitree_go2_ppo_runner_cfg


def unitree_go2_pgtt_env_cfg(play: bool = False):
  """Create a PGTT task while leaving the existing rough task unchanged."""
  cfg = unitree_go2_rough_env_cfg(play=play)

  # Match the paper/deployment's 11-by-9 local elevation grid.
  height_scanner = next(
    sensor for sensor in cfg.scene.sensors if sensor.name == "height_scanner"
  )
  height_scanner.pattern = GridPatternCfg(size=(1.0, 0.8), resolution=0.1)
  cfg.scene.sensors = tuple(
    sensor
    for sensor in cfg.scene.sensors
    if sensor.name in {
      "feet_ground_contact",
      "nonfoot_ground_touch",
      "height_scanner",
    }
  )

  cfg.commands["gait_phase"] = GaitPhaseCommandCfg(
    resampling_time_range=(1.0e9, 1.0e9),
    frequency_range=(1.0, 3.0),
  )

  twist_command = cfg.commands["twist"]
  twist_command.resampling_time_range = (20.0, 20.0)
  twist_command.rel_standing_envs = 0.0
  twist_command.rel_heading_envs = 0.0
  twist_command.heading_command = False
  twist_command.debug_vis = False
  twist_command.ranges.lin_vel_x = (-1.0, 1.0)
  twist_command.ranges.lin_vel_y = (-1.0, 1.0)
  twist_command.ranges.ang_vel_z = (-1.0, 1.0)
  twist_command.ranges.heading = None

  # The actor receives only deployable proprioception, phase, heightmap, frequency,
  # previous action, and the velocity command. The critic adds privileged base velocity.
  cfg.observations["policy"] = ObservationGroupCfg(
    terms={
      "base_ang_vel": ObservationTermCfg(
        func=mdp.base_ang_vel,
        noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
      ),
      "projected_gravity": ObservationTermCfg(
        func=mdp.projected_gravity,
        noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
      ),
      "joint_pos": ObservationTermCfg(
        func=mdp.joint_pos_rel,
        noise=UniformNoiseCfg(n_min=-0.03, n_max=0.03),
      ),
      "joint_vel": ObservationTermCfg(
        func=mdp.joint_vel_rel,
        noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5),
      ),
      "phase": ObservationTermCfg(func=gait_phase_features),
      "heightmap": ObservationTermCfg(
        func=pgtt_heightmap,
        params={"sensor_name": "height_scanner"},
        noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
      ),
      "gait_frequency": ObservationTermCfg(func=gait_frequency),
      "last_action": ObservationTermCfg(func=mdp.last_action),
      "command": ObservationTermCfg(
        func=mdp.generated_commands,
        params={"command_name": "twist"},
      ),
    },
    concatenate_terms=True,
    enable_corruption=True,
  )
  cfg.observations["critic"] = ObservationGroupCfg(
    terms={"base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel)},
    concatenate_terms=True,
    enable_corruption=False,
  )
  cfg.observations.pop("extero", None)

  # Use the paper's shared reward terms rather than inheriting the baseline's
  # repository-specific shaping terms.
  cfg.rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=pgtt_track_linear_velocity,
      weight=1.0,
      params={"command_name": "twist", "sigma": 0.25},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=pgtt_track_angular_velocity,
      weight=0.5,
      params={"command_name": "twist", "sigma": 0.25},
    ),
    "linear_velocity_z": RewardTermCfg(
      func=pgtt_linear_velocity_z_cost,
      weight=-2.0,
    ),
    "angular_velocity_xy": RewardTermCfg(
      func=pgtt_angular_velocity_xy_cost,
      weight=-0.05,
    ),
    "orientation": RewardTermCfg(
      func=mdp.flat_orientation_l2,
      weight=-0.2,
    ),
    "termination": RewardTermCfg(func=mdp.is_terminated, weight=-1.0),
    "joint_power": RewardTermCfg(
      func=pgtt_joint_power_cost,
      weight=-2e-5,
    ),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "joint_limits": RewardTermCfg(
      func=pgtt_joint_limits_cost,
      weight=-1.0,
    ),
    "default_pose": RewardTermCfg(
      func=pgtt_default_pose_cost,
      weight=-0.5,
    ),
    "joint_torque": RewardTermCfg(
      func=pgtt_joint_torque_cost,
      weight=-1e-5,
    ),
    "pgtt_foot_phase": RewardTermCfg(
      func=pgtt_foot_phase_reward,
      weight=1.0,
      params={"sensor_name": "height_scanner"},
    ),
    "pgtt_swing_contact": RewardTermCfg(
      func=pgtt_swing_contact_penalty,
      weight=-0.25,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }

  # The paper evaluates pushes rather than applying the baseline's push event
  # throughout training. Keep terrain-level curriculum, but remove baseline
  # command/reward-weight schedules that target rewards no longer in this config.
  cfg.events.pop("push_robot", None)
  cfg.curriculum.pop("command_vel", None)
  for term_name in (
    "foot_slip",
    "joint_torque_l2",
    "target_smoothness",
    "shank_knee_collision",
    "joint_motion",
  ):
    cfg.curriculum.pop(term_name, None)
  # Keep the baseline guardrails that prevent exploiting body-ground contact.
  cfg.terminations["fell_over"].params["limit_angle"] = math.pi / 2

  if play:
    cfg.observations["policy"].enable_corruption = False

  return cfg


def unitree_go2_pgtt_ppo_runner_cfg():
  """Use the Go2 PPO defaults with asymmetric PGTT actor/critic observations."""
  cfg = unitree_go2_ppo_runner_cfg()
  cfg.experiment_name = "go2_pgtt"
  cfg.obs_groups = {"policy": ("policy",), "critic": ("policy", "critic")}
  return cfg
