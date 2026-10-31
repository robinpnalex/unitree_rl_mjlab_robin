"""PGTT observations and reward terms for the Unitree Go2 task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import math
import torch

from mjlab.entity import Entity
from mjlab.sensor import ContactSensor, RayCastSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_LEG_ORDER = ("FR", "FL", "RR", "RL")


def pgtt_track_linear_velocity(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  sigma: float = 0.25,
) -> torch.Tensor:
  """Paper's Gaussian tracking reward for local-frame planar velocity."""
  robot: Entity = env.scene.entities["robot"]
  command = env.command_manager.get_command(command_name)
  error = command[:, :2] - robot.data.root_link_lin_vel_b[:, :2]
  return torch.exp(-torch.sum(error.square(), dim=-1) / sigma)


def pgtt_track_angular_velocity(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  sigma: float = 0.25,
) -> torch.Tensor:
  """Paper's Gaussian tracking reward for yaw angular velocity."""
  robot: Entity = env.scene.entities["robot"]
  command = env.command_manager.get_command(command_name)
  error = command[:, 2] - robot.data.root_link_ang_vel_b[:, 2]
  return torch.exp(-error.square() / sigma)


def pgtt_linear_velocity_z_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot: Entity = env.scene.entities["robot"]
  return robot.data.root_link_lin_vel_w[:, 2].square()


def pgtt_angular_velocity_xy_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot: Entity = env.scene.entities["robot"]
  return robot.data.root_link_ang_vel_b[:, :2].square().sum(dim=-1)


def pgtt_joint_power_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot: Entity = env.scene.entities["robot"]
  return (robot.data.joint_torques.abs() * robot.data.joint_vel.abs()).sum(dim=-1)


def pgtt_default_pose_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot: Entity = env.scene.entities["robot"]
  default_joint_pos = robot.data.default_joint_pos
  assert default_joint_pos is not None
  return (robot.data.joint_pos - default_joint_pos).square().sum(dim=-1)


def pgtt_joint_limits_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot: Entity = env.scene.entities["robot"]
  limits = robot.data.soft_joint_pos_limits
  assert limits is not None
  joint_pos = robot.data.joint_pos
  outside = (joint_pos < limits[..., 0]) | (joint_pos > limits[..., 1])
  return outside.float().sum(dim=-1)


def pgtt_joint_torque_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot: Entity = env.scene.entities["robot"]
  return robot.data.joint_torques.square().sum(dim=-1)


def gait_phase_features(
  env: ManagerBasedRlEnv, command_name: str = "gait_phase"
) -> torch.Tensor:
  """Return per-leg cosine features followed by sine features."""
  command = env.command_manager.get_term(command_name)
  return torch.cat((torch.cos(command.phase), torch.sin(command.phase)), dim=-1)


def gait_frequency(
  env: ManagerBasedRlEnv, command_name: str = "gait_phase"
) -> torch.Tensor:
  """Return the sampled gait frequency as a one-dimensional observation."""
  command = env.command_manager.get_term(command_name)
  return command.frequency.unsqueeze(-1)


def pgtt_heightmap(
  env: ManagerBasedRlEnv, sensor_name: str = "height_scanner"
) -> torch.Tensor:
  """Return flattened yaw-aligned terrain heights relative to the scanner origin."""
  sensor: RayCastSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  relative_heights = sensor_data.hit_pos_w[..., 2] - sensor_data.pos_w[:, 2:3]
  valid = sensor_data.distances >= 0.0
  return torch.where(valid, torch.nan_to_num(relative_heights), 0.0)


def _local_terrain_height_variation(
  env: ManagerBasedRlEnv, sensor_name: str
) -> torch.Tensor:
  """Compute a front/rear and left/right height range for each foot.

  The authors' implementation uses the four robot-centric heightmap quadrants as
  per-leg neighborhoods. The order here is FR, FL, RR, RL.
  """
  sensor: RayCastSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  hit_pos_w = sensor_data.hit_pos_w
  hit_pos_sensor = quat_apply_inverse(
    sensor_data.quat_w.unsqueeze(1),
    hit_pos_w - sensor_data.pos_w.unsqueeze(1),
  )

  x = hit_pos_sensor[..., 0]
  y = hit_pos_sensor[..., 1]
  terrain_z = hit_pos_w[..., 2]
  valid = (sensor_data.distances >= 0.0) & torch.isfinite(terrain_z)
  quadrants = (
    (x >= 0.0) & (y < 0.0),  # FR
    (x >= 0.0) & (y >= 0.0),  # FL
    (x < 0.0) & (y < 0.0),  # RR
    (x < 0.0) & (y >= 0.0),  # RL
  )

  variations = []
  for quadrant in quadrants:
    mask = quadrant & valid
    highest = torch.where(mask, terrain_z, -torch.inf).amax(dim=-1)
    lowest = torch.where(mask, terrain_z, torch.inf).amin(dim=-1)
    variation = torch.where(mask.any(dim=-1), highest - lowest, 0.0)
    variations.append(variation)
  return torch.stack(variations, dim=-1).clamp_min(0.0)


def _desired_foot_height(
  phase: torch.Tensor,
  terrain_variation: torch.Tensor,
  stance_height: float,
  swing_apex: float,
  stance_ratio: float,
) -> torch.Tensor:
  """Evaluate the paper's two zero-tangent cubic Hermite swing splines."""
  stance_end = 2.0 * math.pi * stance_ratio
  swing_duration = math.pi * (1.0 - stance_ratio)
  swing_peak = math.pi * (1.0 + stance_ratio)
  adapted_apex = swing_apex + terrain_variation

  swing_up_u = ((phase - stance_end) / swing_duration).clamp(0.0, 1.0)
  swing_down_u = ((phase - swing_peak) / swing_duration).clamp(0.0, 1.0)
  swing_up = stance_height + (adapted_apex - stance_height) * (
    3.0 * swing_up_u.square() - 2.0 * swing_up_u.pow(3)
  )
  swing_down = adapted_apex + (stance_height - adapted_apex) * (
    3.0 * swing_down_u.square() - 2.0 * swing_down_u.pow(3)
  )
  return torch.where(
    phase < stance_end,
    stance_height,
    torch.where(phase < swing_peak, swing_up, swing_down),
  )


def pgtt_foot_phase_reward(
  env: ManagerBasedRlEnv,
  sensor_name: str = "height_scanner",
  command_name: str = "gait_phase",
  stance_height: float = -0.30,
  swing_apex: float = -0.20,
  stance_ratio: float = 0.5,
  sigma: float = 0.05,
) -> torch.Tensor:
  """Reward each foot for tracking its terrain-adaptive phase height."""
  robot: Entity = env.scene.entities["robot"]
  command = env.command_manager.get_term(command_name)
  site_ids = [robot.site_names.index(leg) for leg in _LEG_ORDER]
  hip_ids = [robot.body_names.index(f"{leg}_hip") for leg in _LEG_ORDER]

  foot_pos_w = robot.data.site_pos_w[:, site_ids]
  hip_pos_w = robot.data.body_link_pos_w[:, hip_ids]
  hip_quat_w = robot.data.body_link_quat_w[:, hip_ids]
  foot_pos_hip = quat_apply_inverse(hip_quat_w, foot_pos_w - hip_pos_w)
  foot_height_hip = foot_pos_hip[..., 2]

  terrain_variation = _local_terrain_height_variation(env, sensor_name)
  desired_height = _desired_foot_height(
    command.phase,
    terrain_variation,
    stance_height,
    swing_apex,
    stance_ratio,
  )
  squared_error = (desired_height - foot_height_hip).square()
  return torch.exp(-squared_error / sigma).sum(dim=-1)


def pgtt_swing_contact_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
  command_name: str = "gait_phase",
) -> torch.Tensor:
  """Count foot contacts that occur during the phase-defined swing interval."""
  sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_term(command_name)
  assert sensor.data.found is not None
  contact = sensor.data.found > 0
  swing = command.phase >= math.pi
  return (contact & swing).float().sum(dim=-1)
