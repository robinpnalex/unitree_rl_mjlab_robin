"""PGTT-specific Go2 rough-terrain task configuration."""

from mjlab.envs import mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import GridPatternCfg
from mjlab.utils.noise import UniformNoiseCfg

from mjlab.tasks.velocity.mdp.pgtt import (
  gait_frequency,
  gait_phase_features,
  pgtt_foot_phase_reward,
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

  cfg.commands["gait_phase"] = GaitPhaseCommandCfg(
    resampling_time_range=(1.0e9, 1.0e9),
    frequency_range=(1.0, 3.0),
  )

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

  cfg.rewards["pgtt_foot_phase"] = RewardTermCfg(
    func=pgtt_foot_phase_reward,
    weight=1.0,
    params={"sensor_name": "height_scanner"},
  )
  cfg.rewards["pgtt_swing_contact"] = RewardTermCfg(
    func=pgtt_swing_contact_penalty,
    weight=-0.25,
    params={"sensor_name": "feet_ground_contact"},
  )

  if play:
    cfg.observations["policy"].enable_corruption = False

  return cfg


def unitree_go2_pgtt_ppo_runner_cfg():
  """Use the Go2 PPO defaults with asymmetric PGTT actor/critic observations."""
  cfg = unitree_go2_ppo_runner_cfg()
  cfg.experiment_name = "go2_pgtt"
  cfg.obs_groups = {"policy": ("policy",), "critic": ("policy", "critic")}
  return cfg
