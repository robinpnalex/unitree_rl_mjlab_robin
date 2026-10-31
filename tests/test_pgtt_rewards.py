import unittest
from types import SimpleNamespace

import torch

from mjlab.tasks.velocity.mdp.pgtt import (
  pgtt_joint_power_cost,
  pgtt_joint_torque_cost,
)


class _RewardRobotData:
  def __init__(self, actuator_force: torch.Tensor, joint_vel: torch.Tensor):
    self.actuator_force = actuator_force
    self.joint_vel = joint_vel

  @property
  def joint_torques(self):
    raise AssertionError("PGTT actuation rewards must not access joint_torques")


class PgttActuationRewardTest(unittest.TestCase):
  def setUp(self):
    data = _RewardRobotData(
      actuator_force=torch.tensor([[2.0, -3.0], [1.0, -2.0]]),
      joint_vel=torch.tensor([[4.0, -5.0], [3.0, -6.0]]),
    )
    robot = SimpleNamespace(data=data)
    self.env = SimpleNamespace(scene=SimpleNamespace(entities={"robot": robot}))

  def test_joint_power_uses_actuator_force_and_returns_finite_batch(self):
    reward = pgtt_joint_power_cost(self.env)

    self.assertEqual(tuple(reward.shape), (2,))
    self.assertTrue(torch.isfinite(reward).all())
    torch.testing.assert_close(reward, torch.tensor([23.0, 15.0]))

  def test_torque_cost_uses_actuator_force_and_returns_finite_batch(self):
    reward = pgtt_joint_torque_cost(self.env)

    self.assertEqual(tuple(reward.shape), (2,))
    self.assertTrue(torch.isfinite(reward).all())
    torch.testing.assert_close(reward, torch.tensor([13.0, 5.0]))

  def test_actuator_joint_dimension_mismatch_has_clear_error(self):
    self.env.scene.entities["robot"].data.actuator_force = torch.ones((2, 3))

    with self.assertRaisesRegex(RuntimeError, "one actuator force per joint"):
      pgtt_joint_power_cost(self.env)


if __name__ == "__main__":
  unittest.main()
