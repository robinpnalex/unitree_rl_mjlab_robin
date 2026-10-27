"""Per-leg gait phase command used by the PGTT task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import math
import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class GaitPhaseCommandCfg(CommandTermCfg):
  """Configuration for an episode-reset gait phase clock."""

  frequency_range: tuple[float, float] = (1.0, 3.0)
  phase_offsets: tuple[float, float, float, float] = (
    0.0,
    math.pi,
    math.pi,
    0.0,
  )

  def build(self, env: ManagerBasedRlEnv) -> GaitPhaseCommand:
    return GaitPhaseCommand(self, env)


class GaitPhaseCommand(CommandTerm):
  """Advance four leg phases at a per-environment sampled frequency."""

  cfg: GaitPhaseCommandCfg

  def __init__(self, cfg: GaitPhaseCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.phase = torch.zeros((self.num_envs, 4), device=self.device)
    self.frequency = torch.zeros(self.num_envs, device=self.device)
    self._offsets = torch.tensor(cfg.phase_offsets, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat(
      (
        torch.cos(self.phase),
        torch.sin(self.phase),
        self.frequency.unsqueeze(-1),
      ),
      dim=-1,
    )

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.frequency[env_ids] = torch.empty(
      len(env_ids), device=self.device
    ).uniform_(*self.cfg.frequency_range)
    self.phase[env_ids] = self._offsets

  def _update_command(self) -> None:
    self.phase.add_(2.0 * math.pi * self.frequency.unsqueeze(-1) * self._env.step_dt)
    self.phase.remainder_(2.0 * math.pi)
