"""Custom rewards for the LegoTripod turning task.

Mirrors the MetaMachine PlateauSpinComponent with target_spin=-3.0:

    spin_value = dot(-projected_gravity_b, ang_vel_b)
    reward = plateau(-spin_value, 3.0)

i.e., reward spinning clockwise (negative yaw rate) as seen from above.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _plateau(vel: torch.Tensor, max_desired_vel: float) -> torch.Tensor:
    """Vectorised plateau reward.

    Linearly increases from 0 to 1 as `vel` goes from 0 to `max_desired_vel`,
    then saturates at 1.  Negative values return 0.
    """
    return torch.clamp(vel / max_desired_vel, min=0.0, max=1.0)


def plateau_spin(
    env: ManagerBasedRlEnv,
    target_spin: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Plateau-style reward for spinning around the gravity axis.

    Matches the MetaMachine PlateauSpinComponent:
        spin_value = dot(-projected_gravity_b, ang_vel_b)

    With target_spin < 0 (e.g. -3.0):
        reward = plateau(-spin_value, -target_spin)
    i.e. clockwise rotation (negative yaw) is rewarded.

    Args:
        env: The environment.
        target_spin: Target angular velocity (rad/s).  Negative = clockwise.
        asset_cfg: Entity config to pull root state from.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    asset: Entity = env.scene[asset_cfg.name]

    # projected_gravity_b: (num_envs, 3)  -- gravity direction in body frame
    proj_grav = asset.data.projected_gravity_b
    # ang_vel_b: (num_envs, 3)
    ang_vel_b = asset.data.root_link_ang_vel_b

    # spin_value = dot(-projected_gravity_b, ang_vel_b)
    # This is the angular velocity component along the (negative) gravity axis.
    spin_value = torch.sum(-proj_grav * ang_vel_b, dim=-1)  # (num_envs,)

    if target_spin > 0:
        return _plateau(spin_value, target_spin)
    elif target_spin < 0:
        return _plateau(-spin_value, -target_spin)
    else:
        return -torch.square(spin_value)
