"""Custom terminations for the LegoTripod task.

Terminates when the belly geom contacts the floor (body contact termination).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def belly_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str = "belly_contact",
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate when the belly geom touches the floor.

    Args:
        env: The environment.
        sensor_name: Name of the ContactSensor that watches belly/floor contacts.
        force_threshold: Minimum contact force magnitude to count as contact (N).

    Returns:
        Boolean tensor of shape (num_envs,).
    """
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is not None:
        force_mag = torch.norm(data.force, dim=-1)  # (num_envs, num_slots, 3) → (num_envs, num_slots)
        return (force_mag > force_threshold).any(dim=-1)
    assert data.found is not None
    return torch.any(data.found > 0, dim=-1)
