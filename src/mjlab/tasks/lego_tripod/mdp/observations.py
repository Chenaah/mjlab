"""Custom observations for the LegoTripod turning task.

The lego tripod uses a modular observation scheme where:
- Per-module (module 0 only): projected gravity (3D) + gyro (3D) = 6D
- Global: dof_pos (3D) + dof_vel (3D) + last_action (3D) = 9D
Total: 15D per timestep (history of 3 steps → 45D to actor)

Observation alignment with MetaMachine
---------------------------------------
MetaMachine constructs the full 15D observation as a single frame, then pushes
it into an ``ObservationBuffer`` that stores 3 frames in **frame-major** order::

    [frame_t-2(15D), frame_t-1(15D), frame_t-0(15D)]   → 45D

mjlab's built-in per-term history produces **term-major** order instead::

    [A_t-2, A_t-1, A_t-0, B_t-2, B_t-1, B_t-0, ...]   → 45D

To match MetaMachine exactly, we expose a single ``combined_frame_obs`` function
that returns the full 15D vector.  When ``history_length=3`` is applied to this
single term, the circular buffer naturally produces frame-major ordering.

Per-component noise is achieved via ``UniformNoiseCfg`` with 15-element tuples
for ``n_min`` / ``n_max``, so the standard ``enable_corruption`` mechanism works.

MetaMachine uses *raw* ``dof_pos`` / ``dof_vel`` (absolute values from MuJoCo)
and stores ``last_action`` as the action *after* scale+clip (symmetric_limit).

The robot has IMU sensors named:
  imu_quat0, imu_gyro0  -- for module 0 (body "l0")
  imu_quat1, imu_gyro1  -- for module 1 (body "l1")
  imu_quat2, imu_gyro2  -- for module 2 (body "l2")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# Gravity vector in world frame (pointing down).
_GRAVITY_W = torch.tensor([0.0, 0.0, -1.0])


# ---------------------------------------------------------------------------
# Per-module observations (module 0 only for this robot)
# ---------------------------------------------------------------------------


def projected_gravity_module0(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Projected gravity in the frame of module 0 (body 'l0').

    Uses the imu_quat0 sensor (framequat of body l0) to rotate gravity into
    the module-local frame.  Shape: (num_envs, 3).
    """
    sensor: BuiltinSensor = env.scene["robot/imu_quat0"]
    quat_w = sensor.data  # (num_envs, 4) -- [w, x, y, z] from MuJoCo framequat
    gravity_w = _GRAVITY_W.to(env.device).expand(env.num_envs, 3)
    return quat_apply_inverse(quat_w, gravity_w)


def gyro_module0(
    env: ManagerBasedRlEnv,
) -> torch.Tensor:
    """Angular velocity from module 0 gyro sensor.

    Uses the imu_gyro0 sensor.  Shape: (num_envs, 3).
    """
    sensor: BuiltinSensor = env.scene["robot/imu_gyro0"]
    return sensor.data  # (num_envs, 3)


# ---------------------------------------------------------------------------
# Joint observations — raw (matching MetaMachine ``dof_pos`` / ``dof_vel``)
# ---------------------------------------------------------------------------


def joint_pos_raw(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Raw (absolute) joint positions — matches MetaMachine ``dof_pos``.

    Unlike ``joint_pos_rel`` this does **not** subtract ``default_joint_pos``.
    Shape: (num_envs, num_joints).
    """
    asset: Entity = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids
    return asset.data.joint_pos[:, jnt_ids]


def joint_vel_raw(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Raw joint velocities — matches MetaMachine ``dof_vel``.

    Equivalent to ``joint_vel_rel`` when ``default_joint_vel == 0``, but kept
    explicit for clarity.  Shape: (num_envs, num_joints).
    """
    asset: Entity = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids
    return asset.data.joint_vel[:, jnt_ids]


# ---------------------------------------------------------------------------
# Action observation — clipped (matching MetaMachine ``last_action``)
# ---------------------------------------------------------------------------


def last_action_clipped(
    env: ManagerBasedRlEnv,
    clip_min: float = -0.8,
    clip_max: float = 0.8,
) -> torch.Tensor:
    """Last action after clipping — matches MetaMachine ``last_action``.

    MetaMachine stores ``last_action`` as ``clip(action * scale, -symmetric_limit,
    symmetric_limit)`` (scale=1.0, symmetric_limit=0.8 for lego_tripod).
    The default ``envs_mdp.last_action`` in mjlab returns the *raw* policy output
    before ``clip_raw`` is applied, which can differ during early training.

    Shape: (num_envs, action_dim).
    """
    raw_action = env.action_manager.action  # raw policy output
    return torch.clamp(raw_action, clip_min, clip_max)


# ---------------------------------------------------------------------------
# Combined frame observation (frame-major history compat with MetaMachine)
# ---------------------------------------------------------------------------


def combined_frame_obs(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    action_clip_min: float = -0.8,
    action_clip_max: float = 0.8,
) -> torch.Tensor:
    """Full 15D observation frame matching MetaMachine layout.

    Returns a single concatenated vector per environment::

        [projected_gravity_m0(3), gyro_m0(3), dof_pos(3), dof_vel(3), last_action(3)]

    When used with ``history_length=3`` on a single term, the circular buffer
    produces **frame-major** ordering which matches MetaMachine's
    ``ObservationBuffer``::

        [frame_t-2(15D), frame_t-1(15D), frame_t-0(15D)]   → 45D

    Noise is applied externally via ``UniformNoiseCfg`` with per-dimension
    tuples so that ``enable_corruption`` works correctly.

    Shape: (num_envs, 15).
    """
    # 1. projected gravity in module-0 frame  (3D)
    pg = projected_gravity_module0(env, asset_cfg)

    # 2. gyro from module-0 sensor  (3D)
    gyro = gyro_module0(env)

    # 3. raw joint positions  (3D)
    jpos = joint_pos_raw(env, asset_cfg)

    # 4. raw joint velocities  (3D)
    jvel = joint_vel_raw(env, asset_cfg)

    # 5. last action after clipping  (3D)
    act = last_action_clipped(env, action_clip_min, action_clip_max)

    return torch.cat([pg, gyro, jpos, jvel, act], dim=-1)
