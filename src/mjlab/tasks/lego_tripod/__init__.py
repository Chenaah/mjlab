"""LegoTripod task registration for mjlab.

Registers the task under the ID ``Mjlab-LegoTripod-Turn``.

Usage::

    python mjlab/src/mjlab/scripts/train.py Mjlab-LegoTripod-Turn
"""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import lego_tripod_turn_env_cfg
from .rl_cfg import lego_tripod_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-LegoTripod-Turn",
    env_cfg=lego_tripod_turn_env_cfg(play=False),
    play_env_cfg=lego_tripod_turn_env_cfg(play=True),
    rl_cfg=lego_tripod_ppo_runner_cfg(),
    runner_cls=None,  # use default MjlabOnPolicyRunner
)
