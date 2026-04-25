# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.agents.rsl_rl_ppo_cfg import (
    AnymalCRoughPPORunnerCfg,
)


@configclass
class AmadeusQuadrupedRoughPPORunnerCfg(AnymalCRoughPPORunnerCfg):
    """PPO baseline config for the external ANYmal-C rough-terrain task."""

    experiment_name = "amadeus_quadruped_rough"
    save_interval = 25
