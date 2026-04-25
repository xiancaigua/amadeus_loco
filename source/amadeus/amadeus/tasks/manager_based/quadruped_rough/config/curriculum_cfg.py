# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp


@configclass
class AmadeusQuadrupedRoughCurriculumCfg:
    """Curriculum manager terms for rough locomotion."""

    terrain_levels = CurrTerm(func=velocity_mdp.terrain_levels_vel)
