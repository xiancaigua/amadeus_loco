# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp


@configclass
class AmadeusQuadrupedRoughTerminationsCfg:
    """Termination manager terms for rough locomotion."""

    time_out = DoneTerm(func=velocity_mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=velocity_mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
