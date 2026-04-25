# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp


@configclass
class AmadeusQuadrupedRoughRewardsCfg:
    """Reward manager terms for rough locomotion."""

    track_lin_vel_xy_exp = RewTerm(
        func=velocity_mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=velocity_mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    lin_vel_z_l2 = RewTerm(func=velocity_mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=velocity_mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=velocity_mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=velocity_mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=velocity_mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=velocity_mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=velocity_mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )
    flat_orientation_l2 = RewTerm(func=velocity_mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=velocity_mdp.joint_pos_limits, weight=0.0)
