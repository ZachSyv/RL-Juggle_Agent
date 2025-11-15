# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os
import math

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# make isaaclab happy
import isaaclab.sim as sim_utils

# set this path in the Path-to-IsaacLab/source/isaaclab/isaaclab/utils/assets.py
from isaaclab.utils.assets import CUSTOM_ASSET_DIR
from isaaclab.actuators import ImplicitActuatorCfg

import pdb

def get_hand_cfg(prim_name, usd_file_name, pos, rot):
    hand_cfg = ArticulationCfg(
        prim_path=f"/World/envs/env_.*/{prim_name}",
        init_state=ArticulationCfg.InitialStateCfg(pos=pos, rot=rot),
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(CUSTOM_ASSET_DIR, usd_file_name),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=True,
                max_depenetration_velocity=1000.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
            ),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
            fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["WR.*", "(FF|MF|RF|LF|TH)J(4|3|2|1)", "(LF|TH)J5", "elbow_(rotate|bend)"],
                effort_limit={
                    "WRJ2": 4.785,
                    "WRJ1": 2.175,
                    "(FF|MF|RF|LF)J1": 0.7245,
                    "FFJ(4|3|2)": 0.9,
                    "MFJ(4|3|2)": 0.9,
                    "RFJ(4|3|2)": 0.9,
                    "LFJ(5|4|3|2)": 0.9,
                    "THJ5": 2.3722,
                    "THJ4": 1.45,
                    "THJ(3|2)": 0.99,
                    "THJ1": 0.81,
                    "elbow_(rotate|bend)": 0.8
                },
                stiffness={
                    "WRJ.*": 5.0,
                    "(FF|MF|RF|LF|TH)J(4|3|2|1)": 1.0,
                    "(LF|TH)J5": 1.0,
                    "elbow_(rotate|bend)": 1.0
                },
                damping={
                    "WRJ.*": 0.5,
                    "(FF|MF|RF|LF|TH)J(4|3|2|1)": 0.1,
                    "(LF|TH)J5": 0.1,
                    "elbow_(rotate|bend)": 1.0
                },
            ),
        },
        actuator_value_resolution_debug_print=False
    )
    return hand_cfg


@configclass
class JugglingAgentEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 1
    observation_space = 4
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 100, render_interval=decimation)

    # robot(s)
    # hand_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Hand")
    left_hand_cfg = get_hand_cfg("left_hand", "shadow_hand_left_with_elbow.usd",
                                 pos=[0, -0.5, 0.5], rot=[-math.sqrt(2) / 2, 0, math.sqrt(2) / 2, 0])
    right_hand_cfg = get_hand_cfg("right_hand", "shadow_hand_right_with_elbow.usd",
                                  pos=[0, 0.5, 0.5], rot=[-math.sqrt(2) / 2, 0, math.sqrt(2) / 2, 0])

    # pdb.set_trace()

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint
    # cart_dof_name = "cart_to_pole"
    # pole_dof_name = "slider_to_cart"
    # - action scale
    # action_scale = 100.0  # [N]
    # - reward scales
    # rew_scale_alive = 1.0
    # rew_scale_terminated = -2.0
    # rew_scale_pole_pos = -1.0
    # rew_scale_cart_vel = -0.01
    # rew_scale_pole_vel = -0.005
    # - reset states/conditions
    # initial_pole_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    # max_cart_pos = 3.0  # reset if cart exceeds this position [m]