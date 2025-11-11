# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os

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
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    # hand_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Hand")
    hand_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Hand",
        init_state=ArticulationCfg.InitialStateCfg(pos=[0.0, 0.0, 0.5], rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(CUSTOM_ASSET_DIR, "juggle_agent_v0.usd"),
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
                joint_names_expr=["robot0_WR.*", "robot0_(FF|MF|RF|LF|TH)J(3|2|1)", "robot0_(LF|TH)J4", "robot0_THJ0"],
                effort_limit={
                    "robot0_WRJ1": 4.785,
                    "robot0_WRJ0": 2.175,
                    "robot0_(FF|MF|RF|LF)J1": 0.7245,
                    "robot0_FFJ(3|2)": 0.9,
                    "robot0_MFJ(3|2)": 0.9,
                    "robot0_RFJ(3|2)": 0.9,
                    "robot0_LFJ(4|3|2)": 0.9,
                    "robot0_THJ4": 2.3722,
                    "robot0_THJ3": 1.45,
                    "robot0_THJ(2|1)": 0.99,
                    "robot0_THJ0": 0.81,
                },
                stiffness={
                    "robot0_WRJ.*": 5.0,
                    "robot0_(FF|MF|RF|LF|TH)J(3|2|1)": 1.0,
                    "robot0_(LF|TH)J4": 1.0,
                    "robot0_THJ0": 1.0,
                },
                damping={
                    "robot0_WRJ.*": 0.5,
                    "robot0_(FF|MF|RF|LF|TH)J(3|2|1)": 0.1,
                    "robot0_(LF|TH)J4": 0.1,
                    "robot0_THJ0": 0.1,
                },
            ),
        },
        actuator_value_resolution_debug_print=True
    )

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