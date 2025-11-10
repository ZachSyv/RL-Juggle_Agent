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
    # robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    hand_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Hand",
        init_state=ArticulationCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(CUSTOM_ASSET_DIR, "juggle_agent_v0.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # solver_position_iteration_count=16,
                # solver_velocity_iteration_count=1,
                # max_angular_velocity=1000.0,
                # max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
            ),
        ),
        actuators={
            "slider_to_cart": ImplicitActuatorCfg(
                joint_names_expr=[""],
                effort_limit_sim=400.0,
                stiffness=0.0,
                damping=10.0,
            ),
            "cart_to_pole": ImplicitActuatorCfg(
                joint_names_expr=[""],
                effort_limit_sim=400.0,
                stiffness=0.0,
                damping=10.0,
            ),
        },
    )

    # pdb.set_trace()

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint
    cart_dof_name = "robot0_(?:WRJ[01]|FFJ[0-3]|MFJ[0-3])"
    pole_dof_name = "robot0_(?:RFJ[0-3]|LFJ[0-4]|THJ[0-4])"
    # - action scale
    action_scale = 100.0  # [N]
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005
    # - reset states/conditions
    # initial_pole_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    # max_cart_pos = 3.0  # reset if cart exceeds this position [m]