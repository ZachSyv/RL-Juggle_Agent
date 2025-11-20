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

    # reward weights
    w_hoarding = 2.0        # should be high to strongly discourage hoarding 2 balls in one hand
    w_jitter = 0.001        # should be low to not overly discourage small adjustments, this is to prevent random drifting
    w_highest = 0.03        # the highest ball, small becaues it's continuously added
    w_rythem = 0.2         # should be moderate to encourage consistent timing
    w_catch = 5.0          # should be high to strongly encourage successful catches
    w_drop = 1.0           # should be moderate

    # geometric parameters
    catch_radius = 0.1  # radius within which a catch is registered, this definitly needs to be configured before we start
    target_height = 0.5 # height at which the ball apex should be, this also definitly needs to be configured before we start
    target_rythem = 0.4 # 60/150 seconds per throw, i.e. 2.5 throws per second
    ground_height = 0.0

    # tolerances
    sigma_rythem = 0.05
    sigma_drop_distance = 0.2
    sigma_apex_height = 0.1
    min_throw_height = 0.2 # minimum height a ball must reach to be considered a valid throw, done to prevent micro-throws
    min_vertical_velocity = 0.1 # minimum vertical velocity at throw time to be considered a valid throw
    