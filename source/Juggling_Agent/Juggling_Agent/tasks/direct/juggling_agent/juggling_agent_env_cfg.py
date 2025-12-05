# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os
import math
from pathlib import Path
#import torch can't use torch in configclass

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# make isaaclab happy
import isaaclab.sim as sim_utils

# set this path in the Path-to-IsaacLab/source/isaaclab/isaaclab/utils/assets.py
# from isaaclab.utils.assets import CUSTOM_ASSET_DIR
from isaaclab.actuators import ImplicitActuatorCfg

import pdb

CUSTOM_ASSET_DIR = str((Path(__file__).parent.parent.parent.parent / "assets").resolve())

def get_hand_cfg(prim_name, usd_file_name, pos, rot, joint_pos):
    hand_cfg = ArticulationCfg(
        prim_path=f"/World/envs/env_.*/{prim_name}",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=pos,
            rot=rot,
            joint_pos=joint_pos,
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(CUSTOM_ASSET_DIR, usd_file_name),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=True,
                max_depenetration_velocity=10.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
                fix_root_link=True,
            ),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
            fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=1.0),
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["WR.*", "(FF|MF|RF|LF|TH)J(4|3|2|1)", "(LF|TH)J5", "elbow_(rotate|bend)"],
                effort_limit_sim={
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
                    "elbow_(rotate|bend)": 40.0
                },
                stiffness={
                    "WRJ.*": 8.0,
                    "(FF|MF|RF|LF|TH)J(4|3|2|1)": 5.0,
                    "(LF|TH)J5": 5.0,
                    "elbow_(rotate|bend)": 15.0
                },
                damping={
                    "WRJ.*": 1.0,
                    "(FF|MF|RF|LF|TH)J(4|3|2|1)": 1.0,
                    "(LF|TH)J5": 1.0,
                    "elbow_(rotate|bend)": 2.0
                },
            ),
        },
        actuator_value_resolution_debug_print=False
    )
    return hand_cfg


def get_ball_cfg(prim_name, radius, pos):
    ball_cfg = RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{prim_name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
        spawn=sim_utils.SphereCfg(
            radius=radius,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.10), # 100g, stanard lightweight juggling ball
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.materials.PreviewSurfaceCfg(diffuse_color=(0.95, 0.9, 0.6)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.8,
                restitution=0.0,       # 0.0 = No bounce, like a beanbag
                friction_combine_mode="max",     # Use the stickiest value of the two touching objects
                restitution_combine_mode="min",  # Use the least bouncy value of the two touching objects
            ),
        )
    )
    return ball_cfg


@configclass
class JugglingAgentEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 7.0

    # - spaces definition
    num_balls = 1
    num_hands = 2
    action_space = 52
    # observation_space = 4
    observation_space = action_space * 3 + 3 * num_balls * 2 + 7 * num_hands # 52 joint pos + 52 joint vel + 3*num_balls ball pos + 3*num_balls ball vel + 3*num_hands pos + 4*num_hands quaternion
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        # Recommended: Boost GPU buffers for the 2048 environments + complex hands
        physx=sim_utils.PhysxCfg(
            # Enable CCD globally for the scene
            enable_ccd=True, 
            
            # Recommended: Boost GPU buffers for the 2048 environments + complex hands
            gpu_max_rigid_patch_count=10 * 2**17,
            gpu_max_rigid_contact_count=10 * 2**17,
            gpu_found_lost_pairs_capacity=10 * 2**17,
            gpu_found_lost_aggregate_pairs_capacity=10 * 2**17,
        )
    )

    # Initial Joint Position
    left_joint_pos = {
        "elbow_bend": math.radians(4.9),
        "elbow_rotate": math.radians(4.9),
        "WRJ1": math.radians(-7.0),
        "WRJ2": math.radians(0.5),
        "THJ1": math.radians(42.0),
        "THJ2": math.radians(10.0),
        "THJ3": math.radians(0.0),
        "THJ4": math.radians(40.0),
        "THJ5": math.radians(-2.0),
        "FFJ1": math.radians(37.5),
        "FFJ2": math.radians(52.0),
        "FFJ3": math.radians(-1.3),
        "FFJ4": math.radians(-18.5),
        "MFJ1": math.radians(37.1),
        "MFJ2": math.radians(55.2),
        "MFJ3": math.radians(12.1),
        "MFJ4": math.radians(14.0),
        "RFJ1": math.radians(10.9),
        "RFJ2": math.radians(35.0),
        "RFJ3": math.radians(90.0),
        "RFJ4": math.radians(-16.5),
        "LFJ1": math.radians(36.3),
        "LFJ2": math.radians(11.1),
        "LFJ3": math.radians(90.0),
        "LFJ4": math.radians(-19.9),
        "LFJ5": math.radians(6.0),
    }
    right_joint_pos = {
        "elbow_bend": math.radians(4.9),
        "elbow_rotate": math.radians(4.9),
        "WRJ1": math.radians(-6.9),
        "WRJ2": math.radians(0.0),
        "THJ1": math.radians(30.2),
        "THJ2": math.radians(39.9),
        "THJ3": math.radians(0.0),
        "THJ4": math.radians(69.9),
        "THJ5": math.radians(-20.0),
        "FFJ1": math.radians(43.1),
        "FFJ2": math.radians(13.1),
        "FFJ3": math.radians(86.0),
        "FFJ4": math.radians(-19.9),
        "MFJ1": math.radians(39.3),
        "MFJ2": math.radians(17.7),
        "MFJ3": math.radians(82.6),
        "MFJ4": math.radians(0.0),
        "RFJ1": math.radians(32.4),
        "RFJ2": math.radians(23.6),
        "RFJ3": math.radians(81.9),
        "RFJ4": math.radians(-8.0),
        "LFJ1": math.radians(52.0),
        "LFJ2": math.radians(72.5),
        "LFJ3": math.radians(90.0),
        "LFJ4": math.radians(15.0),
        "LFJ5": math.radians(4.0),
    }

    hand_pos = [(0, -0.3, 0.4), (0, 0.3, 0.4)]

    # ball spawn offsets relative to hands (x, y, z)
    # first two relative to left hand, third relative to right hand
    ball_offset = [
        (-0.36823, -0.03328, 0.0068),
        (-0.29534, 0.01543, 0.02894),
        (-0.31156, 0.0074, 0.02503),
    ]
    ball_anchor = [0, 0, 1]
    ball_radius = 0.0375
    # do we need to add ball attributes here?

    # can't use torch in configclass
    # ball_spawn_offsets = torch.tensor(ball_offset)  # (num_balls, 3)
    # ball_anchors = torch.tensor(ball_anchor)  # (num_balls, 3)
    # hand_bases = torch.tensor(hand_pos)  # (2, 3)
    # anchor_pos = hand_bases[ball_anchors]  # (num_balls, 3)
    # init_ball_pos = anchor_pos + ball_spawn_offsets

    init_ball_pos = []
    for i in range(num_balls):
        anchor = ball_anchor[i]
        pos = [
            hand_pos[anchor][0] + ball_offset[i][0],
            hand_pos[anchor][1] + ball_offset[i][1],
            hand_pos[anchor][2] + ball_offset[i][2],
        ]
        init_ball_pos.append(pos)

    # robot(s)
    # hand_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Hand")
    left_hand_cfg = get_hand_cfg("left_hand", "shadow_hand_left_with_elbow.usd",
                                 pos=hand_pos[0], rot=[-math.sqrt(2) / 2, 0, math.sqrt(2) / 2, 0],
                                 joint_pos=left_joint_pos)
    right_hand_cfg = get_hand_cfg("right_hand", "shadow_hand_right_with_elbow.usd",
                                  pos=hand_pos[1], rot=[-math.sqrt(2) / 2, 0, math.sqrt(2) / 2, 0],
                                  joint_pos=right_joint_pos)

    ball1_cfg = get_ball_cfg("ball_1", ball_radius, pos=init_ball_pos[0])
    # ball2_cfg = get_ball_cfg("ball_2", ball_radius, pos=init_ball_pos[1])
    # ball3_cfg = get_ball_cfg("ball_3", ball_radius, pos=init_ball_pos[2])

    # pdb.set_trace()

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True) # increase num envs to 4096 if you have more GPU memory

    # reward weights
    w_hoarding = 1.5        # should be high to strongly discourage hoarding 2 balls in one hand
    w_jitter = 0.05        # should be low to not overly discourage small adjustments, this is to prevent random drifting. Continuously added
    w_highest = 2.5        # the highest ball, small becaues it's continuously added
    w_rythem = 1.0         # should be moderate to encourage consistent timing
    w_catch = 50.0          # should be high to strongly encourage successful catches
    w_drop = 10.0           # should be moderate, high enough to provide guidance on where the ball should be, but smaller than catch reward

    # geometric parameters
    catch_radius = 0.0675  # radius from the hand to the ball which a catch is registered, this definitly needs to be configured before we start TODO
    target_height = 0.95 # height at which the ball apex should be, this also definitly needs to be configured before we start TODO
    target_delta_height = 0.5 # target_height - hand_height, hand height is approx 0.45m when in rest position
    target_rythem = 0.4 # 60/150 seconds per throw, i.e. 2.5 throws per second
    ground_height = 0.0
    out_of_bounds_radius = 2.0 # radius from the origin in the xy-plane, if a ball gets thrown beyond this, the episode terminates. Implimented to prevent the agent from launching balls and going "hey, no negative rewards were given, so I can just keep throwing them away"
    # tolerances
    #sigma_rythem = 0.2
    sigma_rythem = 0.1
    sigma_drop_distance = 0.1
    sigma_apex_height = 0.1
    hoarding_time_threshold = 0.1 # time threshold before hoarding penalty starts to be applied
    
    min_throw_height = 0.575 # minimum height a ball must reach to be considered a valid throw, done to prevent micro-throws. Set to 0.1 above the hand height
    min_vertical_velocity = 0.1 # minimum vertical velocity at throw time to be considered a valid throw
