# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os
import math
from pathlib import Path

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg
from isaaclab.actuators import ImplicitActuatorCfg

import isaaclab.sim as sim_utils

# set this path in the Path-to-IsaacLab/source/isaaclab/isaaclab/utils/assets.py
# from isaaclab.utils.assets import CUSTOM_ASSET_DIR
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
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=True,
                max_depenetration_velocity=2.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=4,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
                fix_root_link=True,
            ),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
            fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=1.0),
        ),
        actuators={
            # "fingers": ImplicitActuatorCfg(
            # "wrist_and_elbow": ImplicitActuatorCfg(
            #     # joint_names_expr=["WR.*", "(FF|MF|RF|LF|TH)J(4|3|2|1)", "(LF|TH)J5", "elbow_(rotate|bend)"],
            #     joint_names_expr=["WR.*", "elbow_(rotate|bend)"],
            #     effort_limit_sim={
            #         "WRJ2": 4.785,
            #         "WRJ1": 2.175,
            #         # "(FF|MF|RF|LF)J1": 0.7245,
            #         # "FFJ(4|3|2)": 0.9,
            #         # "MFJ(4|3|2)": 0.9,
            #         # "RFJ(4|3|2)": 0.9,
            #         # "LFJ(5|4|3|2)": 0.9,
            #         # "THJ5": 2.3722,
            #         # "THJ4": 1.45,
            #         # "THJ(3|2)": 0.99,
            #         # "THJ1": 0.81,
            #         # "elbow_(rotate|bend)": 40.0
            #     },
            #     stiffness={
            #         "WRJ.*": 6.0, 
            #         # "(FF|MF|RF|LF|TH)J(4|3|2|1)": 0.0,
            #         # "(LF|TH)J5": 0.0,
            #         "elbow_(rotate|bend)": 10.0
            #     },
            #     damping={
            #         "WRJ.*": 0.8,
            #         # "(FF|MF|RF|LF|TH)J(4|3|2|1)": 0.05,
            #         # "(LF|TH)J5": 0.05,
            #         "elbow_(rotate|bend)": 2.0
            #     },
            # ),
            "elbows": ImplicitActuatorCfg(
                joint_names_expr=["elbow_(rotate|bend)"],
                effort_limit_sim=40.0, # Elbows are strong
                stiffness=10.0,
                damping=2.0,           # Higher damping for stability
            ),
            "wrists": ImplicitActuatorCfg(
                joint_names_expr=["WRJ.*"],
                effort_limit_sim={
                    "WRJ2": 4.785,
                    "WRJ1": 2.175,
                },
                stiffness=6.0,
                damping=0.5,
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[
                    # All finger joints
                    "(FF|MF|RF|LF|TH)J(5|4|3|2|1)",
                ],
                # A single "grasp" torque applied across all finger joints
                effort_limit_sim={
                    "(FF|MF|RF|LF)J1": 0.7245,
                    "FFJ(4|3|2)": 0.9,
                    "MFJ(4|3|2)": 0.9,
                    "RFJ(4|3|2)": 0.9,
                    "LFJ(5|4|3|2)": 0.9,
                    "THJ5": 2.3722,
                    "THJ4": 1.45,
                    "THJ(3|2)": 0.99,
                    "THJ1": 0.81,
                },
                stiffness={
                    "(FF|MF|RF|LF|TH)J(4|3|2|1)": 4.0,
                    "(LF|TH)J5": 5.0,
                },
                damping={
                    "(FF|MF|RF|LF|TH)J(4|3|2|1)": 0.2,
                    "(LF|TH)J5": 0.2,
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
    decimation = 1
    episode_length_s = 7.0

    num_balls = 1
    num_hands = 2
    action_space = num_hands * 5 # 5 actions per hand, 2 for elbow, 2 for wrist, 1 for open/close fingers
    observation_space = 52 * 2 + 10 + 3 * num_balls * 2 + 7 * num_hands # 52 joint pos + 52 joint vel + 3*num_balls ball pos + 3*num_balls ball vel + 3*num_hands pos + 4*num_hands quaternion
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physx=sim_utils.PhysxCfg(
            # Enable CCD globally for the scene
            enable_ccd=True, 
            # Boost GPU buffers for the larger environments
            gpu_max_rigid_patch_count=10 * 2**19,
            gpu_max_rigid_contact_count=10 * 2**19,
            gpu_found_lost_pairs_capacity=10 * 2**19,
            gpu_found_lost_aggregate_pairs_capacity=10 * 2**19,
        )
    )

    flat_joint_pos = {
        # WRIST: Keep neutral or slightly extended to flatten the palm relative to the arm
        "WRJ1": 0.0,
        "WRJ2": 0.0, 
        
        # THUMB: Abduct (Spread out) to make a wider shelf, but don't curl.
        "THJ1": math.radians(60.0), # Spread out
        "THJ2": math.radians(0.0),  # No flexion
        "THJ3": math.radians(0.0),
        "THJ4": math.radians(0.0),
        "THJ5": math.radians(0.0),

        # FINGERS: All 0.0 (Fully Extended Flat)
        # J1 is Abduction (Spread), J2-J4 are Flexion (Curl)
        "FFJ1": math.radians(0.0), "FFJ2": 0.0, "FFJ3": 0.0, "FFJ4": 0.0,
        "MFJ1": math.radians(0.0), "MFJ2": 0.0, "MFJ3": 0.0, "MFJ4": 0.0,
        "RFJ1": math.radians(0.0), "RFJ2": 0.0, "RFJ3": 0.0, "RFJ4": 0.0,
        "LFJ1": math.radians(0.0), "LFJ2": 0.0, "LFJ3": 0.0, "LFJ4": 0.0, "LFJ5": 0.0,
        
        "elbow_bend": math.radians(0.0),
        "elbow_rotate": math.radians(0.0),
    }

    # Initial Joint Position
    # left_joint_pos = {
    #     "elbow_bend": math.radians(4.9),
    #     "elbow_rotate": math.radians(4.9),
    #     "WRJ1": math.radians(-7.0),
    #     "WRJ2": math.radians(0.5),
    #     "THJ1": math.radians(42.0),
    #     "THJ2": math.radians(10.0),
    #     "THJ3": math.radians(0.0),
    #     "THJ4": math.radians(40.0),
    #     "THJ5": math.radians(-2.0),
    #     "FFJ1": math.radians(37.5),
    #     "FFJ2": math.radians(52.0),
    #     "FFJ3": math.radians(-1.3),
    #     "FFJ4": math.radians(-18.5),
    #     "MFJ1": math.radians(37.1),
    #     "MFJ2": math.radians(55.2),
    #     "MFJ3": math.radians(12.1),
    #     "MFJ4": math.radians(14.0),
    #     "RFJ1": math.radians(10.9),
    #     "RFJ2": math.radians(35.0),
    #     "RFJ3": math.radians(90.0),
    #     "RFJ4": math.radians(-16.5),
    #     "LFJ1": math.radians(36.3),
    #     "LFJ2": math.radians(11.1),
    #     "LFJ3": math.radians(90.0),
    #     "LFJ4": math.radians(-19.9),
    #     "LFJ5": math.radians(6.0),
    # }
    # right_joint_pos = {
    #     "elbow_bend": math.radians(4.9),
    #     "elbow_rotate": math.radians(4.9),
    #     "WRJ1": math.radians(-6.9),
    #     "WRJ2": math.radians(0.0),
    #     "THJ1": math.radians(30.2),
    #     "THJ2": math.radians(39.9),
    #     "THJ3": math.radians(0.0),
    #     "THJ4": math.radians(69.9),
    #     "THJ5": math.radians(-20.0),
    #     "FFJ1": math.radians(43.1),
    #     "FFJ2": math.radians(13.1),
    #     "FFJ3": math.radians(86.0),
    #     "FFJ4": math.radians(-19.9),
    #     "MFJ1": math.radians(39.3),
    #     "MFJ2": math.radians(17.7),
    #     "MFJ3": math.radians(82.6),
    #     "MFJ4": math.radians(0.0),
    #     "RFJ1": math.radians(32.4),
    #     "RFJ2": math.radians(23.6),
    #     "RFJ3": math.radians(81.9),
    #     "RFJ4": math.radians(-8.0),
    #     "LFJ1": math.radians(52.0),
    #     "LFJ2": math.radians(72.5),
    #     "LFJ3": math.radians(90.0),
    #     "LFJ4": math.radians(15.0),
    #     "LFJ5": math.radians(4.0),
    # }
    left_joint_pos = flat_joint_pos
    right_joint_pos = flat_joint_pos

    hand_pos = [(0, -0.3, 0.4), (0, 0.3, 0.4)]

    # ball spawn offsets relative to hands (x, y, z)
    # first two relative to left hand, third relative to right hand
    ball_offset = [
        (-0.33, 0.0, 0.055),#(-0.36823, -0.03328, 0.0068),
        (-0.29534, 0.01543, 0.02894),
        (-0.3, 0.00, 0.055),
    ]
    ball_anchor = [0, 0, 1]
    ball_radius = 0.0375

    init_ball_pos = []
    for i in range(num_balls):
        anchor = ball_anchor[i]
        pos = [
            hand_pos[anchor][0] + ball_offset[i][0],
            hand_pos[anchor][1] + ball_offset[i][1],
            hand_pos[anchor][2] + ball_offset[i][2],
        ]
        init_ball_pos.append(pos)

    # asset configs
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

    # sensors
    contact_sensor_left_palm: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/left_hand/palm",
        history_length=1,
        track_air_time=False,
        # Only report contact if the other object is a ball
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"], 
        debug_vis=False,
    )
    contact_sensor_left_metacarpal: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/left_hand/.*metacarpal", #
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_left_thumb_proximal: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/left_hand/thproximal",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_left_thumb_middle: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/left_hand/thmiddle",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_left_thumb_distal: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/left_hand/thdistal",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_left_index_proximal: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/left_hand/ffproximal",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_left_index_middle: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/left_hand/ffmiddle",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_left_index_distal: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/left_hand/ffdistal",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_left_middle_proximal: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/left_hand/mfproximal",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_left_middle_middle: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/left_hand/mfmiddle",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_left_middle_distal: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/left_hand/mfdistal",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_left_ring_proximal: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/left_hand/rfproximal",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_left_ring_middle: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/left_hand/rfmiddle",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_left_ring_distal: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/left_hand/rfdistal",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_left_pinky_proximal: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/left_hand/lfproximal",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_left_pinky_middle: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/left_hand/lfmiddle",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_left_pinky_distal: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/left_hand/lfdistal",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )


    contact_sensor_right_palm: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/right_hand/palm",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_right_metacarpal: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/right_hand/.*metacarpal",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_right_thumb_proximal: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/right_hand/thproximal",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_right_thumb_middle: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/right_hand/thmiddle",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_right_thumb_distal: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/right_hand/thdistal",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_right_index_proximal: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/right_hand/ffproximal",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_right_index_middle: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/right_hand/ffmiddle",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_right_index_distal: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/right_hand/ffdistal",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_right_middle_proximal: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/right_hand/mfproximal",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_right_middle_middle: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/right_hand/mfmiddle",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_right_middle_distal: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/right_hand/mfdistal",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_right_ring_proximal: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/right_hand/rfproximal",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_right_ring_middle: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/right_hand/rfmiddle",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_right_ring_distal: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/right_hand/rfdistal",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_right_pinky_proximal: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/right_hand/lfproximal",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_right_pinky_middle: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/right_hand/lfmiddle",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )
    contact_sensor_right_pinky_distal: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/right_hand/lfdistal",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/ball_.*"],
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8192, env_spacing=4.0, replicate_physics=True) # increase num envs to 4096 if you have more GPU memory

    # reward weights
    # Continuous Penalties
    w_hoarding = 30.0        # should be high to strongly discourage hoarding 2 balls in one hand
    w_jitter = 0.01        # should be low to not overly discourage small adjustments, this is to prevent random drifting. Continuously added
    w_hands_touching = 20.0 # hands touching is very bad, high penalty

    # Discrete Penalties
    w_drop = 30.0           # moderate penalty that lowers depending on how far from the hand the ball is dropped
    w_drift = 25.0         # penatly for throwing forwards/backwards. Strong to encourage throws on the y-axis

    # Continuous Rewards
    w_catch_prediction = 0.5

    # Discrete Rewards
    w_delta_throw = 75.0     # reward for throwing the ball up, larger sigma to allow for early learning, and since the real goal is apex height
    w_accuracy = 25.0       # reward for throwing the ball to where the target hand will be at catch time
    w_apex_height = 100.0    # should be high to encourage throwing the ball up to the target height. The trick to consistent juggling is getting a consistent apex height
    w_throw_power = 5.0
    w_rythem = 0.0# 1.0 useless with 1 ball        # should be moderate to encourage consistent timing
    w_catch = 250.0          # main goal, paid out over the next catch_payout_time steps. should be high to strongly encourage successful catches
    


    # geometric parameters
    target_height = 1.0#0.95 # height at which the ball apex should be
    target_delta_height = 0.5 # target_height - hand_height, hand height is approx 0.45m when in rest position
    target_rythem = 0.4 # 60/150 seconds per throw, i.e. 2.5 throws per second
    ground_height = 0.0
    out_of_bounds_radius = 2.0 # radius from the origin in the xy-plane, if a ball gets thrown beyond this, the episode terminates. Implimented to prevent the agent from launching balls and going "hey, no negative rewards were given, so I can just keep throwing them away"
    center_of_hand_bias = 0.775 #center offset towards knuckles for a more accurate center of hand position
    spawn_randomized_offset_range_x = 0.045 # 4.5cm, random offset applied to ball spawn position to prevent overfitting
    spawn_randomized_offset_range_y = 0.0225   # 2.25cm
    spawn_randomized_offset_range_z = 0.0   # 0cm, keep z consistent to prevent dropping in/clipping issues
    action_scale = 1.5


    # tolerances
    sigma_catch_position = 0.1
    sigma_catch_height = 0.2
    sigma_rythem = 0.1
    sigma_drop_distance = 0.25
    sigma_apex_height = 0.15#0.1
    sigma_delta_throw = 0.8
    sigma_throw_accuracy = 0.25

    # thresholds and limits
    catch_payout_time = 0.1         # how long after a catch the catch reward is paid out over, designed to spread the reward out so the agent holds onto the ball and controls the catch
    hoarding_time_threshold = 0.5   # time threshold before hoarding penalty starts to be applied
    
    close_threshold = 0.2         # threshold for finger joint to be considered "closed"
    open_threshold = -0.2           # threshold for finger joint to be considered "open"
    contact_threshold = 0.05      # in Newtons, sensor contact threshold to consider a ball "in contact" with the hand

    max_catch_height = 0.525
    max_drift_velocity = 0.2
    max_drop_penalty = -1.0
    min_drop_penalty = -0.1
    max_speed_reward = 3.0
    min_delta_throw_height = -0.1    # minimum height difference between throw and catch to be considered a valid throw
    #min_delta_throw_height = 0.1    # minimum height a ball must reach to be considered a valid throw, done to prevent micro-throws.
    min_hand_dist = 0.4             # radius around each hand which the other hand should not enter
    min_throw_velocity = 1.0
    min_vertical_velocity = -0.1    # minimum vertical velocity at throw time to be considered a valid throw
