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

def get_robot_cfg(prim_name, usd_file_name, pos, rot):
    robot_cfg = ArticulationCfg(
        prim_path=f"/World/envs/env_.*/{prim_name}",
        init_state=ArticulationCfg.InitialStateCfg(pos=pos, rot=rot),
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
            "arm_and_wrist": ImplicitActuatorCfg(
                # Use regex to capture both left and right arms in the single robot
                joint_names_expr=[".*_elbow_joint", ".*_wrist_.*joint"],
                stiffness=300.0,
                damping=40.0,
                # Remove effort_limit_sim and velocity_limit_sim so it inherits 
                # the true physical limits from the Unitree URDF
            ),
            "fingers_main": ImplicitActuatorCfg(
                # Matches thumb (1), index (2), and middle (3) pitch joints for both L and R
                joint_names_expr=[".*_[123].*"],
                stiffness=10.0,
                damping=1.0,
            ),
            "fingers_aux": ImplicitActuatorCfg(
                # Matches ring (4) and pinky (5) pitch joints for both L and R
                joint_names_expr=[".*_[45].*"],
                stiffness=10.0,
                damping=1.0,
            ),
            # Lock the shoulders if you don't want them moving during training
            "shoulders": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder_.*"],
                stiffness=400.0,
                damping=40.0,
            ),
        },
        actuator_value_resolution_debug_print=False
    )
    return robot_cfg

# def get_hand_cfg(prim_name, usd_file_name, pos, rot, joint_pos):
#     hand_cfg = ArticulationCfg(
#         prim_path=f"/World/envs/env_.*/{prim_name}",
#         init_state=ArticulationCfg.InitialStateCfg(
#             pos=pos,
#             rot=rot,
#             joint_pos=joint_pos,
#         ),
#         spawn=sim_utils.UsdFileCfg(
#             usd_path=os.path.join(CUSTOM_ASSET_DIR, usd_file_name),
#             activate_contact_sensors=True,
#             rigid_props=sim_utils.RigidBodyPropertiesCfg(
#                 disable_gravity=True,
#                 retain_accelerations=True,
#                 max_depenetration_velocity=2.0,
#             ),
#             articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#                 enabled_self_collisions=True,
#                 solver_position_iteration_count=12,
#                 solver_velocity_iteration_count=4,
#                 sleep_threshold=0.005,
#                 stabilization_threshold=0.0005,
#                 fix_root_link=True,
#             ),
#             joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
#             fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=1.0),
#         ),
#         actuators={
#             # "fingers": ImplicitActuatorCfg(
#             # "wrist_and_elbow": ImplicitActuatorCfg(
#             #     # joint_names_expr=["WR.*", "(FF|MF|RF|LF|TH)J(4|3|2|1)", "(LF|TH)J5", "elbow_(rotate|bend)"],
#             #     joint_names_expr=["WR.*", "elbow_(rotate|bend)"],
#             #     effort_limit_sim={
#             #         "WRJ2": 4.785,
#             #         "WRJ1": 2.175,
#             #         # "(FF|MF|RF|LF)J1": 0.7245,
#             #         # "FFJ(4|3|2)": 0.9,
#             #         # "MFJ(4|3|2)": 0.9,
#             #         # "RFJ(4|3|2)": 0.9,
#             #         # "LFJ(5|4|3|2)": 0.9,
#             #         # "THJ5": 2.3722,
#             #         # "THJ4": 1.45,
#             #         # "THJ(3|2)": 0.99,
#             #         # "THJ1": 0.81,
#             #         # "elbow_(rotate|bend)": 40.0
#             #     },
#             #     stiffness={
#             #         "WRJ.*": 6.0, 
#             #         # "(FF|MF|RF|LF|TH)J(4|3|2|1)": 0.0,
#             #         # "(LF|TH)J5": 0.0,
#             #         "elbow_(rotate|bend)": 10.0
#             #     },
#             #     damping={
#             #         "WRJ.*": 0.8,
#             #         # "(FF|MF|RF|LF|TH)J(4|3|2|1)": 0.05,
#             #         # "(LF|TH)J5": 0.05,
#             #         "elbow_(rotate|bend)": 2.0
#             #     },
#             # ),
#             "elbows": ImplicitActuatorCfg(
#                 joint_names_expr=["elbow_(rotate|bend)"],
#                 effort_limit_sim=50.0, # Elbows are strong
#                 stiffness=100.0,
#                 damping=10.0,           # Higher damping for stability
#                 velocity_limit_sim=5.0,    # Limit velocity for stability
#             ),
#             "wrists": ImplicitActuatorCfg(
#                 joint_names_expr=["WRJ.*"],
#                 effort_limit_sim={
#                     "WRJ2": 5.785,
#                     "WRJ1": 2.175,
#                 },
#                 stiffness=40.0,
#                 damping=2.00,
#                 velocity_limit_sim=8.0,
#             ),
#             "fingers": ImplicitActuatorCfg(
#                 joint_names_expr=[
#                     # All finger joints
#                     "(FF|MF|RF|LF|TH)J(5|4|3|2|1)",
#                 ],
#                 # A single "grasp" torque applied across all finger joints
#                 effort_limit_sim={
#                     "(FF|MF|RF|LF)J1": 0.7245,
#                     "FFJ(4|3|2)": 0.9,
#                     "MFJ(4|3|2)": 0.9,
#                     "RFJ(4|3|2)": 0.9,
#                     "LFJ(5|4|3|2)": 0.9,
#                     "THJ5": 2.3722,
#                     "THJ4": 1.45,
#                     "THJ(3|2)": 0.99,
#                     "THJ1": 0.81,
#                 },
#                 stiffness={
#                     "(FF|MF|RF|LF|TH)J(4|3|2|1)": 10.0,
#                     "(LF|TH)J5": 10.0,
#                 },
#                 damping={
#                     "(FF|MF|RF|LF|TH)J(4|3|2|1)": 0.5,
#                     "(LF|TH)J5": 0.5,
#                 },
#                 velocity_limit_sim=5.0,
#             ),
#         },
#         actuator_value_resolution_debug_print=False
#     )
#     return hand_cfg


ball_radius = 0.03175 #the balls we have are 6.35cm in diameter
def get_ball_cfg(prim_name, radius, pos):
    ball_cfg = RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{prim_name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
        spawn=sim_utils.SphereCfg(
            radius=radius,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.13), # 130g, standard juggling ball
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=1.0,
                linear_damping=0.01, # slight air resistance for stability, since we're not modeling ball deformation which would provide additional drag
                angular_damping=0.01, # slight air resistance for stability
                #enable_ccd=True,
            ),
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
    decimation = 4
    episode_length_s = 5.0

    num_balls = 1
    should_cross = (num_balls % 2 == 1)  # cross if odd number of balls
    num_hands = 2

    # values taken directly from the unitree g1 urdf, don't modify
    arm_dof = 7 # shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_joint,  wrist_pitch, wrist_roll, wrist_yaw)
    arm_joints = 7

    # Dex5-1P
    finger_dof = 4
    thumb_dof = 4
    total_finger_joints = 20

    # Inspire hand
    # finger_dof = 1
    # thumb_dof = 2
    # total_finger_joints = 12 # 2 per finger, 4 in thumb

    num_joints = (arm_joints + total_finger_joints) * 2

    # modifiable properties
    finger_groups = 2 # {thumb, index, middle}, {ring, pinky}
    finger_controlable_dof = 1 # control all finger joints with a single action per finger group

    action_space = arm_dof + finger_groups*finger_controlable_dof # only learn for a single hand, we then mirror the actions to the other hand
    observation_space = num_joints + num_joints + action_space + 3 * num_balls * 2 + 3 * num_hands + 4 * num_hands # joint pos + joint vel + action + 3*num_balls ball pos + 3*num_balls ball vel + 3*num_hands pos + 4*num_hands quaternion
    # observation_space = 137
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physx=sim_utils.PhysxCfg(
            # Enable CCD globally for the scene
            enable_ccd=True, 
            # Boost GPU buffers for the larger environments
            gpu_max_rigid_patch_count=10 * 2**20,
            gpu_max_rigid_contact_count=10 * 2**20,
            gpu_found_lost_pairs_capacity=10 * 2**20,
            gpu_found_lost_aggregate_pairs_capacity=10 * 2**20,
        )
    )

    # Shadow hand positions:
    # flat_joint_pos = {
    #     # WRIST: Keep neutral or slightly extended to flatten the palm relative to the arm
    #     "WRJ1": 0.0,
    #     "WRJ2": 0.0, 
        
    #     # THUMB: Abduct (Spread out) to make a wider shelf, but don't curl.
    #     "THJ1": math.radians(60.0), # Spread out
    #     "THJ2": math.radians(0.0),  # No flexion
    #     "THJ3": math.radians(0.0),
    #     "THJ4": math.radians(0.0),
    #     "THJ5": math.radians(0.0),

    #     # FINGERS: All 0.0 (Fully Extended Flat)
    #     # J1 is Abduction (Spread), J2-J4 are Flexion (Curl)
    #     "FFJ1": math.radians(0.0), "FFJ2": 0.0, "FFJ3": 0.0, "FFJ4": 0.0,
    #     "MFJ1": math.radians(0.0), "MFJ2": 0.0, "MFJ3": 0.0, "MFJ4": 0.0,
    #     "RFJ1": math.radians(0.0), "RFJ2": 0.0, "RFJ3": 0.0, "RFJ4": 0.0,
    #     "LFJ1": math.radians(0.0), "LFJ2": 0.0, "LFJ3": 0.0, "LFJ4": 0.0, "LFJ5": 0.0,
        
    #     "elbow_bend": math.radians(0.0),
    #     "elbow_rotate": math.radians(0.0),
    # }
    # closed_joint_pos = {
    #     "THJ1": 1.25, # thumb tip, full flextion interferes with fingers
    #     "THJ2": 0.697,
    #     "THJ3": 0.208,
    #     "THJ4": 1.2, #0,7 # thumb base flexation upwards
    #     "THJ5": -0.5, #thumb base flexation inwards, want it more outwards so it dosen't interfere with fingers

    #     # FINGERS:
    #     "FFJ1": 0.6, "FFJ2": 0.7, "FFJ3": 1.57, "FFJ4": 0.0,
    #     "MFJ1": 0.6, "MFJ2": 0.7, "MFJ3": 1.57, "MFJ4": 0.0,
    #     "RFJ1": 0.6, "RFJ2": 0.7, "RFJ3": 1.57, "RFJ4": -0.2,
    #     # pinky
    #     "LFJ1": 0.6, "LFJ2": 0.5, "LFJ3": 1.57, "LFJ4": -0.349, "LFJ5": 0.2,
    # }

    # Dex5-1 joint positions:
    flat_joint_pos = {
        "Yaw_11.*": math.radians(38.977), "Roll_12.*": 0.0, "Pitch_13.*": 0.0, "Pitch_14.*": 0.0, # thumb, abduct (Spread out) to make a wider shelf, but don't curl.
        "Roll_21.*": 0.0, "Pitch_22.*": 0.0, "Pitch_23.*": 0.0, "Pitch_24.*": 0.0, # index
        "Roll_31.*": 0.0, "Pitch_32.*": 0.0, "Pitch_33.*": 0.0, "Pitch_34.*": 0.0, # middle
        "Roll_41.*": math.radians(-22.0), "Pitch_42.*": 0.0, "Pitch_43.*": 0.0, "Pitch_44.*": 0.0, # ring 
        "Roll_51.*": math.radians(-22.0), "Pitch_52.*": 0.0, "Pitch_53.*": 0.0, "Pitch_54.*": 0.0, # pinky
    }
    closed_joint_pos = {
        "Yaw_11.*": math.radians(38.0), "Roll_12.*": math.radians(-103.9), "Pitch_13.*": math.radians(20.0), "Pitch_14.*": math.radians(50.0), # thumb
        "Roll_21.*": 0.0, "Pitch_22.*": math.radians(90.0), "Pitch_23.*": math.radians(90.0), "Pitch_24.*": math.radians(40.0), # index
        "Roll_31.*": 0.0, "Pitch_32.*": math.radians(90.0), "Pitch_33.*": math.radians(90.0), "Pitch_34.*": math.radians(40.0), # middle
        "Link_41.*": math.radians(-22.0), "Pitch_42.*": math.radians(90.0), "Pitch_43.*": math.radians(90.0), "Pitch_44.*": math.radians(40.0), # ring 
        "Roll_51.*": math.radians(-22.0), "Pitch_52.*": math.radians(90.0), "Pitch_53.*": math.radians(90.0), "Pitch_54.*": math.radians(40.0), # pinky
    }

    left_joint_pos = flat_joint_pos
    right_joint_pos = flat_joint_pos

    hand_pos = [(0, -0.25, 0.4), (0, 0.25, 0.4)]

    # ball spawn offsets relative to hands (x, y, z)
    # first two relative to left hand, third relative to right hand
    ball_offset = [
        (0.0, 0.0, 0.1),
        #(-0.33, 0.0, 0.055),#(-0.36823, -0.03328, 0.0068),
        # (-0.29534, 0.01543, 0.02894),
        # (-0.3, 0.00, 0.055),
    ]
    ball_anchor = [0, 0, 1]

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
    # left_hand_cfg = get_hand_cfg("left_hand", "shadow_hand_left_with_elbow.usd",
    #                              pos=hand_pos[0], rot=[-math.sqrt(2) / 2, 0, math.sqrt(2) / 2, 0],
    #                              joint_pos=left_joint_pos)
    robot_cfg = get_robot_cfg(
        prim_name="g1_robot", 
        usd_file_name="g1_with_dex5_hands.usd", # Replace with your actual merged USD name
        pos=[0.0, 0.0, 0.8], # Height of the G1 torso in the air
        rot=[1.0, 0.0, 0.0, 0.0] 
    )

    ball1_cfg = get_ball_cfg("ball_1", ball_radius, pos=init_ball_pos[0])
    # for ball in range(num_balls):
    #     ball_cfg = get_ball_cfg(f"ball_{ball+1}", ball_radius, pos=init_ball_pos[ball])
    

    target_ball_paths = [f"/World/envs/env_.*/ball_{i+1}" for i in range(num_balls)]
    # sensors
    contact_sensor_left: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/g1_robot/.*/Dex5_URDF_L/*",
        history_length=1,
        track_air_time=False,
        # Only report contact if the other object is a ball
        filter_prim_paths_expr=target_ball_paths, 
        debug_vis=False,
    )
    contact_sensor_right: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/g1_robot/.*/Dex5_URDF_R/*",
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=target_ball_paths, 
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16384, env_spacing=4.0, replicate_physics=True)

    # reward weights
    # Continuous Penalties
    w_hoarding = 20.0        # should be high to strongly discourage hoarding 2 balls in one hand
    w_jitter = 0.1        # should be low to not overly discourage small adjustments, this is to prevent random drifting. Continuously added
    w_hands_touching = 15.0 # hands touching is very bad, high penalty

    # Discrete Penalties
    w_drop = 1.0           # flat penalty to prevent the agent from giving up and dropping the ball
    w_lazy_drop = 5.0

    # Discrete Rewards
    w_wide_delta_throw = 30.0     # reward for throwing the ball up, larger sigma to allow for early learning, and since the real goal is apex height
    w_wide_accuracy = 10.0       # reward for throwing the ball to where the target hand will be at catch time

    w_precise_delta_throw = 100.0
    w_precise_accuracy = 50.0
    w_apex_height = 100.0

    #w_rythem = # should be moderate to encourage consistent timing
    w_catch = 500.0          # main goal, paid out over the next catch_payout_time steps. should be high to strongly encourage successful catches

    jitter_joint_weights = {
        ".*_shoulder_.*": 10.0,
        ".*_elbow_.*": 5.0,
        ".*_wrist_.*": 0.5,
        # any lower body joint: 25.0 
        
        # Dex5 Fingers (Format: [JointType]_[FingerNum][JointNum][Side])
        # Example: Pitch_14L is Thumb, Roll_51R is Pinky.
        # Base finger weight (0.1) * Finger Scalar
        ".*_1[1-4][LR]": 0.1 * 1.0,  # Thumb 
        ".*_2[1-4][LR]": 0.1 * 1.0,  # Index 
        ".*_3[1-4][LR]": 0.1 * 1.0,  # Middle
        ".*_4[1-4][LR]": 0.1 * 1.0,  # Ring  
        ".*_5[1-4][LR]": 0.1 * 1.0,  # Pinky 
    }

    # geometric parameters
    target_height = 0.9 #1.0# height at which the ball apex should be
    target_delta_height = 0.45 # target_height - hand_height - tollerance for the windup
    #target_rythem = 0.4 # 60/150 seconds per throw, i.e. 2.5 throws per second
    ground_height = 0.0
    out_of_bounds_radius = 2.0 # radius from the origin in the xy-plane, if a ball gets thrown beyond this, the episode terminates. Implimented to prevent the agent from launching balls and going "hey, no negative rewards were given, so I can just keep throwing them away"
    
    # hand specific parameters, needs to be changed if the hand model is changed
    center_of_hand_bias = 0.775 #center offset towards knuckles for a more accurate center of hand position
    spawn_randomized_offset_range_x = 0.045 # 4.5cm, random offset applied to ball spawn position to prevent overfitting
    spawn_randomized_offset_range_y = 0.0225   # 2.25cm
    spawn_randomized_offset_range_z = 0.0   # 0cm, keep z consistent to prevent dropping in/clipping issues

    action_scale = 1.5
    action_smoothing = 0.5

    num_ball_up_reward_scale = [1.0, 0.1, 0.0, 0.0] # how much the throw rewards are scaled based on the number of balls currently going up. 0,1,2,3


    # tolerances
    sigma_catch_height = 0.05
    #sigma_rythem = 0.1
    sigma_apex_height = 0.05
    sigma_wide_delta_throw = 0.8
    sigma_delta_throw = 0.3
    sigma_wide_throw_accuracy = 0.55
    sigma_throw_accuracy = 0.075
    sigma_delta_throw_catch_wide = 0.25
    sigma_delta_throw_catch = 0.1

    # thresholds and limits
    catch_payout_time = 0.25         # how long after a catch the catch reward is paid out over, designed to spread the reward out so the agent holds onto the ball and controls the catch
    hoarding_time_threshold = 0.75   # time threshold before hoarding penalty starts to be applied
    start_hold_threshold = 0.75

    contact_threshold = 0.01      # in Newtons, sensor contact threshold to consider a ball "in contact" with the hand

    catch_precise_height_ratio = 0.95
    min_delta_throw_height = -0.1    # minimum height difference between throw and catch to be considered a valid throw
    min_hand_dist = 0.2             # radius around each hand which the other hand should not enter
    min_throw_velocity = 0.25
    min_vertical_velocity = -0.1    # minimum vertical velocity at throw time to be considered a valid throw
    above_hand_threshold = 0.05        # height above hand to consider ball "above" the hand for catching purposes
