# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .juggling_agent_env_cfg import JugglingAgentEnvCfg


def assign_bias(bias, idx_list):
    return [x + bias for x in idx_list]


class JugglingAgentEnv(DirectRLEnv):
    cfg: JugglingAgentEnvCfg

    def __init__(self, cfg: JugglingAgentEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # I think setup scene is already called in the super init so this should be fine
        device = self.device

        # import pdb;
        # pdb.set_trace()

        self.left_hand_idx, _ =  self.left_hand.find_joints(".*")
        self.right_hand_idx, _ = self.right_hand.find_joints(".*")

        # create bias for each obj
        # self.left_hand_bias = 0
        # self.right_hand_bias = len(self.left_hand_idx)

        # create ball view
        # if self.sim.physics_sim_view is None:
        #     self.sim.reset()
        # self.ball_view = self.sim.physics_sim_view.create_rigid_body_view("/World/envs/env_*/ball_*")
        # if self.sim.physics_sim_view is None:
        #     self.sim.reset()
        # self.ball_view = self.sim.physics_sim_view.create_rigid_body_view("/World/envs/env_*/ball_*")

        # build the initial joint pos
        self.init_left_joint_pos = self._build_init_joint_pose(self.left_hand, self.cfg.left_joint_pos)
        self.init_right_joint_pos = self._build_init_joint_pose(self.right_hand, self.cfg.right_joint_pos)
        
        self.left_limits = self.left_hand.data.joint_effort_limits
        self.right_limits = self.right_hand.data.joint_effort_limits

        self.ball_spawn_offsets = torch.tensor(self.cfg.ball_offset, device=self.device, dtype=torch.float32)
        self.ball_anchors = torch.tensor(self.cfg.ball_anchor, device=self.device, dtype=torch.float32)

        wrist_L_ids, _ = self.left_hand.find_bodies(".*palm")
        wrist_R_ids, _ = self.right_hand.find_bodies(".*palm")
        
        self.left_wrist_idx = wrist_L_ids[0]
        self.right_wrist_idx = wrist_R_ids[0]

        # 2. Find Knuckle (Top of Palm)         # middle finger base connection is better reference for hand center
        knuckle_L_ids, _ = self.left_hand.find_bodies(".*mfproximal")
        knuckle_R_ids, _ = self.right_hand.find_bodies(".*mfproximal")
        
        self.left_knuckle_idx = knuckle_L_ids[0]
        self.right_knuckle_idx = knuckle_R_ids[0]

        self.init_ball_pos = torch.tensor(self.cfg.init_ball_pos, device=device, dtype=torch.float32)
        # assert self.cfg.action_space == len(self.left_hand_idx) + len(self.right_hand_idx), 'action dim mismatch'

        self.reward_buffer = torch.zeros(self.num_envs, device=device)

        self.hoarding_threshold = 0 if self.cfg.num_balls == 1 else 1

        self.action_dim = self.cfg.action_space
        self.actions = torch.zeros((self.num_envs, self.action_dim), device=device)
        self.prev_actions = torch.zeros((self.num_envs, self.action_dim), device=device)
        self.actions_smooth = torch.zeros((self.num_envs, self.action_dim), device=device)

        self.joint_pos = torch.zeros([self.num_envs, self.action_dim], device=device)
        self.joint_vel = torch.zeros([self.num_envs, self.action_dim], device=device)

        # initialize scalar buffers to avoid reallocation
        self.distance_target2ground = torch.tensor(1.0 / (self.cfg.target_height - self.cfg.ground_height), device=device) # precalulate for efficiency
        self.cross_pos = torch.tensor(1.0, device=device)
        self.cross_neg = torch.tensor(-0.25, device=device) # adjust penalty for same hand catch, may inhibit learning

        # precomput parts of gaussian reward
        self.sigma_apex_height_coeff = torch.tensor(-1.0 / (2 * (self.cfg.sigma_apex_height ** 2)), device=device)
        self.sigma_drop_distance_coeff = torch.tensor(-1.0 / (2 * (self.cfg.sigma_drop_distance ** 2)), device=device)
        self.sigma_rythem_coeff = torch.tensor(-1.0 / (2 * (self.cfg.sigma_rythem ** 2)), device=device)

        self._allocate_tensors()

    def _setup_scene(self):
        self.left_hand = Articulation(self.cfg.left_hand_cfg)
        self.right_hand = Articulation(self.cfg.right_hand_cfg)

        self.ball1 = RigidObject(self.cfg.ball1_cfg)
        # self.ball2 = RigidObject(self.cfg.ball2_cfg)
        # self.ball3 = RigidObject(self.cfg.ball3_cfg)

        self.scene.articulations["left_hand"] = self.left_hand
        self.scene.articulations["right_hand"] = self.right_hand

        self.scene.rigid_objects["ball1"] = self.ball1
        # self.scene.rigid_objects["ball2"] = self.ball2
        # self.scene.rigid_objects["ball3"] = self.ball3

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # spawn balls in the source env and clone to others
        # ball_cfg = sim_utils.SphereCfg( # we already do this in the config class
        #     radius=self.cfg.ball_radius,
        #     mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        #     rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        #     collision_props=sim_utils.CollisionPropertiesCfg(),
        #     visual_material=sim_utils.materials.PreviewSurfaceCfg(diffuse_color=(0.95, 0.9, 0.6)),
        # )
        # for i in range(self.cfg.num_balls):
        #     ball_cfg.func(f"/World/envs/env_.*/ball_{i}", ball_cfg, translation=(0.0, 0.0, 0.0))

        # self.balls = [self.ball1, self.ball2, self.ball3]
        self.balls = [self.ball1]

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # print('='*20)
        # print(self.ball1.root_physx_view.get_transforms())
        # print(self.ball1.root_physx_view.get_transforms().shape) (num_env, 7)
        # print('='*20)
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        #print("Applying action:", self.actions)
        # 1. Hyperparameter: Smoothing Factor (Alpha)
        # 0.0 = Frozen, 1.0 = No Smoothing. 
        alpha = 0.8
        
        # 2. Apply Low-Pass Filter
        current_action = self.actions
        # if not hasattr(self, 'actions_smooth'):
        #     self.actions_smooth = torch.zeros_like(current_action)
            
        self.actions_smooth = alpha * current_action + (1.0 - alpha) * self.actions_smooth

        # 3. Slice and Apply the SMOOTH actions
        num_left = len(self.left_hand_idx)
        
        # Use actions_smooth here instead of self.actions!
        left_action = self.actions_smooth[:, :num_left] * self.left_limits
        right_action = self.actions_smooth[:, num_left:] * self.right_limits

        self.left_hand.set_joint_effort_target(left_action, joint_ids=self.left_hand_idx)
        self.right_hand.set_joint_effort_target(right_action, joint_ids=self.right_hand_idx)
        # self.left_hand.set_joint_effort_target(
        #     self.actions[:, assign_bias(self.left_hand_bias, self.left_hand_idx)] * 1, joint_ids=self.left_hand_idx)
        # self.right_hand.set_joint_effort_target(
        #     self.actions[:, assign_bias(self.right_hand_bias, self.right_hand_idx)] * 1, joint_ids=self.right_hand_idx)
        # []
    
    def _allocate_tensors(self):

        num_envs = self.num_envs
        device = self.device #I think this is defined automatically in the parent class

        # Ball state tensors
        self.ball_pos = torch.zeros((num_envs, self.cfg.num_balls, 3), device=device) # 3 is for x,y,z
        self.ball_vel = torch.zeros((num_envs, self.cfg.num_balls, 3), device=device)
        
        # Hand state tensors
        self.hand_pos = torch.zeros((num_envs, self.cfg.num_hands, 3), device=device)

        # Event tracking tensors
        self.catch_events = torch.full((num_envs, self.cfg.num_balls), -1, device=device, dtype=torch.bool)  # -1 means no catch
        self.drop_events = torch.full((num_envs, self.cfg.num_balls), -1, device=device, dtype=torch.bool)   # -1 means no drop
        self.throw_events = torch.full((num_envs, self.cfg.num_balls), -1, device=device, dtype=torch.bool)  # -1 means no throw

        # Height tracking tensors
        self.ball_peak_height = torch.zeros((num_envs, self.cfg.num_balls), device=device)
        self.ball_initial_height = torch.zeros((num_envs, self.cfg.num_balls), device=device)
        # Drop position tracking
        self.ball_drop_pos = torch.zeros((num_envs, self.cfg.num_balls, 3), device=device)

        # Throw/Catch hand tracking
        self.ball_throw_hand = torch.full((num_envs, self.cfg.num_balls), -1, device=device, dtype=torch.long)  # 0 or 1 for left/right hand
        self.ball_catch_hand = torch.full((num_envs, self.cfg.num_balls), -1, device=device, dtype=torch.long)  # -1 for no hand, 0 or 1 for left/right hand
        self.prev_in_hand = torch.zeros((num_envs, self.cfg.num_balls, self.cfg.num_hands), device=device, dtype=torch.bool)
        self.in_hand = torch.zeros((num_envs, self.cfg.num_balls, self.cfg.num_hands), device=device, dtype=torch.bool)

        self.ball_pos_flat = self.ball_pos.view(self.num_envs, -1)
        self.ball_vel_flat = self.ball_vel.view(self.num_envs, -1)
        self.hand_pos_flat = self.hand_pos.view(self.num_envs, -1)

        num_ball_up_rewards = [0.0, 1.0, 0.5, 0.0]  # index 0: 0 balls up, index 1: 1 ball up, index 2: 2 balls up, index 3: 3 balls up
        self.num_ball_up_rewards_lookup = torch.tensor(num_ball_up_rewards, device=device)

        self.ball_throw_time = torch.zeros((num_envs, self.cfg.num_balls), device=device, dtype=torch.float)
        self.hand_throw_last_time = torch.full((num_envs, self.cfg.num_hands), -1.0, device=device, dtype=torch.float)
        self.hand_throw_intervals = torch.full((num_envs, self.cfg.num_hands), -1.0, device=device, dtype=torch.float)
        self.hoarding_timer = torch.zeros((num_envs, self.cfg.num_hands), device=device, dtype=torch.float)

    def compute_target_hand_position(self, env_ids, ball_ids):
        throw_hand = self.ball_throw_hand[env_ids, ball_ids]
        opposite = 1 - throw_hand # 0 is left hand, 1 is right hand, so 1-throw_hand gives opposite hand index
        return self.hand_pos[env_ids, opposite]

    def _get_observations(self) -> dict:
        left_pos = self.left_hand.data.joint_pos
        left_vel = self.left_hand.data.joint_vel
        right_pos = self.right_hand.data.joint_pos
        right_vel = self.right_hand.data.joint_vel

        hand_split = left_pos.shape[1] # split index for left and right hand joints

        self.joint_pos[:, :hand_split] = left_pos
        self.joint_pos[:, hand_split:] = right_pos
        self.joint_vel[:, :hand_split] = left_vel
        self.joint_vel[:, hand_split:] = right_vel

        # this was tracking the elbow, not the hand
        # self.hand_pos[:, 0] = self.left_hand.data.root_pos_w
        # self.hand_pos[:, 1] = self.right_hand.data.root_pos_w
        l_wrist = self.left_hand.data.body_pos_w[:, self.left_wrist_idx]
        r_wrist = self.right_hand.data.body_pos_w[:, self.right_wrist_idx]
        
        l_knuckle = self.left_hand.data.body_pos_w[:, self.left_knuckle_idx]
        r_knuckle = self.right_hand.data.body_pos_w[:, self.right_knuckle_idx]

        # TUNABLE OFFSET: How far forward is the center?
        # This shifts the center back by ~2cm compared to just tracking the knuckle.
        bias = 0.775 

        self.hand_pos[:, 0] = (l_wrist * (1 - bias)) + (l_knuckle * bias)
        self.hand_pos[:, 1] = (r_wrist * (1 - bias)) + (r_knuckle * bias)

        self.ball_pos = torch.stack([ball.data.root_pos_w for ball in self.balls], dim=1)
        self.ball_vel = torch.stack([ball.data.root_vel_w[:, :3] for ball in self.balls], dim=1)
        
        left_quaternion = self.left_hand.data.body_quat_w[:, self.left_wrist_idx]
        right_quaternion = self.right_hand.data.body_quat_w[:, self.right_wrist_idx]

        self.detect_events()
        
        obs = torch.cat(
            (
                self.joint_pos,
                self.joint_vel,
                self.prev_actions.detach(),
                self.actions_smooth.detach(),
                self.ball_pos_flat,
                self.ball_vel_flat,
                self.hand_pos_flat,
                left_quaternion,
                right_quaternion,
            ),
            dim=1,
        )
        obs = torch.nan_to_num(obs)

        observations = {"policy": obs}
        # if torch.isnan(obs).any() or torch.isinf(obs).any():
        #     print("[WARNING] NaN/Inf detected in observations! Clamping to zero.")
        #     obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            
        return observations

    def detect_events(self):
        self.catch_events[:] = False
        self.drop_events[:] = False
        self.throw_events[:] = False

        # distance_to_hand = torch.norm(
        #     self.ball_pos.unsqueeze(2) - self.hand_pos.unsqueeze(1), # unsqueeze for broadcasting, unsqueeze ball_pos from (num_envs, num_balls, 3) to (num_envs, num_balls, 1, 3), unsqueeze hand_pos
        #     dim=-1)

        # self.in_hand = distance_to_hand < self.cfg.catch_radius
        diff = self.ball_pos.unsqueeze(2) - self.hand_pos.unsqueeze(1)
        distance_to_hand = torch.norm(diff, dim=-1)
        height_diff = diff[..., 2]
        is_close = distance_to_hand < self.cfg.catch_radius
        is_above = height_diff > -0.02  # Clip off the bottom 1/3 of the sphere
        self.in_hand = is_close & is_above
        vel_z = self.ball_vel[..., 2]
        
        ###############
        # Detect throw #
        ###############
        # for ball in range(self.cfg.num_balls):
        #     was_in_hand = self.prev_in_hand[:, ball]
        #     is_in_hand_now = (
        #         (torch.norm(self.ball_pos[:, ball] - self.hand_pos[:, 0], dim=-1) < self.cfg.catch_radius) | # in left hand
        #         (torch.norm(self.ball_pos[:, ball] - self.hand_pos[:, 1], dim=-1) < self.cfg.catch_radius)   # in right hand
        #     )
        #     throw_mask = was_in_hand & (~is_in_hand_now) # ~ is logical NOT for torch tensors
        #     if throw_mask.any():
        #         self.throw_events[throw_mask] = ball
        #         hand_L_position = (torch.norm( # check if the ball was in left hand, otherwise it was in right hand but we don't need to check that again because we already know it was in a hand
        #             self.ball_pos[:, ball] - self.hand_pos[:, 0], dim=-1
        #         ) < self.cfg.catch_radius)

        #         self.ball_throw_hand[throw_mask, ball] = torch.where( # 0 for left hand, 1 for right hand
        #             hand_L_position[throw_mask],
        #             torch.zeros_like(hand_L_position[throw_mask], device=self.device, dtype=torch.long),
        #             torch.ones_like(hand_L_position[throw_mask], device=self.device, dtype=torch.long),
        #         )

        # I think this does the same as the above loop but faster using tensor operations
        was_any_in_hand = self.prev_in_hand.any(dim=2)  
        any_in_hand_now = self.in_hand.any(dim=2)  # (num_envs, num_balls)
        throw_mask = (was_any_in_hand & (~any_in_hand_now)) & (vel_z > self.cfg.min_vertical_velocity)
        

        #throw_mask = (was_left & ~now_left & ~now_right) | (was_right & ~now_left & ~now_right)
        # envs_with_throws = throw_mask.any(dim=1)
        self.throw_events.copy_(throw_mask)
        throw_idxs = throw_mask.nonzero(as_tuple=True)
        if len(throw_idxs[0]) > 0:
            envs_with_throws_ids, ball_ids = throw_idxs

            prev_holder = self.prev_in_hand[envs_with_throws_ids, ball_ids]

            self.ball_throw_hand[envs_with_throws_ids, ball_ids] = prev_holder.float().argmax(dim=1)

            current_time = self.sim.current_time

            # Compute delta_t per env-per-hand
            last_t = self.hand_throw_last_time[envs_with_throws_ids, self.ball_throw_hand[envs_with_throws_ids, ball_ids]]
            self.hand_throw_intervals[envs_with_throws_ids, self.ball_throw_hand[envs_with_throws_ids, ball_ids]] = current_time - last_t

            # Update last throw time
            self.ball_throw_time[envs_with_throws_ids, ball_ids] = current_time
            self.hand_throw_last_time[envs_with_throws_ids, self.ball_throw_hand[envs_with_throws_ids, ball_ids]] = current_time

            self.ball_initial_height[envs_with_throws_ids, ball_ids] = self.ball_pos[envs_with_throws_ids, ball_ids, 2]

            # reset peaks for thrown balls
            self.ball_peak_height[envs_with_throws_ids, ball_ids] = self.ball_pos[envs_with_throws_ids, ball_ids, 2]

        #######################
        # track peak heights #
        #######################
        # for ball in range(self.cfg.num_balls):
        #     self.ball_peak_height[:, ball] = torch.max(
        #         self.ball_peak_height[:, ball],
        #         self.ball_pos[:, ball, 2]
        #     )
        # I think this does the same as the above loop but faster using tensor operations
        self.ball_peak_height = torch.max(self.ball_peak_height, self.ball_pos[:, :, 2])
        

        # update stale peaks (I think this breaks it)
        # ball_height = self.ball_pos[:, :, 2]
        # ball_vel_z = self.ball_vel[:, :, 2]

        # falling = ball_vel_z < -0.01
        # below_peak = (self.ball_peak_height - ball_height) > 0.05
        # not_held = ~self.in_hand.any(dim=2)

        # stale = falling & below_peak & not_held
        # self.ball_peak_height[stale] = ball_height[stale]


        ###############
        # Detect catch #
        ###############
        # for ball in range(self.cfg.num_balls):
        #     dist_to_hands = torch.norm(
        #         self.ball_pos[:, ball].unsqueeze(1) - self.hand_pos, dim=-1
        #     )  # single ball pos has dim (num_env, 3). unsqueeze ball position to (num_envs, 1, 3) for broadcasting

        #     caught_L = (dist_to_hands[:, 0] < self.cfg.catch_radius)
        #     caught_R = (dist_to_hands[:, 1] < self.cfg.catch_radius)

        #     catch_mask = caught_L | caught_R
        #     if catch_mask.any():
        #         self.catch_events[catch_mask] = ball
        #         self.ball_catch_hand[catch_mask, ball] = torch.where(
        #             caught_L[catch_mask], # 0 for left hand, 1 for right hand
        #             torch.tensor(0, device=self.device),
        #             torch.tensor(1, device=self.device)
        #         )

        # I think this does the same as the above loop but faster using tensor operations

        caught = self.in_hand.any(dim=2)

        
        was_not_in_hand = ~self.prev_in_hand.any(dim=2)
        was_thrown = (self.ball_throw_time > 0.0)

        # only catch if ball is going down or nearly still
        descending = vel_z < self.cfg.min_vertical_velocity

        valid_catch_mask = was_not_in_hand & caught & descending & was_thrown


        self.catch_events.copy_(valid_catch_mask)

        # envs_caught = valid_catch.any(dim=1)
        catch_idxs = valid_catch_mask.nonzero(as_tuple=True)
        if len(catch_idxs[0]) > 0:
            env_caught_ids, ball_ids = catch_idxs
            #ball_caught_ids = caught[env_caught_ids].float().argmax(dim=1) # get first ball that was caught
            # catch_mask = caught[env_caught_ids]
            # throw_times = self.ball_throw_time[env_caught_ids]
            # throw_times_masked = torch.where(catch_mask, throw_times.unsqueeze(-1), -1e9)
            # # Find the max time for each ball across both hands
            # ball_times_max, _ = throw_times_masked.max(dim=2)
            # ball_caught_ids = ball_times_max.argmax(dim=1)
            # self.catch_events[env_caught_ids] = ball_caught_ids
            
            #determin the hand that caught
            hand_dist = distance_to_hand[env_caught_ids, ball_ids]
            closest_hand = hand_dist.argmin(dim=1)
            self.ball_catch_hand[env_caught_ids, ball_ids] = closest_hand

            # hand_caught = caught[env_caught_ids, ball_caught_ids]  
            # hand_idx = hand_caught.int().argmax(dim=1)  # 0 for left hand, 1 for right hand
            #self.ball_catch_hand[env_caught_ids, ball_ids] = caught[env_caught_ids, ball_ids].long().argmax(dim=1)


        ###############
        # Detect drop #
        ###############

        valid_throw = self.ball_peak_height > self.cfg.min_throw_height

        ball_height_pos = self.ball_pos[:, :, 2]  
        dropped = (~self.in_hand.any(dim=2)) & (ball_height_pos < self.cfg.ground_height + 0.1 )  # small buffer to avoid numerical issues, may need tuning

        
        # for ball in range(self.cfg.num_balls):
        #     dropped_mask = dropped[:, ball]
        #     if dropped_mask.any():
        #         self.drop_events[dropped_mask] = ball
        #         self.ball_drop_pos[dropped_mask, ball] = self.ball_pos[dropped_mask, ball]
        #         self.ball_target_hand_pos[dropped_mask] = self.compute_target_hand_position(dropped_mask, ball)
        
        # I think this does the same as the above loop but faster using tensor operations
        valid_drop = dropped & valid_throw
        # env_dropped = valid_drop.any(dim=1)
        self.drop_events.copy_(valid_drop)

        if valid_drop.any():
            #env_dropped_ids = env_dropped.nonzero(as_tuple=True)[0]

            # gets the ball that was thrown
            #drop_mask = valid_drop[env_dropped_ids]
            #throw_times = self.ball_throw_time[env_dropped_ids]
            #mask = torch.where(drop_mask, throw_times, 1e9)
            #dropped_ball_ids = mask.argmin(dim=1)

            #self.drop_events[env_dropped_ids] = dropped_ball_ids
            
            #self.ball_drop_pos[env_dropped_ids, dropped_ball_ids] = self.ball_pos[env_dropped_ids, dropped_ball_ids]
            self.ball_drop_pos[valid_drop] = self.ball_pos[valid_drop]
        # update prev_in_hand, we need to know if the ball was in hand in the previous step to detect throw events
        # self.prev_in_hand = torch.zeros((self.num_envs, self.cfg.num_balls), device=self.device, dtype=torch.bool)
        # for ball in range(self.cfg.num_balls):
        #     self.prev_in_hand[:, ball] = (
        #         (torch.norm(self.ball_pos[:, ball] - self.hand_pos[:, 0], dim=-1) < self.cfg.catch_radius) | 
        #         (torch.norm(self.ball_pos[:, ball] - self.hand_pos[:, 1], dim=-1) < self.cfg.catch_radius)
        #     )
        # I think this does the same as the above loop but faster using tensor operations
        self.prev_in_hand.copy_(self.in_hand)



    def _get_rewards(self) -> torch.Tensor:
        self.reward_buffer.fill_(0.0)

        ###################
        # hoarding penalty#
        ###################
        # if more than one ball is in a hand after the first second, begin to apply a penalty. This is to prevent the agent from just holding onto balls instead of juggling them


        # calculate if the ball is in the hand, use the catch_radius variable
          
        # in_hand = torch.zeros((num_envs, self.cfg.num_balls, self.cfg.num_hands), device=device, dtype=torch.bool)
        # for ball in range(self.cfg.num_balls):
        #     for hand in range(self.cfg.num_hands):
        #         distance_to_hand = torch.norm(
        #             ball_pos[:, ball] - hand_pos[:, hand],
        #             dim=-1
        #         )
        #         in_hand[:, ball, hand] = distance_to_hand < self.catch_radius
        
        # I belive this works the same as the above nested loop, but takes advantage of tensor broadcasting to make it much faster
        
        # in_hand is already computed in detect_events function
        balls_in_L = (self.in_hand[:, :, 0]).sum(dim=1)
        balls_in_R = (self.in_hand[:, :, 1]).sum(dim=1)

        is_hoarding = (balls_in_L > self.hoarding_threshold) | (balls_in_R > self.hoarding_threshold) # peniltize holding a ball
        self.hoarding_timer[is_hoarding] += self.step_dt
        self.hoarding_timer[~is_hoarding] = 0.0

        hoarding_time = torch.clamp(self.hoarding_timer - self.cfg.hoarding_time_threshold, min=0.0)
        
        hoarding_penalty = torch.clamp(self.cfg.w_hoarding * (hoarding_time.square()), max=5.0)

        self.reward_buffer -= hoarding_penalty

        ###################
        # jitter penalty  #
        ###################

        delta_a = self.actions - self.prev_actions
        jitter = torch.sum(delta_a**2, dim=-1)
        self.reward_buffer += -self.cfg.w_jitter * jitter


        #######################
        # highest ball reward #
        #######################
        # a continuous reward that is applied every step based on the height of only the highest ball, and does not reward of multiple balls are going upwards. this is to encourage the agent to initially throw balls up

        # always applied
        # reward the height of only the higest ball up to a specified apex defined in the cfg file
        height_cords = self.ball_pos[:, :, 2]  
        vertical_velocities = self.ball_vel[:, :, 2]

        going_up = vertical_velocities > self.cfg.min_vertical_velocity
        # above_min_height = height_cords > self.cfg.min_throw_height
        above_min_height = height_cords > 0 # disable min height, we already use is_held to prevent rewarding balls that are still in hand
        is_held = self.in_hand.any(dim=2)
        
        up_mask = going_up & above_min_height & (~is_held)  # ball is going up, above min height, and not in hand

        num_up = up_mask.sum(dim=1)

        # Safety: Clamp index so we don't crash if num_up > len(lookup), though this should not happen
        safe_indices = torch.clamp(num_up, max=len(self.num_ball_up_rewards_lookup)-1)
        num_scale = self.num_ball_up_rewards_lookup[safe_indices]
        
        masked_heights = height_cords.clone()
        masked_heights[~up_mask] = -1.0 # set to invalid height
        highest_ball_height, _ = masked_heights.max(dim=1)
        
        # potentially change to delta height and delta target height?

        highest_ball_height = highest_ball_height.clamp_(min=self.cfg.ground_height, max=self.cfg.target_height)

        #one_going_up = (up_mask.sum(dim=1) == 1)  # only one ball is going up

        # check to see if the ball going up is in the air (not in hand)
        # height_cords_up = (height_cords * up_mask.float()).sum(dim=1)

        # Use clip for height reward, should be nicer for early lerning but maybe switch to Gaussian if not working well?
        height_r = (highest_ball_height - self.cfg.ground_height) * self.distance_target2ground
        height_r_norm = height_r.clamp_(0.0, 1.0)

        # height Gaussian reward
        #height_r_norm = torch.exp((height_cords_up - self.cfg.target_height).square() * self.sigma_apex_height_coeff)

        self.reward_buffer += self.cfg.w_highest * height_r_norm * num_scale

        ###################
        # catch reward    #
        ###################
        # largest reward, applies a reward based on the max height of the ball that was just caught, and whether it was caught by the opposite hand

        # only added on catch event
        # two parts here, get the max height of the caught ball
        #                 check if the ball is caught by the opposite hand
        catch_idxs = self.catch_events.nonzero(as_tuple=True)
        if len(catch_idxs[0]) > 0:

            env_ids, ball_ids = catch_idxs

            peak = self.ball_peak_height[env_ids, ball_ids]
            start_height = self.ball_initial_height[env_ids, ball_ids]
            delta_height = torch.clamp(peak - start_height, min=0.0)

            # height Gaussian
            Gh = torch.exp((delta_height - self.cfg.target_delta_height).square() * self.sigma_apex_height_coeff)

            # hand indices
            throw_hand = self.ball_throw_hand[env_ids, ball_ids]
            catch_hand = self.ball_catch_hand[env_ids, ball_ids]
                
            cross = torch.where(throw_hand != catch_hand, self.cross_pos, self.cross_neg)

            catch_r = self.cfg.w_catch * Gh * cross
            self.reward_buffer[env_ids] += catch_r

            # reset the peak height for the caught ball
            self.ball_peak_height[env_ids, ball_ids] = 0.0

        ###################
        # drop reward  #
        ###################
        # a smaller reward designed to encourage the agent when to have the ball drop closer to the target hand position, so that it eventually learns to catch it. Same logic as catch reward, but much smaller reward weight

        # only added on drop event
        drop_idxs = self.drop_events.nonzero(as_tuple=True)
        if len(drop_idxs[0]) > 0:
            env_ids, ball_ids = drop_idxs
            peak = self.ball_peak_height[env_ids, ball_ids]
            start_height = self.ball_initial_height[env_ids, ball_ids]
            delta_height = torch.clamp(peak - start_height, min=0.0)
            
            drop_pos = self.ball_drop_pos[env_ids, ball_ids]

            # height gaussian
            Gh = torch.exp((delta_height - self.cfg.target_delta_height).square() * self.sigma_apex_height_coeff)
            # distance gaussian to target hand
            target_hand_pos = self.compute_target_hand_position(env_ids, ball_ids)
            dist = torch.norm(drop_pos - target_hand_pos, dim=-1)
            Gd = torch.exp((dist).square() * self.sigma_drop_distance_coeff)

            drop_r = self.cfg.w_drop * Gh * Gd
            self.reward_buffer[env_ids] += drop_r

            # reset the peak height for the dropped ball
            self.ball_peak_height[env_ids, ball_ids] = 0.0

        ###################
        # rythem reward   #
        ###################
        # a small reward to encourage consistent rythem, applied on throw event based on time since last throw compared to target rythem. Designed to encourage consistent throws, and to make the pattern more stable, and realistic

        
        throw_idxs = self.throw_events.nonzero(as_tuple=True)
        if len(throw_idxs[0]) > 0:
            env_ids, ball_id = throw_idxs
            hands = self.ball_throw_hand[env_ids, ball_id]
            delta_t = self.hand_throw_intervals[env_ids, hands]
            target_interval = self.cfg.target_rythem * 2.0  # since each hand throws every other throw, the target interval for each hand is double the target rythem
            GT = torch.exp((delta_t - target_interval).square() * self.sigma_rythem_coeff)
            self.reward_buffer[env_ids] += self.cfg.w_rythem * GT


        self.prev_actions = self.actions.clone()

        return self.reward_buffer

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # self.joint_pos = torch.cat([self.left_hand.data.joint_pos, self.right_hand.data.joint_pos], dim=1)
        # self.joint_vel = torch.cat([self.left_hand.data.joint_vel, self.right_hand.data.joint_vel], dim=1)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        #out_of_bounds = torch.zeros(1, dtype=torch.bool) # I don't think we need this anymore

        ball_height = self.ball_pos[:, :, 2]
        is_ball_dropped = (ball_height < (self.cfg.ground_height + 0.1)) & (~self.in_hand.any(dim=2))  # small buffer to avoid numerical issues, may need tuning
        
        #check if ball is thrown out of bounds
        relative_ball_pos = self.ball_pos - self.scene.env_origins.unsqueeze(1)
        ball_dist_xy = torch.norm(relative_ball_pos[:, :, :2], dim=-1)
        is_ball_out_of_bounds = ball_dist_xy > self.cfg.out_of_bounds_radius
        
        agent_terminated = (is_ball_dropped | is_ball_out_of_bounds).any(dim=1)

        # out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self.placeholder_idx1]) >= 0, dim=1)
        # out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self.placeholder_idx2]) > math.pi / 2, dim=1)
        return agent_terminated, time_out
        #return out_of_bounds, time_out

    def _build_init_joint_pose(self, hand: Articulation, targets: dict[str, float]):
        """Create cached joint position/velocity tensors for all envs."""
        name_to_idx = {n: i for i, n in enumerate(hand.joint_names)}
        joint_pos = torch.zeros((hand.num_joints,), device=self.device)
        for name, val in targets.items():
            if name in name_to_idx:
                joint_pos[name_to_idx[name]] = val
        joint_pos = joint_pos.repeat(self.num_envs, 1)
        return joint_pos

    def _apply_init_joint_pose(self, hand: Articulation, joint_pos: torch.Tensor, env_ids):
        """Apply initial joint pose targets from cached tensors for selected envs."""
        if env_ids is None:
            pos_target = joint_pos
        else:
            pos_target = joint_pos[env_ids]
        vel_target = torch.zeros_like(pos_target)
        hand.set_joint_position_target(pos_target, env_ids=env_ids)
        hand.set_joint_velocity_target(vel_target, env_ids=env_ids)

        hand.write_joint_state_to_sim(pos_target, vel_target, None, env_ids=env_ids)

    def _reset_ball_pos(self, env_ids: Sequence[int] | None):
        """Reset ball transforms and velocities"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        for i, ball in enumerate(self.balls):
            default_state = ball.data.default_root_state[env_ids].clone()
            default_state[:, :3] = self.scene.env_origins[env_ids] # first 3 cols in root state are position x,y,z
            default_state[:, :3] += self.init_ball_pos[None, i, :]  # add initial offset
            # default velocity
            default_state[:, 7:] = 0.0 # last 6 cols in default root state are linear and angular velocity
            ball.write_root_state_to_sim(default_state, env_ids=env_ids)


    def _reset_idx(self, env_ids: Sequence[int] | None):
        # if env_ids is None:
        #     env_ids = self.left_hand._ALL_INDICES
        super()._reset_idx(env_ids)
        
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # Reset hand joints
        self._apply_init_joint_pose(self.left_hand, self.init_left_joint_pos, env_ids)
        self._apply_init_joint_pose(self.right_hand, self.init_right_joint_pos, env_ids)       

        # Reset positions of balls
        self._reset_ball_pos(env_ids)

        for _ in range(1):  # let the balls resettle
            self.sim.step()

        self.scene.update(dt=self.cfg.sim.dt)

        self.detect_events()  # update in_hand status after reset


        # for ball in range(self.cfg.num_balls):
        #     hand_mask = self.in_hand[env_ids, ball]
        #     hand_id = hand_mask.int().argmax(dim=1)  # 0 for left hand, 1 for right hand
        #     self.ball_throw_hand[env_ids, ball] = hand_id
        # I think this does the same as the above loop but faster using tensor operations
        # anchors = torch.tensor(self.cfg.ball_anchor, device=self.device, dtype=torch.long)
        # self.ball_throw_hand[env_ids] = anchors[None, :].expand(len(env_ids), -1)
        active_anchors = self.cfg.ball_anchor[:self.cfg.num_balls]
        
        anchors = torch.tensor(active_anchors, device=self.device, dtype=torch.long)
        self.ball_throw_hand[env_ids] = anchors[None, :].expand(len(env_ids), -1)
        
        self.ball_catch_hand[env_ids] = -1  # reset to -1 (no catch)


        # Reset tracking variables
        self.catch_events[env_ids] = False
        self.drop_events[env_ids] = False
        self.throw_events[env_ids] = False
        self.ball_peak_height[env_ids] = self.ball_pos[env_ids, :, 2]
        self.ball_initial_height[env_ids] = self.ball_pos[env_ids, :, 2]
        
        self.prev_actions[env_ids] = 0.0
        self.actions[env_ids] = 0.0

        # if not hasattr(self, 'actions_smooth'):
        #     self.actions_smooth = torch.zeros_like(self.actions)
        self.actions_smooth[env_ids] = 0.0
        
        # Reset timers
        # self.throw_last_time[env_ids] = 0.0
        # self.throw_intervals[env_ids] = 0.0
        self.hand_throw_last_time[env_ids] = self.sim.current_time
        self.hand_throw_intervals[env_ids] = 0.0
        self.ball_throw_time[env_ids] = 0.0
        self.hoarding_timer[env_ids] = 0.0

        # self.detect_events()  
        self.prev_in_hand[env_ids] = self.in_hand[env_ids]
