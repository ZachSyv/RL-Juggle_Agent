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
        self.left_hand_bias = 0
        self.right_hand_bias = len(self.left_hand_idx)

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

        self.ball_spawn_offsets = torch.tensor(self.cfg.ball_offset, device=self.device, dtype=torch.float32)
        self.ball_anchors = torch.tensor(self.cfg.ball_anchor, device=self.device, dtype=torch.long)
        self.hand_bases = torch.tensor(self.cfg.hand_pos, device=self.device, dtype=torch.float32)

        # palm_L_ids, _ = self.left_hand.find_bodies(".*palm")
        # palm_R_ids, _ = self.right_hand.find_bodies(".*palm")
        # middle finger base connection is better reference for hand center
        # palm_L_ids, _ = self.left_hand.find_bodies(".*mfproximal")
        # palm_R_ids, _ = self.right_hand.find_bodies(".*mfproximal")
        
        # self.left_palm_idx = palm_L_ids[0]
        # self.right_palm_idx = palm_R_ids[0]

        wrist_L_ids, _ = self.left_hand.find_bodies(".*palm")
        wrist_R_ids, _ = self.right_hand.find_bodies(".*palm")
        
        self.left_wrist_idx = wrist_L_ids[0]
        self.right_wrist_idx = wrist_R_ids[0]

        # 2. Find Knuckle (Top of Palm)
        knuckle_L_ids, _ = self.left_hand.find_bodies(".*mfproximal")
        knuckle_R_ids, _ = self.right_hand.find_bodies(".*mfproximal")
        
        self.left_knuckle_idx = knuckle_L_ids[0]
        self.right_knuckle_idx = knuckle_R_ids[0]

        self.init_ball_pos = torch.tensor(self.cfg.init_ball_pos, device=device, dtype=torch.float32)
        # assert self.cfg.action_space == len(self.left_hand_idx) + len(self.right_hand_idx), 'action dim mismatch'

        self.reward_buffer = torch.zeros(self.num_envs, device=device)

        self.action_dim = self.cfg.action_space
        self.actions = torch.zeros((self.num_envs, self.action_dim), device=device)
        self.prev_actions = torch.zeros((self.num_envs, self.action_dim), device=device)

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
        self.ball2 = RigidObject(self.cfg.ball2_cfg)
        self.ball3 = RigidObject(self.cfg.ball3_cfg)

        self.scene.articulations["left_hand"] = self.left_hand
        self.scene.articulations["right_hand"] = self.right_hand

        self.scene.rigid_objects["ball1"] = self.ball1
        self.scene.rigid_objects["ball2"] = self.ball2
        self.scene.rigid_objects["ball3"] = self.ball3

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

        self.balls = [self.ball1, self.ball2, self.ball3]

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
        # 1. Hyperparameter: Smoothing Factor (Alpha)
        # 0.0 = Frozen, 1.0 = No Smoothing. 
        # 0.6 is a good balance for snappy but smooth motion.
        alpha = 0.6 
        
        # 2. Apply Low-Pass Filter
        # New Target = 60% New Command + 40% Old Command
        current_action = self.actions.clone()
        if not hasattr(self, 'actions_smooth'):
            self.actions_smooth = torch.zeros_like(current_action)
            
        self.actions_smooth = alpha * current_action + (1.0 - alpha) * self.actions_smooth

        # 3. Slice and Apply the SMOOTH actions
        num_left = len(self.left_hand_idx)
        
        # Use actions_smooth here instead of self.actions!
        left_action = self.actions_smooth[:, :num_left] * 1.0 # Keep scale at 1.0
        right_action = self.actions_smooth[:, num_left:] * 1.0 

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
        self.catch_events = torch.full((num_envs,), -1, device=device, dtype=torch.long)  # -1 means no catch
        self.drop_events = torch.full((num_envs,), -1, device=device, dtype=torch.long)   # -1 means no drop
        self.throw_events = torch.full((num_envs,), -1, device=device, dtype=torch.long)  # -1 means no throw

        # Height tracking tensors
        self.ball_peak_height = torch.zeros((num_envs, self.cfg.num_balls), device=device)

        # Drop position tracking
        self.ball_drop_pos = torch.zeros((num_envs, self.cfg.num_balls, 3), device=device)

        # Throw/Catch hand tracking
        self.ball_throw_hand = torch.zeros((num_envs, self.cfg.num_balls), device=device, dtype=torch.long)  # 0 or 1 for left/right hand
        self.ball_catch_hand = torch.zeros((num_envs, self.cfg.num_balls), device=device, dtype=torch.long)  # 0 or 1 for left/right hand
        self.prev_in_hand = torch.zeros((num_envs, self.cfg.num_balls, self.cfg.num_hands), device=device, dtype=torch.bool)

        # throw rythem tracking, both hands share the same rythem timer. May want to change to be per-hand and add a offset for one hand
        self.throw_last_time = torch.zeros(num_envs, device=device)
        self.throw_intervals = torch.zeros(num_envs, device=device)

    def compute_target_hand_position(self, env_ids, ball_ids):
        throw_hand = self.ball_throw_hand[env_ids, ball_ids]
        opposite = 1 - throw_hand # 0 is left hand, 1 is right hand, so 1-throw_hand gives opposite hand index
        return self.hand_pos[env_ids, opposite]

    def _get_observations(self) -> dict:
        # import pdb; pdb.set_trace()

        # obs = torch.cat(
        #     (
        #         self.joint_pos[:, self.left_hand_idx],
        #         self.joint_vel[:, self.left_hand_idx],
        #         self.joint_pos[:, self.right_hand_idx],
        #         self.joint_vel[:, self.right_hand_idx],
        #     ),
        #     dim=-1,
        # )

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
        # 0.8 means "80% of the way from wrist to knuckle"
        # This shifts the center back by ~2cm compared to just tracking the knuckle.
        bias = 0.775 

        self.hand_pos[:, 0] = (l_wrist * (1 - bias)) + (l_knuckle * bias)
        self.hand_pos[:, 1] = (r_wrist * (1 - bias)) + (r_knuckle * bias)

        self.ball_pos = torch.stack([ball.data.root_pos_w for ball in self.balls], dim=1)
        self.ball_vel = torch.stack([ball.data.root_vel_w[:, :3] for ball in self.balls], dim=1)
        
        self.detect_events()
        
        obs = torch.cat(
            (
                self.joint_pos,
                self.joint_vel,
                self.hand_pos.reshape(self.num_envs, -1),
                self.ball_pos.reshape(self.num_envs, -1),
                self.ball_vel.reshape(self.num_envs, -1),
            ),
            dim=1,
        )
        observations = {"policy": obs}
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            print("[WARNING] NaN/Inf detected in observations! Clamping to zero.")
            obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            
        observations = {"policy": obs}
        return observations

    def detect_events(self):
        self.catch_events[:] = -1
        self.drop_events[:] = -1
        self.throw_events[:] = -1

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
        ###############
        # Detect drop #
        ###############

        ball_height_pos = self.ball_pos[:, :, 2]  
        dropped = ball_height_pos < self.cfg.ground_height + 0.01  # small buffer to avoid numerical issues, may need tuning
        
        # for ball in range(self.cfg.num_balls):
        #     dropped_mask = dropped[:, ball]
        #     if dropped_mask.any():
        #         self.drop_events[dropped_mask] = ball
        #         self.ball_drop_pos[dropped_mask, ball] = self.ball_pos[dropped_mask, ball]
        #         self.ball_target_hand_pos[dropped_mask] = self.compute_target_hand_position(dropped_mask, ball)
        
        # I think this does the same as the above loop but faster using tensor operations
        env_dropped = dropped.any(dim=1)

        if env_dropped.any():
            env_dropped_ids = env_dropped.nonzero(as_tuple=True)[0]
            dropped_ball_ids = dropped[env_dropped_ids].float().argmax(dim=1) # get first ball that was dropped, may need to change later to handle multiple drops
            self.drop_events[env_dropped] = dropped_ball_ids
            self.ball_drop_pos[env_dropped_ids, dropped_ball_ids] = self.ball_pos[env_dropped_ids, dropped_ball_ids]

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
        self.catch_events[:] = -1 # default to -1, no catch

        caught = self.in_hand

        balls_caught = caught.any(dim=2)  # shape (num_envs, num_balls)
        envs_caught = balls_caught.any(dim=1)

        if envs_caught.any():
            env_caught_ids = envs_caught.nonzero(as_tuple=True)[0]
            ball_caught_ids = balls_caught[env_caught_ids].float().argmax(dim=1) # get first ball that was caught

            self.catch_events[env_caught_ids] = ball_caught_ids

            hand_caught = caught[env_caught_ids, ball_caught_ids]  
            hand_idx = hand_caught.int().argmax(dim=1)  # 0 for left hand, 1 for right hand
            self.ball_catch_hand[env_caught_ids, ball_caught_ids] = hand_idx

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
        throw_mask = was_any_in_hand & (~any_in_hand_now) # ~ is logical NOT for torch tensors
        envs_with_throws = throw_mask.any(dim=1)
        if envs_with_throws.any():
            envs_with_throws_ids = envs_with_throws.nonzero(as_tuple=True)[0]
            thrown_ball_ids =  throw_mask[envs_with_throws_ids].int().argmax(dim=1)

            prev_in_hand_throw = self.prev_in_hand[envs_with_throws_ids, thrown_ball_ids]

            throw_hand_idx = prev_in_hand_throw.int().argmax(dim=1)  # 0 for left hand, 1 for right hand
            
            self.throw_events[envs_with_throws_ids] = thrown_ball_ids
            self.ball_throw_hand[envs_with_throws_ids, thrown_ball_ids] = throw_hand_idx


        # update prev_in_hand, we need to know if the ball was in hand in the previous step to detect throw events
        # self.prev_in_hand = torch.zeros((self.num_envs, self.cfg.num_balls), device=self.device, dtype=torch.bool)
        # for ball in range(self.cfg.num_balls):
        #     self.prev_in_hand[:, ball] = (
        #         (torch.norm(self.ball_pos[:, ball] - self.hand_pos[:, 0], dim=-1) < self.cfg.catch_radius) | 
        #         (torch.norm(self.ball_pos[:, ball] - self.hand_pos[:, 1], dim=-1) < self.cfg.catch_radius)
        #     )
        # I think this does the same as the above loop but faster using tensor operations
        self.prev_in_hand.copy_(self.in_hand) # already computed above

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
        

    def _get_rewards(self) -> torch.Tensor:
        # total_reward = compute_rewards(
        #     1.0,
        #     -2.0, -1.0, -0.01, -0.005,
        #     self.joint_pos[:, self.placeholder_idx2[0]],
        #     self.joint_vel[:, self.placeholder_idx2[0]],
        #     self.joint_pos[:, self.placeholder_idx1[0]],
        #     self.joint_vel[:, self.placeholder_idx1[0]],
        #     self.reset_terminated,
        # )
        # return total_reward

        self.reward_buffer.fill_(0.0)

        num_envs = self.num_envs
        device = self.device
        ball_pos = self.ball_pos
        ball_vel = self.ball_vel
        hand_pos = self.hand_pos
        actions = self.actions
        prev_actions = self.prev_actions

        #reward = torch.zeros(num_envs, device=self.device)

        #return reward

        ###################
        # hoarding penalty#
        ###################
        # if more than one ball is in a hand after the first second, apply a penalty. This is to prevent the agent from just holding onto balls instead of juggling them


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
        
        # self.step_dt is automatically calculated as (sim_dt * decimation)
        time_mask = (self.episode_length_buf * self.step_dt) > 1.0
        # in_hand is already computed in detect_events function
        balls_in_L = self.in_hand[:, :, 0].sum(dim=1)
        balls_in_R = self.in_hand[:, :, 1].sum(dim=1)

        hoarding = (balls_in_L > 1) | (balls_in_R > 1)
        self.reward_buffer += -self.cfg.w_hoarding * (hoarding & time_mask).float()

        ###################
        # jitter penalty  #
        ###################

        delta_a = actions - prev_actions
        jitter = torch.sum(delta_a**2, dim=-1)
        self.reward_buffer += -self.cfg.w_jitter * jitter


        #######################
        # highest ball reward #
        #######################
        # a continuous reward that is applied every step based on the height of only the highest ball, and does not reward of multiple balls are going upwards. this is to encourage the agent to initially throw balls up

        # always applied
        # reward the height of only the higest ball up to a specified apex defined in the cfg file
        height_cords = ball_pos[:, :, 2]  
        vertical_velocities = ball_vel[:, :, 2]

        going_up = vertical_velocities > 0
        above_min_height = height_cords > self.cfg.min_throw_height
        is_held = self.in_hand.any(dim=2)
        
        up_mask = going_up & above_min_height & (~is_held)  # ball is going up, above min height, and not in hand

        one_going_up = (up_mask.sum(dim=1) == 1)  # only one ball is going up
        index_going_up = torch.argmax(up_mask.float(), dim=1)

        # check to see if the ball going up is in the air (not in hand)
        height_cords_up = (height_cords * up_mask.float()).sum(dim=1)

        # Use clip for height reward, should be nicer for early lerning but maybe switch to Gaussian if not working well?
        height_r = (height_cords_up - self.cfg.ground_height) * self.distance_target2ground
        height_r_norm = torch.clamp(height_r, 0.0, 1.0)

        # height Gaussian reward
        #height_r_norm = torch.exp((height_cords_up - self.cfg.target_height).square() * self.sigma_apex_height_coeff)

        self.reward_buffer += self.cfg.w_highest * height_r_norm * one_going_up.float()

        ###################
        # catch reward    #
        ###################
        # largest reward, applies a reward based on the max height of the ball that was just caught, and whether it was caught by the opposite hand

        # only added on catch event
        # two parts here, get the max height of the caught ball
        #                 check if the ball is caught by the opposite hand

        catch_mask = self.catch_events >= 0
        if catch_mask.any():

            batch = catch_mask.nonzero(as_tuple=True)[0]
            ball_id = self.catch_events[batch] 
            peak = self.ball_peak_height[batch, ball_id]

            # height Gaussian
            Gh = torch.exp((peak - self.cfg.target_height).square() * self.sigma_apex_height_coeff)

            # hand indices
            throw_hand = self.ball_throw_hand[batch, ball_id]
            catch_hand = self.ball_catch_hand[batch, ball_id]
                
            cross = torch.where(throw_hand != catch_hand, self.cross_pos, self.cross_neg)

            catch_r = self.cfg.w_catch * Gh * cross
            self.reward_buffer[batch] += catch_r

        ###################
        # drop reward  #
        ###################
        # a smaller reward designed to encourage the agent when to have the ball drop closer to the target hand position, so that it eventually learns to catch it. Same logic as catch reward, but much smaller reward weight

        # only added on drop event
        drop_mask = self.drop_events >= 0
        if drop_mask.any():
            batch = drop_mask.nonzero(as_tuple=True)[0]
            ball_id = self.drop_events[batch] 
            peak = self.ball_peak_height[batch, ball_id]
            drop_pos = self.ball_drop_pos[batch, ball_id]

            # height gaussian
            Gh = torch.exp((peak - self.cfg.target_height).square() * self.sigma_apex_height_coeff)

            # distance gaussian to target hand
            target_hand_pos = self.compute_target_hand_position(batch, ball_id)
            dist = torch.norm(drop_pos - target_hand_pos, dim=-1)
            Gd = torch.exp((dist).square() * self.sigma_drop_distance_coeff)

            drop_r = self.cfg.w_drop * Gh * Gd
            self.reward_buffer[batch] += drop_r


        ###################
        # rythem reward   #
        ###################
        # a small reward to encourage consistent rythem, applied on throw event based on time since last throw compared to target rythem. Designed to encourage consistent throws, and to make the pattern more stable, and realistic

        throw_max = self.throw_events >= 0
        if throw_max.any():
            env_ids = throw_max.nonzero(as_tuple=True)[0]
            # delta t = time since last throw for the same hand
            delta_t = self.throw_intervals[env_ids] # TODO, need to actually track this variable properly
            GT = torch.exp((delta_t - self.cfg.target_rythem).square() * self.sigma_rythem_coeff)
            self.reward_buffer[env_ids] += self.cfg.w_rythem * GT


        self.prev_actions = actions.clone()

        return self.reward_buffer

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = torch.cat([self.left_hand.data.joint_pos, self.right_hand.data.joint_pos], dim=1)
        self.joint_vel = torch.cat([self.left_hand.data.joint_vel, self.right_hand.data.joint_vel], dim=1)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.zeros(1, dtype=torch.bool) # I don't think we need this anymore

        ball_height = self.ball_pos[:, :, 2]
        is_ball_dropped = ball_height < (self.cfg.ground_height + 0.1)  # small buffer to avoid numerical issues, may need tuning
        
        #check if ball is thrown out of bounds
        ball_dist_xy = torch.norm(self.ball_pos[:, :, :2], dim=-1)
        is_ball_out_of_bounds = ball_dist_xy > self.cfg.out_of_bounds_radius
        
        agent_terminated = (is_ball_dropped | is_ball_out_of_bounds).any(dim=1)

        # out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self.placeholder_idx1]) >= 0, dim=1)
        # out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self.placeholder_idx2]) > math.pi / 2, dim=1)
        return agent_terminated, time_out, 
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

        # env_ids = torch.as_tensor(env_ids, device=self.device)

        # origins = self.scene.env_origins[env_ids]  # (n, 3)
        # ball_pos = origins[:, None, :] + self.init_ball_pos  # (n, num_balls, 3)
        #
        # flat_pos = ball_pos.reshape(-1, 3)  # (n * num_balls, 3)
        # # build root pose tensor expected by PhysX view (pos + xyzw quat)
        # flat_pos = torch.nn.functional.pad(flat_pos, (0, 4), value=0.0)  # (n * num_balls, 7)
        # flat_pos[:, 6] = 1.0
        # flat_vel = torch.zeros((flat_pos.shape[0], 6), device=self.device)  # (n * num_balls, 6)
        #
        # # view indices align as env-major: env_id * num_balls + ball_id
        # view_ids = (env_ids[:, None] * self.cfg.num_balls + torch.arange(self.cfg.num_balls,
        #                                                                  device=self.device)).reshape(-1)
        #
        # self.ball_view.set_transforms(flat_pos, indices=view_ids)
        # self.ball_view.set_velocities(flat_vel, indices=view_ids)
        # origins = self.scene.env_origins[env_ids]  # (n, 3)
        # ball_pos = origins[:, None, :] + self.init_ball_pos  # (n, num_balls, 3)

        # flat_pos = ball_pos.reshape(-1, 3)  # (n * num_balls, 3)
        # # build root pose tensor expected by PhysX view (pos + xyzw quat)
        # flat_pos = torch.nn.functional.pad(flat_pos, (0, 4), value=0.0)  # (n * num_balls, 7)
        # flat_pos[:, 6] = 1.0
        # flat_vel = torch.zeros((flat_pos.shape[0], 6), device=self.device)  # (n * num_balls, 6)

        # # view indices align as env-major: env_id * num_balls + ball_id
        # view_ids = (env_ids[:, None] * self.cfg.num_balls + torch.arange(self.cfg.num_balls,
        #                                                                  device=self.device)).reshape(-1)

        # self.ball_view.set_transforms(flat_pos, indices=view_ids)
        # self.ball_view.set_velocities(flat_vel, indices=view_ids)

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
        self.ball_peak_height[env_ids] = 0.0
        
        # Reset tracking variables
        self.catch_events[env_ids] = -1
        self.drop_events[env_ids] = -1
        self.throw_events[env_ids] = -1
        self.prev_in_hand[env_ids] = False
        
        self.prev_actions[env_ids] = 0.0
        self.actions[env_ids] = 0.0

        if not hasattr(self, 'actions_smooth'):
            self.actions_smooth = torch.zeros_like(self.actions)
        self.actions_smooth[env_ids] = 0.0
        
        # Reset timers
        self.throw_last_time[env_ids] = 0.0
        self.throw_intervals[env_ids] = 0.0

        # joint_pos = self.left_hand.data.default_joint_pos[env_ids]
        # joint_pos[:, self.placeholder_idx2] += sample_uniform(
        #     -0.25 * math.pi,
        #     0.25 * math.pi,
        #     joint_pos[:, self.placeholder_idx2].shape,
        #     joint_pos.device,
        # )
        # joint_vel = self.left_hand.data.default_joint_vel[env_ids]
        #
        # default_root_state = self.left_hand.data.default_root_state[env_ids]
        # default_root_state[:, :3] += self.scene.env_origins[env_ids]
        #
        # self.joint_pos[env_ids] = joint_pos
        # self.joint_vel[env_ids] = joint_vel

        # self.left_hand.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        # self.left_hand.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        # self.left_hand.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # env_ids = env_ids.to(self.device)
        #
        # # Reset positions of balls
        # #self.ball_pos[env_ids] = 
        # self.ball_vel[env_ids] = 0
        # self.ball_peak_height[env_ids] = 0
        #
        # # Reset tracking variables
        # self.catch_events[env_ids] = -1
        # self.drop_events[env_ids] = -1
        # self.throw_events[env_ids] = -1
        # self.prev_in_hand[env_ids] = False
        #
        # self.prev_actions[env_ids] = 0
        # self.actions[env_ids] = 0
        #
        # # Reset timers
        # self.throw_last_time[env_ids] = 0
        # self.throw_intervals[env_ids] = 0
        #
        # return super().reset_idx(env_ids)
