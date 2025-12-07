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
from isaaclab.utils.math import sample_uniform, quat_apply

from .juggling_agent_env_cfg import JugglingAgentEnvCfg

class JugglingAgentEnv(DirectRLEnv):
    cfg: JugglingAgentEnvCfg

    def __init__(self, cfg: JugglingAgentEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        device = self.device

        self.left_hand_idx, _ =  self.left_hand.find_joints(".*")
        self.right_hand_idx, _ = self.right_hand.find_joints(".*")

        flex_regex = "(FF|MF|RF|LF)J(4|3|2)|THJ.*|LFJ5"
        abduction_regex = "(FF|MF|RF|LF)J1"
        
        self.left_elbow_joint, _ = self.left_hand.find_joints("elbow_(rotate|bend)")
        self.left_wrist_joints, _ = self.left_hand.find_joints("WR.*")
        self.left_finger_flex_joints, _ = self.left_hand.find_joints(flex_regex)
        self.left_finger_abduction_joints, _ = self.left_hand.find_joints(abduction_regex)
        
        self.right_elbow_joint, _ = self.right_hand.find_joints("elbow_(rotate|bend)")
        self.right_wrist_joints, _ = self.right_hand.find_joints("WR.*")
        self.right_finger_flex_joints, _ = self.right_hand.find_joints(flex_regex)
        self.right_finger_abduction_joints, _ = self.right_hand.find_joints(abduction_regex)

        self.init_left_joint_pos = self._build_init_joint_pose(self.left_hand, self.cfg.left_joint_pos)
        self.init_right_joint_pos = self._build_init_joint_pose(self.right_hand, self.cfg.right_joint_pos)
        
        self.left_lower_limits = self.left_hand.data.soft_joint_pos_limits[..., 0].clone()
        self.left_upper_limits = self.left_hand.data.soft_joint_pos_limits[..., 1].clone()
        self.right_lower_limits = self.right_hand.data.soft_joint_pos_limits[..., 0].clone()
        self.right_upper_limits = self.right_hand.data.soft_joint_pos_limits[..., 1].clone()

        self.ball_spawn_offsets = torch.tensor(self.cfg.ball_offset, device=self.device, dtype=torch.float32)
        self.ball_anchors = torch.tensor(self.cfg.ball_anchor, device=self.device, dtype=torch.float32)

        # we use a combintation of palm and knuckle position to get a more accurate hand center
        wrist_L_ids, _ = self.left_hand.find_bodies(".*palm")
        wrist_R_ids, _ = self.right_hand.find_bodies(".*palm")
        self.left_wrist_idx = wrist_L_ids[0]
        self.right_wrist_idx = wrist_R_ids[0]

        knuckle_L_ids, _ = self.left_hand.find_bodies(".*mfproximal")
        knuckle_R_ids, _ = self.right_hand.find_bodies(".*mfproximal")
        self.left_knuckle_idx = knuckle_L_ids[0]
        self.right_knuckle_idx = knuckle_R_ids[0]

        self.init_ball_pos = torch.tensor(self.cfg.init_ball_pos, device=device, dtype=torch.float32)
        self.reward_buffer = torch.zeros(self.num_envs, device=device)
        self.hoarding_threshold = 0

        self.action_dim = self.cfg.action_space
        self.actions = torch.zeros((self.num_envs, self.action_dim), device=device)
        self.actions_smooth = torch.zeros((self.num_envs, self.action_dim), device=device)
        self.prev_actions_smooth = torch.zeros((self.num_envs, self.action_dim), device=device)

        # Precompute constants
        self.distance_target2ground = torch.tensor(1.0 / (self.cfg.target_height - self.cfg.ground_height), device=device)
        self.cross_pos = torch.tensor(1.0, device=device)
        self.cross_neg = torch.tensor(-1.0, device=device)

        self.sigma_apex_height_coeff = torch.tensor(-1.0 / (2 * (self.cfg.sigma_apex_height ** 2)), device=device)
        self.sigma_drop_distance_coeff = torch.tensor(-1.0 / (2 * (self.cfg.sigma_drop_distance ** 2)), device=device)
        self.sigma_rythem_coeff = torch.tensor(-1.0 / (2 * (self.cfg.sigma_rythem ** 2)), device=device)

        self._allocate_tensors()

    def _setup_scene(self):
        self.left_hand = Articulation(self.cfg.left_hand_cfg)
        self.right_hand = Articulation(self.cfg.right_hand_cfg)
        self.ball1 = RigidObject(self.cfg.ball1_cfg)

        self.scene.articulations["left_hand"] = self.left_hand
        self.scene.articulations["right_hand"] = self.right_hand
        self.scene.rigid_objects["ball1"] = self.ball1
        # self.scene.rigid_objects["ball2"] = self.ball1
        # self.scene.rigid_objects["ball3"] = self.ball1

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.balls = [self.ball1]
        self.scene.clone_environments(copy_from_source=False)
        
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        current_action = self.actions
        alpha = 0.8 # smoothing factor
        self.actions_smooth = alpha * current_action + (1.0 - alpha) * self.actions_smooth
        
        def scale(x, lower, upper): # we scale so the values are both normalized and within joint limits
            return 0.5 * (x + 1) * (upper - lower) + lower
        
        action_left = self.actions_smooth[:, :5]
        target_left = self.target_L
        target_left.fill_(0.0)
        
        target_left[:, self.left_elbow_joint] = scale(
            action_left[:, 0:2], 
            self.left_lower_limits[:, self.left_elbow_joint], 
            self.left_upper_limits[:, self.left_elbow_joint]
        )
        target_left[:, self.left_wrist_joints] = scale(
            action_left[:, 2:4], 
            self.left_lower_limits[:, self.left_wrist_joints], 
            self.left_upper_limits[:, self.left_wrist_joints]
        )
        target_left[:, self.left_finger_flex_joints] = scale(
            action_left[:, 4].unsqueeze(-1), 
            self.left_lower_limits[:, self.left_finger_flex_joints], 
            self.left_upper_limits[:, self.left_finger_flex_joints]
        )


        action_right = self.actions_smooth[:, 5:]
        target_right = self.target_R
        target_right.fill_(0.0)
        
        target_right[:, self.right_elbow_joint] = scale(
            action_right[:, 0:2], 
            self.right_lower_limits[:, self.right_elbow_joint], 
            self.right_upper_limits[:, self.right_elbow_joint]
        )
        target_right[:, self.right_wrist_joints] = scale(
            action_right[:, 2:4], 
            self.right_lower_limits[:, self.right_wrist_joints], 
            self.right_upper_limits[:, self.right_wrist_joints]
        )
        target_right[:, self.right_finger_flex_joints] = scale(
            action_right[:, 4].unsqueeze(-1), 
            self.right_lower_limits[:, self.right_finger_flex_joints], 
            self.right_upper_limits[:, self.right_finger_flex_joints]
        )

        self.left_hand.set_joint_position_target(target_left, joint_ids=self.left_hand_idx)
        self.right_hand.set_joint_position_target(target_right, joint_ids=self.right_hand_idx)

        self.prev_actions_smooth = self.actions_smooth.clone()

    def _allocate_tensors(self):
        num_envs = self.num_envs
        device = self.device

        self.ball_pos = torch.zeros((num_envs, 3), device=device)
        self.ball_vel = torch.zeros((num_envs, 3), device=device)
        self.hand_pos = torch.zeros((num_envs, self.cfg.num_hands, 3), device=device)

        self.catch_events = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.drop_events = torch.zeros((num_envs, ), dtype=torch.bool, device=device)
        self.throw_events = torch.zeros((num_envs, ), dtype=torch.bool, device=device)

        self.ball_peak_height = torch.zeros(num_envs, device=device)
        self.ball_initial_height = torch.zeros(num_envs, device=device)
        self.ball_drop_pos = torch.zeros((num_envs, 3), device=device)

        self.ball_throw_hand = torch.full((num_envs, ), -1, device=device, dtype=torch.long)
        self.ball_catch_hand = torch.full((num_envs, ), -1, device=device, dtype=torch.long)
        self.prev_in_hand = torch.zeros((num_envs, self.cfg.num_hands), device=device, dtype=torch.bool)
        self.in_hand = torch.zeros((num_envs, self.cfg.num_hands), device=device, dtype=torch.bool)

        self.ball_pos_flat = torch.zeros((num_envs, 3), device=device)
        self.ball_vel_flat = torch.zeros((num_envs, 3), device=device)
        self.hand_pos_flat = torch.zeros((num_envs, 6), device=device)

        total_dof = 52
        self.joint_pos = torch.zeros((num_envs, total_dof), device=device)
        self.joint_vel = torch.zeros((num_envs, total_dof), device=device)

        self.target_L = torch.zeros((num_envs, len(self.left_hand_idx)), device=self.device)
        self.target_R = torch.zeros((num_envs, len(self.right_hand_idx)), device=self.device)

        num_ball_up_rewards = [0.0, 1.0, 0.5, 0.0] # 0, 1, 2, 3 or balls up
        self.num_ball_up_rewards_lookup = torch.tensor(num_ball_up_rewards, device=device)

        self.ball_throw_time = torch.zeros(num_envs, device=device, dtype=torch.float)
        self.hand_throw_last_time = torch.full((num_envs, self.cfg.num_hands), -1.0, device=device, dtype=torch.float)
        self.hand_throw_intervals = torch.full((num_envs, self.cfg.num_hands), -1.0, device=device, dtype=torch.float)
        self.holding_duration = torch.zeros((num_envs, ), device=device, dtype=torch.float)

    def _get_target_hand_pos(self):
        # We ensure no -1 indices (though reset handles this) by clamping
        throw_hand = self.ball_throw_hand.clamp(min=0, max=1)
        target_hand_idx = 1 - throw_hand # Target is opposite of throw hand (1 - 0 = 1, 1 - 1 = 0)

        idx = target_hand_idx.view(-1, 1, 1).expand(-1, 1, 3)
        
        target_pos = torch.gather(self.hand_pos, 1, idx).squeeze(1)
        return target_pos

    def _get_observations(self) -> dict:
        left_pos = self.left_hand.data.joint_pos
        left_vel = self.left_hand.data.joint_vel
        right_pos = self.right_hand.data.joint_pos
        right_vel = self.right_hand.data.joint_vel

        hand_split = left_pos.shape[1]

        self.joint_pos[:, :hand_split] = left_pos
        self.joint_pos[:, hand_split:] = right_pos
        self.joint_vel[:, :hand_split] = left_vel
        self.joint_vel[:, hand_split:] = right_vel

        l_wrist = self.left_hand.data.body_pos_w[:, self.left_wrist_idx]
        r_wrist = self.right_hand.data.body_pos_w[:, self.right_wrist_idx]
        l_knuckle = self.left_hand.data.body_pos_w[:, self.left_knuckle_idx]
        r_knuckle = self.right_hand.data.body_pos_w[:, self.right_knuckle_idx]

        bias = 0.775 # center offset towards knuckles
        self.hand_pos[:, 0] = (l_wrist * (1 - bias)) + (l_knuckle * bias)
        self.hand_pos[:, 1] = (r_wrist * (1 - bias)) + (r_knuckle * bias)

        self.ball_pos = self.ball1.data.root_pos_w
        self.ball_vel = self.ball1.data.root_vel_w[:, :3]

        self.ball_pos_flat[:] = self.ball_pos
        self.ball_vel_flat[:] = self.ball_vel
        self.hand_pos_flat[:] = self.hand_pos.reshape(self.num_envs, -1)

        self.left_quaternion = self.left_hand.data.body_quat_w[:, self.left_wrist_idx]
        self.right_quaternion = self.right_hand.data.body_quat_w[:, self.right_wrist_idx]

        self.detect_events()
        
        obs = torch.cat(
            (
                self.joint_pos,
                self.joint_vel,
                self.actions_smooth.detach(), # need detach to prevent gradients flowing back
                self.ball_pos_flat,
                self.ball_vel_flat,
                self.hand_pos_flat,
                self.left_quaternion,
                self.right_quaternion,
            ),
            dim=1,
        )
        obs = torch.nan_to_num(obs)
        observations = {"policy": obs}
        return observations

    def detect_events(self):
        self.catch_events[:] = False
        self.drop_events[:] = False
        self.throw_events[:] = False

        # Hand states
        diff = self.ball_pos[:, None, :] - self.hand_pos
        distance_to_hand = torch.norm(diff, dim=-1)
        height_diff = diff[..., 2]
        
        is_close = distance_to_hand < self.cfg.catch_radius
        is_above = height_diff > -0.02
        self.in_hand = is_close & is_above
        
        in_any_hand = self.in_hand.any(dim=1)
        
        self.holding_duration = torch.where(in_any_hand, self.holding_duration + self.step_dt, 0.0)
        
        vel_ball_z = self.ball_vel[..., 2] # Vertical velocity of ball

        ################
        # Detect throw #
        ################
        was_any_in_hand = self.prev_in_hand.any(dim=1)
        any_in_hand_now = self.in_hand.any(dim=1)
        throw_mask = (was_any_in_hand & (~any_in_hand_now)) & (vel_ball_z > self.cfg.min_vertical_velocity)
        
        self.throw_events.copy_(throw_mask)

        curr_t = self.sim.current_time
        
        # Find which hand held it last (Argmax over bool gives index of True)
        # This gives us a hand index for EVERY env, even those who didn't throw (we mask them later)
        thrower_hand_idx = self.prev_in_hand.float().argmax(dim=1)
        
        self.ball_throw_hand = torch.where(throw_mask, thrower_hand_idx, self.ball_throw_hand)
        
        self.ball_throw_time = torch.where(throw_mask, curr_t, self.ball_throw_time)
        self.ball_initial_height = torch.where(throw_mask, self.ball_pos[:, 2], self.ball_initial_height)
        self.ball_peak_height = torch.where(throw_mask, self.ball_pos[:, 2], self.ball_peak_height)

        # Update Timers (Hands)
        # We need to update specifically the left or right index
        throw_hand = torch.nn.functional.one_hot(thrower_hand_idx, num_classes=self.cfg.num_hands).bool()
        
        update_mask = throw_mask.unsqueeze(1) & throw_hand
        
        delta_t = curr_t - self.hand_throw_last_time
        self.hand_throw_intervals = torch.where(update_mask, delta_t, self.hand_throw_intervals)
        self.hand_throw_last_time = torch.where(update_mask, curr_t, self.hand_throw_last_time)

        self.ball_peak_height = torch.max(self.ball_peak_height, self.ball_pos[:, 2])

        ################
        # Detect Catches
        ################
        # caught = self.in_hand.any(dim=1)
        # stable_catch = (self.holding_duration >= self.cfg.hold_time_threshold) & (self.holding_duration < (self.cfg.hold_time_threshold + self.step_dt + 1e-6))
        # was_thrown = (self.ball_throw_time > 0.0)
        # descending = vel_ball_z < self.cfg.min_vertical_velocity
        
        # valid_catch_mask = stable_catch & caught & descending & was_thrown
        # self.catch_events.copy_(valid_catch_mask)
        
        closest_hand = distance_to_hand.argmin(dim=1)
        self.ball_catch_hand = torch.where(in_any_hand, closest_hand, self.ball_catch_hand)

        ################
        # Detect Drops
        ################
        valid_throw = self.ball_peak_height > self.cfg.min_throw_height
        ball_height_pos = self.ball_pos[:, 2]
        dropped = (~self.in_hand.any(dim=1)) & (ball_height_pos < self.cfg.ground_height + 0.1)
        
        valid_drop = dropped & valid_throw
        self.drop_events.copy_(valid_drop)
        
        self.ball_drop_pos = torch.where(valid_drop.unsqueeze(1), self.ball_pos, self.ball_drop_pos)

        # Update Prev
        self.prev_in_hand.copy_(self.in_hand)

    def _get_rewards(self) -> torch.Tensor:
        self.reward_buffer.fill_(0.0)

        # initialize some commonly used variables
        throw_hand_idx = self.ball_throw_hand.clamp(min=0, max=1)
        target_hand_idx = 1 - throw_hand_idx
        target_hand_pos = self._get_target_hand_pos()
        dist_to_target = torch.norm(self.ball_pos - target_hand_pos, dim=-1)
        is_approaching = (self.ball_vel[:, 2] < -0.1) & (dist_to_target < 0.4) & (~self.in_hand.any(dim=1))

        # --- Continuous Penalties ---
        
        ################
        #   Hoarding   #
        ################
        ''' Penalty for holding onto the ball that needs to be thrown for too long '''

        hoarding_time = torch.clamp(self.holding_duration - self.cfg.hoarding_time_threshold, min=0.0)
        r_hoard = -torch.clamp(self.cfg.w_hoarding * (hoarding_time.square()), max=5.0)
        self.reward_buffer += r_hoard

        ################
        #   Jitter    #
        ################
        ''' Penalty for rapid changes in actions, encouraging smooth movements '''
        delta_a = self.actions_smooth - self.prev_actions_smooth
        jitter = torch.sum(delta_a**2, dim=-1)
        r_jitter = -self.cfg.w_jitter * jitter
        self.reward_buffer += r_jitter

        ##################
        # Hand Proximity #
        ##################
        ''' Penalty for hands being too close together. Implimented to eliminate a annoying behaviour where the agent would "catch" the ball with both hands '''
        left_hand_pos = self.hand_pos[:, 0]
        right_hand_pos = self.hand_pos[:, 1]
        
        hand_separation = torch.norm(left_hand_pos - right_hand_pos, dim=-1)
        
        is_hand_touching = (hand_separation < self.cfg.min_hand_dist)
        r_hand_touching = -self.cfg.w_hands_touching * is_hand_touching.float()
        valid_stance = (~is_hand_touching).float()

        self.reward_buffer += r_hand_touching

        # --- Continuous Rewards ---

        ##########
        # Height #
        ##########
        ''' Reward for the highest ball that is currently going up. The full reward is only given if only one ball is going up, and is scaled down for multiple balls going up. '''
        height_cords = self.ball_pos[:, 2].unsqueeze(1)
        vertical_velocities = self.ball_vel[:, 2].unsqueeze(1)
        
        going_up = vertical_velocities > self.cfg.min_vertical_velocity
        is_held = self.in_hand.any(dim=1, keepdim=True)
        up_mask = going_up & (~is_held)
        
        num_up = up_mask.sum(dim=1)
        safe_indices = torch.clamp(num_up, max=len(self.num_ball_up_rewards_lookup)-1)
        num_scale = self.num_ball_up_rewards_lookup[safe_indices]

        masked_heights = torch.where(up_mask, height_cords, -1.0)
        highest_ball_height = masked_heights.max(dim=1).values.clamp(
            min=self.cfg.ground_height, max=self.cfg.target_height
        )
        
        # height_r = (highest_ball_height - self.cfg.ground_height) * self.distance_target2ground
        # height_r_norm = height_r.clamp(0.0, 1.0)
        r_height = torch.exp((highest_ball_height - self.cfg.target_height).square() * self.sigma_apex_height_coeff)
        
        r_height = self.cfg.w_highest * r_height * num_scale
        self.reward_buffer += r_height

        ##########
        # Rhythm #
        ##########
        ''' Reward for throwing the ball at regular intervals, encouraging a consistent juggling rhythm that resembles human juggling patterns '''
        throw_hand_idx = self.ball_throw_hand
        idx_expanded_t = throw_hand_idx.unsqueeze(1)
        delta_t_val = torch.gather(self.hand_throw_intervals, 1, idx_expanded_t).squeeze(1)
        
        target_interval = self.cfg.target_rythem * 2.0
        GT = torch.exp((delta_t_val - target_interval).square() * self.sigma_rythem_coeff)
        
        r_rhythm = (self.cfg.w_rythem * GT) * self.throw_events.float()
        self.reward_buffer += r_rhythm


        #############
        # Open Hand #
        #############
        ''' Reward for opening the target hand when the ball is falling, to facilitate successful catches '''
        # Actions are (N, 10). Left Fingers is index 4, Right Fingers is index 9.
        # We can gather them efficiently.
        # Create indices: Left=4, Right=9
        #action_indices = torch.where(target_hand_idx == 0, 4, 9).unsqueeze(1)
        #Get Orientation (Palm Up)
        # We need the quaternion of the TARGET hand
        # left_quaternion / right_quaternion are (N, 4)
        target_quat = torch.where(target_hand_idx.unsqueeze(1) == 0, self.left_quaternion, self.right_quaternion)
        #target_finger_action = torch.gather(self.actions_smooth, 1, action_indices).squeeze(1)
        palm_normal_local = torch.tensor([0.0, -1.0, 0.0], device=self.device).expand(self.num_envs, -1)
        palm_normal_world = quat_apply(target_quat, palm_normal_local)
        palm_up_score = palm_normal_world[:, 2]

        z_threshold = 0.5 # palm mostly up
        valid_orientation_factor = torch.clamp((palm_up_score - z_threshold) * 5.0, min=0.0, max=1.0)
        
        # gaussian based on ideal finger flexation
        # error_sq = (target_finger_action - self.cfg.target_finger_flexation).square()
        # open_bonus = torch.exp(-error_sq * self.cfg.sigma_finger_flexation)
        r_open = self.cfg.w_hand_up * is_approaching.float() * valid_orientation_factor * valid_stance
        self.reward_buffer += r_open


        # --- Sparse Rewards ---

        ##########
        #  Catch #
        ##########
        ''' Reward for successfully catching the ball. Multiplies the height of the throw with a large cross-hand bonus. Negative reward if caught with the same hand. '''       
        is_holding = self.in_hand.any(dim=1)
        was_thrown = (self.ball_throw_time > 0.0)
        in_payout_window = (self.holding_duration < self.cfg.hold_time_threshold)
        valid_catch_mask = is_holding & was_thrown & in_payout_window

        # calculate Gh
        peak = self.ball_peak_height
        start_height = self.ball_initial_height
        delta_height = torch.clamp(peak - start_height, min=0.0)
        Gh = torch.exp((delta_height - self.cfg.target_delta_height).square() * self.sigma_apex_height_coeff)
        
        throw_hand = self.ball_throw_hand
        catch_hand = self.ball_catch_hand
        cross = torch.where(throw_hand != catch_hand, self.cross_pos, self.cross_neg)
        
        r_catch = self.cfg.w_catch * Gh * cross * valid_catch_mask.float()
        self.reward_buffer += r_catch
        # catch_val = self.cfg.w_catch * Gh * cross
        # # Apply mask
        # r_catch = catch_val * self.catch_events.float()
        # self.reward_buffer += r_catch
        
        # self.ball_peak_height = torch.where(self.catch_events, 0.0, self.ball_peak_height)

        ##################
        # Lateral Throws #
        ##################
        ''' Reward for throwing the ball laterally towards the opposite hand, encouraging throws on the correct axis '''
        direction_target = torch.where(throw_hand == 0, 1.0, -1.0) # Left hand throws to +Y, Right hand to -Y
        vel_y = self.ball_vel[:, 1]
        # Reward only on throw event. Clamp so we don't punish "wrong" direction (just 0 reward).
        r_lateral = self.cfg.w_lateral * (vel_y * direction_target).clamp(min=0.0) * self.throw_events.float()
        self.reward_buffer += r_lateral

        ############
        #   Drop   #
        ############
        ''' Reward for dropping the ball close to the target hand, encouraging accurate throws. Scaled by the height of the throw to prevent just dropping onto the floor. '''
        drop_pos = self.ball_drop_pos

        dist_drop = torch.norm(drop_pos - target_hand_pos, dim=-1)
        Gd = torch.exp(dist_drop.square() * self.sigma_drop_distance_coeff)
        
        drop_bias = 0.7
        height_bias = 0.3
        drop_val = self.cfg.w_drop * (height_bias * Gh + drop_bias * Gd)
        
        r_drop = drop_val * self.drop_events.float()
        self.reward_buffer += r_drop
        self.ball_peak_height = torch.where(self.drop_events, 0.0, self.ball_peak_height)

 
        # -------------------------------------------------------
        # LOGGING (This puts data into TensorBoard/Console)
        # -------------------------------------------------------
        # We use 'episode' prefix so RL-Games aggregates it properly
        
        # Log the MEAN reward per step across all environments
        self.extras["logs/rewards_hoard"] = r_hoard.mean()
        self.extras["logs/rewards_jitter"] = r_jitter.mean()
        self.extras["logs/rewards_height"] = r_height.mean()
        self.extras["logs/rewards_hands_touching"] = r_hand_touching.mean()
        self.extras["logs/rewards_open"] = r_open.mean()
        
        # For sparse events (Catch/Drop), we might want the SUM or MEAN
        # Mean tells us "Average reward per step per agent"
        self.extras["logs/rewards_catch"] = r_catch.mean()
        self.extras["logs/rewards_drop"] = r_drop.mean()
        self.extras["logs/rewards_rhythm"] = r_rhythm.mean()
        self.extras["logs/rewards_lateral"] = r_lateral.mean()
        
        # Log event counts
        self.extras["logs/event_catch_count"] = self.catch_events.float().mean()
        self.extras["logs/event_drop_count"] = self.drop_events.float().mean()

        self.prev_actions_smooth = self.actions_smooth.clone()
        return self.reward_buffer

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        ball_height = self.ball_pos[:, 2]
        is_ball_dropped = (ball_height < (self.cfg.ground_height + 0.1)) & (~self.in_hand.any(dim=1))
        
        relative_ball_pos = self.ball_pos - self.scene.env_origins
        ball_dist_xy = torch.norm(relative_ball_pos[:, :2], dim=-1)
        is_ball_out_of_bounds = ball_dist_xy > self.cfg.out_of_bounds_radius
        
        agent_terminated = is_ball_dropped | is_ball_out_of_bounds
        return agent_terminated, time_out

    def _build_init_joint_pose(self, hand: Articulation, targets: dict[str, float]):
        name_to_idx = {n: i for i, n in enumerate(hand.joint_names)}
        joint_pos = torch.zeros((hand.num_joints,), device=self.device)
        for name, val in targets.items():
            if name in name_to_idx:
                joint_pos[name_to_idx[name]] = val
        joint_pos = joint_pos.repeat(self.num_envs, 1)
        return joint_pos

    def _apply_init_joint_pose(self, hand: Articulation, joint_pos: torch.Tensor, env_ids, joint_ids):
        if env_ids is None:
            pos_target = joint_pos
        else:
            pos_target = joint_pos[env_ids]
        vel_target = torch.zeros_like(pos_target)
        hand.set_joint_position_target(pos_target, env_ids=env_ids, joint_ids = joint_ids)
        hand.set_joint_velocity_target(vel_target, env_ids=env_ids, joint_ids=joint_ids)
        hand.write_joint_state_to_sim(pos_target, vel_target, joint_ids=joint_ids, env_ids=env_ids)

    def _reset_ball_pos(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        ball = self.ball1
        default_state = ball.data.default_root_state[env_ids].clone()
        default_state[:, :3] = self.scene.env_origins[env_ids] 
        default_state[:, :3] += self.init_ball_pos[0, :] 
        default_state[:, 7:] = 0.0 
        ball.write_root_state_to_sim(default_state, env_ids=env_ids)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        self._apply_init_joint_pose(self.left_hand, self.init_left_joint_pos, env_ids, self.left_hand_idx)
        self._apply_init_joint_pose(self.right_hand, self.init_right_joint_pos, env_ids, self.right_hand_idx)
        
        self._reset_ball_pos(env_ids)
        
        for _ in range(1):
            self.sim.step()

        self.scene.update(dt=self.cfg.sim.dt)
        self.detect_events()

        active_anchors = self.cfg.ball_anchor[0]
        anchors = torch.tensor(active_anchors, device=self.device, dtype=torch.long)
        self.ball_throw_hand[env_ids] = anchors
        
        self.ball_catch_hand[env_ids] = -1

        self.catch_events[env_ids] = False
        self.drop_events[env_ids] = False
        self.throw_events[env_ids] = False
        
        self.ball_peak_height[env_ids] = self.ball_pos[env_ids, 2]
        self.ball_initial_height[env_ids] = self.ball_pos[env_ids, 2]
        
        self.actions[env_ids] = 0.0
        self.actions_smooth[env_ids] = 0.0
        self.prev_actions_smooth[env_ids] = 0.0
        
        self.hand_throw_last_time[env_ids] = self.sim.current_time
        self.hand_throw_intervals[env_ids] = 0.0
        self.ball_throw_time[env_ids] = 0.0
        self.holding_duration[env_ids] = 0.0

        self.prev_in_hand[env_ids] = self.in_hand[env_ids]
