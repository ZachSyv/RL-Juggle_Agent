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
from isaaclab.sensors import ContactSensor

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

        self.left_finger_closed = self.left_upper_limits[:, self.left_finger_flex_joints]
        self.left_finger_open = self.left_lower_limits[:, self.left_finger_flex_joints]

        self.right_finger_closed = self.right_upper_limits[:, self.right_finger_flex_joints]
        self.right_finger_open = self.right_lower_limits[:, self.right_finger_flex_joints]
        
        self.ball_spawn_offsets = torch.tensor(self.cfg.ball_offset, device=self.device, dtype=torch.float32)
        self.ball_anchors = torch.tensor(self.cfg.ball_anchor, device=self.device, dtype=torch.float32)

        left_hand_start_pos = self.cfg.hand_pos[0]
        left_ball_start_pos = self.cfg.ball_offset[0]
        self.left_start_pos = torch.tensor(
            [left_hand_start_pos[0] + left_ball_start_pos[0], left_hand_start_pos[1] + left_ball_start_pos[1], left_hand_start_pos[2] + left_ball_start_pos[2]],
            device=device, 
            dtype=torch.float32)

        right_hand_start_pos = self.cfg.hand_pos[1]
        right_ball_start_pos = self.cfg.ball_offset[2]
        self.right_start_pos = torch.tensor(
            [right_hand_start_pos[0] + right_ball_start_pos[0], right_hand_start_pos[1] + right_ball_start_pos[1], right_hand_start_pos[2] + right_ball_start_pos[2]],
            device=device, 
            dtype=torch.float32)

        # we use a combintation of palm and knuckle position to get a more accurate hand center
        wrist_L_ids, _ = self.left_hand.find_bodies(".*palm")
        wrist_R_ids, _ = self.right_hand.find_bodies(".*palm")
        self.left_wrist_idx = wrist_L_ids[0]
        self.right_wrist_idx = wrist_R_ids[0]

        knuckle_L_ids, _ = self.left_hand.find_bodies(".*mfproximal")
        knuckle_R_ids, _ = self.right_hand.find_bodies(".*mfproximal")
        self.left_knuckle_idx = knuckle_L_ids[0]
        self.right_knuckle_idx = knuckle_R_ids[0]

        self.initial_hand_positions = torch.tensor(self.cfg.hand_pos, device=device, dtype=torch.float32)

        self.init_ball_pos = torch.tensor(self.cfg.init_ball_pos, device=device, dtype=torch.float32)
        self.reward_buffer = torch.zeros(self.num_envs, device=device)
        self.hoarding_threshold = self.cfg.num_balls // 2 # number of balls allowed to be held before hoarding penalty applies

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
        # self.scene.rigid_objects["ball2"] = self.ball2
        # self.scene.rigid_objects["ball3"] = self.ball3
        self.balls = [self.ball1]
        self.scene.clone_environments(copy_from_source=False)

        self.contact_sensor_left_palm = ContactSensor(self.cfg.contact_sensor_left_palm)
        self.scene.sensors["contact_sensor_left_palm"] = self.contact_sensor_left_palm
        self.contact_sensor_left_metacarpal = ContactSensor(self.cfg.contact_sensor_left_metacarpal)
        self.scene.sensors["contact_sensor_left_metacarpal"] = self.contact_sensor_left_metacarpal
        self.left_hand_sensors = []
        for name in ["thumb", "index", "middle", "ring", "pinky"]:
            for joint in ["proximal", "middle", "distal"]:
                cfg_name = f"contact_sensor_left_{name}_{joint}"
                sensor = ContactSensor(getattr(self.cfg, cfg_name))
                self.scene.sensors[cfg_name] = sensor
                self.left_hand_sensors.append(sensor)
        self.contact_sensor_right_palm = ContactSensor(self.cfg.contact_sensor_right_palm)
        self.scene.sensors["contact_sensor_right_palm"] = self.contact_sensor_right_palm
        self.contact_sensor_right_metacarpal = ContactSensor(self.cfg.contact_sensor_right_metacarpal)
        self.scene.sensors["contact_sensor_right_metacarpal"] = self.contact_sensor_right_metacarpal
        self.right_hand_sensors = []
        for name in ["thumb", "index", "middle", "ring", "pinky"]:
            for joint in ["proximal", "middle", "distal"]:
                cfg_name = f"contact_sensor_right_{name}_{joint}"
                sensor = ContactSensor(getattr(self.cfg, cfg_name))
                self.scene.sensors[cfg_name] = sensor
                self.right_hand_sensors.append(sensor)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
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

        self.grip_state[:, 0] = torch.where(
            action_left[:, 4] > self.cfg.close_threshold, True,
            torch.where(action_left[:, 4] < self.cfg.open_threshold, False, self.grip_state[:, 0])
        )
        close_left = self.grip_state[:, 0]
        target_left[:, self.left_finger_flex_joints] = torch.where(
            close_left.unsqueeze(-1),
            self.left_finger_closed,
            self.left_finger_open
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

        self.grip_state[:, 1] = torch.where(
            action_right[:, 4] > self.cfg.close_threshold, True,
            torch.where(action_right[:, 4] < self.cfg.open_threshold, False, self.grip_state[:, 1])
        )
        close_right = self.grip_state[:, 1]
        target_right[:, self.right_finger_flex_joints] = torch.where(
            close_right.unsqueeze(-1),
            self.right_finger_closed,
            self.right_finger_open
        )

        self.left_hand.set_joint_position_target(target_left, joint_ids=self.left_hand_idx)
        self.right_hand.set_joint_position_target(target_right, joint_ids=self.right_hand_idx)

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
        self.catch_peak_height = torch.zeros(num_envs, device=device)

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

        self.grip_state = torch.zeros((num_envs, self.cfg.num_hands), device=device, dtype=torch.bool)

        self.target_L = torch.zeros((num_envs, len(self.left_hand_idx)), device=self.device)
        self.target_R = torch.zeros((num_envs, len(self.right_hand_idx)), device=self.device)

        self.initial_hand_target_position = torch.zeros((num_envs, self.cfg.num_hands, 3), device=device, dtype=torch.float)

        num_ball_up_rewards = [1.0, 1.0, 0.5, 0.0] # 0, 1, 2, 3 or balls up
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
        target_pos = torch.gather(self.initial_hand_target_position, 1, idx).squeeze(1)
        return target_pos

    def predict_landing_position(self, pos, vel, target_z, g=9.81):
        delta_z = target_z - pos[:, 2]
        velocity_z = vel[:, 2]

        # Physics: 0.5*g*t^2 - vz*t + dz = 0
        # We want the time 't' when the ball reaches target_z
        # Using quadratic formula: t = (vz + sqrt(vz^2 - 2*g*dz)) / g
        
        discriminant = velocity_z**2 - 2 * g * delta_z
        
        # If discriminant < 0, the ball never reaches the height (bad throw)
        valid_mask = discriminant >= 0
        
        # We only care about valid throws for the math
        # Clamp discriminant to 0 to avoid NaNs in sqrt, we will mask invalid ones later
        safe_discriminant = torch.clamp(discriminant, min=0.0)
        
        # Calculate time to impact
        # Note: We usually want the "plus" solution (the landing), not the launch
        t_flight = (velocity_z + torch.sqrt(safe_discriminant)) / g
        
        # Predict x and y at that time
        pred_x = pos[:, 0] + vel[:, 0] * t_flight
        pred_y = pos[:, 1] + vel[:, 1] * t_flight
        
        landing_pos = torch.stack([pred_x, pred_y, target_z], dim=1)
        return landing_pos, valid_mask    


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

        self.hand_pos[:, 0] = (l_wrist * (1 - self.cfg.center_of_hand_bias)) + (l_knuckle * self.cfg.center_of_hand_bias)
        self.hand_pos[:, 1] = (r_wrist * (1 - self.cfg.center_of_hand_bias)) + (r_knuckle * self.cfg.center_of_hand_bias)

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

        # Sensor Logic
        is_touching_left = torch.zeros((self.num_envs, ), device=self.device, dtype=torch.bool)
        is_touching_right = torch.zeros((self.num_envs, ), device=self.device, dtype=torch.bool)

        left_forces_palm = torch.norm(self.contact_sensor_left_palm.data.force_matrix_w.sum(dim=1), dim=-1).flatten()
        is_touching_left |= left_forces_palm > self.cfg.contact_threshold

        left_forces_metacarpal = torch.norm(self.contact_sensor_left_metacarpal.data.force_matrix_w.sum(dim=1), dim=-1).flatten()
        is_touching_left |= left_forces_metacarpal > self.cfg.contact_threshold

        left_finger_forces = [s.data.force_matrix_w for s in self.left_hand_sensors]
        left_all_finger_forces = torch.stack(left_finger_forces, dim=1)
        left_flat_forces = left_all_finger_forces.sum(dim=2)
        left_forces = torch.norm(left_flat_forces, dim=-1)
        max_left_forces = left_forces.max(dim=1).values.flatten()
        is_touching_left |= max_left_forces > self.cfg.contact_threshold


        right_forces_palm = torch.norm(self.contact_sensor_right_palm.data.force_matrix_w.sum(dim=1), dim=-1).flatten()
        is_touching_right |= right_forces_palm > self.cfg.contact_threshold

        right_forces_metacarpal = torch.norm(self.contact_sensor_right_metacarpal.data.force_matrix_w.sum(dim=1), dim=-1).flatten()
        is_touching_right |= right_forces_metacarpal > self.cfg.contact_threshold

        right_finger_forces = [s.data.force_matrix_w for s in self.right_hand_sensors]
        right_all_finger_forces = torch.stack(right_finger_forces, dim=1)
        right_flat_forces = right_all_finger_forces.sum(dim=2)
        right_forces = torch.norm(right_flat_forces, dim=-1)
        max_right_forces = right_forces.max(dim=1).values.flatten()
        is_touching_right |= max_right_forces > self.cfg.contact_threshold
        
        # Hand states
        ball_z = self.ball_pos[:, 2].unsqueeze(1)
        hand_z = self.hand_pos[:, :, 2]
        is_above = ball_z > (hand_z - 0.02)

        self.in_hand[:, 0] = is_touching_left & is_above[:, 0]
        self.in_hand[:, 1] = is_touching_right & is_above[:, 1]
        
        in_any_hand = self.in_hand.any(dim=1)
        
        self.holding_duration = torch.where(in_any_hand, self.holding_duration + self.step_dt, 0.0)
        
        ################
        # Detect throw #
        ################
        was_any_in_hand = self.prev_in_hand.any(dim=1)
        any_in_hand_now = self.in_hand.any(dim=1)
        vel_ball_z = self.ball_vel[..., 2]

        # calculate delta peak height so we can filter drops out of throws
        g = 9.81
        delta_peak_height = (torch.sign(vel_ball_z) * (vel_ball_z ** 2)) / (2 * g)
        is_valid_throw_height = delta_peak_height >= self.cfg.min_delta_throw_height

        throw_mask = (was_any_in_hand & (~any_in_hand_now)) & (vel_ball_z > self.cfg.min_vertical_velocity) & is_valid_throw_height

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
        was_any_in_hand = self.prev_in_hand.any(dim=1)
        any_in_hand_now = self.in_hand.any(dim=1)
        
        just_caught = any_in_hand_now & (~was_any_in_hand)
        self.catch_peak_height = torch.where(just_caught, self.ball_peak_height, self.catch_peak_height)
        self.ball_peak_height = torch.where(just_caught, self.ball_pos[:, 2], self.ball_peak_height)
        
        current_holding_hand = self.in_hand.float().argmax(dim=1)
        self.ball_catch_hand = torch.where(in_any_hand, current_holding_hand, self.ball_catch_hand)

        ################
        # Detect Drops
        ################
        ball_height_pos = self.ball_pos[:, 2]
        dropped = (~self.in_hand.any(dim=1)) & (ball_height_pos < self.cfg.ground_height + 0.1)
        
        self.drop_events.copy_(dropped)
        
        self.ball_drop_pos = torch.where(dropped.unsqueeze(1), self.ball_pos, self.ball_drop_pos)

        self.prev_in_hand.copy_(self.in_hand)

    def _get_rewards(self) -> torch.Tensor:
        self.reward_buffer.fill_(0.0)

        # initialize some commonly used variables
        throw_hand_idx = self.ball_throw_hand.clamp(min=0, max=1)
        target_hand_idx = 1 - throw_hand_idx
        target_pos_fixed = self._get_target_hand_pos()
        target_hand_pos = target_hand_idx.view(-1, 1, 1).expand(-1, 1, 3)
        target_hand_pos = torch.gather(self.initial_hand_target_position, 1, target_hand_pos).squeeze(1)
        dist_to_target = torch.norm(self.ball_pos - target_hand_pos, dim=-1)
        is_approaching = (self.ball_vel[:, 2] < -0.1) & (~self.in_hand.any(dim=1))
        is_above_hand = (self.ball_pos[:, 2] > target_hand_pos[:, 2] + 0.01)
        was_thrown = (self.ball_throw_time > 0.0)

        # --- Continuous Penalties ---
        
        ################
        #   Hoarding   #
        ################
        ''' Penalty for holding onto the ball that needs to be thrown for too long '''

        hoarding_time = torch.clamp(self.holding_duration - self.cfg.hoarding_time_threshold, min=0.0)
        r_hoard = -self.cfg.w_hoarding * (hoarding_time.square())
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

        ##################################
        # Hand Catch Position Prediction #
        ##################################
        landing_pos, valid_mask = self.predict_landing_position(
            self.ball_pos,
            self.ball_vel,
            target_z=target_hand_pos[:, 2],
        )
        catch_error_xy = landing_pos[:, :2] - target_hand_pos[:, :2]
        dist_catch_xy = torch.norm(catch_error_xy, dim=-1)
        catch_prediction_mask = valid_mask & is_approaching & is_above_hand & was_thrown

        r_catch_position = (torch.exp(-((dist_catch_xy ** 2) / (2 * (self.cfg.sigma_catch_position ** 2))))) * self.cfg.w_catch_prediction
        self.reward_buffer += r_catch_position * catch_prediction_mask.float()


        ##########
        # Rhythm #
        ##########
        ''' Reward for throwing the ball at regular intervals, encouraging a consistent juggling rhythm that resembles human juggling patterns '''
        # throw_hand_idx = self.ball_throw_hand
        # idx_expanded_t = throw_hand_idx.unsqueeze(1)
        # delta_t_val = torch.gather(self.hand_throw_intervals, 1, idx_expanded_t).squeeze(1)
        
        # target_interval = self.cfg.target_rythem * 2.0
        # rythem_error = (delta_t_val - target_interval).square()
        # GT = torch.exp(-rythem_error / (2 * (self.cfg.sigma_rythem ** 2)))
        
        # r_rhythm = (self.cfg.w_rythem * GT) * self.throw_events.float()
        # self.reward_buffer += r_rhythm


        # --- Sparse Rewards ---

        ##########
        #  Catch #
        ##########
        ''' Reward for successfully catching the ball. Multiplies the height of the throw with a large cross-hand bonus. Negative reward if caught with the same hand. '''       
        is_holding = self.in_hand.any(dim=1)
        in_payout_window = (self.holding_duration < self.cfg.catch_payout_time)
        valid_catch_mask = is_holding & was_thrown & in_payout_window

        # calculate Gh
        peak = self.catch_peak_height
        start_height = self.ball_initial_height
        delta_height = torch.clamp(peak - start_height, min=0.0)
        catch_height_error = (delta_height - self.cfg.target_delta_height).square()
        Gh = torch.exp((-catch_height_error) / (2 * (self.cfg.sigma_apex_height ** 2)))
        
        catch_height = self.ball_pos[:, 2]
        excess_catch_height = torch.clamp(catch_height - self.cfg.max_catch_height, min=0.0)
        catch_height_penalty = torch.exp(-excess_catch_height.square() / (2 * (self.cfg.sigma_catch_height ** 2)))

        throw_hand = self.ball_throw_hand
        catch_hand = self.ball_catch_hand
        cross = torch.where(throw_hand != catch_hand, self.cross_pos, self.cross_neg)
        
        r_catch = self.cfg.w_catch * Gh * cross * valid_catch_mask.float() * catch_height_penalty
        self.reward_buffer += r_catch

        #########
        # Throw #
        #########
        ball_speed = torch.norm(self.ball_vel, dim=-1)

        vertical_velocities = self.ball_vel[:, 2].unsqueeze(1)

        g = 9.81
        going_up = vertical_velocities > self.cfg.min_vertical_velocity
        is_held = self.in_hand.any(dim=1, keepdim=True)
        up_mask = going_up & (~is_held)
        target_velocity = math.sqrt(2 * g * self.cfg.target_delta_height)
        will_go_high_enough = torch.clamp(vertical_velocities/target_velocity, min=0.0, max=1.0).squeeze(1)
        
        valid_throw_velocity = ball_speed >= self.cfg.min_throw_velocity
        clamped_speed = torch.clamp(ball_speed, max=self.cfg.max_speed_reward)
        r_power = torch.log(1 + clamped_speed) * up_mask.squeeze(1).float() * will_go_high_enough

        drift_velocity = torch.abs(self.ball_vel[:, 0])
        excess_drift = torch.clamp(drift_velocity - self.cfg.max_drift_velocity, min=0.0)
        r_drift = excess_drift.square()

        
        # delta height score based on vertical velocity, we don't use the predicted height because the squaring makes it too punishing early on
        current_ball_height = self.ball_pos[:, 2].unsqueeze(1)
        vertical_velocity_error = (vertical_velocities - target_velocity)
        r_delta_height = torch.exp(-vertical_velocity_error.square() / (2 * (self.cfg.sigma_delta_throw ** 2))).squeeze(1)

        # calculate if the ball is going up or down
        # kinetic_term = (torch.sign(vertical_velocities) * vertical_velocities.square()) / (2 * g)
        # predicted_abs_peak_height = current_ball_height + kinetic_term
        #predicted_delta_peak_height = predicted_abs_peak_height - current_ball_height

        # absolute height score
        predicted_peak_height = current_ball_height + (torch.sign(vertical_velocities) * vertical_velocities.square()) / (2 * g)
        apex_height_error = predicted_peak_height - self.cfg.target_height
        r_apex_height = (torch.exp(-apex_height_error.square() / (2 * (self.cfg.sigma_apex_height ** 2))).squeeze(1)) #* valid_throw_velocity.float()

        # num balls scale        
        num_up = up_mask.sum(dim=1)
        safe_indices = torch.clamp(num_up, max=len(self.num_ball_up_rewards_lookup)-1)
        num_scale = self.num_ball_up_rewards_lookup[safe_indices]

        # ballistic prediction
        landing_pos, valid_mask = self.predict_landing_position(
            self.ball_pos, 
            self.ball_vel, 
            target_z=target_pos_fixed[:, 2],
        )

        error_xy = landing_pos[:, :2] - target_pos_fixed[:, :2]
        dist_xy = torch.norm(error_xy, dim=-1)
        GT = torch.exp(-dist_xy.square() / (2 * (self.cfg.sigma_throw_accuracy ** 2)))
        r_accuracy = GT * valid_mask.float() * will_go_high_enough

        r_throw = (r_accuracy * self.cfg.w_accuracy + r_apex_height * self.cfg.w_apex_height + r_delta_height * self.cfg.w_delta_throw + r_power * self.cfg.w_throw_power - r_drift * self.cfg.w_drift) * num_scale * self.throw_events.float()
        self.reward_buffer += r_throw

        ############
        #   Drop   #
        ############
        ''' Penalty for dropping the ball, smaller the closer to the target hand, encouraging accurate throws. '''
        drop_pos = self.ball_drop_pos

        dist_drop = torch.norm(drop_pos - target_pos_fixed, dim=-1)
        Gd = torch.exp(-dist_drop.square() / (2 * (self.cfg.sigma_drop_distance ** 2)))
        
        # calculate Linear Interpolation penatly
        r_drop = (((self.cfg.min_drop_penalty * Gd) + (self.cfg.max_drop_penalty * (1.0 - Gd))) * self.drop_events.float() * self.cfg.w_drop)
        self.reward_buffer += r_drop 
        self.ball_peak_height = torch.where(self.drop_events, 0.0, self.ball_peak_height)

 
        # -------------------------------------------------------
        # LOGGING (This puts data into TensorBoard/Console)
        # -------------------------------------------------------
        # We use 'episode' prefix so RL-Games aggregates it properly
        
        # Log the MEAN reward per step across all environments
        self.extras["logs/rewards_hoard"] = r_hoard.mean()
        self.extras["logs/rewards_jitter"] = r_jitter.mean()
        self.extras["logs/rewards_catch_position"] = r_catch_position.mean()
        self.extras["logs/rewards_hands_touching"] = r_hand_touching.mean()
        
        # For sparse events (Catch/Drop), we might want the SUM or MEAN
        # Mean tells us "Average reward per step per agent"
        self.extras["logs/rewards_catch"] = r_catch.mean()
        self.extras["logs/rewards_drop"] = r_drop.mean()
        self.extras["logs/rewards_throw"] = r_throw.mean()
        self.extras["logs/rewards_power"] = r_power.mean() * self.cfg.w_throw_power
        self.extras["logs/rewards_drift"] = -r_drift.mean() * self.cfg.w_drift
        self.extras["logs/potential_rewards_accuracy"] = r_accuracy.mean() * self.cfg.w_accuracy
        self.extras["logs/potential_rewards_apex_height"] = r_apex_height.mean() * self.cfg.w_apex_height
        self.extras["logs/potential_rewards_delta_height"] = r_delta_height.mean() * self.cfg.w_delta_throw
        self.extras["logs/true_rewards_accuracy"] = r_accuracy.mean() * self.cfg.w_accuracy * self.throw_events.float().mean()
        self.extras["logs/true_rewards_apex_height"] = r_apex_height.mean() * self.cfg.w_apex_height * self.throw_events.float().mean()
        self.extras["logs/true_rewards_delta_height"] = r_delta_height.mean() * self.cfg.w_delta_throw * self.throw_events.float().mean()

        # Log event counts
        #self.extras["logs/event_catch_count"] = self.catch_events.float().mean()
        #self.extras["logs/event_drop_count"] = self.drop_events.float().mean()

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
        start_hand = self.ball_throw_hand[env_ids]
        pos_left_batch = self.left_start_pos.repeat(len(env_ids), 1)
        pos_right_batch = self.right_start_pos.repeat(len(env_ids), 1)
        chosen_start_pos = torch.where(start_hand.unsqueeze(-1) == 0, pos_left_batch, pos_right_batch)
        
        spawn_noise = (torch.rand_like(chosen_start_pos) - 0.5) * 2.0 * self.cfg.spawn_randomized_offset_range
        spawn_noise[:, 2] = 0.0  # No vertical noise
        chosen_start_pos += spawn_noise

        ball = self.ball1
        default_state = ball.data.default_root_state[env_ids].clone()
        default_state[:, :3] = self.scene.env_origins[env_ids] 
        default_state[:, :3] += chosen_start_pos
        default_state[:, 7:] = 0.0 
        ball.write_root_state_to_sim(default_state, env_ids=env_ids)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        random_hand_start = torch.randint(0, 2, (len(env_ids),), device=self.device)
        self.ball_throw_hand[env_ids] = random_hand_start
        self._reset_ball_pos(env_ids)
            
        self._apply_init_joint_pose(self.left_hand, self.init_left_joint_pos, env_ids, self.left_hand_idx)
        self._apply_init_joint_pose(self.right_hand, self.init_right_joint_pos, env_ids, self.right_hand_idx)
        
        for _ in range(1):
            self.sim.step()

        self.scene.update(dt=self.cfg.sim.dt)

        l_wrist = self.left_hand.data.body_pos_w[env_ids, self.left_wrist_idx]
        r_wrist = self.right_hand.data.body_pos_w[env_ids, self.right_wrist_idx]
        l_knuckle = self.left_hand.data.body_pos_w[env_ids, self.left_knuckle_idx]
        r_knuckle = self.right_hand.data.body_pos_w[env_ids, self.right_knuckle_idx]
        
        # Calculate Palm Center
        left_palm_pos = (l_wrist * (1 - self.cfg.center_of_hand_bias)) + (l_knuckle * self.cfg.center_of_hand_bias)
        right_palm_pos = (r_wrist * (1 - self.cfg.center_of_hand_bias)) + (r_knuckle * self.cfg.center_of_hand_bias)
        
        # Save to our fixed buffer
        self.initial_hand_target_position[env_ids, 0] = left_palm_pos
        self.initial_hand_target_position[env_ids, 1] = right_palm_pos

        self.detect_events()

        # active_anchors = self.cfg.ball_anchor[0]
        # anchors = torch.tensor(active_anchors, device=self.device, dtype=torch.long)
        
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
