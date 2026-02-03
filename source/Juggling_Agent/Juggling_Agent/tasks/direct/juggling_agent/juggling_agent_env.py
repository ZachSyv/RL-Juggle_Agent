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

        flex_regex = "(FF|MF|RF|LF)J(3|2|1)|THJ.*|LFJ5"
        abduction_regex = "(FF|MF|RF|LF)J4"

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

        self.closed_left_joint_pos = self._build_init_joint_pose(self.left_hand, self.cfg.closed_joint_pos)
        self.closed_right_joint_pos = self._build_init_joint_pose(self.right_hand, self.cfg.closed_joint_pos)

        self.left_lower_limits = self.left_hand.data.soft_joint_pos_limits[..., 0].clone()
        self.left_upper_limits = self.left_hand.data.soft_joint_pos_limits[..., 1].clone()
        self.right_lower_limits = self.right_hand.data.soft_joint_pos_limits[..., 0].clone()
        self.right_upper_limits = self.right_hand.data.soft_joint_pos_limits[..., 1].clone()

        self.left_finger_closed = self.closed_left_joint_pos[:, self.left_finger_flex_joints]
        self.left_finger_open = self.init_left_joint_pos[:, self.left_finger_flex_joints]

        self.right_finger_closed = self.closed_right_joint_pos[:, self.right_finger_flex_joints]
        self.right_finger_open = self.init_right_joint_pos[:, self.right_finger_flex_joints]
        
        self.ball_spawn_offsets = torch.tensor(self.cfg.ball_offset, device=self.device, dtype=torch.float32)
        self.ball_anchors = torch.tensor(self.cfg.ball_anchor, device=self.device, dtype=torch.float32)

        left_hand_start_pos = self.cfg.hand_pos[0]
        left_ball_start_pos = self.cfg.ball_offset[0]
        self.left_start_pos = torch.tensor(
            [left_hand_start_pos[0] + left_ball_start_pos[0], left_hand_start_pos[1] + left_ball_start_pos[1], left_hand_start_pos[2] + left_ball_start_pos[2]],
            device=device, 
            dtype=torch.float32)

        right_hand_start_pos = self.cfg.hand_pos[1]
        right_ball_start_pos = self.cfg.ball_offset[0] #[2]
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
        #self.sigma_drop_distance_coeff = torch.tensor(-1.0 / (2 * (self.cfg.sigma_drop_distance ** 2)), device=device)
        #self.sigma_rythem_coeff = torch.tensor(-1.0 / (2 * (self.cfg.sigma_rythem ** 2)), device=device)

        self._allocate_tensors()
        # self.total_env_steps = 0

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
        self.contact_sensor_left_wrist = ContactSensor(self.cfg.contact_sensor_left_wrist)
        self.scene.sensors["contact_sensor_left_wrist"] = self.contact_sensor_left_wrist
        self.contact_sensor_left_forearm = ContactSensor(self.cfg.contact_sensor_left_forearm)
        self.scene.sensors["contact_sensor_left_forearm"] = self.contact_sensor_left_forearm
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
        self.contact_sensor_right_wrist = ContactSensor(self.cfg.contact_sensor_right_wrist)
        self.scene.sensors["contact_sensor_right_wrist"] = self.contact_sensor_right_wrist
        self.contact_sensor_right_forearm = ContactSensor(self.cfg.contact_sensor_right_forearm)
        self.scene.sensors["contact_sensor_right_forearm"] = self.contact_sensor_right_forearm
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
        alpha = self.cfg.action_smoothing
        self.actions_smooth = alpha * current_action + (1.0 - alpha) * self.actions_smooth
        
        action_left = self.actions_smooth[:, :5]
        target_left = self.target_L
        target_left.copy_(self.init_left_joint_pos)


        elbow_ids = self.left_elbow_joint
        target_left[:, elbow_ids] += action_left[:, 0:2] * self.cfg.action_scale
        
        # Clamp to physical limits so we don't break the robot
        target_left[:, elbow_ids] = torch.max(
            torch.min(target_left[:, elbow_ids], self.left_upper_limits[:, elbow_ids]), 
            self.left_lower_limits[:, elbow_ids]
        )

        # 2. WRIST: Apply Action as Offset
        wrist_ids = self.left_wrist_joints
        target_left[:, wrist_ids] += action_left[:, 2:4] * self.cfg.action_scale
        
        target_left[:, wrist_ids] = torch.max(
            torch.min(target_left[:, wrist_ids], self.left_upper_limits[:, wrist_ids]), 
            self.left_lower_limits[:, wrist_ids]
        )

        # 3. FINGERS
        finger_action_L = action_left[:, 4].unsqueeze(-1)
        percent_closed_left = (torch.clamp((finger_action_L + 1.0)/ 2.0, min=0.0, max=1.0))
        diff_left = self.left_finger_closed - self.left_finger_open
        target_left[:, self.left_finger_flex_joints] = self.left_finger_open + (diff_left * percent_closed_left)

        # --- RIGHT HAND ---
        action_right = self.actions_smooth[:, 5:]
        target_right = self.target_R
        # Start with DEFAULT
        target_right.copy_(self.init_right_joint_pos)
        
        # 1. ELBOW
        elbow_ids_r = self.right_elbow_joint
        target_right[:, elbow_ids_r] += action_right[:, 0:2] * self.cfg.action_scale
        
        target_right[:, elbow_ids_r] = torch.max(
            torch.min(target_right[:, elbow_ids_r], self.right_upper_limits[:, elbow_ids_r]), 
            self.right_lower_limits[:, elbow_ids_r]
        )

        # 2. WRIST
        wrist_ids_r = self.right_wrist_joints
        target_right[:, wrist_ids_r] += action_right[:, 2:4] * self.cfg.action_scale
        
        target_right[:, wrist_ids_r] = torch.max(
            torch.min(target_right[:, wrist_ids_r], self.right_upper_limits[:, wrist_ids_r]), 
            self.right_lower_limits[:, wrist_ids_r]
        )

        # 3. FINGERS
        finger_action_R = action_right[:, 4].unsqueeze(-1)
        percent_closed_right = (torch.clamp((finger_action_R + 1.0)/ 2.0, min=0.0, max=1.0))
        diff_right = self.right_finger_closed - self.right_finger_open
        target_right[:, self.right_finger_flex_joints] = self.right_finger_open + (diff_right * percent_closed_right)
        # Apply
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
        self.is_touching = torch.zeros((num_envs, self.cfg.num_hands), device=device, dtype=torch.bool)

        self.ball_pos_flat = torch.zeros((num_envs, 3), device=device)
        self.ball_vel_flat = torch.zeros((num_envs, 3), device=device)
        self.hand_pos_flat = torch.zeros((num_envs, 6), device=device)

        total_dof = 52
        self.joint_pos = torch.zeros((num_envs, total_dof), device=device)
        self.joint_vel = torch.zeros((num_envs, total_dof), device=device)

        #self.grip_state = torch.zeros((num_envs, self.cfg.num_hands), device=device, dtype=torch.bool)

        self.target_L = torch.zeros((num_envs, len(self.left_hand_idx)), device=self.device)
        self.target_R = torch.zeros((num_envs, len(self.right_hand_idx)), device=self.device)

        self.initial_hand_target_position = torch.zeros((num_envs, self.cfg.num_hands, 3), device=device, dtype=torch.float)

        self.num_ball_up_rewards_lookup = torch.tensor(self.cfg.num_ball_up_reward_scale, device=device)
        self.ball_throw_num_already_up_scale = torch.ones((num_envs, ), device=device, dtype=torch.float)

        self.ball_throw_time = torch.zeros(num_envs, device=device, dtype=torch.float)
        self.hand_throw_last_time = torch.full((num_envs, self.cfg.num_hands), -1.0, device=device, dtype=torch.float)
        self.hand_throw_intervals = torch.full((num_envs, self.cfg.num_hands), -1.0, device=device, dtype=torch.float)
        self.holding_duration = torch.zeros((num_envs, ), device=device, dtype=torch.float)
        self.is_caught = torch.zeros((num_envs, ), device=device, dtype=torch.bool)
        self.is_thrown = torch.zeros((num_envs, ), device=device, dtype=torch.bool)

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

    # def _get_curriculum_sigma(self, start, end):
    #     progress = self.total_env_steps / self.cfg.curriculum_learning_steps
    #     progress = min(max(progress, 0.0), 1.0)
    #     sigma = start + (end - start) * progress
    #     return sigma

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

        env_origins = self.scene.env_origins

        l_wrist_local = l_wrist - env_origins
        r_wrist_local = r_wrist - env_origins
        l_knuckle_local = l_knuckle - env_origins
        r_knuckle_local = r_knuckle - env_origins

        self.hand_pos[:, 0] = (l_wrist_local * (1 - self.cfg.center_of_hand_bias)) + (l_knuckle_local * self.cfg.center_of_hand_bias)
        self.hand_pos[:, 1] = (r_wrist_local * (1 - self.cfg.center_of_hand_bias)) + (r_knuckle_local * self.cfg.center_of_hand_bias)

        self.ball_pos = self.ball1.data.root_pos_w
        self.ball_pos = self.ball_pos - env_origins
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

        ball_z = self.ball_pos[:, 2].unsqueeze(1)
        hand_z = self.hand_pos[:, :, 2]
        is_above = ball_z > (hand_z - self.cfg.above_hand_threshold)

        # Sensor Logic
        is_touching_left = torch.zeros((self.num_envs, ), device=self.device, dtype=torch.bool)
        is_touching_right = torch.zeros((self.num_envs, ), device=self.device, dtype=torch.bool)

        left_forces_palm = torch.norm(self.contact_sensor_left_palm.data.force_matrix_w.sum(dim=1), dim=-1).flatten()
        is_touching_left |= left_forces_palm > self.cfg.contact_threshold

        left_forces_wrist = torch.norm(self.contact_sensor_left_wrist.data.force_matrix_w.sum(dim=1), dim=-1).flatten()
        is_touching_left |= left_forces_wrist > self.cfg.contact_threshold

        left_forces_metacarpal = torch.norm(self.contact_sensor_left_metacarpal.data.force_matrix_w.sum(dim=1), dim=-1).flatten()
        is_touching_left |= left_forces_metacarpal > self.cfg.contact_threshold

        left_finger_forces = [s.data.force_matrix_w for s in self.left_hand_sensors]
        left_all_finger_forces = torch.stack(left_finger_forces, dim=1)
        left_flat_forces = left_all_finger_forces.sum(dim=2)
        left_forces = torch.norm(left_flat_forces, dim=-1)
        max_left_forces = left_forces.max(dim=1).values.flatten()
        is_touching_left |= max_left_forces > self.cfg.contact_threshold

        self.in_hand[:, 0] = is_touching_left & is_above[:, 0]

        left_forces_forearm = torch.norm(self.contact_sensor_left_forearm.data.force_matrix_w.sum(dim=1), dim=-1).flatten()
        is_touching_left |= left_forces_forearm > self.cfg.contact_threshold

        self.is_touching[:, 0] = is_touching_left

        right_forces_palm = torch.norm(self.contact_sensor_right_palm.data.force_matrix_w.sum(dim=1), dim=-1).flatten()
        is_touching_right |= right_forces_palm > self.cfg.contact_threshold

        right_forces_wrist = torch.norm(self.contact_sensor_right_wrist.data.force_matrix_w.sum(dim=1), dim=-1).flatten()
        is_touching_right |= right_forces_wrist > self.cfg.contact_threshold

        right_forces_metacarpal = torch.norm(self.contact_sensor_right_metacarpal.data.force_matrix_w.sum(dim=1), dim=-1).flatten()
        is_touching_right |= right_forces_metacarpal > self.cfg.contact_threshold

        right_finger_forces = [s.data.force_matrix_w for s in self.right_hand_sensors]
        right_all_finger_forces = torch.stack(right_finger_forces, dim=1)
        right_flat_forces = right_all_finger_forces.sum(dim=2)
        right_forces = torch.norm(right_flat_forces, dim=-1)
        max_right_forces = right_forces.max(dim=1).values.flatten()
        is_touching_right |= max_right_forces > self.cfg.contact_threshold

        self.in_hand[:, 1] = is_touching_right & is_above[:, 1]

        right_forces_forearm = torch.norm(self.contact_sensor_right_forearm.data.force_matrix_w.sum(dim=1), dim=-1).flatten()
        is_touching_right |= right_forces_forearm > self.cfg.contact_threshold

        self.is_touching[:, 1] = is_touching_right
        
        # Hand states
        
        in_any_hand = self.in_hand.any(dim=1)
        is_touching_any = self.is_touching.any(dim=1)
        
        self.holding_duration = torch.where(is_touching_any, self.holding_duration + self.step_dt, 0.0)
        
        ################
        # Detect throw #
        ################
        was_any_in_hand = self.prev_in_hand.any(dim=1)
        any_in_hand_now = self.in_hand.any(dim=1)
        ball_speed = torch.norm(self.ball_vel, dim=-1)
        vel_ball_z = self.ball_vel[..., 2]

        # calculate delta peak height so we can filter drops out of throws
        g = 9.81
        delta_peak_height = (torch.sign(vel_ball_z) * (vel_ball_z ** 2)) / (2 * g)
        is_valid_throw_height = delta_peak_height >= self.cfg.min_delta_throw_height

        throw_mask = (was_any_in_hand & (~any_in_hand_now)) & (vel_ball_z > self.cfg.min_vertical_velocity) & is_valid_throw_height & (ball_speed > self.cfg.min_throw_velocity)

        self.throw_events.copy_(throw_mask)
        self.is_thrown = torch.where(self.throw_events, True, self.is_thrown)

        self.vertical_velocities = self.ball_vel[:, 2].unsqueeze(1)
        self.going_up = self.vertical_velocities > self.cfg.min_vertical_velocity
        self.is_held = self.in_hand.any(dim=1, keepdim=True)

        current_up_mask = self.going_up & (~self.is_held)
        num_going_up = current_up_mask.sum(dim=1)
        num_others_going_up = num_going_up - self.throw_events.long()

        safe_indices = torch.clamp(num_others_going_up, max=len(self.num_ball_up_rewards_lookup)-1)
        current_throw_num_ball_scale = self.num_ball_up_rewards_lookup[safe_indices]

        just_thrown = self.throw_events
        self.ball_throw_num_already_up_scale = torch.where(just_thrown, current_throw_num_ball_scale, self.ball_throw_num_already_up_scale)

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
        
        self.is_caught = torch.where(just_caught, self.is_thrown, self.is_caught)
        self.is_thrown = torch.where(just_caught, False, self.is_thrown)
        self.ball_throw_time = torch.where(just_caught, 0.0, self.ball_throw_time)
        
        current_holding_hand = self.in_hand.float().argmax(dim=1)
        self.ball_catch_hand = torch.where(in_any_hand, current_holding_hand, self.ball_catch_hand)

        ################
        # Detect Drops
        ################
        ball_height_pos = self.ball_pos[:, 2]
        dropped = (~self.is_touching.any(dim=1)) & (ball_height_pos < self.cfg.ground_height + 0.1)
        
        self.drop_events.copy_(dropped)
        
        self.ball_drop_pos = torch.where(dropped.unsqueeze(1), self.ball_pos, self.ball_drop_pos)

        self.prev_in_hand.copy_(self.in_hand)

        self.is_caught = torch.where(dropped, False, self.is_caught)

    def _get_rewards(self) -> torch.Tensor:
        self.reward_buffer.fill_(0.0)

        # initialize some commonly used variables
        g = 9.81
        throw_hand_idx = self.ball_throw_hand.clamp(min=0, max=1)
        target_hand_idx = 1 - throw_hand_idx
        target_pos_fixed = self._get_target_hand_pos()
        target_hand_pos = target_hand_idx.view(-1, 1, 1).expand(-1, 1, 3)
        target_hand_pos = torch.gather(self.hand_pos, 1, target_hand_pos).squeeze(1)
        dist_to_target = torch.norm(self.ball_pos - target_hand_pos, dim=-1)
        #is_approaching = (self.ball_vel[:, 2] < -0.1) & (~self.in_hand.any(dim=1))
        #s_above_hand = (self.ball_pos[:, 2] > target_hand_pos[:, 2])
        was_thrown = (self.ball_throw_time > 0.0)
        num_going_up_penalty = self.ball_throw_num_already_up_scale

        # --- Continuous Penalties ---
        
        ################
        #   Hoarding   #
        ################
        ''' Penalty for holding onto the ball that needs to be thrown for too long '''

        hoarding_time = torch.clamp(self.holding_duration - self.cfg.hoarding_time_threshold, min=0.0)
        r_hoard = -self.cfg.w_hoarding * (hoarding_time.square())

        current_episode_time = self.episode_length_buf * self.step_dt
        in_startup_window = current_episode_time < self.cfg.start_hold_threshold
        r_hoard = torch.where(in_startup_window, torch.zeros_like(r_hoard), r_hoard)

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
        ammount_of_violatoin = torch.clamp(self.cfg.min_hand_dist - hand_separation, min=0.0)
        r_hand_touching = -self.cfg.w_hands_touching * (ammount_of_violatoin ** 2)

        self.reward_buffer += r_hand_touching

        # --- Continuous Rewards ---

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
        valid_catch_mask = is_holding & in_payout_window & self.is_caught

        # calculate Gh
        peak = self.catch_peak_height
        start_height = self.ball_initial_height
        delta_height = torch.clamp(peak - start_height, min=0.0)

        height_ratio = torch.clamp(delta_height / self.cfg.target_delta_height, min=0.0, max=1.0)
        height_gate = height_ratio.square()

        catch_throw_height_error = (delta_height - self.cfg.target_delta_height).square()
        Gh_percise = torch.exp((-catch_throw_height_error) / (2 * (self.cfg.sigma_delta_throw_catch ** 2)))
        #Gh_wide = torch.exp((-catch_throw_height_error) / (2 * (self.cfg.sigma_delta_throw_catch_wide ** 2)))
        #Gh = (Gh_percise * self.cfg.catch_wide_precise_ratio + Gh_wide * (1 - self.cfg.catch_wide_precise_ratio))
        throw_quality = Gh_percise * self.cfg.catch_precise_height_ratio + height_gate * (1 - self.cfg.catch_precise_height_ratio)

        catch_height = self.ball_pos[:, 2]
        target_catch_height = self.ball_initial_height
        catch_height_error = catch_height - target_catch_height
        catch_height_penalty = torch.exp(-catch_height_error.square() / (2 * (self.cfg.sigma_catch_height ** 2)))

        throw_hand = self.ball_throw_hand
        catch_hand = self.ball_catch_hand

        is_cross_catch = throw_hand != catch_hand
        is_correct_hand = is_cross_catch == self.cfg.should_cross

        r_wrong_catch = -throw_quality * (~is_correct_hand).float()
        r_good_catch = (throw_quality * catch_height_penalty * height_gate) * is_correct_hand.float() * num_going_up_penalty
        r_catch = self.cfg.w_catch * (r_good_catch + r_wrong_catch) * valid_catch_mask.float()
        self.reward_buffer += r_catch

        #########
        # Throw #
        #########
        up_mask = self.going_up & (~self.is_held)
        target_hand_z = target_pos_fixed[:, 2]

        # delta height score based on vertical velocity, we don't use the predicted height because the squaring makes it too punishing early on
        target_velocity = math.sqrt(2 * g * self.cfg.target_delta_height)
        vertical_velocity_error = (self.vertical_velocities - target_velocity)
        r_wide_delta_height = torch.exp(-vertical_velocity_error.square() / (2 * (self.cfg.sigma_wide_delta_throw ** 2))).squeeze(1)
        r_precise_delta_height = torch.exp(-vertical_velocity_error.square() / (2 * (self.cfg.sigma_delta_throw ** 2))).squeeze(1)
        r_delta_height = (r_wide_delta_height * self.cfg.w_wide_delta_throw + r_precise_delta_height * self.cfg.w_precise_delta_throw) * up_mask.squeeze(1).float()

        # absolute height score
        current_ball_height = self.ball_pos[:, 2].unsqueeze(1)
        predicted_peak_height = current_ball_height + (torch.sign(self.vertical_velocities) * self.vertical_velocities.square()) / (2 * g)
        apex_height_error = predicted_peak_height - self.cfg.target_height
        r_apex_height = self.cfg.w_apex_height * torch.exp(-apex_height_error.square() / (2 * (self.cfg.sigma_apex_height ** 2))).squeeze(1) * up_mask.squeeze(1).float()

        # ballistic prediction
        landing_pos, valid_mask = self.predict_landing_position(
            self.ball_pos, 
            self.ball_vel, 
            target_z=target_hand_z,
        )

        height_clamp = torch.clamp(self.vertical_velocities/target_velocity, min=0.1, max=1.0).squeeze(1)
        error_xy = landing_pos[:, :2] - target_pos_fixed[:, :2]
        dist_xy = torch.norm(error_xy, dim=-1)
        wide_GT = torch.exp(-dist_xy.square() / (2 * (self.cfg.sigma_wide_throw_accuracy ** 2)))
        precise_GT = torch.exp(-dist_xy.square() / (2 * (self.cfg.sigma_throw_accuracy ** 2)))
        r_accuracy = (wide_GT * self.cfg.w_wide_accuracy + precise_GT * self.cfg.w_precise_accuracy) * valid_mask.float() * height_clamp.square()

        # allignment_scaler = wide_GT
        # r_apex_height_scaled = r_apex_height * allignment_scaler.square()
        # r_delta_height_wide_scaled = r_wide_delta_height * allignment_scaler
        # r_delta_height_precise_scaled = r_precise_delta_height * allignment_scaler.square()
        # r_delta_height_scaled = (r_delta_height_wide_scaled * self.cfg.w_wide_delta_throw + r_delta_height_precise_scaled * self.cfg.w_precise_delta_throw) * up_mask.squeeze(1).float()

        r_throw = (r_accuracy + r_apex_height + r_delta_height) * num_going_up_penalty * self.throw_events.float()
        self.reward_buffer += r_throw

        ############
        #   Drop   #
        ############
        ''' Penalty for dropping the ball, smaller the closer to the target hand, encouraging accurate throws. '''

        is_missed_catch = self.drop_events & self.is_thrown
        is_lazy_drop = self.drop_events & (~self.is_thrown)

        # flat drop penalty
        r_drop = -((self.cfg.w_drop * is_missed_catch.float()) + ( self.cfg.w_lazy_drop * is_lazy_drop.float()))
        self.reward_buffer += r_drop
        self.is_thrown = torch.where(self.drop_events, False, self.is_thrown)
        self.ball_peak_height = torch.where(self.drop_events, 0.0, self.ball_peak_height)

        # -------------------------------------------------------
        # LOGGING (This puts data into TensorBoard/Console)
        # -------------------------------------------------------
        # We use 'episode' prefix so RL-Games aggregates it properly
        
        # Log the MEAN reward per step across all environments
        self.extras["logs/rewards_hoard"] = r_hoard.mean()
        self.extras["logs/rewards_jitter"] = r_jitter.mean()
        #self.extras["logs/rewards_catch_position"] = r_catch_position.mean()
        self.extras["logs/rewards_hands_touching"] = r_hand_touching.mean()
        
        # For sparse events (Catch/Drop), we might want the SUM or MEAN
        # Mean tells us "Average reward per step per agent"
        self.extras["logs/rewards_catch"] = r_catch.mean()
        self.extras["logs/rewards_wrong_catch"] = r_wrong_catch.mean()
        self.extras["logs/rewards_good_catch"] = r_good_catch.mean()
        self.extras["logs/rewards_drop"] = r_drop.mean()
        self.extras["logs/rewards_throw"] = r_throw.mean()
        # self.extras["logs/rewards_target_direction"] = r_target_direction.mean() * self.cfg.w_throw_direction
        # self.extras["logs/rewards_target_lift"] = r_target_lift.mean() * self.cfg.w_throw_lift
        #self.extras["logs/rewards_power"] = r_power.mean() * self.cfg.w_throw_power
        #self.extras["logs/rewards_drift"] = -r_drift.mean() * self.cfg.w_drift
        self.extras["logs/potential_rewards_accuracy"] = r_accuracy.mean()
        self.extras["logs/potential_rewards_wide_accuracy"] = (r_accuracy * (self.cfg.w_wide_accuracy / (self.cfg.w_wide_accuracy + self.cfg.w_precise_accuracy))).mean()
        self.extras["logs/potential_rewards_precise_accuracy"] = (r_accuracy * (self.cfg.w_precise_accuracy / (self.cfg.w_wide_accuracy + self.cfg.w_precise_accuracy))).mean()
        self.extras["logs/potential_rewards_apex_height"] = r_apex_height.mean()
        #self.extras["logs/potential_rewards_apex_height_scaled"] = r_apex_height_scaled.mean()
        self.extras["logs/potential_rewards_delta_height"] = r_delta_height.mean()
        self.extras["logs/potential_rewards_wide_delta_height"] = (r_wide_delta_height * self.cfg.w_wide_delta_throw).mean()
        self.extras["logs/potential_rewards_precise_delta_height"] = (r_precise_delta_height * self.cfg.w_precise_delta_throw).mean()
        self.extras["logs/true_rewards_accuracy"] = r_accuracy.mean() * self.throw_events.float().mean()
        self.extras["logs/true_rewards_apex_height"] = r_apex_height.mean() * self.throw_events.float().mean()
        self.extras["logs/true_rewards_delta_height"] = r_delta_height.mean() * self.throw_events.float().mean()
        self.extras["logs/Gh"] = height_gate.mean()
        self.extras["logs/throw_quality"] = throw_quality.mean()
        # self.extras["logs/current_sigma_delta_height"] = current_sigma_delta_height
        # self.extras["logs/current_sigma_apex_height"] = current_sigma_apex_height
        # self.extras["logs/current_sigma_throw_accuracy"] = current_sigma_throw_accuracy

        # Log event counts
        #self.extras["logs/event_catch_count"] = self.catch_events.float().mean()
        #self.extras["logs/event_drop_count"] = self.drop_events.float().mean()

        # # debug sensors
        # list_of_sensors_Debug = [
        #     ("LEFT_PALM", self.contact_sensor_left_palm),
        #     ("LEFT_WRIST", self.contact_sensor_left_wrist),
        #     ("LEFT_FOREARM", self.contact_sensor_left_forearm),
        #     ("LEFT_METACARPAL", self.contact_sensor_left_metacarpal),
        #     ("RIGHT_PALM", self.contact_sensor_right_palm),
        #     ("RIGHT_WRIST", self.contact_sensor_right_wrist),
        #     ("RIGHT_FOREARM", self.contact_sensor_right_forearm),
        #     ("RIGHT_METACARPAL", self.contact_sensor_right_metacarpal),
        # ]
        # print("---- Sensor Forces ----")

        # for name, sensor in list_of_sensors_Debug:
        #     force = torch.norm(sensor.data.force_matrix_w[0].sum(dim=0), dim=-1).flatten()
        #     print("Sensor {} Force: {:.4f}".format(name, force.item()))

        # print("-----------------------")
        # print("Hoarding Reward", r_hoard[0].item())
        # print("Catch Reward", r_catch[0].item())

        self.prev_actions_smooth = self.actions_smooth.clone()
        return self.reward_buffer

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        ball_height = self.ball_pos[:, 2]
        is_ball_dropped = (ball_height < (self.cfg.ground_height + 0.1)) & (~self.in_hand.any(dim=1))
        
        ball_dist_xy = torch.norm(self.ball_pos[:, :2], dim=-1)
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

        random_spawn_scales = torch.tensor([self.cfg.spawn_randomized_offset_range_x, self.cfg.spawn_randomized_offset_range_y, self.cfg.spawn_randomized_offset_range_z], device=self.device)
        spawn_noise = (torch.rand_like(chosen_start_pos) - 0.5) * 2.0
        spawn_noise *= random_spawn_scales
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
            
        self._apply_init_joint_pose(self.left_hand, self.init_left_joint_pos, env_ids, self.left_hand_idx)
        self._apply_init_joint_pose(self.right_hand, self.init_right_joint_pos, env_ids, self.right_hand_idx)
        
        random_hand_start = torch.randint(0, 2, (len(env_ids),), device=self.device)
        self.ball_throw_hand[env_ids] = random_hand_start
        self._reset_ball_pos(env_ids)
        
        for _ in range(1):
            self.sim.step()

        self.scene.update(dt=self.cfg.sim.dt)

        l_wrist = self.left_hand.data.body_pos_w[env_ids, self.left_wrist_idx]
        r_wrist = self.right_hand.data.body_pos_w[env_ids, self.right_wrist_idx]
        l_knuckle = self.left_hand.data.body_pos_w[env_ids, self.left_knuckle_idx]
        r_knuckle = self.right_hand.data.body_pos_w[env_ids, self.right_knuckle_idx]

        env_origins = self.scene.env_origins[env_ids]
        l_wrist_local = l_wrist - env_origins
        r_wrist_local = r_wrist - env_origins
        l_knuckle_local = l_knuckle - env_origins
        r_knuckle_local = r_knuckle - env_origins
        
        # Calculate Palm Center
        left_palm_pos = (l_wrist_local * (1 - self.cfg.center_of_hand_bias)) + (l_knuckle_local * self.cfg.center_of_hand_bias)
        right_palm_pos = (r_wrist_local * (1 - self.cfg.center_of_hand_bias)) + (r_knuckle_local * self.cfg.center_of_hand_bias)
        
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
