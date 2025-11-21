# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .juggling_agent_env_cfg import JugglingAgentEnvCfg


class JugglingAgentEnv(DirectRLEnv):
    cfg: JugglingAgentEnvCfg

    def __init__(self, cfg: JugglingAgentEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # self._cart_dof_idx, _ = self.robot.find_joints(self.cfg.cart_dof_name)
        # self._pole_dof_idx, _ = self.robot.find_joints(self.cfg.pole_dof_name)
        self.placeholder_idx1 = [0]
        self.placeholder_idx2 = [1]

        # self.joint_pos = torch.zeros([2048, 26], device="cuda:0")
        # self.joint_vel = torch.zeros([2048, 26], device="cuda:0")
        
        # I think setup scene is already called in the super init so this should be fine
        device = self.device

        self.joint_pos = torch.zeros([self.num_envs, 26], device=device)
        self.joint_vel = torch.zeros([self.num_envs, 26], device=device)
        
        # import pdb; pdb.set_trace()

    def _setup_scene(self):
        device = self.device
        self.left_hand = Articulation(self.cfg.left_hand_cfg)
        self.right_hand = Articulation(self.cfg.right_hand_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["left_hand"] = self.left_hand
        self.scene.articulations["right_hand"] = self.right_hand
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.action_dim = self.left_hand.action_dim + self.right_hand.action_dim
        self.actions = torch.zeros((self.num_envs, self.action_dim), device=device)
        self.prev_actions = torch.zeros((self.num_envs, self.action_dim), device=device)

        # TODO add 3 balls

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # self.left_hand.set_joint_effort_target(self.actions * 100, joint_ids=self.placeholder_idx1)
        # self.right_hand.set_joint_effort_target(self.actions * 100, joint_ids=self.placeholder_idx1)
        []
        # TODO
        # apply the actions to both hands
        # self.left_hand.apply_action(...)
        # self.right_hand.apply_action(...)

    # do we need a post physics step to update ball and hand positions? TODO?

    
    def _allocate_tensors(self):

        num_envs = self.num_envs
        device = self.device #I think this is defined automatically in the parent class

        # Ball state tensors
        self.ball_pos = torch.zeros((num_envs, self.cfg.num_balls, 3), device=device) # 3 is for x,y,z
        self.ball_vel = torch.zeros((num_envs, self.cfg.num_balls, 3), device=device)
        
        # Hand state tensors
        self.hand_pos = torch.zeros((num_envs, self.cfg.num_hands, 3), device=device)

        # Event tracking tensors
        self.catch_events = torch.full((num_envs,), -1, device=device, dtype=torch.int32)  # -1 means no catch
        self.drop_events = torch.full((num_envs,), -1, device=device, dtype=torch.int32)   # -1 means no drop
        self.throw_events = torch.full((num_envs,), -1, device=device, dtype=torch.int32)  # -1 means no throw

        # Height tracking tensors
        self.ball_peak_height = torch.zeros((num_envs, self.cfg.num_balls), device=device)

        # Drop position tracking
        self.ball_drop_pos = torch.zeros((num_envs, self.cfg.num_balls, 3), device=device)

        # Throw/Catch hand tracking
        self.ball_throw_hand = torch.zeros((num_envs, self.cfg.num_balls), device=device, dtype=torch.int32)  # 0 or 1 for left/right hand
        self.ball_catch_hand = torch.zeros((num_envs, self.cfg.num_balls), device=device, dtype=torch.int32)  # 0 or 1 for left/right hand
        self.prev_in_hand = torch.zeros((num_envs, self.cfg.num_balls), device=device, dtype=torch.bool)

        # target hand position for ball, used in drop reward calculation
        self.ball_target_hand_pos = torch.zeros((num_envs, self.cfg.num_hands, 3), device=device)

        # throw rythem tracking, both hands share the same rythem timer. May want to change to be per-hand and add a offset for one hand
        self.throw_last_time = torch.zeros(num_envs, device=device)
        self.throw_intervals = torch.zeros(num_envs, device=device)

        # TODO initial ball positions

    def compute_target_hand_position(self, mask, ball_id):
        # Simple rule: ball should go to opposite hand from throw
        throw_hand = self.ball_throw_hand[mask, ball_id]
        opposite = 1 - throw_hand # 0 is left hand, 1 is right hand, so 1-throw_hand gives opposite hand index
        return self.hand_pos[mask, opposite]

    def _get_observations(self) -> dict:
        # obs = torch.cat(
        #     (
        #         self.joint_pos[:, self.placeholder_idx2[0]].unsqueeze(dim=1),
        #         self.joint_vel[:, self.placeholder_idx2[0]].unsqueeze(dim=1),
        #         self.joint_pos[:, self.placeholder_idx1[0]].unsqueeze(dim=1),
        #         self.joint_vel[:, self.placeholder_idx1[0]].unsqueeze(dim=1),
        #     ),
        #     dim=-1,
        # )
        obs = torch.cat(
            (
                self.hand_pos.reshape(self.num_envs, -1),
                self.ball_pos.reshape(self.num_envs, -1),
                self.ball_vel.reshape(self.num_envs, -1),
            ),
            dim=1,
        )
        observations = {"policy": obs}
        return observations

    def detect_events(self):
        self.catch_events[:] = -1
        self.drop_events[:] = -1
        self.throw_events[:] = -1

        ###############
        # Detect drop #
        ###############

        ball_height_pos = self.ball_pos[:, :, 2]  
        dropped = ball_height_pos < self.cfg.ground_height + 0.01  # small buffer to avoid numerical issues, may need tuning
        
        for ball in range(self.cfg.num_balls):
            dropped_mask = dropped[:, ball]
            if dropped_mask.any():
                self.drop_events[dropped_mask] = ball
                self.ball_drop_pos[dropped_mask, ball] = self.ball_pos[dropped_mask, ball]
                self.ball_target_hand_pos[dropped_mask] = self.compute_target_hand_position(dropped_mask, ball)


        ###############
        # Detect catch #
        ###############
        for ball in range(self.cfg.num_balls):
            dist_to_hands = torch.norm(
                self.ball_pos[:, ball].unsqueeze(1) - self.hand_pos, dim=-1
            )  # single ball pos has dim (num_env, 3). unsqueeze ball position to (num_envs, 1, 3) for broadcasting

            caught_L = (dist_to_hands[:, 0] < self.cfg.catch_radius)
            caught_R = (dist_to_hands[:, 1] < self.cfg.catch_radius)

            catch_mask = caught_L | caught_R
            if catch_mask.any():
                self.catch_events[catch_mask] = ball
                self.ball_catch_hand[catch_mask, ball] = torch.where(
                    caught_L[catch_mask], # 0 for left hand, 1 for right hand
                    torch.tensor(0, device=self.device),
                    torch.tensor(1, device=self.device)
                )

        ###############
        # Detect throw #
        ###############
        for ball in range(self.cfg.num_balls):
            was_in_hand = self.prev_in_hand[:, ball]
            is_in_hand_now = (
                (torch.norm(self.ball_pos[:, ball] - self.hand_pos[:, 0], dim=-1) < self.cfg.catch_radius) | # in left hand
                (torch.norm(self.ball_pos[:, ball] - self.hand_pos[:, 1], dim=-1) < self.cfg.catch_radius)   # in right hand
            )
            throw_mask = was_in_hand & (~is_in_hand_now) # ~ is logical NOT for torch tensors
            if throw_mask.any():
                self.throw_events[throw_mask] = ball
                hand_L_position = (torch.norm( # check if the ball was in left hand, otherwise it was in right hand but we don't need to check that again because we already know it was in a hand
                    self.ball_pos[:, ball] - self.hand_pos[:, 0], dim=-1
                ) < self.cfg.catch_radius)

                self.ball_throw_hand[throw_mask, ball] = torch.where( # 0 for left hand, 1 for right hand
                    hand_L_position[throw_mask],
                    torch.zeros_like(hand_L_position[throw_mask], device=self.device, dtype=torch.long),
                    torch.ones_like(hand_L_position[throw_mask], device=self.device, dtype=torch.long),
                )
        
        # update prev_in_hand, we need to know if the ball was in hand in the previous step to detect throw events
        self.prev_in_hand = torch.zeros((self.num_envs, self.cfg.num_balls), device=self.device, dtype=torch.bool)
        for ball in range(self.cfg.num_balls):
            self.prev_in_hand[:, ball] = (
                (torch.norm(self.ball_pos[:, ball] - self.hand_pos[:, 0], dim=-1) < self.cfg.catch_radius) | 
                (torch.norm(self.ball_pos[:, ball] - self.hand_pos[:, 1], dim=-1) < self.cfg.catch_radius)
            )

        #######################
        # track peak heights #
        #######################
        for ball in range(self.cfg.num_balls):
            self.ball_peak_height[:, ball] = torch.max(
                self.ball_peak_height[:, ball],
                self.ball_pos[:, ball, 2]
            )
        

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

        num_envs = self.num_envs
        device = self.device
        ball_pos = self.ball_pos
        ball_vel = self.ball_vel
        hand_pos = self.hand_pos
        actions = self.actions
        prev_actions = self.prev_actions

        reward = torch.zeros(num_envs, device=self.device)

        ###################
        # hoarding penalty#
        ###################

        # calculate if the ball is in the hand, use the catch_radius variable
        # should only be calculated after 1 second has passed to allow for initial positioning TODO
        
        # in_hand = torch.zeros((num_envs, self.cfg.num_balls, self.cfg.num_hands), device=device, dtype=torch.bool)
        # for ball in range(self.cfg.num_balls):
        #     for hand in range(self.cfg.num_hands):
        #         distance_to_hand = torch.norm(
        #             ball_pos[:, ball] - hand_pos[:, hand],
        #             dim=-1
        #         )
        #         in_hand[:, ball, hand] = distance_to_hand < self.catch_radius
        
        # I belive this works the same as the above nested loop, but takes advantage of tensor broadcasting to make it much faster
        
        #if self.episode_length_buf > ?  TODO, only start applying hoarding penalty after 1 second, need to figure out hz first 
        in_hand = (
            torch.norm( # broadcasts the num balls and num hands dimensions, then subtracts to get the distance between each ball and each hand
                ball_pos.unsqueeze(2) - hand_pos.unsqueeze(1), # unsqueeze for broadcasting, unsqueeze ball_pos from (num_envs, num_balls, 3) to (num_envs, num_balls, 1, 3), unsqueeze hand_pos from (num_envs, num_hands, 3) to (num_envs, 1, num_hands, 3)
                dim=-1
            ) < self.cfg.catch_radius
        ) 

        balls_in_L = in_hand[:, :, 0].sum(dim=1)
        balls_in_R = in_hand[:, :, 1].sum(dim=1)

        hoarding = (balls_in_L > 1) | (balls_in_R > 1)
        reward += -self.cfg.w_hoarding * hoarding.float()

        ###################
        # jitter penalty  #
        ###################

        delta_a = actions - prev_actions
        jitter = torch.sum(delta_a**2, dim=-1)
        reward += -self.cfg.w_jitter * jitter


        #######################
        # highest ball reward #
        #######################
        # always applied
        # reward the height of only the higest ball up to a specified apex defined in the cfg file
        height_cords = ball_pos[:, :, 2]  
        vertical_velocities = ball_vel[:, :, 2]

        up_mask = vertical_velocities > self.cfg.min_vertical_velocity
        one_going_up = (up_mask.sum(dim=1) == 1)  # only one ball is going up
        index_going_up = torch.argmax(up_mask.float(), dim=1)

        # check to see if the ball going up is in the air (not in hand)
        batch_ids = torch.arange(num_envs, device=device)

        height_cords_up = height_cords[batch_ids, index_going_up]
        above_min = height_cords_up > self.cfg.min_throw_height

        # Use clip for height reward, should be nicer for early lerning but maybe switch to Gaussian if not working well?
        height_r = (height_cords_up - self.cfg.ground_height) / (self.cfg.target_height - self.cfg.ground_height)
        height_r_norm = torch.clamp(height_r, 0.0, 1.0)

        reward += self.cfg.w_highest * height_r_norm * one_going_up.float() * above_min.float()

        ###################
        # catch reward    #
        ###################

        # only added on catch event
        # two parts here, get the max height of the caught ball
        #                 check if the ball is caught by the opposite hand

        catch_mask = self.catch_events >= 0
        if catch_mask.any():

            batch = catch_mask.nonzero(as_tuple=True)[0]
            ball_id = self.catch_events[batch] 
            peak = self.ball_peak_height[batch, ball_id]

            # height Gaussian
            Gh = torch.exp(- (peak - self.cfg.target_height)**2 / (2 * self.cfg.sigma_apex_height**2))

            # hand indices
            throw_hand = self.ball_throw_hand[batch, ball_id]
            catch_hand = self.ball_catch_hand[batch, ball_id]
                
            cross = torch.where(
                throw_hand != catch_hand,
                torch.tensor(1.0, device=self.device),
                torch.tensor(-0.25, device=self.device) # adjust penalty for same hand catch, may inhibit learning
            )

            reward[catch_mask] += self.cfg.w_catch * Gh * cross

        ###################
        # drop reward  #
        ###################

        # only added on drop event
        drop_mask = self.drop_events >= 0
        if drop_mask.any():
            batch = drop_mask.nonzero(as_tuple=True)[0]
            ball_id = self.drop_events[batch] 

            peak = self.ball_peak_height[batch, ball_id]
            drop_pos = self.ball_drop_pos[batch, ball_id]

            # height gaussian
            Gh = torch.exp(- (peak - self.cfg.target_height)**2 / (2 * self.cfg.sigma_apex_height**2))

            # distance gaussian to target hand
            target_hand_pos = self.compute_target_hand_position(batch, ball_id)
            dist = torch.norm(drop_pos - target_hand_pos, dim=-1)
            Gd = torch.exp(- (dist)**2 / (2 * self.cfg.sigma_drop_distance**2))

            reward[drop_mask] += self.cfg.w_drop * Gh * Gd


        ###################
        # rythem reward   #
        ###################

        throw_max = self.throw_events >= 0
        if throw_max.any():
            # delta t = time since last throw for the same hand
            delta_t = self.throw_intervals[throw_max] # TODO, need to actually track this variable properly
            GT = torch.exp(- (delta_t - self.cfg.target_rythem)**2 / (2 * self.cfg.sigma_rythem**2))
            reward[throw_max] += self.cfg.w_rythem * GT


        self.prev_actions = actions.clone()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.left_hand.data.joint_pos
        self.joint_vel = self.left_hand.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self.placeholder_idx1]) >= 0, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self.placeholder_idx2]) > math.pi / 2, dim=1)
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # if env_ids is None:
        #     env_ids = self.left_hand._ALL_INDICES
        # super()._reset_idx(env_ids)

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
        
        env_ids = env_ids.to(self.device)

        # Reset positions of balls
        #self.ball_pos[env_ids] = TODO
        self.ball_vel[env_ids] = 0
        self.ball_peak_height[env_ids] = 0

        # Reset tracking variables
        self.catch_events[env_ids] = -1
        self.drop_events[env_ids] = -1
        self.throw_events[env_ids] = -1
        self.prev_in_hand[env_ids] = False

        self.prev_actions[env_ids] = 0
        self.actions[env_ids] = 0

        # Reset timers
        self.throw_last_time[env_ids] = 0
        self.throw_intervals[env_ids] = 0

        return super().reset_idx(env_ids)

# I don't think we need this function anymore since the reward is computed inline
# @torch.jit.script
# def compute_rewards(
#     rew_scale_alive: float,
#     rew_scale_terminated: float,
#     rew_scale_pole_pos: float,
#     rew_scale_cart_vel: float,
#     rew_scale_pole_vel: float,
#     pole_pos: torch.Tensor,
#     pole_vel: torch.Tensor,
#     cart_pos: torch.Tensor,
#     cart_vel: torch.Tensor,
#     reset_terminated: torch.Tensor,
# ):