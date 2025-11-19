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

        self.joint_pos = torch.zeros([2048, 26], device="cuda:0")
        self.joint_vel = torch.zeros([2048, 26], device="cuda:0")

        # import pdb; pdb.set_trace()

    def _setup_scene(self):
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

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # self.left_hand.set_joint_effort_target(self.actions * 100, joint_ids=self.placeholder_idx1)
        # self.right_hand.set_joint_effort_target(self.actions * 100, joint_ids=self.placeholder_idx1)
        []

    
    def _allocate_tensors(self):

        num_envs = self.num_envs
        device = self.device

        # Ball state tensors
        self.ball_pos = torch.zeros((num_envs, self.cfg.num_balls, 3), device=device) # 3 is for x,y,z
        self.ball_vel = torch.zeros((num_envs, self.cfg.num_balls, 3), device=device)
        
        # Hand state tensors
        self.hand_pos = torch.zeros((num_envs, self.cfg.num_hands, 3), device=device)


        # Previous actions
        #self.action_dim = ? TODO, find out the action dimension
        self.actions = torch.zeros((num_envs, self.action_dim), device=device)
        self.prev_actions = torch.zeros((num_envs, self.action_dim), device=device)

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
        # Detect drop # TODO
        ###############
        
        #for ball in range(self.cfg.num_balls):
            # Check if ball has hit the ground
            # if so,
                # self.drop_events = ball id
                # self.ball_drop_pos = current ball position
                # self.ball_target_hand_pos = position of the intended catch hand

        ###############
        # Detect catch #
        ###############
        #for ball in range(self.cfg.num_balls):
            # Check if ball is caught TODO

            # caught_L = ... TODO
            # caught_R = ... TODO

            # catch_mask = caught_L | caught_R
            # if catch_mask.any():
                # self.catch_events[catch_mask] = ball
                # self.ball_catch_hand[catch_mask, ball] = torch.where(
                #     caught_L[catch_mask], # 0 for left hand, 1 for right hand
                #     torch.tensor(0, device=self.device),
                #     torch.tensor(1, device=self.device)
                # )

        ###############
        # Detect throw #
        ###############
        #for ball in range(self.cfg.num_balls): TODO


        # update prev_in_hand, we need to know if the ball was in hand in the previous step to detect throw events
        self.prev_in_hand = torch.zeros((self.num_envs, self.cfg.num_balls), device=self.device, dtype=torch.bool)
        for ball in range(self.cfg.num_balls):
            self.prev_in_hand[:, ball] = (
                (torch.norm(self.ball_pos[:, ball] - self.hand_pos[:, 0], dim=-1) < self.catch_radius) | 
                (torch.norm(self.ball_pos[:, ball] - self.hand_pos[:, 1], dim=-1) < self.catch_radius)
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
        total_reward = compute_rewards(
            1.0,
            -2.0, -1.0, -0.01, -0.005,
            self.joint_pos[:, self.placeholder_idx2[0]],
            self.joint_vel[:, self.placeholder_idx2[0]],
            self.joint_pos[:, self.placeholder_idx1[0]],
            self.joint_vel[:, self.placeholder_idx1[0]],
            self.reset_terminated,
        )
        return total_reward

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
        self.ball_pos[env_ids] = self.initial_ball_positions[env_ids]
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

@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    num_envs = self.num_envs
    ball_pos = self.ball_pos
    ball_vel = self.ball_vel
    hand_pos = self.hand_pos
    actions = self.actions
    prev_actions = self.prev_actions

    reward = torch.zeros(num_envs, device=self.device)

    ###################
    # hoarding penalty#
    ###################

    # calculate if the ball is in the hand, use the catch_radius variable TODO
    # should only be calculated after 1 second has passed to allow for initial positioning
    
    # might be this?
    #in_hand = (ball_pos - hand_pos) < self.catch_radius

    # balls_in_L =
    # balls_in_R =

    # this part is finished
    hoarding = (balls_in_L > 1) | (balls_in_R > 1)
    reward += -self.w_hoarding * hoarding.float()

    ###################
    # jitter penalty  #
    ###################

    delta_a = actions - prev_actions
    jitter = torch.sum(delta_a**2, dim=-1)
    reward += -self.w_jitter * jitter


    ###################
    # highest ball reward # TODO
    ###################


    ###################
    # catch reward    #
    ###################

    # only added on catch event
    # two parts here, get the max height of the caught ball
    #                 check if the ball is caught by the opposite hand

    # TODO, define catch events, it should be a tensor of shape (num_envs,) with -1 if no catch, otherwise the ball id
    catch_mask = self.catch_events >= 0
    if catch_mask.any():

        ball_id = self.catch_events[catch_mask] 
        peak = self.ball_peak_height[catch_mask, ball_id]

        # height Gaussian
        # need to define h_target TODO
        Gh = torch.exp(- (peak - self.h_target)**2 / (2 * self.sigma_h**2))

        # hand indices
        # need to define ball_throw_hand and ball_catch_hand TODO
        throw_hand = self.ball_throw_hand[catch_mask, ball_id]   # 0 or 1
        catch_hand = self.ball_catch_hand[catch_mask, ball_id]   # 0 or 1
            
        cross = torch.where(
        throw_hand != catch_hand,
        torch.tensor(1.0, device=self.device),
        torch.tensor(-0.5, device=self.device)
    )

    reward[catch_mask] += self.w_catch * Gh * cross

    ###################
    # drop reward  #
    ###################

    # only added on drop event
    # TODO
    drop_mask = self.drop_events >= 0 # TODO, define drop events, happens when the ball y coordinate hits the ground (the ground_height variable defined in cfg). 
    #Also should be a tensor of shape (num_envs,) with -1 if no drop, otherwise the ball id
    
    # if drop_mask.any():
    #     ball_id = self.drop_events[drop_mask] 

    #     peak = self.ball_peak_height[drop_mask, ball_id]
    #     drop_pos = self.ball_drop_pos[drop_mask, ball_id]

    #     # height Gaussian
    #     Gh = torch.exp(- (peak - target_hand)**2 / (2 * self.sigma_h**2))

    #     target_hand_pos TODO
    #     dist = torch.norm(drop_pos - target_hand_pos, dim=-1)
    #     Gd = torch.exp(- (dist)**2 / (2 * self.sigma_d**2))


    #     reward[drop_mask] += self.w_drop * Gh * Gd

    #     # distance from dropped ball to the intended catch hand
    #     target_hand_pos = 


    ###################
    # rythem reward   #
    ###################

    throw_max = self.throw_events >= 0 # TODO, define throw events, happens when the ball y coordinate exceeds the min_throw_height variable defined in cfg.
    if throw_max.any():
        # TODO delta t = time since last throw for the same hand
        # 

        # GT = torch.exp(- (delta_t - self.t_target)**2 / (2 * self.sigma_t**2))
        # reward[throw_max] += self.w_rythem * GT

    self.prev_actions = actions.clone()

    # rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    # rew_termination = rew_scale_terminated * reset_terminated.float()
    # rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    # rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    # rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    # total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return reward
    # return torch.zeros(pole_pos.shape)