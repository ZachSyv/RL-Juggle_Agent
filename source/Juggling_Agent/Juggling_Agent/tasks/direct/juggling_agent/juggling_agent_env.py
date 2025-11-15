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

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos[:, self.placeholder_idx2[0]].unsqueeze(dim=1),
                self.joint_vel[:, self.placeholder_idx2[0]].unsqueeze(dim=1),
                self.joint_pos[:, self.placeholder_idx1[0]].unsqueeze(dim=1),
                self.joint_vel[:, self.placeholder_idx1[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

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
        if env_ids is None:
            env_ids = self.left_hand._ALL_INDICES
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
    # rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    # rew_termination = rew_scale_terminated * reset_terminated.float()
    # rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    # rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    # rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    # total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel

    return torch.zeros(pole_pos.shape)