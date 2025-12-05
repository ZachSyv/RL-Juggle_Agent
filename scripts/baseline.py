# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run baseline hand control with custom initial positions."""

"""Launch Isaac Sim Simulator first."""

import argparse
import math

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Baseline agent with custom hand control for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import Juggling_Agent.tasks  # noqa: F401


def generate_hand_actions(t, action_dim, control_mode="zero"):
    """Generate hand actions based on the specified control mode.

    Returns:
        Tensor of actions for all joints
    """

    return torch.zeros(action_dim)


def get_custom_joint_positions():
    joint_pos = {
        "elbow_bend": math.radians(0),
        "elbow_rotate": math.radians(0),
        "WRJ1": math.radians(0),
        "WRJ2": math.radians(0),
        "THJ1": math.radians(36.72),
        "THJ2": math.radians(31.76),
        "THJ3": math.radians(0),
        "THJ4": math.radians(25.82),
        "THJ5": math.radians(-6.91),
        "FFJ1": math.radians(17.72),
        "FFJ2": math.radians(15.06),
        "FFJ3": math.radians(73.45),
        "FFJ4": math.radians(-19.99),
        "MFJ1": math.radians(11.33),
        "MFJ2": math.radians(24.8),
        "MFJ3": math.radians(74.0),
        "MFJ4": math.radians(-8.9),
        "RFJ1": math.radians(16.32),
        "RFJ2": math.radians(24.39),
        "RFJ3": math.radians(70.06),
        "RFJ4": math.radians(-19.99),
        "LFJ1": math.radians(46.5),
        "LFJ2": math.radians(90.0),
        "LFJ3": math.radians(90.0),
        "LFJ4": math.radians(-0.6),
        "LFJ5": math.radians(4.0),
    }

    return joint_pos, joint_pos


def get_custom_ball_spawn():
    ball_offset = (-0.2945, 0.01, 0.063)
    ball_anchor = 0  # left hand

    return ball_offset, ball_anchor


def modify_env_config(env_cfg, joint_config=None, ball_config=None):
    """
    Modify the environment configuration with custom initial positions.
    """
    if joint_config is not None:
        left_joint_pos, right_joint_pos = joint_config

        # Update joint positions in the config
        for joint_name, value in left_joint_pos.items():
            env_cfg.left_joint_pos[joint_name] = value

        for joint_name, value in right_joint_pos.items():
            env_cfg.right_joint_pos[joint_name] = value

        print(f"[INFO]: Updated custom joint positions")

    if ball_config is not None:
        ball_offset, ball_anchor = ball_config

        # Update ball spawn configuration
        env_cfg.ball_offset[0] = ball_offset
        env_cfg.ball_anchor[0] = ball_anchor

        # Recalculate init_ball_pos
        anchor_hand_pos = env_cfg.hand_pos[ball_anchor]
        env_cfg.init_ball_pos[0] = (
            anchor_hand_pos[0] + ball_offset[0],
            anchor_hand_pos[1] + ball_offset[1],
            anchor_hand_pos[2] + ball_offset[2],
        )

        print(f"[INFO]: Updated custom ball spawn location: {env_cfg.init_ball_pos[0]}")

    return env_cfg


def main():
    """Baseline agent throw and catch one ball without fingers moving."""

    # Parse environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    joint_config = get_custom_joint_positions()
    ball_config = get_custom_ball_spawn()

    # Apply custom configurations
    env_cfg = modify_env_config(env_cfg, joint_config=joint_config, ball_config=ball_config)

    control_mode = "left_throw"

    env = gym.make(args_cli.task, cfg=env_cfg)

    # Print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    print(f"[INFO]: Control mode: {control_mode}")

    # Reset environment
    env.reset()

    # Time counter for control patterns
    t = 0.0
    dt = env_cfg.sim.dt * env_cfg.decimation  # Time step in seconds

    # Simulate environment
    while simulation_app.is_running():
        # Run everything in inference mode
        with torch.inference_mode():
            action_sample = generate_hand_actions(t, env.action_space.shape[-1], control_mode)
            actions = action_sample.unsqueeze(0).expand(env.action_space.shape[0], -1).to(env.unwrapped.device)

            env.step(actions)
            t += dt

    # Close the simulator
    env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
