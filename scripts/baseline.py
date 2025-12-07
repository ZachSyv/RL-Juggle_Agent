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


def get_ball_in_hand_status(env):
    """
    Check if ball is currently in left or right hand from the environment.
    """
    unwrapped = env.unwrapped
    # Get in_hand status for first environment (env 0)
    # in_hand shape: (num_envs, num_hands) where hands are [0=left, 1=right]
    in_left_hand = unwrapped.in_hand[0, 0].item()  # Boolean
    in_right_hand = unwrapped.in_hand[0, 1].item()  # Boolean
    return in_left_hand, in_right_hand


def throw_motion(t, windup_duration, throw_duration, hold_duration, windup_strength, throw_strength, rotation_strength):
    """
    Generate throwing motion timing for a hand.
    Returns the action values for elbow_bend and elbow_rotate based on current phase.
    """
    cycle_time = hold_duration + windup_duration + throw_duration
    phase = t % cycle_time

    if phase < hold_duration:
        # Hold first
        return 0.0, 0.0
    elif phase < hold_duration + windup_duration:
        # Wind-up phase: slowly lower the hand (positive value)
        return windup_strength, rotation_strength
    elif phase < hold_duration + windup_duration + throw_duration:
        # Throw phase: quickly throw up (negative value)
        return throw_strength, rotation_strength
    else:
        # Back to neutral
        return 0.0, 0.0


def generate_hand_actions(t, action_dim, throwing_hand="left", throw_params=None):
    """Generate hand actions for juggling.

    Args:
        t: Current time in seconds
        action_dim: Dimension of the action space
        throwing_hand: "left" or "right" - which hand is currently throwing
        throw_params: Dictionary with throw parameters

    Returns:
        Tensor of actions for all joints
    """
    if throw_params is None:
        throw_params = {}

    actions = torch.zeros(action_dim)

    # Get throw motion values
    elbow_bend_action, rotation_action = throw_motion(
        t,
        windup_duration=throw_params.get("windup_duration", 0.5),
        throw_duration=throw_params.get("throw_duration", 0.2),
        hold_duration=throw_params.get("hold_duration", 1.0),
        windup_strength=throw_params.get("windup_strength", 0.5),
        throw_strength=throw_params.get("throw_strength", -0.8),
        rotation_strength=throw_params.get("rotation_strength", -0.2)
    )

    # Catching hand gets minor adjustment force
    catch_force = throw_params.get("catch_force", 0.12)

    if throwing_hand == "left":
        actions[1] = elbow_bend_action   # left elbow_bend
        actions[0] = rotation_action     # left elbow_rotate

        actions[27] = catch_force        # right elbow_bend - minor downward adjustment
    else:  # throwing_hand == "right"
        actions[27] = elbow_bend_action  # right elbow_bend
        actions[26] = -rotation_action   # right elbow_rotate

        actions[1] = catch_force         # left elbow_bend - minor downward adjustment

    return actions


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

    return joint_pos


def get_custom_ball_spawn():
    ball_offset = (-0.2945, 0.01, 0.063)
    ball_anchor = 0  # left hand

    return ball_offset, ball_anchor


def modify_env_config(env_cfg, joint_config=None, ball_config=None):
    """
    Modify the environment configuration with custom initial positions.
    """
    if joint_config is not None:
        joint_pos = joint_config

        # Update joint positions in the config
        for joint_name, value in joint_pos.items():
            env_cfg.left_joint_pos[joint_name] = value

        for joint_name, value in joint_pos.items():
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

    env_cfg.episode_length_s = 30.0

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

    # Choose control mode: "zero" or "left_throw"
    control_mode = "left_throw"

    # Throw parameters
    throw_params = {
        "windup_duration": 0.5,       # Duration to lower hand (wind-up) in seconds
        "windup_strength": 0.165,     # Positive value to lower hand during wind-up
        "throw_duration": 0.15,       # Duration of throw motion in seconds
        "throw_strength": -3.5,       # Negative value for elbow_bend to throw up
        "hold_duration": 1.0,         # Duration to hold before repeating
        "rotation_strength": -0.1,    # Extra force for elbow_rotate during windup/throw
        "catch_force": 0.12           # Minor force on catching hand
    }

    env = gym.make(args_cli.task, cfg=env_cfg)

    # Print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    print(f"[INFO]: Starting with LEFT hand throwing, RIGHT hand catching")

    # Reset environment
    env.reset()

    # Time counter for control patterns
    t = 0.0
    dt = env_cfg.sim.dt * env_cfg.decimation  # Time step in seconds

    # Juggling state tracking
    throwing_hand = "left"  # Start with left hand throwing
    prev_in_left = False
    prev_in_right = False

    # Status display tracking
    # status_interval = 0.1  # Print status every 0.1 seconds
    # last_status_time = 0.0

    # Simulate environment
    while simulation_app.is_running():
        # Run everything in inference mode
        with torch.inference_mode():
            # Get ball in hand status
            in_left, in_right = get_ball_in_hand_status(env)

            # Detect catch and switch throwing hand
            if throwing_hand == "left":
                if in_right and not prev_in_right:
                    # Right hand just caught the ball!
                    throwing_hand = "right"
                    t = 0.0  # Reset timer for new throw cycle
                    print(f"\n>>> CATCH! Switching to RIGHT hand throwing <<<\n")
            else:  # throwing_hand == "right"
                if in_left and not prev_in_left:
                    # Left hand just caught the ball!
                    throwing_hand = "left"
                    t = 0.0  # Reset timer for new throw cycle
                    print(f"\n>>> CATCH! Switching to LEFT hand throwing <<<\n")

            # Update previous status
            prev_in_left = in_left
            prev_in_right = in_right

            # Generate actions based on which hand is throwing
            action_sample = generate_hand_actions(t, env.action_space.shape[-1], throwing_hand, throw_params)
            actions = action_sample.unsqueeze(0).expand(env.action_space.shape[0], -1).to(env.unwrapped.device)

            env.step(actions)

            # Display ball status periodically
            # if t - last_status_time >= status_interval:
            #     last_status_time = t
            #
            #     # Display status
            #     status_str = f"[t={t:.2f}s] Throwing: {throwing_hand.upper():5s} | Ball: "
            #     if in_left:
            #         status_str += "IN LEFT HAND"
            #     elif in_right:
            #         status_str += "IN RIGHT HAND"
            #     else:
            #         status_str += "IN AIR"
            #     print(status_str)

            t += dt

    # Close the simulator
    env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
