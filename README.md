# Juggling Agent

## Current 1 ball version:

Most up to date code is on the 1-ball branch.

https://github.com/user-attachments/assets/5be8330f-05b7-4a3e-ad5b-a6c22eaa3850


## Command to Train
Move into the directory containing IsaacLab (mine is isaacsim/Isaaclab)

Assuming the project is stored parallel to isaacsim directory.

### Install package
./isaaclab.sh -p -m pip install -force-reinstall -e ../../JugglingAgent/RL-Juggle_Agent/source/Juggling_Agent

### Train model in sim

./isaaclab.sh -p ../../JugglingAgent/RL-Juggle_Agent/scripts/rl_games/train.py --task=Isaac-JugglingAgent-v0 --headless
