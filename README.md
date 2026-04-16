<div style="text-align: left;">
  <img src="https://img.shields.io/badge/IsaacLab%20-v2.3.2-blue" alt="IsaacLab v2.3.0" style="margin-bottom: 1px;">
  <img src="https://img.shields.io/badge/rsl_rl%20-v3.3.0-blue" alt="rsl-rl v3.3.0" style="margin-bottom: 1px;">
  <div style="display: flex; justify-content: space-around;">
    <img src="./gifs/train.gif" alt="Train" width="30%">
    <img src="./gifs/sim-to-sim.gif" alt="Sim-to-Sim" width="32.5%">
    <img src="./gifs/sim-to-real_.gif" alt="Sim-to-Real" width="32%">
  </div>
</div>

## Overview


Reinforcement learning implementation of the **quadruped get-up task** in IsaacLab. It includes different robots, with scripts for sim-to-sim and sim-to-real transfer.

Features:
- [Concurrent State Estimator](https://arxiv.org/pdf/2202.05481)
- [Rapid Motor Adaptation](https://arxiv.org/pdf/2107.04034)
- Identification of robot parameters for sim2real using [pace](https://github.com/leggedrobotics/pace-sim2real) via our repo [sim2real-robot-identification](https://github.com/iit-DLSLab/sim2real-robot-identification)
- Sim-to-Sim in [Mujoco](https://github.com/google-deepmind/mujoco)
- Sim-to-Real using ROS2 

Real-world deployment via:
- [unitree-ros2-dls](https://github.com/iit-DLSLab/unitree-ros2-dls) for unitree robot communication

A list of robots and environments available is described below:

| Robot Model         | Environment Name Pattern                                   |
|---------------------|------------------------------------------------------------|
| [Aliengo](https://github.com/iit-DLSLab/gym-quadruped/tree/master/gym_quadruped/robot_model/aliengo), [Go2](https://github.com/iit-DLSLab/gym-quadruped/tree/master/gym_quadruped/robot_model/go2), [B2](https://github.com/iit-DLSLab/gym-quadruped/tree/master/gym_quadruped/robot_model/b2)| GetUp-**RobotModel**-Flat-Blind <br> GetUp-**RobotModel**-Rough-Blind <br>|


## Installation and Runs

If you want only to deploy a trained policy on your robot, continue on [README_DEPLOY](https://github.com/iit-DLSLab/get-up-dls-isaaclab/blob/main/README_DEPLOY.md) otherwise on [README_TRAIN](https://github.com/iit-DLSLab/get-up-dls-isaaclab/blob/main/README_TRAIN.md).

**For the train, check first the compatibility with IsaacLab and rsl-rl at the top of this readme. They indicate the releases that we tested.**


## How to contribute

PRs are very welcome (search for **TODO** in the issue, or add what you like)!


## Maintainer

This repository is maintained by [Giulio Turrisi](https://github.com/giulioturrisi).
