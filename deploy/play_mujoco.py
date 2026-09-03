# Description: This script is used to simulate the full model of the robot in mujoco

# Authors:
# Giulio Turrisi

import time
import numpy as np
from tqdm import tqdm
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/../")
sys.path.append(dir_path+"/../scripts/rsl_rl")

# Simulation related imports
import mujoco
import mujoco.viewer
sys.path.append(dir_path+"/mujoco_utils/")
import mujoco_utils

# GetUp Policy imports
from getup_policy_wrapper import GetUpPolicyWrapper

import config


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)

    robot_name = config.robot
    scene_name = config.scene
    simulation_dt = 0.002


    # Create the mujoco model ---------------------------------------------------------------------
    mjModel = mujoco.MjModel.from_xml_path(dir_path + "/mujoco_utils/robot_model/" + robot_name + "/" + scene_name + ".xml")
    mjModel.opt.timestep = simulation_dt
    mjData = mujoco.MjData(mjModel)
    keyframe_id = mujoco.mj_name2id(mjModel, mujoco.mjtObj.mjOBJ_KEY, "home")
    mjData.qpos = mjModel.key_qpos[keyframe_id]
    mujoco.mj_forward(mjModel, mjData)

    viewer = mujoco.viewer.launch_passive(
        mjModel,
        mjData,
        show_left_ui=False,
        show_right_ui=False,
    )
    mujoco.mjv_defaultFreeCamera(mjModel, viewer.cam)
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False



    # Initialization of variables used in the main control loop --------------------------------
    getup_policy = GetUpPolicyWrapper(mjModel=mjModel)


    # --------------------------------------------------------------
    RENDER_FREQ = 30  # Hz
    last_render_time = time.time()
    step_num = 1

    while viewer.is_running():
        step_start = time.time()

        # Get the current state of the robot -----------------------------------------------------
        qpos, qvel = mjData.qpos, mjData.qvel
        imu_angular_velocity = mjData.sensordata[3:6]
        imu_orientation = mjData.sensordata[9:13]

        joints_pos_leg = qpos[7:19]
        joints_vel_leg = qvel[6:18]

        # RL controller --------------------------------------------------------------
        if step_num % round(1 / (getup_policy.RL_FREQ * simulation_dt)) == 0 and step_num > 5000:

            desired_joint_pos = getup_policy.compute_control(
                        joints_pos_leg=joints_pos_leg,
                        joints_vel_leg=joints_vel_leg,
                        imu_angular_velocity=imu_angular_velocity,
                        imu_orientation=imu_orientation)

        # PD controller --------------------------------------------------------------
        else:
            desired_joint_pos = getup_policy.desired_joint_pos


        Kp = getup_policy.Kp_walking
        Kd = getup_policy.Kd_walking

        tau_leg = Kp * (desired_joint_pos - joints_pos_leg) - Kd * joints_vel_leg


        # Set control and mujoco step ----------------------------------------------------------------------
        mjData.ctrl[0:12] = tau_leg
        mujoco.mj_step(mjModel, mjData)
        step_num += 1


        # Sleep to match real-time ---------------------------------------------------------
        loop_elapsed_time = time.time() - step_start

        if(loop_elapsed_time < simulation_dt):
            time.sleep(simulation_dt - (loop_elapsed_time))

        # Render only at a certain frequency -----------------------------------------------------------------
        if time.time() - last_render_time > 1.0 / RENDER_FREQ or step_num == 1:
            viewer.cam.lookat[:] = mujoco_utils.base_pos(mjData)
            viewer.sync()
            last_render_time = time.time()
