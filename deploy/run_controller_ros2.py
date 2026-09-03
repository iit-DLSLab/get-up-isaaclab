# Description: This script is used to run the policy on the real robot

# Authors:
# Giulio Turrisi
import os
import sys
import shlex
import subprocess
from pathlib import Path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

dir_path = Path(__file__).resolve().parent
sys.path.append(str(dir_path / ".."))

ros_ws = dir_path / "ros2_ws"
setup_bash = ros_ws / "install" / "setup.bash"

if not setup_bash.exists():
    print("Building the msgs first...")
    subprocess.run(["colcon", "build"], cwd=ros_ws, check=True)

if os.environ.get("GET_UP_SOURCED") != "1":
    print("Sourcing ROS2 workspace and restarting script...")
    cmd = (
        f"source {shlex.quote(str(setup_bash))} && "
        "export GET_UP_SOURCED=1 && "
        f"exec {shlex.quote(sys.executable)} "
        + " ".join(shlex.quote(arg) for arg in [str(Path(__file__).resolve()), *sys.argv[1:]])
    )
    os.execv("/bin/bash", ["bash", "-c", cmd])


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from dls2_interface.msg import BaseState, BlindState, Imu, ControlSignal

import time
import numpy as np
np.set_printoptions(precision=3, suppress=True)

import threading

import copy

# Simulation related imports
import mujoco
import mujoco.viewer
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path + "/mujoco_utils/")
import mujoco_utils


# GetUp Policy imports
from getup_policy_wrapper import GetUpPolicyWrapper

import config

# Set the priority of the process
pid = os.getpid()
print("PID: ", pid)
os.system("renice -n -21 -p " + str(pid))
os.system("echo -20 > /proc/" + str(pid) + "/autogroup")
#for real time, launch it with chrt -r 99 python3 run_controller.py


USE_MUJOCO_RENDER = False


class ControllerROS2(Node):
    def __init__(self):
        super().__init__('ControllerROS2')

        # Mujoco model and data
        self.mjModel = mujoco.MjModel.from_xml_path(file_path + "/mujoco_utils/robot_model/" + config.robot + "/" + config.scene + ".xml")
        self.mjData = mujoco.MjData(self.mjModel)
        keyframe_id = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_KEY, "down")
        self.mjData.qpos = self.mjModel.key_qpos[keyframe_id]
        mujoco.mj_forward(self.mjModel, self.mjData)

        self.last_render_time = time.time()
        if USE_MUJOCO_RENDER:
            self.viewer = mujoco.viewer.launch_passive(
                self.mjModel,
                self.mjData,
                show_left_ui=False,
                show_right_ui=False,
            )
            mujoco.mjv_defaultFreeCamera(self.mjModel, self.viewer.cam)


        # Subscribers and Publishers
        self.subscription_base_state = self.create_subscription(BaseState,"/base_state", self.get_base_state_callback, 1)
        self.subscription_blind_state = self.create_subscription(BlindState,"/blind_state_legged", self.get_blind_state_callback, 1)
        self.subscription_imu = self.create_subscription(Imu,"/imu", self.get_imu_callback, 1)

        self.subscription_joy = self.create_subscription(Joy,"/joy", self.get_joy_callback, 1)
        self.last_joy_time = None

        self.publisher_control_signal = self.create_publisher(ControlSignal,"/control_signal_legged", 1)
        self.sequence_id = 0 # To keep track of the last msg sent, useful for debugging and synchronization
        RL_FREQ = 1./(config.training_env["sim"]["dt"]*config.training_env["decimation"])  # Hz, frequency of the RL controller
        self.timer = self.create_timer(1.0/RL_FREQ, self.compute_rl_control)


        # Safety check to not do anything until a first base and blind state are received
        self.first_message_base_arrived = False
        self.first_message_joints_arrived = False
        self.first_message_imu_arrived = False

        # Timing stuff
        self.loop_time = 0.002
        self.last_start_time = None

        # Base State
        self.position = np.zeros(3)
        self.orientation = np.zeros(4)
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)

        # Blind State
        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)

        # IMU
        self.imu_linear_acceleration = np.zeros(3)
        self.imu_angular_velocity = np.zeros(3)
        self.imu_orientation = np.zeros(4)

        # Commands
        self.ref_base_lin_vel_H = np.zeros(3)
        self.ref_base_ang_yaw_dot = 0.0


        # Initialization of variables used in the main control loop --------------------------------
        self.get_up_policy = GetUpPolicyWrapper(mjModel=self.mjModel)


        goDown_qpos = self.mjModel.key_qpos[keyframe_id]
        self.stand_up_and_down_actions = goDown_qpos[7:19].copy()
        self.joint_positions = goDown_qpos[7:19].copy()


        # Interactive Command Line ----------------------------
        from console import Console
        self.console = Console(controller_node=self)
        thread_console = threading.Thread(target=self.console.interactive_command_line)
        thread_console.daemon = True
        thread_console.start()


    def get_joy_callback(self, msg):
        """
        Callback function to handle joystick input. Joystick used is a
        8Bitdi Ultimate 2C Wireless Controller.
        """

        filter_joystick = 0.7
        self.ref_base_lin_vel_H[0] = self.ref_base_lin_vel_H[0]*filter_joystick + (msg.axes[1]/3.5)*(1-filter_joystick)  # Forward/Backward
        self.ref_base_lin_vel_H[1] = self.ref_base_lin_vel_H[1]*filter_joystick + (msg.axes[0]/3.5)*(1-filter_joystick)  # Left/Right
        self.ref_base_ang_yaw_dot = self.ref_base_ang_yaw_dot*filter_joystick + (msg.axes[3]/2.)*(1-filter_joystick)  # Yaw

        self.last_joy_time = time.time()

        #kill the node if the button is pressed
        if msg.buttons[8] == 1:
            self.get_logger().info("Joystick button pressed, shutting down the node.")
            # This will kill the robot hal
            os.system("kill -9 $(ps -u | grep -m 1 hal | grep -o \"^[^ ]* *[0-9]*\" | grep -o \"[0-9]*\")")
            # This will kill the process running this script
            os.system("pkill -f play_ros2.py")
            exit(0)



    def get_base_state_callback(self, msg):
        self.position = np.array(msg.pose.position) #world frame
        # For the quaternion, the order is [x, y, z, w] on DLS2 but here we want [w, x, y, z] (mujoco convention)
        self.orientation = np.roll(np.array(msg.pose.orientation), 1) #world frame
        self.linear_velocity = np.array(msg.velocity.linear) #world frame
        self.angular_velocity = np.array(msg.velocity.angular) #base frame

        self.first_message_base_arrived = True



    def get_blind_state_callback(self, msg):
        self.joint_positions = np.array(msg.joints_position)
        self.joint_velocities = np.array(msg.joints_velocity)

        self.first_message_joints_arrived = True


    def get_imu_callback(self, msg):
        self.imu_linear_acceleration = np.array(msg.linear_acceleration)
        self.imu_angular_velocity = np.array(msg.angular_velocity)
        # For the quaternion, the order is [x, y, z, w] on DLS2 but here we want [w, x, y, z] (mujoco convention)
        self.imu_orientation = np.roll(np.array(msg.orientation), 1)

        self.first_message_imu_arrived = True


    def compute_rl_control(self):
        # Update the loop time
        start_time = time.perf_counter()
        if(self.last_start_time is not None):
            self.loop_time = (start_time - self.last_start_time)
        self.last_start_time = start_time
        simulation_dt = self.loop_time


        # Safety check to not do anything until a first base and blind state are received
        if(self.first_message_imu_arrived==False or self.first_message_joints_arrived==False):
            return

        # Update the mujoco model
        # Note that in case of IMU or concurrent state estimator, these info below are not used,
        # In the case we have a state estimator, this is usefull only for debugging visually
        self.mjData.qpos[0:3] = copy.deepcopy(self.position)
        self.mjData.qvel[0:3] = copy.deepcopy(self.linear_velocity)
        self.mjData.qpos[3:7] = copy.deepcopy(self.imu_orientation)
        self.mjData.qvel[3:6] = copy.deepcopy(self.imu_angular_velocity)


        # These info instead are used for sure in all the cases
        self.mjData.qpos[7:19] = copy.deepcopy(self.joint_positions)
        self.mjData.qvel[6:18] = copy.deepcopy(self.joint_velocities)
        self.mjModel.opt.timestep = simulation_dt
        mujoco.mj_forward(self.mjModel, self.mjData)

        # Safety check for joystick timeout
        if(self.last_joy_time is not None and time.time() - self.last_joy_time > 1.0):
            self.ref_base_lin_vel_H[0] = 0.0
            self.ref_base_lin_vel_H[1] = 0.0
            self.ref_base_ang_yaw_dot = 0.0
            print("Joystick timeout, stopping the robot")
            self.last_joy_time = None


        get_up_policy = self.get_up_policy

        qpos, qvel = self.mjData.qpos, self.mjData.qvel

        joints_pos_leg = qpos[7:19]
        joints_vel_leg = qvel[6:18]

        # variable saved for goDown and goUp motion
        self.joint_positions = joints_pos_leg.copy()


        if(self.console.isRLActivated):

            desired_joint_pos = get_up_policy.compute_control(
                        joints_pos_leg=joints_pos_leg,
                        joints_vel_leg=joints_vel_leg,
                        imu_angular_velocity=self.imu_angular_velocity,
                        imu_orientation=self.imu_orientation)

            # Impedence Loop
            Kp = get_up_policy.Kp_walking
            Kd = get_up_policy.Kd_walking


        else:
            desired_joint_pos = self.stand_up_and_down_actions

            # Impedence Loop
            Kp = get_up_policy.Kp_stand_up_and_down
            Kd = get_up_policy.Kd_stand_up_and_down

        # Publish the desired joint positions to the control signal --------------------------------
        control_signal_msg = ControlSignal()
        control_signal_msg.timestamp = float(self.get_clock().now().nanoseconds)
        control_signal_msg.sequence_id = int(self.sequence_id % 1000)  # To avoid overflow, we reset the sequence id after it reaches a certain value
        self.sequence_id += 1
        control_signal_msg.joints_position = np.array(desired_joint_pos).flatten().tolist()
        control_signal_msg.joints_velocity = np.zeros(12).tolist()
        control_signal_msg.joints_torques = np.zeros(12).tolist()
        control_signal_msg.kp = (np.ones(12) * Kp).tolist()
        control_signal_msg.kd = (np.ones(12) * Kd).tolist()

        self.publisher_control_signal.publish(control_signal_msg)



        # Render the simulation at a certain frequency -----------------------------------------------------------
        if USE_MUJOCO_RENDER:
            RENDER_FREQ = 30  # Hz
            if time.time() - self.last_render_time > 1.0 / RENDER_FREQ:
                self.viewer.cam.lookat[:] = mujoco_utils.base_pos(self.mjData)
                self.viewer.sync()
                self.last_render_time = time.time()




#---------------------------
if __name__ == '__main__':

    print('Hello from get-up-dls-isaaclab ros node.')

    rclpy.init()
    controller_ros2_node = ControllerROS2()
    rclpy.spin(controller_ros2_node)

    controller_ros2_node.destroy_node()
    rclpy.shutdown()

    print("ControllerROS2 node is stopped")
    exit(0)
