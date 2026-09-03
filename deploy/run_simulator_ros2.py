# Description: This script is used to run the ros2 simulator

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
from dls2_interface.msg import BaseState, BlindState, Imu, ControlSignal
from unitree_go.msg import LowState, MotorState, IMUState

import time
import numpy as np
np.set_printoptions(precision=3, suppress=True)

# Simulation related imports
import mujoco
import mujoco.viewer
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path + "/mujoco_utils/")
import mujoco_utils

# Config imports
import config as cfg

# Set the priority of the process
pid = os.getpid()
print("PID: ", pid)
os.system("renice -n -21 -p " + str(pid))
os.system("echo -20 > /proc/" + str(pid) + "/autogroup")
#for real time, launch it with chrt -r 99 python3 run_controller.py


USE_SCHEDULER = True # Use the scheduler to compute the control signal
SCHEDULER_FREQ = 500 # Frequency of the scheduler
RENDER_FREQ = 30

# Shell for the controllers ----------------------------------------------
class SimulatorROS2(Node):
    def __init__(self):
        super().__init__('SimulatorROS2')

        # Subscribers and Publishers
        self.publisher_base_state = self.create_publisher(BaseState,"/base_state", 1)
        self.publisher_blind_state = self.create_publisher(BlindState,"/blind_state_legged", 1)
        self.publisher_imu = self.create_publisher(Imu,"/imu", 1)

        self.publisher_low_state = self.create_publisher(LowState,"/lowstate", 1)

        self.subscriber_control_signal = self.create_subscription(ControlSignal,"/control_signal_legged", self.get_control_signal_callback, 1)

        self.timer = self.create_timer(1.0/SCHEDULER_FREQ, self.compute_simulator_step_callback)

        # Timing stuff
        self.loop_time = 0.002
        self.last_start_time = None
        self.last_mpc_loop_time = 0.0


        # Mujoco model and data
        self.mjModel = mujoco.MjModel.from_xml_path(file_path + "/mujoco_utils/robot_model/" + cfg.robot + "/" + cfg.scene + ".xml")
        self.mjModel.opt.timestep = 1.0/SCHEDULER_FREQ
        self.mjData = mujoco.MjData(self.mjModel)
        keyframe_id = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.mjData.qpos = self.mjModel.key_qpos[keyframe_id]
        mujoco.mj_forward(self.mjModel, self.mjData)

        self.viewer = mujoco.viewer.launch_passive(
            self.mjModel,
            self.mjData,
            show_left_ui=False,
            show_right_ui=False,
        )
        mujoco.mjv_defaultFreeCamera(self.mjModel, self.viewer.cam)
        self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False

        self.last_render_time = time.time()

        self.foot_body_ids = {
            leg: mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_BODY, leg + "_foot")
            for leg in ["FL", "FR", "RL", "RR"]
        }

        # Desired PD
        self.desired_joints_position = np.zeros(12)
        self.desired_joints_velocity = np.zeros(12)

        # Desired gains
        self.Kp = 0
        self.Kd = 0


    def get_control_signal_callback(self, msg):

        self.desired_joints_position = np.array(msg.joints_position)

        self.Kp = np.array(msg.kp)[0]
        self.Kd = np.array(msg.kd)[0]


    def compute_simulator_step_callback(self):

        qpos, qvel = self.mjData.qpos, self.mjData.qvel
        base_lin_vel = mujoco_utils.base_lin_vel(self.mjData, frame='world')
        base_ang_vel = mujoco_utils.base_ang_vel(self.mjData, frame='base')
        base_pos = mujoco_utils.base_pos(self.mjData)

        # Publish Base State ------------------------------------------------
        base_state_msg = BaseState()
        base_state_msg.pose.position = base_pos
        base_state_msg.pose.orientation = np.roll(self.mjData.qpos[3:7],-1)
        base_state_msg.velocity.linear = base_lin_vel
        base_state_msg.velocity.angular = base_ang_vel
        self.publisher_base_state.publish(base_state_msg)


        # Publish Blind State ------------------------------------------------
        blind_state_msg = BlindState()
        blind_state_msg.joints_position = self.mjData.qpos[7:19].tolist()
        blind_state_msg.joints_velocity = self.mjData.qvel[6:18].tolist()
        blind_state_msg.joints_effort = self.mjData.actuator_force.tolist()
        self.publisher_blind_state.publish(blind_state_msg)


        # Publish IMU ------------------------------------------------
        imu_msg = Imu()
        imu_msg.linear_acceleration = self.mjData.sensordata[0:3]
        imu_msg.angular_velocity = self.mjData.sensordata[3:6]
        # To be compliant with our hal, we expect the xyzw order,
        # but mujoco gives us wxyz, so we roll the array to get the correct order
        imu_msg.orientation = np.roll(np.array(self.mjData.sensordata[9:13]), -1)
        self.publisher_imu.publish(imu_msg)


        # Compute feet contact forces ------------------------------------------------
        feet_GRF = {leg: np.zeros(3) for leg in ["FL", "FR", "RL", "RR"]}
        leg_from_body_id = {body_id: leg for leg, body_id in self.foot_body_ids.items()}
        for contact_id in range(self.mjData.ncon):
            contact = self.mjData.contact[contact_id]
            body1_id = self.mjModel.geom_bodyid[contact.geom1]
            body2_id = self.mjModel.geom_bodyid[contact.geom2]
            if 0 in [body1_id, body2_id]:
                foot_body_id = body2_id if body1_id == 0 else body1_id
                if foot_body_id in leg_from_body_id:
                    force_contact = np.zeros(6)
                    mujoco.mj_contactForce(self.mjModel, self.mjData, contact_id, force_contact)
                    rotation_contact = contact.frame.reshape(3, 3)
                    feet_GRF[leg_from_body_id[foot_body_id]] += rotation_contact.T @ force_contact[:3]


        # Publish Low State of Unitree ------------------------------------------------
        # FR, FL, RR, RL convention to follow the unitree standard msgs
        lowstate_msg = LowState()
        for i in range(3):
            lowstate_msg.motor_state[i].q = self.mjData.qpos[10+i]
            lowstate_msg.motor_state[i].dq = self.mjData.qvel[9+i]
        for i in range(3):
            lowstate_msg.motor_state[i+3].q = self.mjData.qpos[7+i]
            lowstate_msg.motor_state[i+3].dq = self.mjData.qvel[6+i]
        for i in range(3):
            lowstate_msg.motor_state[i+6].q = self.mjData.qpos[16+i]
            lowstate_msg.motor_state[i+6].dq = self.mjData.qvel[15+i]
        for i in range(3):
            lowstate_msg.motor_state[i+9].q = self.mjData.qpos[13+i]
            lowstate_msg.motor_state[i+9].dq = self.mjData.qvel[12+i]

        lowstate_msg.foot_force = np.array([abs(feet_GRF["FR"][2]), abs(feet_GRF["FL"][2]), abs(feet_GRF["RR"][2]), abs(feet_GRF["RL"][2])], dtype=np.int16)
        lowstate_msg.imu_state.accelerometer = self.mjData.sensordata[0:3].astype(np.float32)
        lowstate_msg.imu_state.gyroscope = self.mjData.sensordata[3:6].astype(np.float32)
        lowstate_msg.imu_state.quaternion = np.roll(np.array(self.mjData.sensordata[9:13].astype(np.float32)), -1)
        self.publisher_low_state.publish(lowstate_msg)


        # Step the environment --------------------------------------------------------------------------------
        joints_pos = qpos[7:19]
        joints_vel = qvel[6:18]

        action = self.Kp*(self.desired_joints_position - joints_pos) - self.Kd*joints_vel

        max_torque = self.mjModel.actuator_ctrlrange[0:12, 1]
        action = np.clip(action, -max_torque*0.95, max_torque*0.95)
        self.mjData.ctrl[0:12] = action
        mujoco.mj_step(self.mjModel, self.mjData)


        # Render only at a certain frequency -----------------------------------------------------------------
        if time.time() - self.last_render_time > 1.0 / RENDER_FREQ:
            self.viewer.cam.lookat[:] = base_pos
            self.viewer.sync()
            self.last_render_time = time.time()


def main():
    print('Hello from the SimulatorROS2 node.')
    rclpy.init()

    simulator_ros2_node = SimulatorROS2()

    rclpy.spin(simulator_ros2_node)
    simulator_ros2_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
