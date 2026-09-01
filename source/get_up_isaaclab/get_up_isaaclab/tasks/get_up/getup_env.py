# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, RayCasterCamera, RayCasterCameraCfg, MultiMeshRayCasterCamera, MultiMeshRayCasterCameraCfg, TiledCameraCfg, TiledCamera, patterns, Imu
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from isaaclab import cloner

from .go2_env_cfg import Go2FlatEnvCfg, Go2RoughBlindEnvCfg
from .pegasus_env_cfg import PegasusFlatEnvCfg, PegasusRoughBlindEnvCfg

from get_up_isaaclab.tasks.supervised_learning_networks import SimpleNN


def _normalize_actuator_gain(gain: torch.Tensor, nominal_gain: torch.Tensor) -> torch.Tensor:
    """Normalize an explicit actuator gain without dividing by a zero nominal gain."""
    valid = nominal_gain.abs() > torch.finfo(nominal_gain.dtype).eps
    denominator = torch.where(valid, nominal_gain, torch.ones_like(nominal_gain))
    return torch.where(valid, gain / denominator, torch.zeros_like(gain))


class GetUpEnv(DirectRLEnv):
    cfg: Go2FlatEnvCfg | Go2RoughBlindEnvCfg | PegasusFlatEnvCfg | PegasusRoughBlindEnvCfg

    def __init__(self, cfg: Go2FlatEnvCfg | Go2RoughBlindEnvCfg | PegasusFlatEnvCfg | PegasusRoughBlindEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._previous_previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )


        # Desired Hip Offset
        self._desired_hip_offset = torch.tensor([-self.cfg.desired_hip_offset, self.cfg.desired_hip_offset, -self.cfg.desired_hip_offset, self.cfg.desired_hip_offset], device=self.device)
        

        # Observation history
        self._observation_history = torch.zeros(self.num_envs, cfg.history_length, cfg.single_observation_space, device=self.device)

        # RMA
        if(cfg.use_rma == True):
            self._rma_network = SimpleNN(cfg.rma_observation_space, cfg.rma_output_space)
            self._rma_network.to(self.device)
            self._observation_history_rma = torch.zeros(self.num_envs, cfg.history_length, cfg.single_rma_observation_space, device=self.device)
            if self.cfg.observation_noise_model:
                self._observation_noise_model_rma: NoiseModel = self.cfg.observation_noise_model.class_type(
                    self.cfg.observation_noise_model, num_envs=self.num_envs, device=self.device
                )


        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_height_exp",
                "track_orientation_exp",

                "action_rate_l2",
                "action_smoothness_l2",
                
                "joints_acc_l2",
                "joints_torques_l2",
                "joints_energy_l1",
                
                "feet_to_hip_distance_l2",
            ]
        }
        # Get specific body indices
        self._base_contact_sensor_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_contact_sensor_ids, _ = self._contact_sensor.find_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True)
        self._hip_contact_sensor_ids, _ = self._contact_sensor.find_bodies(["FL_hip", "FR_hip", "RL_hip", "RR_hip"], preserve_order=True)
        self._thigh_contact_sensor_ids, _ = self._contact_sensor.find_bodies(["FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"], preserve_order=True)
        self._undesired_contact_body_ids = self._base_contact_sensor_id + self._hip_contact_sensor_ids + self._thigh_contact_sensor_ids

        
        self._feet_ids_robot, _ = self._robot.find_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True)
        self._hip_ids_robot, _ = self._robot.find_bodies(["FL_hip", "FR_hip", "RL_hip", "RR_hip"], preserve_order=True)

        # Ensure the order is consistent with the one expected in the cfg
        self._ids_joints_order = self._robot.find_joints(name_keys=self.cfg.desired_joints_order, preserve_order=True)[0]

        # Nominal (pre-randomization) explicit-actuator PD gains. Captured here, before any
        # "reset" mode event has run, since asset.data.default_joint_stiffness/damping is a
        # deprecated live snapshot that reads 0 for explicit actuators like PaceDCMotor
        # (the solver's own PD gains are zeroed; the actuator computes effort in Python).
        self._nominal_actuator_stiffness = {}
        self._nominal_actuator_damping = {}
        for joint_type in ("hip", "thigh", "calf"):
            actuator = self._robot.actuators[joint_type]
            self._nominal_actuator_stiffness[joint_type] = actuator.stiffness.clone()
            self._nominal_actuator_damping[joint_type] = actuator.damping.clone()


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # we add a height scanner for the proprioceptive getup
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner

        # we add an imu
        self._imu = Imu(self.cfg.imu)
        self.scene.sensors["imu"] = self._imu

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # clone and replicate environments
        src, dest = "/World/envs/env_0", "/World/envs/env_{}"
        positions = cloner.grid_transforms(
            self.scene.num_envs, self.scene.cfg.env_spacing, device=self.device
        )[0]
        global_paths = (self.cfg.terrain.prim_path,)
        plan = cloner.clone_plan_from_env_0(
            src, dest, self.scene.num_envs, self.device, positions, global_paths=global_paths
        )
        cloner.replicate(plan, stage=self.scene.stage)

        # PhysX replication requires explicit collision filtering between environments.
        if "physx" in self.scene.physics_backend:
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor):
        self._previous_previous_actions = self._previous_actions.clone()
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone()
        default_joint_pos_ordered = self._robot.data.default_joint_pos[:, self._ids_joints_order]
        
        # Clip the action
        self._actions = torch.clamp(self._actions, -self.cfg.desired_clip_actions, self.cfg.desired_clip_actions)

        # Filter the action
        if(self.cfg.use_filter_actions):
            alpha = 0.8
            temp = alpha * self._actions + (1 - alpha) * self._previous_actions
            self._processed_actions = self.cfg.action_scale * temp + default_joint_pos_ordered
        else:
            self._processed_actions = self.cfg.action_scale * self._actions + default_joint_pos_ordered


    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions, joint_ids=self._ids_joints_order)



    def _get_observations(self) -> dict:    
        
        # Standard Obs for the Actor/Critic
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._imu.data.ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._robot.data.joint_pos[:, self._ids_joints_order] - self._robot.data.default_joint_pos[:, self._ids_joints_order],
                    self._robot.data.joint_vel[:, self._ids_joints_order],
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        if(self.cfg.use_observation_history):
            #the bottom element is the newest observation!!
            self._observation_history = torch.cat((self._observation_history[:,1:,:], obs.unsqueeze(1)), dim=1)
            obs = torch.flatten(self._observation_history, start_dim=1)



        # If RMA, we add some other predicted obs
        if(self.cfg.use_rma):
            # Predict the RMA observation
            obs_rma = self._get_rma()
            obs = torch.cat((obs, obs_rma), dim=-1)


        # Final observations dictionary
        observations = {"policy": obs}       
        

        # Critic OBS could be different if needed
        if(self.cfg.use_asymmetric_ppo):
            obs_critic = self._get_privileged_observation()
            observations["critic"] = torch.cat((obs, obs_critic), dim=-1)
        else:
            observations["critic"] = obs
        # ------------------------------------------------------------------------------------------


        return observations


    def _get_rewards(self) -> torch.Tensor:

        # terrain orientation
        height_data_scanner = self._height_scanner.data.ray_hits_w[..., 2]
        height_data_scanner = torch.nan_to_num(height_data_scanner, nan=0.0, posinf=1.0, neginf=-1.0)
        height_data_scanner = torch.clip(height_data_scanner, min=-5, max=5) # Handle inf values
        mean_height_ray = torch.mean(height_data_scanner, dim=1)

        height_map_resolution = self._height_scanner.cfg.pattern_cfg.resolution
        height_map_x_points = int(round(self._height_scanner.cfg.pattern_cfg.size[0] / height_map_resolution)) + 1
        height_map_y_points = int(round(self._height_scanner.cfg.pattern_cfg.size[1] / height_map_resolution))
        distance_between_front_and_back = (height_map_x_points/2)* height_map_resolution

        cols_back = torch.arange(0, height_data_scanner.shape[1], height_map_x_points).unsqueeze(1) + torch.arange(int(height_map_x_points/2))
        cols_back = cols_back.flatten().to(height_data_scanner.device)
        selected_height_data_back = height_data_scanner[:, cols_back]

        cols_front = torch.arange(int(height_map_x_points/2), height_data_scanner.shape[1], height_map_x_points).unsqueeze(1) + torch.arange(int(height_map_x_points/2))
        cols_front = cols_front.flatten().to(height_data_scanner.device)
        selected_height_data_front = height_data_scanner[:, cols_front]

        mean_height_ray_front = torch.mean(selected_height_data_front, dim=1)
        mean_height_ray_back = torch.mean(selected_height_data_back, dim=1)
        delta_z = mean_height_ray_front - mean_height_ray_back
        delta_s = torch.tensor(distance_between_front_and_back).to(self.device)
        terrain_pitch = -torch.atan2(delta_z, delta_s)
        #terrain_pitch = torch.atan2(torch.sin(terrain_pitch), torch.cos(terrain_pitch))

        """cols_right = torch.arange(0, height_data_scanner.shape[1]//2, 1).unsqueeze(1) 
        cols_right = cols_right.flatten().to(height_data_scanner.device)
        selected_height_data_right = height_data_scanner[:, cols_right]

        cols_left = torch.arange(0, height_data_scanner.shape[1]//2, 1).unsqueeze(1) + height_data_scanner.shape[1]//2
        cols_left = cols_left.flatten().to(height_data_scanner.device)
        selected_height_data_left = height_data_scanner[:, cols_left]

        delta_z_roll = torch.mean(selected_height_data_left, dim=1) - torch.mean(selected_height_data_right, dim=1)
        delta_s_roll = torch.tensor((height_map_y_points-1)* height_map_resolution).to(self.device)
        terrain_roll = torch.atan2(delta_z_roll, delta_s_roll)
        # TODO check if we need roll in base frame
        """
        terrain_roll = torch.zeros_like(terrain_pitch)


        root_roll_w, root_pitch_w, _ = math_utils.euler_xyz_from_quat(self._robot.data.root_quat_w.torch)
        root_roll_w = torch.atan2(torch.sin(root_roll_w), torch.cos(root_roll_w))
        root_pitch_w = torch.atan2(torch.sin(root_pitch_w), torch.cos(root_pitch_w))
        
        base_orientation_error =  torch.square(terrain_pitch - root_pitch_w) + torch.square(terrain_roll - root_roll_w)
        base_orientation_mapped = torch.exp(-base_orientation_error / 0.1)


        # track_height
        height_data_scanner = self._height_scanner.data.ray_hits_w[..., 2]
        height_data_scanner = torch.nan_to_num(height_data_scanner, nan=0.0, posinf=1.0, neginf=-1.0)
        height_data_scanner = torch.clip(height_data_scanner, min=-5, max=5) # Handle inf values
        mean_height_ray = torch.mean(height_data_scanner, dim=1)

        height_error = torch.square(self.cfg.desired_base_height + mean_height_ray - self._robot.data.root_state_w[:, 2])
        height_error_mapped = torch.exp(-height_error / 0.1)

        should_optimize_height = (torch.abs(terrain_pitch - root_pitch_w) < 0.5) * (torch.abs(terrain_roll - root_roll_w) < 0.5)
        height_error_mapped = height_error_mapped * should_optimize_height

        
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        action_smoothness = torch.sum(torch.square(self._actions - 2*self._previous_actions + self._previous_previous_actions), dim=1)
        

        # joint acceleration
        joints_accel = torch.sum(torch.square(self._robot.data.joint_acc[:, self._ids_joints_order]), dim=1)


        # joint torques
        joints_torques = torch.sum(torch.square(self._robot.data.applied_torque[:, self._ids_joints_order]), dim=1)


        # energy = torque * velocity
        joints_energy = torch.sum(
            torch.abs(
                self._robot.data.applied_torque[:, self._ids_joints_order]
                * self._robot.data.joint_vel[:, self._ids_joints_order]
            ),
            dim=1,
        )



        # feet to hip distance
        ROT_W2H = math_utils.matrix_from_quat(math_utils.yaw_quat(self._robot.data.root_quat_w.torch))
        feet_to_base_w = self._robot.data.body_pos_w[:, self._feet_ids_robot, :3] - self._robot.data.root_state_w[:, :3].unsqueeze(1)
        feet_to_base_h = torch.matmul(ROT_W2H.transpose(1,2), feet_to_base_w.transpose(1, 2))
        
        hip_to_base_w = self._robot.data.body_pos_w[:, self._hip_ids_robot, :3] - self._robot.data.root_state_w[:, :3].unsqueeze(1)
        hip_to_base_h = torch.matmul(ROT_W2H.transpose(1,2), hip_to_base_w.transpose(1, 2))
        
        desired_hip_offset = self._desired_hip_offset
        feet_to_hip_distance_x = torch.square(feet_to_base_h[:, 0] - hip_to_base_h[:, 0])
        feet_to_hip_distance_y = torch.square(feet_to_base_h[:, 1] + desired_hip_offset.unsqueeze(0) - hip_to_base_h[:, 1])
        feet_to_hip_distance = -torch.mean(torch.sqrt(feet_to_hip_distance_x + feet_to_hip_distance_y), dim=1)
        
        should_stance = (torch.abs(terrain_pitch - root_pitch_w) < 0.5) * (torch.abs(terrain_roll - root_roll_w) < 0.5)
        feet_to_hip_distance = feet_to_hip_distance * should_stance


        rewards = {
            "track_height_exp": height_error_mapped * self.cfg.height_reward_scale * self.step_dt,
            "track_orientation_exp": base_orientation_mapped * self.cfg.orientation_reward_scale * self.step_dt,

            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "action_smoothness_l2": action_smoothness * self.cfg.action_smoothness_reward_scale * self.step_dt,

            "joints_acc_l2": joints_accel * self.cfg.joints_accel_reward_scale * self.step_dt,
            "joints_torques_l2": joints_torques * self.cfg.joints_torque_reward_scale * self.step_dt,
            "joints_energy_l1": joints_energy * self.cfg.joints_energy_reward_scale * self.step_dt,

            "feet_to_hip_distance_l2": feet_to_hip_distance * self.cfg.feet_to_hip_distance_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Check for NaNs and Infs
        if torch.isnan(reward).any() or torch.isinf(reward).any():
            print("NaN or Inf detected in reward computation. Setting reward to zero for affected environments.")
            breakpoint()  # For debugging purposes
            reward = torch.where(torch.isnan(reward) | torch.isinf(reward), torch.zeros_like(reward), reward)
        
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        return died, time_out


    def _reset_idx(self, env_ids: torch.Tensor | None):
        # Isaac Lab 3.0 compat: ids may arrive (or _ALL_INDICES may be) warp arrays
        def _to_torch_ids(ids):
            if ids is not None and not torch.is_tensor(ids):
                import warp as wp
                ids = wp.to_torch(ids)
            return ids.to(dtype=torch.long) if ids is not None else ids

        env_ids = _to_torch_ids(env_ids)
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = _to_torch_ids(self._robot._ALL_INDICES)


        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs: 
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._previous_previous_actions[env_ids] = 0.0
        



        if(self.cfg.use_rma):
            if self.cfg.observation_noise_model:
                self._observation_noise_model_rma.reset(env_ids)

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos += torch.zeros_like(joint_pos).uniform_(-3.14159, 3.14159)
        # we need to project them inside the robots limits
        joints_limits = self._robot.data.default_joint_pos_limits
        joints_legs_limits = joints_limits[:,self._ids_joints_order]
        joint_pos[:, self._ids_joints_order] = torch.clamp(joint_pos[:, self._ids_joints_order], joints_legs_limits[0,:,0], joints_legs_limits[0,:,1])

        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 0:2] += torch.zeros_like(default_root_state[:, 0:2]).uniform_(-2.0, 2.0)
        
        # Apply random orientation except for environments with IDs in [1, 100]
        rand_quats = math_utils.random_orientation(env_ids.shape[0], device=self.device)
        env_ids_device = env_ids.to(self.device)
        no_random_mask = (env_ids_device >= 0) & (env_ids_device <= 500)
        if no_random_mask.any() and self.num_envs > 500:
            rand_quats[no_random_mask] = default_root_state[no_random_mask, 3:7]
        default_root_state[:, 3:7] = rand_quats
        
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        
        if(self._terrain.cfg.terrain_generator is not None and self._terrain.cfg.terrain_generator.curriculum == True):
            extras["Episode_Curriculum/terrain_levels"] = torch.mean(self._terrain.terrain_levels.float())
        
        self.extras["log"].update(extras)


    def _get_rma(self, ):
        joint_pos_error_ordered = self._robot.data.joint_pos[:, self._ids_joints_order] - self._robot.data.default_joint_pos[:, self._ids_joints_order]
        joint_vel_ordered = self._robot.data.joint_vel[:, self._ids_joints_order]
        # Learning privileged information via supervised learning
        obs_rma = torch.cat(
            [
                tensor
                for tensor in (
                    self._imu.data.lin_acc_b,
                    self._imu.data.ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    joint_pos_error_ordered,
                    joint_vel_ordered * self.cfg.observation_joint_vel_scale,
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        #the bottom element is the newest observation!!
        self._observation_history_rma = torch.cat((self._observation_history_rma[:,1:,:], obs_rma.unsqueeze(1)), dim=1)
        obs = torch.flatten(self._observation_history_rma, start_dim=1)

        # Add noise to the observation - this is usually done in direct_rl.py in IsaacLab, but 
        # the obs of concurrent SE does not pass from there - its prediciton yes instead!
        if self.cfg.observation_noise_model:          
            obs = self._observation_noise_model_rma(obs.clone())  
        
        outputs_rma = self._get_privileged_observation()

        self._rma_network.dataset.add_sample(obs, outputs_rma)

        # Prediction
        num_episode_from_start = self.common_step_counter / 24. #self.max_episode_length #HACK this should be taken from rsl rl
        num_final_episode_from_start = 8000.
        if num_episode_from_start > self.cfg.rma_ep_saving_start:
            with torch.no_grad(): 
                prediction_rma = self._rma_network(obs)
            obs_rma = prediction_rma
        else:
            obs_rma = outputs_rma

        # Train at some interval
        if (num_episode_from_start % self.cfg.rma_ep_saving_interval == 0 and 
            num_episode_from_start > self.cfg.rma_ep_saving_start - 1 and 
                num_episode_from_start < num_final_episode_from_start - 500):  # Adjust the interval as needed
            self._rma_network.train_network(batch_size=self.cfg.rma_batch_size, 
                                            epochs=self.cfg.rma_train_epochs, 
                                            learning_rate=self.cfg.rma_lr, 
                                            device=self.device)
            # Save the network
            self._rma_network.save_network("rma.pth", self.device)
        
        return obs_rma


    def _get_privileged_observation(self):
        asset_cfg = SceneEntityCfg("robot", joint_names=[".*"])
        asset: Articulation = self.scene[asset_cfg.name]


        # PD of the joints
        hip_stiffness = _normalize_actuator_gain(asset.actuators["hip"].stiffness, self._nominal_actuator_stiffness["hip"])
        thigh_stiffness = _normalize_actuator_gain(asset.actuators["thigh"].stiffness, self._nominal_actuator_stiffness["thigh"])
        calf_stiffness = _normalize_actuator_gain(asset.actuators["calf"].stiffness, self._nominal_actuator_stiffness["calf"])

        hip_damping = _normalize_actuator_gain(asset.actuators["hip"].damping, self._nominal_actuator_damping["hip"])
        thigh_damping = _normalize_actuator_gain(asset.actuators["thigh"].damping, self._nominal_actuator_damping["thigh"])
        calf_damping = _normalize_actuator_gain(asset.actuators["calf"].damping, self._nominal_actuator_damping["calf"])

        # height error
        height_data_scanner = self._height_scanner.data.ray_hits_w[..., 2]
        height_data_scanner = torch.nan_to_num(height_data_scanner, nan=0.0, posinf=1.0, neginf=-1.0)
        height_data_scanner = torch.clip(height_data_scanner, min=-5, max=5) # Handle inf values
        mean_height_ray = torch.mean(height_data_scanner, dim=1)
        height_error = torch.abs(self.cfg.desired_base_height + mean_height_ray - self._robot.data.root_state_w[:, 2])


        # terrain orientation
        height_map_resolution = self._height_scanner.cfg.pattern_cfg.resolution
        height_map_x_points = int(round(self._height_scanner.cfg.pattern_cfg.size[0] / height_map_resolution)) + 1
        height_map_y_points = int(round(self._height_scanner.cfg.pattern_cfg.size[1] / height_map_resolution))
        distance_between_front_and_back = (height_map_x_points/2)* height_map_resolution

        cols_back = torch.arange(0, height_data_scanner.shape[1], height_map_x_points).unsqueeze(1) + torch.arange(int(height_map_x_points/2))
        cols_back = cols_back.flatten().to(height_data_scanner.device)
        selected_height_data_back = height_data_scanner[:, cols_back]

        cols_front = torch.arange(int(height_map_x_points/2), height_data_scanner.shape[1], height_map_x_points).unsqueeze(1) + torch.arange(int(height_map_x_points/2))
        cols_front = cols_front.flatten().to(height_data_scanner.device)
        selected_height_data_front = height_data_scanner[:, cols_front]

        mean_height_ray_front = torch.mean(selected_height_data_front, dim=1)
        mean_height_ray_back = torch.mean(selected_height_data_back, dim=1)
        delta_z = mean_height_ray_front - mean_height_ray_back
        delta_s = torch.tensor(distance_between_front_and_back).to(self.device)
        terrain_pitch = -torch.atan2(delta_z, delta_s)

        contacts_foot = self._contact_sensor.data.net_forces_w_history[:, :, self._feet_contact_sensor_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0

        # Pose height scanner data
        height_data = (
            self._height_scanner.data.pos_w[:, 2].unsqueeze(1)
            - self._height_scanner.data.ray_hits_w[..., 2]
            - 0.5
        )
        height_data = torch.nan_to_num(height_data, nan=0.0, posinf=1.0, neginf=-1.0)
        height_data = height_data.clip(-1.0, 1.0)
    


        obs_privileged = torch.cat((
                            hip_stiffness, thigh_stiffness, calf_stiffness, #P gain
                            hip_damping, thigh_damping, calf_damping, #D gain
                            height_error.unsqueeze(1),
                            terrain_pitch.unsqueeze(1),
                            contacts_foot,
                            height_data
                            ) 
                        , dim=-1)
        return obs_privileged