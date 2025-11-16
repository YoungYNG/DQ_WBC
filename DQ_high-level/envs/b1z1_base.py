import numpy as np
import os
import torch
import cv2
import random
from typing import Dict, Any, Tuple, List, Set
from collections import defaultdict
from termcolor import cprint

from .reward_vec_task import RewardVecTask
from utils.low_level_model import ActorCritic

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from isaacgym.torch_utils import *
from torch import Tensor
from torchvision import transforms
import time
from torchvision.utils import save_image
from legged_gym.envs.manip_loco.b1z1_config import B1Z1RoughCfg
from legged_gym.utils.terrain import Terrain, Terrain_Perlin
from .visual_perb import perturb_depth_with_seg,sample_perturbation_params

# LIN_VEL_X_CLIP = 0.15
LIN_VEL_X_CLIP = 0.15
ANG_VEL_YAW_CLIP = 0.35
ANG_VEL_PITCH_CLIP = 0.35
GAIT_WAIT_TIME = 35

@torch.jit.script
def torch_rand_int(lower, upper, shape, device):
    # type: (int, int, Tuple[int, int], str) -> Tensor
    return torch.randint(lower, upper, shape, device=device)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class B1Z1Base(RewardVecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, 
                 virtual_screen_capture: bool = False, force_render: bool = False, 
                 use_roboinfo: bool = True, observe_gait_commands: bool = True, 
                 no_feature: bool = False, mask_arm: bool = False, depth_random=False,
                 robot_start_pose: tuple =(-2.00, 0.0, 0.66), stu_distill=False, 
                 commands_curriculum=True, pitch_control=False, pred_success=False,
                 rand_control=False, arm_delay=False, num_gripper_dof=1, rand_cmd_scale=False,
                 rand_depth_clip=False, stop_pick=False, eval=False,
                 *args, **kwargs
                ):
        self.cfg = cfg
        self.floating_base = self.cfg["env"].get("floatingBase", False)
        self.use_roboinfo = use_roboinfo
        self.observe_gait_commands = observe_gait_commands and (not self.floating_base)
        self.no_feature = no_feature
        self.mask_arm = mask_arm
        self.num_features = 0
        self.depth_random = depth_random
        self.stu_distill = stu_distill
        self.commands_curriculum = commands_curriculum
        self.pitch_control = pitch_control
        self.pred_success = pred_success
        self.rand_control = rand_control
        self.arm_delay = arm_delay
        self.num_gripper_dof = num_gripper_dof
        self.rand_cmd_scale = rand_cmd_scale
        self.rand_depth_clip = rand_depth_clip
        self.stop_pick = stop_pick
        self.eval = eval
        ##### my add #####
        self.camera_test = False
        self.one_img_features = self.cfg["sensor"]["onboard_camera"]["resolution"][1] * self.cfg["sensor"]["onboard_camera"]["resolution"][0] * 4
        self.one_img_features_1_ch = self.cfg["sensor"]["onboard_camera"]["resolution"][1] * self.cfg["sensor"]["onboard_camera"]["resolution"][0] 
        self.two_img_features = self.cfg["sensor"]["onboard_camera"]["resolution"][1] * self.cfg["sensor"]["onboard_camera"]["resolution"][0] * 8
        self.car_dofs_num = 4
        self.i=0
        self.k = 0
        self.collect_dataset_flag=0
        self.data_all_step_num = 2000
        self.step_forward_record_num = 0
        self.all_steps_data = []
        self.frame_list = []
        self.cube_root_states_list = []
        self.wrist_depth_img_buf_list = []
        self.wrist_seg_img_buf_list = []
        self.action_data_history_buf_list = []
        self.states_history_buf_list = []
        self.last_table_x_lin_pose = 0
        self.camera_history_len = 3
        self.table_level = "Level01"
        self.table_level = self.cfg["env"]["D1_bench_task_level"]
        cprint(f"[ DQ_Bench ] Task_Level : {self.table_level} !", "red")
        self.random_forces_value_max = 7.2
        self.random_forces_value_min = 3.6
        self.forces_xyz_index = 1

        self.num_torques = 18
        if sim_device == "cpu":
            self.sim_id = 0
        else:
            self.sim_id = int(sim_device.split(":")[1])
            self.graph_device = 5
        
        self.debug_vis = self.cfg["env"]["enableDebugVis"]
        self.camera_mode = self.cfg["env"]["cameraMode"]
        print("camera mode is", self.camera_mode)
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]
        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.control_freq_low_init = self.cfg["env"].get("controlFrequencyLow", 7)
        self.control_freq_low = self.control_freq_low_init
        self.hold_steps = self.cfg["env"].get("holdSteps", 10)
        self.img_delay_frame = self.cfg["env"].get("imgDelayFrame", 5) # 0.08s5
        self.enable_camera = self.cfg["sensor"].get("enableCamera", False)
        self.depth_clip_lower = self.cfg["sensor"].get("depth_clip_lower", 0.15)
        self.depth_clip_rand_range = self.cfg["sensor"].get("depth_clip_rand_range", [0.18, 0.25])
        

        if (self.enable_camera or self.camera_test) and self.img_delay_frame:
            assert self.img_delay_frame <= self.control_freq_low, "control_freq_low < img_delay_frame, may have bug when get imgs"
        if self.enable_camera or self.camera_test:
            self.resize_transform = transforms.Resize((self.cfg["sensor"]["resized_resolution"][1], self.cfg["sensor"]["resized_resolution"][0]))
        if (self.enable_camera or self.camera_test) and self.cfg["env"]["numEnvs"] > 40:
            self.cfg["env"]["numEnvs"] = self.cfg["env"]["numEnvs"]
            if self.depth_random:
                self.cfg["env"]["numEnvs"] = 300
            if self.camera_mode == "seperate":
                self.cfg["env"]["numEnvs"] = 240
            print("camera_num_envs is",cfg["env"]["numEnvs"])
            
        self.robot_start_pose = robot_start_pose

        self._extra_env_settings()
        
        self._setup_obs_and_action_info()
        
        self.reward_scales = self.cfg["reward"]["scales"]
        
        self.randomize = False
        
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render, *args, **kwargs)
        
        if not self.floating_base:
            self.low_level_policy = self._load_low_level_model()
        else:
            self.num_gripper_joints = 1
        
        self._prepare_reward_function()
        
        self.dt = self.control_freq_inv * self.sim_params.dt
        
        self._init_tensors()
        self.global_step_counter = cfg["env"].get("globalStepCounter", 0)
        self.local_step_counter = 0
        
        if self.viewer is not None:
            self._init_camera()
        
        self.depth_transform = None
        if self.depth_random:
            # Random Erase + Gaussian Blur + Gaussian Noise + Random Rotation
            from torchvision.transforms import v2
            self.depth_transform = v2.Compose([
                # v2.RandomAffine(translate=(10,10)),
                v2.RandomErasing(p=0.4, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=-2, inplace=False),
                v2.RandomErasing(p=0.4, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0, inplace=False),
                v2.GaussianBlur(kernel_size=5, sigma=(0.01, 0.1)),
                AddGaussianNoise(mean=0, std=0.005),
                v2.RandomRotation(degrees=5)
            ])
            self.mask_transform = v2.Compose([
                v2.GaussianBlur(kernel_size=5, sigma=(0.01, 0.1)),
                AddGaussianNoise(mean=0, std=0.01),
                v2.RandomRotation(degrees=5)
            ])
            
    def render_record(self, mode="rgb_array"):
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        imgs = []
        for i in range(self.num_envs):
            cam = self._rendering_camera_handles[i]
            root_pos = self._robot_root_states[i, :3].cpu().numpy()
            cam_pos = root_pos + np.array([0, 2, 0.3]) # np.array([0, 2, 1])
            # cam_pos = root_pos + np.array([2, 2, 2])
            lookat_pos = root_pos # + np.array([0, 0, 1])
            self.gym.set_camera_location(cam, self.envs[i], gymapi.Vec3(*cam_pos), gymapi.Vec3(*root_pos))
            # self.gym.set_camera_location(cam, self.envs[i], gymapi.Vec3(*cam_pos), gymapi.Vec3(*lookat_pos))
            
            img = self.gym.get_camera_image(self.sim, self.envs[i], cam, gymapi.IMAGE_COLOR)
            w, h = img.shape
            imgs.append(img.reshape([w, h // 4, 4]))
        return imgs

    def _extra_env_settings(self):
        """
        Extrta settings for the environment, different from each environment.
        """
        pass
        
    def _setup_obs_and_action_info(self, removed_dim, num_action=9, num_obs=38):
        """Setup observation and action space:
            self.cfg["env"]["numObservations"] = _
            self.cfg["env"]["numActions"] = _
        """
        _num_action = num_action
        if self.floating_base:
            _num_action += 1
        if self.pitch_control:
            _num_action += 1
        _num_obs = num_obs + _num_action
        if (not self.floating_base) and self.use_roboinfo:
            _num_obs += 24
        self.cfg["env"]["img_numObservations"] = _num_obs## my add
        self.cfg["env"]["numObservations"] = _num_obs
        self.cfg["env"]["numActions"] = _num_action
        if self.enable_camera:
            # TODO: modify input image channels
            if self.camera_mode == "full" or self.camera_mode == "seperate":
                _num_states = self.cfg["sensor"]["resized_resolution"][0] * self.cfg["sensor"]["resized_resolution"][1] * self.camera_history_len * 4 + (_num_obs - removed_dim) - self.num_features
            elif self.camera_mode == "wrist_seg":
                _num_states = self.cfg["sensor"]["resized_resolution"][0] * self.cfg["sensor"]["resized_resolution"][1] * self.camera_history_len * 3 + (_num_obs - removed_dim) - self.num_features
            elif self.camera_mode == "front_only":
                _num_states = self.cfg["sensor"]["resized_resolution"][0] * self.cfg["sensor"]["resized_resolution"][1] * self.camera_history_len * 2 + (_num_obs - removed_dim) - self.num_features
            self.cfg["env"]["numStates"] = _num_states


    def _process_rigid_body_props(self, props, i):
        rng_mass = [0., 15.]
        # rng_mass = [0., 20.]
        rand_mass = np.random.uniform(rng_mass[0], rng_mass[1])
        # rand_mass = np.linspace(rng_mass[0], rng_mass[1], self.num_envs)[i]
        props[1].mass += rand_mass
        # props[1].mass += rng_mass

        rng_com_x = [-0.1, 0.1]
        rng_com_y = [-0.1, 0.1]
        rng_com_z = [-0.1, 0.1]
        rand_com = np.random.uniform([rng_com_x[0], rng_com_y[0], rng_com_z[0]], [rng_com_x[1], rng_com_y[1], rng_com_z[1]], size=(3, ))
        # rand_com = np.array([0.12,-0.12,0.12])
        # rand_com_x = np.linspace(rng_com_x[0], rng_com_x[1], self.num_envs)[i]
        # rand_com_y = np.linspace(rng_com_y[0], rng_com_y[1], self.num_envs)[i]
        # rand_com_z = np.linspace(rng_com_z[0], rng_com_z[1], self.num_envs)[i]
        # rand_com = np.array([rand_com_x, rand_com_y, rand_com_z])
        props[1].com += gymapi.Vec3(*rand_com)
        
        return props
    
    
    def _init_tensors(self):
        """Init necessary tensors and buffers for b1z1
        """
        ##### DQ add #####
        self.cat_forward_camera_RGBD_flatten_tensor = torch.zeros(self.num_envs, self.one_img_features, device=self.device)
        self.table_commands = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False) ## 1 linear , 1 angular , 1 height
        self.table_z_lower = torch.ones(self.num_envs, device=self.device)*0.2
        self.table_z_upper = torch.ones(self.num_envs, device=self.device)*0.9
        self.vel_line_z = torch.rand(self.num_envs,device=self.device) *-0.0008/7*8 + 0.0004/7*8
        self.vel_line_z_fix = torch.full((self.num_envs,), 0.0002222 / 7 * 8, device=self.device)
        self.lin_circle_select = torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.num_steps_to_change_mode = torch.zeros(self.num_envs,device=self.device) 
        self.increase_to_change_step = torch.ones(self.num_envs,device=self.device)
        self.rand_change_step = torch.randint(low=40, high=60, size=(self.num_envs,), device=self.device)
        rand_sign = torch.randint(0, 2, (self.num_envs,), device=self.device) * 2 - 1  # -1 or 1
        rand_magnitude = torch.rand(self.num_envs, device=self.device) * (0.15 - 0.0) + 0  # [0,0.15]
        rand_magnitude_level02 = torch.rand(self.num_envs, device=self.device) * (0.3 - 0.15) + 0.15  # [0.15,0.3]
        rand_magnitude_level03 = torch.rand(self.num_envs, device=self.device) * (0.212 - 0) + 0  # [0.0,0.212]
        self.vel_line_forward_y = rand_sign * rand_magnitude
        self.vel_line_forward_y_level02 = rand_sign * rand_magnitude_level02
        self.vel_line_forward_y_level03 = rand_sign * rand_magnitude_level03
        self.vel_line_forward_x = torch.rand(self.num_envs,device=self.device) *(-0.212-0.212) + 0.212
        self.vel_angular_yaw = torch.rand(self.num_envs,device=self.device) *(-0.3-0.3) + 0.3
        self.wrist_depth_img_buf = torch.zeros(self.num_envs, 6, self.one_img_features_1_ch, device=self.device)
        self.wrist_depth_img_buf_last_time = torch.zeros(self.num_envs, 6, self.one_img_features_1_ch, device=self.device)
        self.wrist_seg_img_buf = torch.zeros(self.num_envs, 6, self.one_img_features_1_ch, device=self.device)
        self.wrist_seg_img_buf_last_time = torch.zeros(self.num_envs, 6, self.one_img_features_1_ch, device=self.device)
        self.action_data_history_buf = torch.zeros(self.num_envs, 6, self.num_actions, device=self.device, dtype=torch.float)
        self.states_history_buf = torch.zeros(self.num_envs, 6, 61, device=self.device, dtype=torch.float)
        self.states_obs_buf = torch.zeros(self.num_envs, 61, device=self.device, dtype=torch.float)
        self.reset_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.one_epoch_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.episode_step_sum = torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)
        self.surpass_dist_flag = torch.ones(self.num_envs,device=self.device,dtype=torch.bool)
        ##### DQ add #####
        
        self._actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self._dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self._rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self._jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, "robot")
        self._force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        
        self._root_states = gymtorch.wrap_tensor(self._actor_root_state)
        
        #### !!!!!!!! reshape the self._actor_root_state tensor. The original shape is (actors_num of all envs,13), the later shape is (envs,actor_nums of each env,13)
        self._robot_root_states = self._root_states.view(self.num_envs, self.num_actors, self._actor_root_state.shape[-1])[..., 0, :]# 0 is the robot index of each env
        self._initial_robot_root_states = self._robot_root_states.clone()
        self._initial_robot_root_states[:, 7:13] = 0.0

        self._robot_actor_ids = self.num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        
        self._dof_state = gymtorch.wrap_tensor(self._dof_state_tensor)
        self.dof_per_env = self._dof_state.shape[0] // self.num_envs
        self._dof_pos = self._dof_state.view(self.num_envs, self.dof_per_env, 2)[..., :self.num_dofs, 0]
        self._dof_vel = self._dof_state.view(self.num_envs, self.dof_per_env, 2)[..., :self.num_dofs, 1]
        self._last_dof_vel = self._dof_vel.clone()
        
        self._initial_dof_pos = self._dof_pos.clone()
        self._initial_dof_vel = self._dof_vel.clone()

        self._rigid_body_state = gymtorch.wrap_tensor(self._rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs ## rigid_body number
        rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)[...,:,:] 
        
        self._rigid_body_pos = rigid_body_state_reshaped[..., 0:3]
        self._rigid_body_rot = rigid_body_state_reshaped[..., 3:7]
        self._rigid_body_vel = rigid_body_state_reshaped[..., 7:10]
        self._rigid_body_ang_vel = rigid_body_state_reshaped[..., 10:13]

        self.jacobian_whole = gymtorch.wrap_tensor(self._jacobian_tensor)
        self.gripper_idx = self.gym.find_actor_rigid_body_index(self.envs[0], self.robot_handles[0], "ee_gripper_link", gymapi.DOMAIN_ENV)
        self.ee_j_eef = self.jacobian_whole[:, self.gripper_idx, :6, -(6 + self.num_gripper_dof):-self.num_gripper_dof]
        self.ee_pos = rigid_body_state_reshaped[:, self.gripper_idx, 0:3]
        self.ee_orn = rigid_body_state_reshaped[:, self.gripper_idx, 3:7]
        
        self.wrist_idx = self.gym.find_actor_rigid_body_index(self.envs[0], self.robot_handles[0], "link06", gymapi.DOMAIN_ENV)
        self.flange_idx = self.gym.find_actor_rigid_body_index(self.envs[0], self.robot_handles[0], "link05", gymapi.DOMAIN_ENV)

        self.gripper_dof_pos = torch.zeros(self.num_envs, self.num_gripper_dof, device=self.device)
        self.gripper_dof_vel = torch.zeros(self.num_envs, self.num_gripper_dof, device=self.device)
        self.dof_limits_lower_tensor = torch.zeros(self.num_envs, self.num_gripper_dof, device=self.device)
        self.dof_limits_upper_tensor = torch.zeros(self.num_envs, self.num_gripper_dof, device=self.device)

        
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float32) # num_actions is 9
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float32)
        # print("num_action is ",self.num_actions)
        self.clipped_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float32)
        
        contact_force_tensor = gymtorch.wrap_tensor(self._contact_force_tensor)
        self._contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)
        
        if not self.floating_base:
            self.force_sensor_tensor = gymtorch.wrap_tensor(self._force_sensor_tensor).view(self.num_envs, -1, 6)
            self.foot_contacts_from_sensor = self.force_sensor_tensor.norm(dim=-1) > 2.0
        
        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_buf = self._terminate_buf.clone()
        
        self.last_low_actions = torch.zeros(self.num_envs, 18, device=self.device, dtype=torch.float32)
        
        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.gait_wait_timer = torch.zeros(self.num_envs, dtype=torch.int, device=self.device,
                                        requires_grad=False)
        self.is_walking = torch.zeros(self.num_envs, dtype=torch.int, device=self.device,
                                        requires_grad=False)
        
        if not self.floating_base:
            self.low_obs_history_buf = torch.zeros(self.num_envs, 10, self.num_proprio, dtype=torch.float, device=self.device, requires_grad=False)
        
        if self.rand_cmd_scale:
            self.command_scale = torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        if self.rand_depth_clip:
            self.depth_clip_lower = self.depth_clip_lower*torch.ones(self.num_envs, 1, 1, dtype=torch.float, device=self.device, requires_grad=False)
        
        # self.motor_strength = torch.ones(self.num_envs, 18, device=self.device, dtype=torch.float32)
        self.motor_strength = torch.cat([
                    torch_rand_float(0.7, 1.3, (self.num_envs, 12), device=self.device),
                    torch_rand_float(0.7, 1.3, (self.num_envs, 6), device=self.device)
                ], dim=1)
        
        if not self.floating_base:
            low_action_scale = [0.4, 0.45, 0.45] * 2 + [0.4, 0.45, 0.45] * 2 + [2.1, 0.6, 0.6, 0, 0, 0]
            self.low_action_scale = torch.tensor(low_action_scale, device=self.device)
            
            self.p_gains = torch.zeros(self.num_torques, dtype=torch.float, device=self.device, requires_grad=False)
            self.d_gains = torch.zeros(self.num_torques, dtype=torch.float, device=self.device, requires_grad=False)
            
            for i in range(self.num_torques):
                name = self.dof_names[i]
                found = False
                for dof_name in self.cfg["env"]["asset"]["control"]["stiffness"].keys():
                    if dof_name in name:
                        self.p_gains[i] = self.cfg["env"]["asset"]["control"]["stiffness"][dof_name]
                        self.d_gains[i] = self.cfg["env"]["asset"]["control"]["damping"][dof_name]
                        found = True
                    if not found:
                        self.p_gains[i] = 0.
                        self.d_gains[i] = 0.
        
        self.default_dof_pos_wo_gripper = self._initial_dof_pos[:, :-self.num_gripper_joints]
        self.dof_pos_wo_gripper = self._dof_pos[:, :-self.num_gripper_joints]
        self.dof_vel_wo_gripper = self._dof_vel[:, :-self.num_gripper_joints]
        self.dof_pos_gripper = self._dof_pos[:, -self.num_gripper_joints:]
        self.dof_vel_gripper = self._dof_vel[:, -self.num_gripper_joints:]
        self.robot_torques_zero = torch.zeros(self.num_envs, self.num_dofs_robot, device=self.device, dtype=torch.float32) ## my add
        self.gripper_torques_zero = torch.zeros(self.num_envs, self.num_gripper_joints, device=self.device, dtype=torch.float32)

        
        self.last_ee_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.curr_ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.curr_ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.ee_goal_global_cart = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.curr_ee_goal_orn_rpy = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.ee_goal_global_orn_quat = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.ee_goal_center_offset = torch.tensor([0.3, 0.0, 0.7], device=self.device).repeat(self.num_envs, 1)
        self.closest_dist = -torch.ones(self.num_envs, device=self.device, dtype=torch.float)
        self.curr_dist = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        self.action_history_buf = torch.zeros(self.num_envs, 3, self.num_actions, device=self.device, dtype=torch.float)
        self.command_history_buf = torch.zeros(self.num_envs, 3, self.num_actions, device=self.device, dtype=torch.float)
        self.episode_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) - 1
        self.success_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.success_onetime_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        self.reach_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.pick_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        self.masked_forward = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float)
        self.masked_wrist = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float)

        # --------- get camera image -------- #
        self.camera_sensor_dict = defaultdict(list)  
        if self.enable_camera or self.camera_test:
            # self.camera_history_len = 1
            if self.camera_mode == "full" or self.camera_mode == "seperate":
                num_channels = 4
            elif self.camera_mode == "wrist_seg":
                num_channels = 3
            elif self.camera_mode == "front_only":
                num_channels = 2
            self.camera_history_buf = torch.zeros(self.num_envs, self.camera_history_len, self.cfg["sensor"]["resized_resolution"][0] * self.cfg["sensor"]["resized_resolution"][1] * num_channels, device=self.device, dtype=torch.float) # TODO: modify input image channels
            for env_i, env_handle in enumerate(self.envs):
                self.camera_sensor_dict["forward_depth"].append(gymtorch.wrap_tensor(
                    self.gym.get_camera_image_gpu_tensor(
                        self.sim,
                        env_handle,
                        self.camera_handles[env_i][0], ## The head camera's index is 0, and the wire's index is 1.
                        gymapi.IMAGE_DEPTH,
                )))
                self.camera_sensor_dict["forward_seg"].append(gymtorch.wrap_tensor(
                    self.gym.get_camera_image_gpu_tensor(
                        self.sim,
                        env_handle,
                        self.camera_handles[env_i][0], ## The head camera's index is 0, and the wire's index is 1.
                        gymapi.IMAGE_SEGMENTATION,
                )))
                self.camera_sensor_dict["forward_color"].append(gymtorch.wrap_tensor(
                    self.gym.get_camera_image_gpu_tensor(
                        self.sim,
                        env_handle,
                        self.camera_handles[env_i][0], ## The head camera's index is 0, and the wire's index is 1.
                        gymapi.IMAGE_COLOR,
                )))
                self.camera_sensor_dict["wrist_depth"].append(gymtorch.wrap_tensor(
                    self.gym.get_camera_image_gpu_tensor(
                        self.sim,
                        env_handle,
                        self.camera_handles[env_i][1], ## The head camera's index is 0, and the wire's index is 1.
                        gymapi.IMAGE_DEPTH,
                )))
                self.camera_sensor_dict["wrist_seg"].append(gymtorch.wrap_tensor(
                    self.gym.get_camera_image_gpu_tensor(
                        self.sim,
                        env_handle,
                        self.camera_handles[env_i][1], ## The head camera's index is 0, and the wire's index is 1.
                        gymapi.IMAGE_SEGMENTATION,
                )))
                self.camera_sensor_dict["wirst_color"].append(gymtorch.wrap_tensor(
                    self.gym.get_camera_image_gpu_tensor(
                        self.sim,
                        env_handle,
                        self.camera_handles[env_i][1], ## The head camera's index is 0, and the wire's index is 1.
                        gymapi.IMAGE_COLOR,
                )))
            if self.camera_test==True:
                self.fix_camera_sensor_dict = defaultdict(list) # depth & seg 的维度都是二维(H,W),这是像素
                for env_i, env_handle in enumerate(self.envs):
                    self.fix_camera_sensor_dict["cube_depth"].append(gymtorch.wrap_tensor(
                        self.gym.get_camera_image_gpu_tensor(
                            self.sim,
                            env_handle,
                            self.fix_camera_handles[env_i], ## The head camera's index is 0, and the wire's index is 1.
                            gymapi.IMAGE_DEPTH,
                    )))
                    self.fix_camera_sensor_dict["cube_seg"].append(gymtorch.wrap_tensor(
                        self.gym.get_camera_image_gpu_tensor(
                            self.sim,
                            env_handle,
                            self.fix_camera_handles[env_i], ## The head camera's index is 0, and the wire's index is 1.
                            gymapi.IMAGE_SEGMENTATION,
                    )))
                    self.fix_camera_sensor_dict["cube_color"].append(gymtorch.wrap_tensor(
                        self.gym.get_camera_image_gpu_tensor(
                            self.sim,
                            env_handle,
                            self.fix_camera_handles[env_i], ## The head camera's index is 0, and the wire's index is 1.
                            gymapi.IMAGE_COLOR,
                    )))
        
    def _init_camera(self):
        cam_pos = gymapi.Vec3(10, 3, 0.4)
        cam_target = gymapi.Vec3(10, 0, 0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return
    
    def create_sim(self):
        self.up_axis_idx = 2 # Y=1, Z=2;
        self.sim = super().create_sim(self.sim_id, self.sim_id, self.physics_engine, self.sim_params)
        #### create the terrain in high-level ####
        self.terrain = Terrain(self.cfg_terrain.terrain, )
        self._create_trimesh()
        #### create the terrain in high-level ####

        self._create_grond_plane()
        self._create_envs()
        cube_count = self.gym.get_actor_rigid_body_index(self.envs[0], self.cube_handles[0], 0, gymapi.DOMAIN_SIM)
        print(f"rigid_body count is ",cube_count)    #280 when the num_envs = 10, each env has 28 rigid_body count

        ###################################################
        self.gym.prepare_sim(self.sim) # my add
        
        if self.randomize:
            self.apply_randomizations(self.randomization_params)
        return
    
    def _create_grond_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        # plane_params.static_friction = self.plane_static_friction
        # plane_params.dynamic_friction = self.plane_dynamic_friction
        # plane_params.restitution = self.plane_restitution
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)
        return
    
    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
            Very slow when horizontal_scale is small
        """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = 1
        tm_params.dynamic_friction = 1
        tm_params.restitution = 0
        print("Adding trimesh to simulation...")
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)  
        print("Trimesh added")
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
    
    def _create_onboard_cameras(self, env_handle, actor_handle, i):
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.height = self.cfg["sensor"]["onboard_camera"]["resolution"][1]
        camera_props.width = self.cfg["sensor"]["onboard_camera"]["resolution"][0]
        # if hasattr(getattr(self.cfg.sensor, sensor_name), "horizontal_fov"):
        if self.cfg["sensor"]["onboard_camera"].get("horizontal_fov", None) is not None:
            camera_props.horizontal_fov = np.random.uniform(
                self.cfg["sensor"]["onboard_camera"]["horizontal_fov"][0],
                self.cfg["sensor"]["onboard_camera"]["horizontal_fov"][1]
            ) if isinstance(self.cfg["sensor"]["onboard_camera"]["horizontal_fov"], (list, tuple)) else self.cfg["sensor"]["onboard_camera"]["horizontal_fov"]
            # camera_props.horizontal_fov = np.linspace(
            #     self.cfg["sensor"]["onboard_camera"]["horizontal_fov"][0],
            #     self.cfg["sensor"]["onboard_camera"]["horizontal_fov"][1], self.num_envs
            # )[i] if isinstance(self.cfg["sensor"]["onboard_camera"]["horizontal_fov"], (list, tuple)) else self.cfg["sensor"]["onboard_camera"]["horizontal_fov"]
        camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
        local_transform = gymapi.Transform()
        local_pos = self.cfg["sensor"]["onboard_camera"]["position"].copy()
        local_pos[0] += np.random.uniform(-0.01, 0.01)
        local_pos[1] += np.random.uniform(-0.01, 0.01)
        local_pos[2] += np.random.uniform(-0.01, 0.01)
        # local_pos[0] += np.linspace(-0.01,0.01,self.num_envs)[i]
        # local_pos[1] += np.linspace(-0.01,0.01,self.num_envs)[i]
        # local_pos[2] += np.linspace(-0.01,0.01,self.num_envs)[i]
        local_transform.p = gymapi.Vec3(*local_pos)
        local_rot = self.cfg["sensor"]["onboard_camera"]["rotation"].copy()
        local_rot[1] += np.random.uniform(-0.087, 0.087)
        # local_rot[1] += np.linspace(-0.087, 0.087, self.num_envs)[i]
        local_transform.r = gymapi.Quat.from_euler_zyx(*local_rot)
        self.gym.attach_camera_to_body(
            camera_handle,
            env_handle,
            actor_handle,
            local_transform,
            gymapi.FOLLOW_TRANSFORM,
        )
        return camera_handle
    
    def _create_wrist_cameras(self, env_handle, actor_handle, i):
        wrist_handle = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, "link06")
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.height = self.cfg["sensor"]["wrist_camera"]["resolution"][1]
        camera_props.width = self.cfg["sensor"]["wrist_camera"]["resolution"][0]
        if self.cfg["sensor"]["wrist_camera"].get("horizontal_fov", None) is not None:
            camera_props.horizontal_fov = np.random.uniform(
                self.cfg["sensor"]["wrist_camera"]["horizontal_fov"][0],
                self.cfg["sensor"]["wrist_camera"]["horizontal_fov"][1]
            ) if isinstance(self.cfg["sensor"]["wrist_camera"]["horizontal_fov"], (list, tuple)) else self.cfg["sensor"]["wrist_camera"]["horizontal_fov"]
            # camera_props.horizontal_fov = np.linspace(
            #     self.cfg["sensor"]["wrist_camera"]["horizontal_fov"][0],
            #     self.cfg["sensor"]["wrist_camera"]["horizontal_fov"][1], self.num_envs
            # )[i] if isinstance(self.cfg["sensor"]["wrist_camera"]["horizontal_fov"], (list, tuple)) else self.cfg["sensor"]["wrist_camera"]["horizontal_fov"]
        camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
        local_pos = self.cfg["sensor"]["wrist_camera"]["position"].copy() # [0.0955, 0.22, -0.03175]
        wrist_cam_randomize = self.cfg["sensor"]["wrist_camera"].get("rand_position", [0.01, 0.01, 0.01]) 
        local_pos[0] += np.random.uniform(-wrist_cam_randomize[0], wrist_cam_randomize[0])
        local_pos[1] += np.random.uniform(-wrist_cam_randomize[1], wrist_cam_randomize[1])
        local_pos[2] += np.random.uniform(-wrist_cam_randomize[2], wrist_cam_randomize[2])
        # local_pos[0] += np.linspace(-wrist_cam_randomize[0], wrist_cam_randomize[0], self.num_envs)[i]
        # local_pos[1] += np.linspace(-wrist_cam_randomize[1], wrist_cam_randomize[1], self.num_envs)[i]
        # local_pos[2] += np.linspace(-wrist_cam_randomize[2], wrist_cam_randomize[2], self.num_envs)[i]
        local_rot = self.cfg["sensor"]["wrist_camera"]["rotation"].copy()
        local_rot[2] += np.random.uniform(-0.087, 0.087) ## [-1.57, 0.0, -0.87]
        # local_rot[2] += np.linspace(-0.087, 0.087, self.num_envs)[i]
        # local_transform.r = gymapi.Quat.from_euler_zyx(*self.cfg["sensor"]["onboard_camera"]["rotation"])
        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(*local_pos)
        local_transform.r = gymapi.Quat.from_euler_zyx(*local_rot)
        self.gym.attach_camera_to_body(
            camera_handle,
            env_handle,
            wrist_handle,
            local_transform,
            gymapi.FOLLOW_TRANSFORM,
        )
        return camera_handle
    
    def _create_fix_cameras(self, env_handle,table_handle,i):
        car_body_handle = self.gym.find_actor_rigid_body_handle(env_handle, table_handle, "main_body")
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.height = self.cfg["sensor"]["fix_camera"]["resolution"][1]
        camera_props.width = self.cfg["sensor"]["fix_camera"]["resolution"][0]
        if self.cfg["sensor"]["fix_camera"].get("horizontal_fov", None) is not None:
            camera_props.horizontal_fov = np.random.uniform(
                self.cfg["sensor"]["fix_camera"]["horizontal_fov"][0],
                self.cfg["sensor"]["fix_camera"]["horizontal_fov"][1]
            ) if isinstance(self.cfg["sensor"]["fix_camera"]["horizontal_fov"], (list, tuple)) else self.cfg["sensor"]["fix_camera"]["horizontal_fov"]
        camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
        # refer_x=
        # refer_y=self._cube_root_states[env_handle,1]
        # refer_z=self._cube_root_states[env_handle,2]
        local_pos = self.cfg["sensor"]["fix_camera"]["position"].copy() # [0.0955, 0.22, -0.03175]
        wrist_cam_randomize = self.cfg["sensor"]["fix_camera"].get("rand_position", [0.01, 0.01, 0.01]) 
        # local_pos[0] += refer_x
        # local_pos[1] += refer_y
        # local_pos[2] += refer_z
        local_pos[0] += np.random.uniform(-wrist_cam_randomize[0], wrist_cam_randomize[0])
        local_pos[1] += np.random.uniform(-wrist_cam_randomize[1], wrist_cam_randomize[1])
        local_pos[2] += np.random.uniform(-wrist_cam_randomize[2], wrist_cam_randomize[2])
        # local_pos[0] += np.linspace(-wrist_cam_randomize[0], wrist_cam_randomize[0], self.num_envs)[i]
        # local_pos[1] += np.linspace(-wrist_cam_randomize[1], wrist_cam_randomize[1], self.num_envs)[i]
        # local_pos[2] += np.linspace(-wrist_cam_randomize[2], wrist_cam_randomize[2], self.num_envs)[i]
        local_rot = self.cfg["sensor"]["fix_camera"]["rotation"].copy()
        local_rot[2] += np.random.uniform(-0.087, 0.087) ## [-1.57, 0.0, -0.87]
        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(*local_pos)
        local_transform.r = gymapi.Quat.from_euler_zyx(*local_rot)
        # self.gym.set_camera_transform(camera_handle, env_handle, local_transform)
        self.gym.attach_camera_to_body(
            camera_handle,
            env_handle,
            car_body_handle,
            local_transform,
            gymapi.FOLLOW_TRANSFORM,
        )
        return camera_handle
    
    # def draw_link06(self, i):
    #     axes_geom = gymutil.AxesGeometry(scale=0.2)
    #     gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], self._rigid_body_state[:, self.wrist_idx, :3])
        
    def _set_default_joint_angles_dict(self):
        if self.floating_base:
            default_joint_angles = { # = target angles [rad] when action = 0.0
                'z1_waist': 0.0,
                'z1_shoulder': 1.48,
                'z1_elbow': -0.63,
                'z1_wrist_angle': -0.84,
                'z1_forearm_roll': 0.0,
                'z1_wrist_rotate': 1.57, #1.57,
                'z1_jointGripper': -1.57,#-0.785,
            }
        else:
            default_joint_angles = { # = target angles [rad] when action = 0.0
                ####### b1z1 #######
                'FL_hip_joint': 0.2,   # [rad]
                'FL_thigh_joint': 0.8,     # [rad]
                'FL_calf_joint': -1.5,   # [rad]

                'RL_hip_joint': 0.2,   # [rad]
                'RL_thigh_joint': 0.8,   # [rad]
                'RL_calf_joint': -1.5,    # [rad]

                'FR_hip_joint': -0.2 ,  # [rad]
                'FR_thigh_joint': 0.8,     # [rad]
                'FR_calf_joint': -1.5,  # [rad]

                'RR_hip_joint': -0.2,   # [rad]
                'RR_thigh_joint': 0.8,   # [rad]
                'RR_calf_joint': -1.5,    # [rad]

                'z1_waist': 0.0,
                'z1_shoulder': 1.48,
                'z1_elbow': -1.5, # -0.63,
                'z1_wrist_angle': 0, # -0.84,
                'z1_forearm_roll': 0.0,
                'z1_wrist_rotate': 1.57, # 0.0,
                'z1_jointGripper': -1.57, # -0.785,
                ####### b1z1 #######

                ####### b2piper #######
                # 'FL_hip_joint': 0.2,   # [rad]
                # 'FL_thigh_joint': 0.8,     # [rad]
                # 'FL_calf_joint': -1.5,   # [rad]

                # 'RL_hip_joint': 0.2,   # [rad]
                # 'RL_thigh_joint': 0.8,   # [rad]
                # 'RL_calf_joint': -1.5,    # [rad]

                # 'FR_hip_joint': -0.2 ,  # [rad]
                # 'FR_thigh_joint': 0.8,     # [rad]
                # 'FR_calf_joint': -1.5,  # [rad]

                # 'RR_hip_joint': -0.2,   # [rad]
                # 'RR_thigh_joint': 0.8,   # [rad]
                # 'RR_calf_joint': -1.5,    # [rad]

                # 'piper_arm00': 0.0 ,
                # 'piper_arm01': 1.6 ,
                # 'piper_arm02': -1.5 ,
                # 'piper_arm03': 0.0 ,
                # 'piper_arm04': 0.0 ,
                # 'piper_arm05': 0.0 ,
                # 'piper_arm06': 0.0 ,
                # 'piper_arm07': 0.0 ,
                ####### b2piper #######
            }
        return default_joint_angles
    
    def _create_envs(self): # load the b1z1's model, don't relate to the cube's envs
        """Create environments, including load assets and
        create actors.
        """
        
        spacing = self.cfg["env"]["envSpacing"]
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        
        num_per_row = int(np.sqrt(self.num_envs))
        # assetRoot: "data/asset"
        # assetFileRobot: "b1z1-col/urdf/b1z1.urdf"
        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file_robot = self.cfg["env"]["asset"]["assetFileRobot"]
        
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = 3
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.density = 1000.0
        asset_options.angular_damping = 0.
        asset_options.linear_damping = 0.
        asset_options.max_angular_velocity = 1000.
        asset_options.max_linear_velocity = 1000.
        asset_options.armature = 0.
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False
        asset_options.use_mesh_materials = True
        # asset_options.vhacd_enabled = True
        # asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file_robot, asset_options)
        self.num_dofs_robot = self.gym.get_asset_dof_count(robot_asset)  ## my change ##
        self.num_dofs = self.num_dofs_robot 
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        self.dof_limits_lower, self.dof_limits_upper, self.torque_limits = [], [], []
        for i in range(self.num_dofs_robot):
            self.dof_limits_lower.append(dof_props_asset['lower'][i])
            self.dof_limits_upper.append(dof_props_asset['upper'][i])
            # self.dof_limits_upper.append(-0.15)
            self.torque_limits.append(dof_props_asset['effort'][i])
            
        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)
        self.torque_limits = to_torch(self.torque_limits, device=self.device)
        
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.body_names_to_idx = self.gym.get_asset_rigid_body_dict(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.dof_names_to_idx = self.gym.get_asset_dof_dict(robot_asset)
        
        for i in range(len(rigid_shape_props_asset)):
            rigid_shape_props_asset[i].friction = 2.0  ## domain change change original friction

        self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props_asset)

        # default dof
        default_joint_angles = self._set_default_joint_angles_dict()
        robot_dof_dict = self.gym.get_asset_dof_dict(robot_asset)
        initial_pos = np.zeros(self.num_dofs_robot, dtype=np.float32) 
        for k, v in default_joint_angles.items():
            initial_pos[robot_dof_dict[k]] = v
        initial_dof_state = np.zeros(self.num_dofs_robot, dtype=gymapi.DofState.dtype) 
        initial_dof_state['pos'] = initial_pos
        self.initial_robo_pos = to_torch(initial_pos, device=self.device)
        
        robot_body_dict = self.gym.get_asset_rigid_body_dict(robot_asset)
        robot_body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        feet_names = [s for s in robot_body_names if "foot" in s]
        self.sensor_indices = []
        for name in feet_names:
            foot_idx = robot_body_dict[name]
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, -0.05))
            sensor_idx = self.gym.create_asset_force_sensor(robot_asset, foot_idx, sensor_pose)
            self.sensor_indices.append(sensor_idx)

        self.robot_handles = []
        self.camera_handles = []
        self.fix_camera_handles = []
        self.envs = []
        
        friction_range = [0.5, 3.0] 
        num_buckets = 128
        friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
        
        # arm_kp_range = np.linspace(300,400, self.num_envs)
        # gripper_kp_range = np.linspace(1.5,3, self.num_envs)
        
        for i in range(self.num_envs): 
            arm_kp = 400 # np.random.uniform(300,400) # domain change ORI
            # arm_kp = np.random.uniform(350,450) # domain change 
            gripper_kp = np.random.uniform(2,5) # domain change ORI
            # gripper_kp = np.random.uniform(1,8) # domain change 
            arm_kd = 40.0 # domain change ORI
            # arm_kd = np.random.uniform(30,50)
            gripper_kd = 2.5 # domain change ORI
            # gripper_kd = np.random.uniform(2,4)
            
            if self.floating_base:
                dof_props_asset['driveMode'][:].fill(gymapi.DOF_MODE_POS)  # set arm to pos control
                dof_props_asset['stiffness'][:-self.num_gripper_dof].fill(arm_kp)
                dof_props_asset['damping'][:-self.num_gripper_dof].fill(40.0)
                # TODO: find proper gripper kp and kd   
                dof_props_asset['stiffness'][-self.num_gripper_dof:].fill(gripper_kp)
                dof_props_asset['damping'][-self.num_gripper_dof:].fill(2.5)
            else:
                dof_props_asset['driveMode'][12:].fill(gymapi.DOF_MODE_POS)  # set arm to pos control
                dof_props_asset['stiffness'][12:-self.num_gripper_dof].fill(arm_kp)
                dof_props_asset['damping'][12:-self.num_gripper_dof].fill(arm_kd)
                # TODO: find proper gripper kp and kd   
                dof_props_asset['stiffness'][-self.num_gripper_dof:].fill(gripper_kp)
                dof_props_asset['damping'][-self.num_gripper_dof:].fill(gripper_kd)
        
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env_ptr)
            col_group = i
            col_filter = 0

            for k in range(len(rigid_shape_props_asset)):
                rigid_shape_props_asset[k].friction = friction_buckets[np.random.randint(0, num_buckets)]

            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props_asset)
            robot_start_pose = gymapi.Transform()
            # robot_start_pose.p = gymapi.Vec3(*self.robot_start_pose) # gymapi.Vec3(-1.55, 0, 0.66) # 0.95 - 1.35
            robot_start_pose.p = gymapi.Vec3(-2.0,2.0,0.66) # gymapi.Vec3(-1.55, 0, 0.66) # 0.95 - 1.35
            robot_start_pose.r = gymapi.Quat(0, 0, 0, 1)
            robot_handle = self.gym.create_actor(env_ptr, robot_asset, robot_start_pose, "robot", col_group, col_filter, 0)
            self.robot_handles.append(robot_handle)

            # self.initial_roll_x, self.initial_pitch_y, self.initial_yaw_z = euler_from_quat(robot_start_pose.r)
            quat_x = robot_start_pose.r.x
            quat_y = robot_start_pose.r.y
            quat_z = robot_start_pose.r.z
            quat_w = robot_start_pose.r.w
            quat_tensor = torch.tensor([quat_x, quat_y, quat_z, quat_w], device=self.device).unsqueeze(0)
            self.initial_roll_x, self.initial_pitch_y, self.initial_yaw_z = euler_from_quat(quat_tensor)
            
            body_props = self.gym.get_actor_rigid_body_properties(env_ptr, robot_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_ptr, robot_handle, body_props, recomputeInertia=True)
            self.gym.set_actor_dof_properties(env_ptr, robot_handle, dof_props_asset)
            self.gym.set_actor_dof_states(env_ptr, robot_handle, initial_dof_state, gymapi.STATE_ALL)

            self._create_extra(i)

            if self.enable_camera or self.camera_test:
                camera_handle = self._create_onboard_cameras(env_ptr, robot_handle, i)
                wrist_camera_handle = self._create_wrist_cameras(env_ptr, robot_handle, i)
                self.camera_handles.append([camera_handle, wrist_camera_handle])

                
        if self.cfg.get("record_video", False):
            camera_props = gymapi.CameraProperties()
            camera_props.width = 720
            camera_props.height = 480
            self._rendering_camera_handles = []
            for i in range(self.num_envs):
                # root_pos = self.root_states[i, :3].cpu().numpy()
                # cam_pos = root_pos + np.array([0, 1, 0.5])
                cam_pos = np.array([0, 1, 0.5])
                camera_handle = self.gym.create_camera_sensor(self.envs[i], camera_props)
                self._rendering_camera_handles.append(camera_handle)
                self.gym.set_camera_location(camera_handle, self.envs[i], gymapi.Vec3(*cam_pos), gymapi.Vec3(*0*cam_pos))

    def _create_extra(self, env_i):
        """Create extra objects for each environment.
        """
        pass
    
    def _load_low_level_model(self, num_priv=5 + 1 + 12, stochastic=False):
        low_level_kwargs = {
            "continue_from_last_std": True,
            "init_std": [[0.8, 1.0, 1.0] * 4 + [1.0] * 6],
            "actor_hidden_dims": [128],
            "critic_hidden_dims": [128],
            "activation": 'elu', # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
            "output_tanh": False,
            "leg_control_head_hidden_dims": [128, 128],
            "arm_control_head_hidden_dims": [128, 128],
            "priv_encoder_dims": [64, 20],
            "num_leg_actions": 12,
            "num_arm_actions": 6,
            "adaptive_arm_gains": False,
            "adaptive_arm_gains_scale": 10.0
        }
        num_actions = 18
        self.num_priv = num_priv
        self.num_gripper_joints = self.num_gripper_dof
        self.num_proprio = 2 + 3 + 18 + 18 + 12 + 4 + 3 + 3 + 3
        self.num_proprio += 5 if self.observe_gait_commands else 0
        self.history_len = 10
        low_actor_critic: ActorCritic = ActorCritic(self.num_proprio,
                                                    self.num_proprio,
                                                    num_actions,
                                                    **low_level_kwargs,
                                                    num_priv=self.num_priv,
                                                    num_hist=self.history_len,
                                                    num_prop=self.num_proprio,
                                                    )
        policy_path = self.cfg["env"]["low_policy_path"]
        loaded_dict = torch.load(policy_path, map_location=self.device)
        low_actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        low_actor_critic = low_actor_critic.to(self.device)
        low_actor_critic.eval()
        print("Low level pretrained policy loaded!")
        if not stochastic:
            return low_actor_critic.act_inference
        else:
            return low_actor_critic.act
        
    def reset(self):
        """override the vec env reset
        """
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()  # find the env_ids's index whose value is not 0
                
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        return super().reset()
    
    def reset_idx(self, env_ids=None):
        if env_ids is None:
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        self._reset_envs(env_ids)
        return
    
    def _reset_actors(self, env_ids):
        self._robot_root_states[env_ids] = self._initial_robot_root_states[env_ids]
        # self._robot_root_states[env_ids, :2] += torch_rand_float(-0.2, 0.2, (len(env_ids), 2), device=self.device) # small randomization
        self._robot_root_states[env_ids, :2] += torch_rand_float(-0., 0., (len(env_ids), 2), device=self.device) # small randomization
        self._robot_root_states[env_ids, 2] = 0.60
        # print("self._robot_root_states[env_ids, :2] is",self._robot_root_states[env_ids, :2])
        # self._robot_root_states[env_ids, 0] += torch_rand_float(-0.2, 0.2, (len(env_ids), 1), device=self.device) # small randomization ############ my change ##########
        # self._robot_root_states[env_ids, 1] += torch_rand_float(-3.0, 0.0, (len(env_ids), 1), device=self.device) # small randomization ############ my change ##########
        # self._robot_root_states[env_ids, 1] += torch.rand(len(env_ids),device=self.device) *(0.2 - (-0.2)) + (-0.2)
        # self._robot_root_states[env_ids, 0] += torch.rand(len(env_ids),device=self.device) *(0.0 - (-4.0)) + (-4.0)

        rand_yaw_robot = torch_rand_float(-0.8, 0.8, (len(env_ids), 1), device=self.device).squeeze(1)
        if self.enable_camera:
            self._robot_root_states[env_ids, 3:7] = quat_from_euler_xyz(0*rand_yaw_robot, 0*rand_yaw_robot, 0.2*rand_yaw_robot)
        else:
            # self._robot_root_states[env_ids, 3:7] = quat_from_euler_xyz(0*rand_yaw_robot, 0*rand_yaw_robot, rand_yaw_robot)
            self._robot_root_states[env_ids, 3:7] = quat_from_euler_xyz(0*rand_yaw_robot, 0*rand_yaw_robot, 0.2*rand_yaw_robot) # ############ my change ###############

        # self._robot_root_states[env_ids, 3:7] = quat_from_euler_xyz(0*rand_yaw_robot, 0*rand_yaw_robot, 0*rand_yaw_robot)
        # self._dof_pos[env_ids] = self._initial_dof_pos[env_ids] * torch_rand_float(0.8, 1.2, (len(env_ids), 1), device=self.device)
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        
        # Randomize arm joint
        self.dof_pos_gripper[env_ids] += torch_rand_float(-0.5, 0.5, (len(env_ids), self.num_gripper_joints), device=self.device) # small randomization

        # Get current ee pos local
        self.update_roboinfo()
        self.last_ee_pos = quat_rotate_inverse(self._robot_root_states[:, 3:7], self.ee_pos - self.arm_base)

        rand_sign = torch.randint(0, 2, (len(env_ids),), device=self.device) * 2 - 1  
        rand_magnitude = torch.rand(len(env_ids), device=self.device) * (0.15 - 0.0) + 0.0  
        rand_magnitude_level02 = torch.rand(len(env_ids), device=self.device) * (0.3 - 0.15) + 0.15  
        rand_magnitude_level03 = torch.rand(len(env_ids), device=self.device) * (0.212 - 0.0) + 0.0
        self.vel_line_forward_y[env_ids] = rand_sign * rand_magnitude
        self.vel_line_forward_y_level02[env_ids] = rand_sign * rand_magnitude_level02
        self.vel_line_forward_y_level03[env_ids] = rand_sign * rand_magnitude_level03

        # self.vel_line_forward_y[env_ids] = torch.rand(len(env_ids),device=self.device) *(-0.2-0.2) + 0.2
        self.vel_line_forward_x[env_ids] = torch.rand(len(env_ids),device=self.device) *(-0.212-0.212) + 0.212
        self.vel_angular_yaw[env_ids] = torch.rand(len(env_ids),device=self.device) *(-0.3-0.3) + 0.3
        self.vel_line_z[env_ids] = torch.rand(len(env_ids),device=self.device) *-0.0008/7*8 + 0.0004/7*8
        self.vel_line_z_fix[env_ids] = torch.full((len(env_ids),), 0.0002222 / 7 * 8, device=self.device)
        self.lin_circle_select[env_ids] = (torch.rand(len(env_ids), device=self.device) > 0.5)
        self.surpass_dist_flag[env_ids] = torch.ones(len(env_ids),device=self.device,dtype=torch.bool)
        self.env_ids_test = env_ids
            


    def _reset_env_tensors(self, env_ids):
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        self.num_steps_to_change_mode[env_ids] = 0 
        self.increase_to_change_step[env_ids] = 1 
        
        self.last_actions[env_ids, :] = 0
        self.last_low_actions[env_ids, :] = 0
        self.clipped_actions[env_ids, :] = 0
        self.commands[env_ids, :] = 0
        self.action_history_buf[env_ids, :, :] = 0
        self.command_history_buf[env_ids, :, :] = 0

        self.curr_dist[env_ids] = 0.
        self.closest_dist[env_ids] = -1.
        
        self.reach_counter[env_ids] = 0
        self.pick_counter[env_ids] = 0
        
        if self.rand_cmd_scale:
            self.command_scale[env_ids] = torch_rand_float(0.5, 1.0, (len(env_ids),1), device=self.device).squeeze()

    def _refresh_sim_tensors(self):
        self._last_dof_vel = self._dof_vel.clone()
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        
        if not self.floating_base:
            self.foot_contacts_from_sensor = self.force_sensor_tensor.norm(dim=-1) > 2.0

        self._update_base_yaw_quat()
    
    def _update_base_yaw_quat(self):
        base_yaw = euler_from_quat(self._robot_root_states[:, 3:7])[2] ## take 2's dim , just robot_base yaw
        self.base_yaw_euler = torch.cat([torch.zeros(self.num_envs, 2, device=self.device), base_yaw.view(-1, 1)], dim=1) ## rpy = (0,0,base_yaw)
        self.base_yaw_quat = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), base_yaw) ## rpy = (0,0,base_yaw) --> quat(4 num)
    
    def _reset_envs(self, env_ids):
        if len(env_ids) > 0:
            self.episode_counter[env_ids] += 1
            self._reset_actors(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)
            
            if self.local_step_counter == 0:
                self.curr_ee_goal_orn_rpy[:, :] = torch.tensor([np.pi/2, 0., 0.], device=self.device)
                self.curr_ee_goal_cart[:] = torch.tensor([0.46, 0.0, 0.66], device=self.device).repeat(self.num_envs, 1)
                self.init_ee_goal_cart = self.curr_ee_goal_cart.clone()
            else:
                self.curr_ee_goal_cart[env_ids, :] = self.init_ee_goal_cart[env_ids, :]
                self.curr_ee_goal_orn_rpy[env_ids, :] = torch.tensor([np.pi/2, 0., 0.], device=self.device)
            # Randomize env
            if self.randomize:
                self.apply_randomizations(self.randomization_params, env_ids)

            ## my add to collect cube image dataset for the grasp predict net ##
            # self.collect_dataset_flag+=1
            # print("self.step_forward_record_num is  ",self.step_forward_record_num)
            # self.reset_step[env_ids] = 1
            
        return
    
    def _compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        obs = self._compute_robot_obs(env_ids)

        return obs

    def _compute_robot_obs(self, env_ids=None):
        """Compute robot observations.
        """
        raise NotImplementedError
    
    def _compute_low_level_observations(self):
        self._step_contact_targets()
        base_ang_vel = quat_rotate_inverse(self._robot_root_states[:, 3:7], self._robot_root_states[:, 10:13])
        commands = self.commands.clone()
        if self.rand_cmd_scale:
            commands[:, 0] *= self.command_scale

        low_level_obs_buf = torch.cat((self.get_body_orientation(), # dim 2
                                       base_ang_vel, # dim 3
                                       reindex_all(self._dof_pos - self._initial_dof_pos)[:, :-self.num_gripper_joints], # dim 19 or 20
                                       reindex_all(self._dof_vel * 0.05)[:, :-self.num_gripper_joints], # dim 19 or 20
                                       reindex_all(self.last_low_actions)[:, :12],
                                       reindex_feet(self.foot_contacts_from_sensor),
                                       commands[:, :3],
                                    #    self.curr_ee_goal_sphere,
                                       self.curr_ee_goal_cart,
                                       0*self.curr_ee_goal_cart,
                                       ), dim=-1)
        if self.mask_arm:
            arm_pos_obs = reindex_all(self._dof_pos - self._initial_dof_pos)[:, :-self.num_gripper_joints]
            arm_pos_obs[:, 12:] = 0
            arm_vel_obs = reindex_all(self._dof_vel * 0.05)[:, :-self.num_gripper_joints]
            arm_vel_obs[:, 12:] = 0
            low_level_obs_buf = torch.cat((self.get_body_orientation(), # dim 2
                                       base_ang_vel, # dim 3
                                       arm_pos_obs, # dim 19 or 20
                                       arm_vel_obs, # dim 19 or 20
                                       reindex_all(self.last_low_actions)[:, :12],
                                       reindex_feet(self.foot_contacts_from_sensor),
                                       commands[:, :3],
                                    #    self.curr_ee_goal_sphere,
                                       torch.tensor([0.46, 0.0, 0.31], device=self.device).repeat(self.num_envs, 1),
                                       0*self.curr_ee_goal_cart,
                                       ), dim=-1)
        if self.observe_gait_commands:
            low_level_obs_buf = torch.cat((low_level_obs_buf,
                                      self.gait_indices.unsqueeze(1), self.clock_inputs), dim=-1)
        
        self.low_obs_history_buf = torch.where(
            (self.progress_buf < 1)[:, None, None],
            torch.stack([low_level_obs_buf] * 10, dim=1),
            self.low_obs_history_buf
        )
        
        self.low_obs_buf = torch.cat([low_level_obs_buf, self.low_obs_history_buf.view(self.num_envs, -1)], dim=-1)
        
        self.low_obs_history_buf = torch.cat([
            self.low_obs_history_buf[:, 1:],
            low_level_obs_buf.unsqueeze(1)
        ], dim=1)
    
    def get_body_orientation(self, return_yaw=False):
        r, p, y = euler_from_quat(self._robot_root_states[:, 3:7])
        body_angles = torch.stack([r, p, y], dim=-1)
        if not return_yaw:
            return body_angles[:, :-1]
        else:
            return body_angles
        
    def get_ee_goal_spherical_center(self):
        center = torch.cat([self._robot_root_states[:, :2], torch.zeros(self.num_envs, 1, device=self.device)], dim=1)
        center = center + quat_apply(self.base_yaw_quat, self.ee_goal_center_offset)
        return center

    def _compute_torques(self, actions):
        actions_scaled = actions * self.motor_strength * self.low_action_scale

        default_torques = self.p_gains * (actions_scaled + self.default_dof_pos_wo_gripper - self.dof_pos_wo_gripper) - self.d_gains * self.dof_vel_wo_gripper
        default_torques[:, -6:] = 0
        torques = torch.cat([default_torques, self.gripper_torques_zero], dim=-1)
        
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
        
    def control_ik(self, dpose):
        j_eef_T = torch.transpose(self.ee_j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (0.05 ** 2)
        A = torch.bmm(self.ee_j_eef, j_eef_T) + lmbda[None, ...]
        u = torch.bmm(j_eef_T, torch.linalg.solve(A, dpose))
        return u.squeeze(-1)
    
    def _draw_camera_sensors(self):
        self.gym.clear_lines(self.viewer)
        sphere_geom = gymutil.WireframeSphereGeometry(0.03, 4, 4, None, color=(0.329, 0.831, 0.29))
        relative_camera_pos = to_torch(self.cfg["sensor"]["onboard_camera"]["position"], device=self.device).repeat(self.num_envs, 1)
        camera_pos = self._robot_root_states[:, :3] + quat_apply(self._robot_root_states[:, 3:7], relative_camera_pos)
        
        relative_wrist_camera_pos = to_torch(self.cfg["sensor"]["wrist_camera"]["position"], device=self.device).repeat(self.num_envs, 1)
        wrist_pos = self._rigid_body_pos[:, self.wrist_idx]
        wrist_rot = self._rigid_body_rot[:, self.wrist_idx]
        wrist_camera_pos = wrist_pos + quat_apply(wrist_rot, relative_wrist_camera_pos)
        
        for i in range(self.num_envs):
            heights = camera_pos[i].cpu().numpy()
            x = heights[0]
            y = heights[1]
            z = heights[2]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
        
        for i in range(self.num_envs):
            wrist_heights = wrist_camera_pos[i].cpu().numpy()
            x = wrist_heights[0]
            y = wrist_heights[1]
            z = wrist_heights[2]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
    
    def _draw_ee_goal_target(self):
        self.gym.clear_lines(self.viewer)
        sphere_geom = gymutil.WireframeSphereGeometry(0.03, 4, 4, None, color=(1, 0.129, 0))
        ee_goal_local = self.curr_ee_goal_cart
        arm_base_local = torch.tensor([0.3, 0.0, 0.09], device=self.device).repeat(self.num_envs, 1)
        arm_base = quat_apply(self._robot_root_states[:, 3:7], arm_base_local) + self._robot_root_states[:, :3]
        ee_goal_global = quat_apply(self._robot_root_states[:, 3:7], ee_goal_local) + arm_base
        for i in range(self.num_envs):
            heights = ee_goal_global[i].cpu().numpy()
            x = heights[0]
            y = heights[1]
            z = heights[2]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def camera_to_target_point_tensor(p_camera, camera_center=(-0.05, 0, 0.285), 
                                    target_origin=(0, 0, 0), rpy=(1.5707963267948966, 0, 0)):
        """
            将摄像机坐标系下的点转换为目标坐标系下的点（带偏转），使用 PyTorch tensor。
            
            参数:
                p_camera: 摄像机坐标系下的点，PyTorch tensor，例如 torch.tensor([x_c, y_c, z_c])
                camera_center: 摄像机中心在世界坐标系下的坐标，默认 (-0.05, 0, 0.285)
                target_origin: 目标坐标系原点，默认 (0, 0, 0)
                rpy: 目标坐标系的偏转（欧拉角，弧度），默认 (1.5708, 0, 0)，XYZ 顺序
            
            返回:
                p_target: 目标坐标系下的点，PyTorch tensor
        """
        # 确保输入是 tensor
        p_camera = torch.as_tensor(p_camera, dtype=torch.float32)
        camera_center = torch.tensor(camera_center, dtype=torch.float32)
        target_origin = torch.tensor(target_origin, dtype=torch.float32)
        # 步骤 1：从摄像机坐标系到世界坐标系（平移）
        p_world = p_camera + camera_center
        # 步骤 2：构建目标坐标系的旋转矩阵（RPY，XYZ 顺序）
        roll, pitch, yaw = rpy
        # 旋转矩阵 R = Rz * Ry * Rx
        cos_r, sin_r = torch.cos(roll), torch.sin(roll)
        cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        Rx = torch.tensor([
            [1, 0, 0],
            [0, cos_r, -sin_r],
            [0, sin_r, cos_r]
        ], dtype=torch.float32)
        Ry = torch.tensor([
            [cos_p, 0, sin_p],
            [0, 1, 0],
            [-sin_p, 0, cos_p]
        ], dtype=torch.float32)
        Rz = torch.tensor([
            [cos_y, -sin_y, 0],
            [sin_y, cos_y, 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        R = Rz @ Ry @ Rx  # 旋转矩阵
        # 步骤 3：从世界坐标系到目标坐标系（逆旋转）
        R_inv = R.T  # 逆矩阵为转置
        p_target = R_inv @ (p_world - target_origin)
        return p_target
    
    def compute_mask_centroids_with_depth(self, mask, depth):
        """
        计算多个环境掩码的中心点坐标及其对应位置的深度值（最近邻取值）。
        
        参数:
            mask (torch.Tensor): 形状为 [num_envs, h, w] 的张量，掩码值通常为 0 或 1。
                                num_envs 是环境数量，h 是高度，w 是宽度。
            depth (torch.Tensor): 形状为 [num_envs, h, w] 的张量，表示每个像素的深度值。
        
        返回:
            centroids (torch.Tensor): 形状为 [num_envs, 3] 的张量，每行是对应环境的中心点 [x, y, z]，
                                    其中 z 是中心点位置的深度值。
        """
        # 检查输入维度一致性
        assert mask.shape == depth.shape, "Mask and depth tensors must have the same shape"
        num_envs, h, w = mask.shape
        
        # 创建网格坐标
        y_coords, x_coords = torch.meshgrid(torch.arange(h, device=mask.device), 
                                            torch.arange(w, device=mask.device), 
                                            indexing='ij')
        
        # 计算每个环境的掩码总和（即掩码中 1 的数量）
        mask_sum = torch.sum(mask, dim=(1, 2))  # 形状 [num_envs]
        
        # 避免除以零，将 mask_sum 为 0 的情况置为 1（中心点将为无效值）
        mask_sum = torch.clamp(mask_sum, min=1.0)
        
        # 计算加权坐标和（x 和 y 方向）
        x_sum = torch.sum(mask * x_coords, dim=(1, 2))  # 形状 [num_envs]
        y_sum = torch.sum(mask * y_coords, dim=(1, 2))  # 形状 [num_envs]
        
        # 计算中心点坐标
        x_centroids = x_sum / mask_sum  # 形状 [num_envs]
        y_centroids = y_sum / mask_sum  # 形状 [num_envs]
        
        # 将中心点坐标取整为最近的整数索引
        x_indices = torch.round(x_centroids).long().clamp(0, w - 1)  # 限制在 [0, w-1]
        y_indices = torch.round(y_centroids).long().clamp(0, h - 1)  # 限制在 [0, h-1]
        
        # 从 depth 张量中提取中心点的深度值
        z_centroids = depth[torch.arange(num_envs), y_indices, x_indices]  # 形状 [num_envs]
        
        # 组合成 [num_envs, 3] 的张量
        centroids = torch.stack([x_centroids, y_centroids, z_centroids], dim=1)
        
        # 如果某个环境的掩码全为 0，返回无效值（例如 NaN）
        centroids[mask_sum == 1] = float('nan')  # 全 0 掩码的中心点和深度设为 NaN
        
        return centroids
        
    def _get_seg_id(self):
        return 2
    
    def _get_camera_obs(self):
        """ Retrieve the camera images from camera sensors and normalize both depth and rgb images;
        """
        seg_id = self._get_seg_id() # return 2   id2 represent the id of the cube object
        seg_image = torch.stack(self.camera_sensor_dict["forward_seg"]).to(self.device)
        forward_mask = (seg_image == seg_id) 
        wrist_seg_image = torch.stack(self.camera_sensor_dict["wrist_seg"]).to(self.device)
        wrist_mask = (wrist_seg_image == seg_id)
        wrist_mask_image = wrist_mask.float() # transfer bool to float, 1 --> 1.0, 0-->0.0
        forward_mask_image = forward_mask.float()
        
        # if self.depth_random:
        #     wrist_mask_image = self.mask_transform(wrist_mask_image)
        #     forward_mask_image = self.mask_transform(forward_mask_image)
            
        ############# mask image ############
        wrist_mask_image[wrist_mask_image > 0.5] = 1.
        wrist_mask_image[wrist_mask_image <= 0.5] = 0.
        forward_mask_image[forward_mask_image > 0.5] = 1.
        forward_mask_image[forward_mask_image <= 0.5] = 0.
        ############# mask image ############

        #### color image and normalization ####
        forward_color_img = torch.stack(self.camera_sensor_dict["forward_color"]).to(self.device)
        wirst_color_img = torch.stack(self.camera_sensor_dict["wirst_color"]).to(self.device)
        forward_color_img_3ch = forward_color_img[..., :3]
        wirst_color_img_3ch = wirst_color_img[..., :3]
        normalized_forward_color_img = (forward_color_img_3ch - 0.) / (255. - 0.)
        normalized_wirst_color_img = (wirst_color_img_3ch - 0.) / (255. - 0.)
        #### color image and normalization ####

        ############# depth image #############
        depth_image = torch.stack(self.camera_sensor_dict["forward_depth"]).to(self.device)
        depth_image[depth_image < -3] = -3 
        depth_image[depth_image > -self.depth_clip_lower] = 0
        if self.depth_random:
            depth_image = self.depth_transform(depth_image)
        depth_image *= -1 
        normalized_depth = (depth_image - 0.) / (3. - 0.) # normalize depth, the range of depth is (0,1)
        
        wirst_depth_image = torch.stack(self.camera_sensor_dict["wrist_depth"]).to(self.device)
        wirst_depth_image[wirst_depth_image < -3] = -3
        wirst_depth_image[wirst_depth_image > -self.depth_clip_lower] = 0
        if self.depth_random:
            wirst_depth_image = self.depth_transform(wirst_depth_image)

        wirst_depth_image *= -1
        normalized_wrist_depth = (wirst_depth_image - 0.) / (3. - 0.) 
        ############# depth image #############
        
        ############## direct seg depth ############## 
        forward_seg_depth = normalized_depth * forward_mask_image
        wrist_seg_depth = normalized_wrist_depth * wrist_mask_image
        
        if self.cfg["sensor"]["wrist_camera"].get("resized_resolution", False):
            cprint(f"wrist_camera resized_resolution is True", "yellow")
            wrist_mask_image = self.resize_transform(wrist_mask_image)
            wrist_seg_depth = self.resize_transform(wrist_seg_depth)
        ############## direct seg depth ##############

        ############ seg depth with noise ###############
        # mode_fwd, sev_fwd = sample_perturbation_params()
        # mode_wrist, sev_wrist = sample_perturbation_params()

        # # forward_seg_depth, forward_mask_image, normalized_depth = perturb_depth_with_seg(
        # #     normalized_depth,
        # #     forward_mask_image,
        # #     mode=mode_fwd,
        # #     severity=sev_fwd,
        # #     device=self.device
        # # )
        # # wrist_seg_depth, wrist_mask_image, normalized_wrist_depth = perturb_depth_with_seg(
        # #     normalized_wrist_depth,
        # #     wrist_mask_image,
        # #     mode=mode_wrist,
        # #     severity=sev_wrist,
        # #     device=self.device
        # # )

        # forward_seg_depth, forward_mask_image, normalized_depth = perturb_depth_with_seg(
        #     normalized_depth,
        #     forward_mask_image,
        #     mode="seg_depth",
        #     severity=0.3,
        #     device=self.device
        # )
        # wrist_seg_depth, wrist_mask_image, normalized_wrist_depth = perturb_depth_with_seg(
        #     normalized_wrist_depth,
        #     wrist_mask_image,
        #     mode="seg_depth",
        #     severity=0.3,
        #     device=self.device
        # )

        ############# seg depth with noises ###############
        
        ######## my add for debug #########
        self.nor_forward_color_flatten = normalized_forward_color_img.flatten(start_dim=1) # (envs,96 * 54 * 3)
        self.nor_wirst_color_flatten = normalized_wirst_color_img.flatten(start_dim=1)# (envs,96 * 54 * 3)
        self.nor_forward_depth_flatten = normalized_depth.flatten(start_dim=1)# (envs,96 * 54 * 1)
        self.nor_wirst_depth_flatten = normalized_wrist_depth.flatten(start_dim=1)# (envs,96 * 54 * 1)
        self.cat_two_camera_RGBD_flatten_tensor = torch.cat((self.nor_forward_color_flatten,self.nor_wirst_color_flatten,self.nor_forward_depth_flatten,self.nor_wirst_depth_flatten),dim = 1) # (envs,96 * 54 * 8)
        self.cat_forward_camera_RGBD_flatten_tensor = torch.cat((self.nor_forward_color_flatten,self.nor_forward_depth_flatten),dim = 1) # (envs,96 * 54 * 4)
        self.cat_wirst_camera_RGBD_flatten_tensor = torch.cat((self.nor_wirst_color_flatten,self.nor_wirst_depth_flatten),dim = 1) # (envs,96 * 54 * 4)
        self.cat_two_camera_RGBD_flatten_tensor_size =  self.cfg["sensor"]["wrist_camera"]["resolution"][0] * self.cfg["sensor"]["wrist_camera"]["resolution"][0] * 10
        self.cat_one_camera_RGBD_flatten_tensor_size =  self.cfg["sensor"]["wrist_camera"]["resolution"][0] * self.cfg["sensor"]["wrist_camera"]["resolution"][0] * 5
        self.cfg["env"]["cat_two_camera_RGBD_flatten_tensor"] = self.cat_two_camera_RGBD_flatten_tensor
        self.cfg["env"]["cat_forward_camera_RGBD_flatten_tensor"] = self.cat_forward_camera_RGBD_flatten_tensor
        ######## my add for debug #########

        return normalized_depth.flatten(start_dim=1), normalized_wrist_depth.flatten(start_dim=1), \
            forward_mask_image.flatten(start_dim=1), wrist_mask_image.flatten(start_dim=1), \
            forward_seg_depth.flatten(start_dim=1), wrist_seg_depth.flatten(start_dim=1) , \
            # normalized_forward_color_img.flatten(start_dim=1), normalized_wirst_color_img.flatten(start_dim=1)
        
    def get_fix_camera_obs(self):
        seg_id = self._get_seg_id() # return 2 
        seg_image = torch.stack(self.fix_camera_sensor_dict["cube_seg"]).to(self.device)
        cube_mask = (seg_image == seg_id) 
        cube_mask_image = cube_mask.float()

        cube_mask_image[cube_mask_image > 0.5] = 1.
        cube_mask_image[cube_mask_image <= 0.5] = 0.

        #### color image and normalization ####

        cube_color_img = torch.stack(self.fix_camera_sensor_dict["cube_color"]).to(self.device)
        cube_color_img_3ch = cube_color_img[..., :3]
        normalized_cube_color_img = (cube_color_img_3ch - 0.) / (255. - 0.)

        #### color image and normalization ####

        depth_image = torch.stack(self.fix_camera_sensor_dict["cube_depth"]).to(self.device)
        depth_image[depth_image < -8] = -8 
        depth_image[depth_image > -self.depth_clip_lower] = 0
        if self.depth_random:
            depth_image = self.depth_transform(depth_image)
        depth_image *= -1 
        normalized_depth = (depth_image - 0.) / (8. - 0.) 

        cube_seg_depth = normalized_depth * cube_mask_image

        cube_mask_image_4dim = cube_mask_image.unsqueeze(-1)
        cube_color_img_3ch_3dim = cube_color_img_3ch[0] # 3dim # just 1 env
        # print(wirst_color_img_3ch_3dim.shape)
        cube_depth_image_4dim = depth_image.unsqueeze(-1)
        cube_depth_image_3dim = depth_image[0].unsqueeze(-1) # 2dim -> 3dim # just 1 env
        cube_mask_image_3dim = cube_mask_image[0].unsqueeze(-1)
        cube_RGBD_HAVE_SEG_img_one_env = torch.cat([cube_color_img_3ch_3dim,cube_depth_image_3dim,cube_mask_image_3dim],dim=-1)
        cube_RGBD_HAVE_SEG_img_all_env = torch.cat([cube_color_img_3ch,cube_depth_image_4dim,cube_mask_image_4dim],dim=-1)

        target_folder = "data_img_totest_01"
        child_folder = "data05"
        # obj_name = "21low"
        obj_name = "30all_nomove"
        batch_num = 5
        os.makedirs(f'/your_path/{target_folder}/{child_folder}',exist_ok="True")
        file_path = f'/your_path/{target_folder}/{child_folder}/{obj_name}_img{batch_num}.pt'
        cube_states_file_path = f'/your_path/{target_folder}/{child_folder}/{obj_name}_cube_root_states{batch_num}.pt'
        self.i+=1
        if(self.i==4):
            print("cube_RGBD_HAVE_SEG_img_all_env shape is ",cube_RGBD_HAVE_SEG_img_all_env.shape)
            print("_cube_root_states shape is ",self._cube_root_states.shape)
            self.frame_list.append(cube_RGBD_HAVE_SEG_img_all_env)
            print(f"frame {self.i} is saved")
            torch.save(self.frame_list, file_path)
            print(f"save to {file_path}")
            self.cube_root_states_list.append(self._cube_root_states)
            torch.save(self.cube_root_states_list, cube_states_file_path)
            print(f"cube States save to {cube_states_file_path}")

        
        return normalized_depth.flatten(start_dim=1), cube_mask_image.flatten(start_dim=1), cube_seg_depth.flatten(start_dim=1),
            # normalized_cube_color_img.flatten(start_dim=1), normalized_wirst_color_img.flatten(start_dim=1)
            
    def _get_wrist_camera_obs(self):
        """ Retrieve the camera images from camera sensors and normalize both depth and rgb images;
        """
        seg_id = self._get_seg_id()
        wrist_seg_image = torch.stack(self.camera_sensor_dict["wrist_seg"]).to(self.device).clone()
        wrist_mask = (wrist_seg_image == seg_id)
        wrist_mask_image = wrist_mask.float()
      
        wrist_mask_image[wrist_mask_image > 0.5] = 1.
        wrist_mask_image[wrist_mask_image <= 0.5] = 0.
        
        wirst_depth_image = torch.stack(self.camera_sensor_dict["wrist_depth"]).to(self.device).clone()
        
        wirst_depth_image[wirst_depth_image < -2] = -2
        wirst_depth_image[wirst_depth_image > -self.depth_clip_lower] = 0
        if self.depth_random:
            wirst_depth_image = self.depth_transform(wirst_depth_image)

        wirst_depth_image *= -1
        normalized_wrist_depth = (wirst_depth_image - 0.) / (2. - 0.)
     
        wrist_seg_depth = normalized_wrist_depth * wrist_mask_image
        
        if self.cfg["sensor"]["wrist_camera"].get("resized_resolution", False):
            wrist_mask_image = self.resize_transform(wrist_mask_image)
            wrist_seg_depth = self.resize_transform(wrist_seg_depth)
        
        return normalized_wrist_depth.flatten(start_dim=1), \
            wrist_mask_image.flatten(start_dim=1), \
            wrist_seg_depth.flatten(start_dim=1)
    
    def _get_front_camera_obs(self):
        """ Retrieve the camera images from camera sensors and normalize both depth and rgb images;
        """
        seg_id = self._get_seg_id()
        seg_image = torch.stack(self.camera_sensor_dict["forward_seg"]).to(self.device).clone()
        forward_mask = (seg_image == seg_id)
      
        forward_mask_image = forward_mask.float()
  
        forward_mask_image[forward_mask_image > 0.5] = 1.
        forward_mask_image[forward_mask_image <= 0.5] = 0.
        
        depth_image = torch.stack(self.camera_sensor_dict["forward_depth"]).to(self.device).clone()
        
        depth_image[depth_image < -2] = -2
        depth_image[depth_image > -self.depth_clip_lower] = 0
        if self.depth_random:
            depth_image = self.depth_transform(depth_image)
            
        depth_image *= -1
        normalized_depth = (depth_image - 0.) / (2. - 0.)
        
        
        forward_seg_depth = normalized_depth * forward_mask_image
        
        return normalized_depth.flatten(start_dim=1), \
            forward_mask_image.flatten(start_dim=1), \
            forward_seg_depth.flatten(start_dim=1)

    def check_termination(self):
        """
        Add early stop to prevent the agent from gaining too much useless data
        """
        base_quat = self._robot_root_states[:, 3:7]
        r, p, _ = euler_from_quat(base_quat)
        z = self._robot_root_states[:, 2]
        
        r_term = torch.abs(r) > 0.8 ## prevent roll too big
        p_term = torch.abs(p) > 0.8 ## prevent picth too big
        z_term = z < 0.1
        self.timeout_buf = self.progress_buf >= self.max_episode_length

        curr_ee_pos_local = quat_rotate_inverse(self._robot_root_states[:, 3:7], self.ee_pos - self.arm_base)
        ik_fail = (self.curr_ee_goal_cart[:, -1:] - curr_ee_pos_local[:, -1:]).norm(dim=-1) > 0.2 ## 防止机械臂末端的z和ee目标的z差值太大，比如机械臂再重力作用下夹爪被迫朝上了，超过阈值就reset
        # ik_fail &= (torch.div(self.curr_ee_goal_cart[:, -1], curr_ee_pos_local[:, -1]) < 0)
        # print("ikfail", ik_fail[0], self.curr_ee_goal_cart[0, -1:], curr_ee_pos_local[0, -1:])
        
        # curr_to_goal = self.curr_ee_goal_cart - curr_ee_pos_local
        # curr_ee_vel_unorm = curr_ee_pos_local - self.last_ee_pos
        # ik_fail = (torch.nn.CosineSimilarity(dim=1, eps=1e-6)(curr_to_goal, curr_ee_vel_unorm) <= 0) & (torch.norm(curr_ee_vel, dim=-1) > 0.02)
        self.reset_buf[:] = self.timeout_buf | r_term | p_term | z_term | ik_fail
        
        self.last_ee_pos = curr_ee_pos_local

    def _prepare_reward_function(self):
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
        
        self.reward_functions, self.reward_names = [], []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))
        
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}
        self.episode_metric_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                                    for name in list(self.reward_scales.keys()) + list(self.reward_scales.keys())}
    
    def compute_reward(self):
        self.rew_buf[:] = 0.0
        
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew, metric = self.reward_functions[i]()    
            rew *= self.reward_scales[name]    
            self.rew_buf += rew
            self.episode_sums[name] += rew
            self.episode_metric_sums[name] += metric
        if self.cfg["reward"]["only_positive_rewards"]:
            self.rew_buf[:] = torch.clip(self.rew_buf, min=0.0)
        
        if "termination" in self.reward_scales:
            rew, metric = self._reward_termination()
            rew *= self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew   
            self.episode_metric_sums["termination"] += metric    
            
    def get_walking_cmd_mask(self, env_ids=None, return_all=False):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        walking_mask0 = torch.abs(self.commands[env_ids, 0]) > LIN_VEL_X_CLIP
        walking_mask1 = torch.abs(self.commands[env_ids, 1]) > ANG_VEL_PITCH_CLIP
        walking_mask2 = torch.abs(self.commands[env_ids, 2]) > ANG_VEL_YAW_CLIP
        walking_mask = walking_mask0 | walking_mask1 | walking_mask2
        if return_all:
            return walking_mask0, walking_mask1, walking_mask2, walking_mask
        return walking_mask
            
    def _step_contact_targets(self):
        if self.observe_gait_commands:
            frequencies = 2
            phases = 0.5
            offsets = 0
            bounds = 0
            durations = 0.5
            self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)
            
            is_walking = self.get_walking_cmd_mask()
            
            # suddenly want to stop
            suddenstop_indices = ((self.gait_wait_timer > 0) | ((~is_walking) & (self.is_walking))).bool()
            if len(self.gait_indices[suddenstop_indices]) > 0:     
                self.gait_indices[suddenstop_indices] += 1
            overdue_indices = self.gait_wait_timer >= GAIT_WAIT_TIME
            if len(self.gait_indices[overdue_indices]) > 0:
                self.gait_indices[overdue_indices] = torch.zeros(self.num_envs, dtype=torch.float, 
                                                                                    device=self.device, requires_grad=False) 
                self.gait_wait_timer[overdue_indices] = 0
            # command to stop, last status is stop
            elseindices = (self.gait_wait_timer == 0) & ((~is_walking) & (~self.is_walking))
            elseindices = elseindices.to(torch.bool)
            if len(self.gait_indices[elseindices]) > 0:
                self.gait_wait_timer[elseindices] = 0
                self.gait_indices[elseindices] = 0

            self.is_walking = is_walking

            foot_indices = [self.gait_indices + phases + offsets + bounds,
                            self.gait_indices + offsets,
                            self.gait_indices + bounds,
                            self.gait_indices + phases]

            self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
            self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
            self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
            self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])
    
    
    def get_all_pos_targets(self, ee_goal_cart, ee_goal_orn_quat):
        dpos = ee_goal_cart - self.ee_pos
        drot = orientation_error(ee_goal_orn_quat, self.ee_orn / torch.norm(self.ee_orn, dim=-1).unsqueeze(-1))
        dpose = torch.cat([dpos, drot], dim=-1).unsqueeze(-1)
        arm_pos_targets = self.control_ik(dpose) + self._dof_pos[:, -(6 + self.num_gripper_joints):-self.num_gripper_joints]
        all_pos_targets = torch.zeros_like(self._dof_pos)
        all_pos_targets[:, -(6 + self.num_gripper_joints):-self.num_gripper_joints] = arm_pos_targets
        all_pos_targets[:, -self.num_gripper_joints:] = self.gripper_dof_pos
        
        return all_pos_targets
    
    def get_torques(self):
        self.torques = self._compute_torques(self.last_low_actions)
        
        return self.torques
    
    def floating_base_control(self):
        actor_root_state_copy = self._root_states.clone()
        x_vel_local = self.commands[:, 0]
        vel_global = quat_apply(self._robot_root_states[:, 3:7], torch.cat([x_vel_local.unsqueeze(-1), torch.zeros_like(x_vel_local.unsqueeze(-1)), torch.zeros_like(x_vel_local.unsqueeze(-1))], dim=-1))
        yaw_vel_local = self.commands[:, 2]
        yaw_vel_global = quat_apply(self._robot_root_states[:, 3:7], torch.cat([torch.zeros_like(yaw_vel_local.unsqueeze(-1)), torch.zeros_like(yaw_vel_local.unsqueeze(-1)), yaw_vel_local.unsqueeze(-1)], dim=-1))
        actor_root_state_copy[self._robot_actor_ids, 7:10] = vel_global.squeeze(-1)
        actor_root_state_copy[self._robot_actor_ids, 9] = self.commands[:, 1]
        actor_root_state_copy[self._robot_actor_ids, 12] = yaw_vel_global[:, -1]
        actor_root_state_copy[self._robot_actor_ids, 2] = torch.clip(actor_root_state_copy[self._robot_actor_ids, 2], 0.4, 0.66)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(actor_root_state_copy),
                                                        gymtorch.unwrap_tensor(self._robot_actor_ids), self.num_envs)
        
    def compute_table_command_level_00(self): ## static
        self.num_steps_to_change_mode[...] += 1
        mask_step_change = (self.num_steps_to_change_mode % (self.rand_change_step*6*self.increase_to_change_step)==0)
        self.increase_to_change_step[mask_step_change]+=1
        self.mask_step_change = mask_step_change
        true_count = mask_step_change.sum().item()
        self.rand_change_step[mask_step_change] = torch.randint(low=40, high=60, size=(true_count,), device=self.device)
        self.table_commands[:,0] = 0
        self.table_commands[:,1] = 0
        self.table_commands[self.lin_circle_select,5] = 0
        self.table_commands[~self.lin_circle_select,5] = 0
        self.table_commands[:,2] = 0


    def compute_table_command_level_01(self): ## fixed 0--0.15m/s in y direction or cicle_trajectory movement
        self.num_steps_to_change_mode[...] += 1
        mask_step_change = (self.num_steps_to_change_mode % (self.rand_change_step*6*self.increase_to_change_step)==0)
        self.increase_to_change_step[mask_step_change]+=1
        self.mask_step_change = mask_step_change
        true_count = mask_step_change.sum().item()
        self.rand_change_step[mask_step_change] = torch.randint(low=40, high=60, size=(true_count,), device=self.device)

        self.table_commands[:,0] = 0
        self.table_commands[:,1] = self.vel_line_forward_y
        self.table_commands[self.lin_circle_select,5] = self.vel_angular_yaw[self.lin_circle_select]
        # self.table_commands[self.lin_circle_select,5] = 0
        self.table_commands[~self.lin_circle_select,5] = 0
        self.table_commands[:,2] = 0
        
    def compute_table_command_level_02(self): ## fixed 0.15--0.30m/s in y direction or cicle_trajectory movement
        self.num_steps_to_change_mode[...] += 1
        mask_step_change = (self.num_steps_to_change_mode % (self.rand_change_step*6*self.increase_to_change_step)==0)
        self.increase_to_change_step[mask_step_change]+=1
        self.mask_step_change = mask_step_change
        true_count = mask_step_change.sum().item()
        self.rand_change_step[mask_step_change] = torch.randint(low=40, high=60, size=(true_count,), device=self.device)
        self.table_commands[:,0] = 0
        self.table_commands[:,1] = self.vel_line_forward_y_level02
        self.table_commands[self.lin_circle_select,5] = self.vel_angular_yaw[self.lin_circle_select]
        self.table_commands[~self.lin_circle_select,5] = 0
        self.table_commands[:,2] = 0
        
    # def compute_table_command_level_03(self): ## low-speed random two-dimensional trajectory
    #     self.num_steps_to_change_mode[...] += 1
    #     # print(f"num_steps_to_change_mode is {self.num_steps_to_change_mode[0]}")
    #     # print("self.num_steps_to_change_mode[0] is ",self.num_steps_to_change_mode[0])
    #     mask_step_change = (self.num_steps_to_change_mode % (self.rand_change_step*6*self.increase_to_change_step)==0)
    #     # print(f"self.num_steps_to_change_mode is {self.num_steps_to_change_mode} ; mask_step_change is {mask_step_change} ; self.rand_change_step*6 is {self.rand_change_step*6*self.increase_to_change_step}")
    #     self.increase_to_change_step[mask_step_change]+=1
    #     self.mask_step_change = mask_step_change
    #     true_count = mask_step_change.sum().item()
    #     self.rand_change_step[mask_step_change] = torch.randint(low=40, high=60, size=(true_count,), device=self.device)
        
    #     self.vel_line_forward_y[mask_step_change] = torch.rand(true_count,device=self.device) *(-0.1-0.1) + 0.1
    #     self.vel_line_forward_x[mask_step_change] = torch.rand(true_count,device=self.device) *(-0.1-0.1) + 0.1
    #     self.vel_angular_yaw[mask_step_change] = torch.rand(true_count,device=self.device) *(-0.2-0.2) + 0.2
    #     self.vel_line_z[mask_step_change] = torch.rand(true_count,device=self.device) *-0.0008/7*8 + 0.0004/7*8
    #     # self.table_commands[:,0] = self.vel_line_forward_x
    #     # self.table_commands[:,1] = self.vel_line_forward_y
    #     # self.table_commands[:,5] = self.vel_angular_yaw
    #     # self.table_commands[:,2] = self.vel_line_z
    #     # self.vel_line_z = torch.where(self.vel_line_z>0.15, self.vel_line_z, torch.zeros_like(self.vel_line_z,device=self.device))
    #     self.table_commands[:,0] = self.vel_line_forward_x
    #     self.table_commands[:,1] = self.vel_line_forward_y
    #     self.table_commands[:,5] = self.vel_angular_yaw
    #     self.table_commands[:,2] = 0
        
    def compute_table_command_level_03(self): ## High-speed random two-dimensional trajectory
        self.num_steps_to_change_mode[...] += 1
        mask_step_change = (self.num_steps_to_change_mode % (self.rand_change_step*6*self.increase_to_change_step)==0)
        self.increase_to_change_step[mask_step_change]+=1
        self.mask_step_change = mask_step_change
        true_count = mask_step_change.sum().item()
        self.rand_change_step[mask_step_change] = torch.randint(low=40, high=60, size=(true_count,), device=self.device)
        
        self.vel_line_forward_y_level03[mask_step_change] = torch.rand(true_count,device=self.device) *(-0.212-0.212) + 0.212
        self.vel_line_forward_x[mask_step_change] = torch.rand(true_count,device=self.device) *(-0.212-0.212) + 0.212
        self.vel_angular_yaw[mask_step_change] = torch.rand(true_count,device=self.device) *(-0.3-0.3) + 0.3
        self.vel_line_z[mask_step_change] = torch.rand(true_count,device=self.device) *-0.0008/7*8 + 0.0004/7*8
        self.table_commands[:,0] = self.vel_line_forward_x
        self.table_commands[:,1] = self.vel_line_forward_y_level03
        self.table_commands[:,5] = self.vel_angular_yaw
        self.table_commands[:,2] = 0
    
    def compute_table_command_level_04(self): 
        self.num_steps_to_change_mode[...] += 1
        mask_step_change = (self.num_steps_to_change_mode % (self.rand_change_step*6*self.increase_to_change_step)==0)
        self.increase_to_change_step[mask_step_change]+=1
        self.mask_step_change = mask_step_change
        true_count = mask_step_change.sum().item()
        self.rand_change_step[mask_step_change] = torch.randint(low=40, high=60, size=(true_count,), device=self.device)
        
        # self.vel_line_forward_y_level03[mask_step_change] = torch.rand(true_count,device=self.device) *(-0.212-0.212) + 0.212
        # self.vel_line_forward_x[mask_step_change] = torch.rand(true_count,device=self.device) *(-0.212-0.212) + 0.212
        # self.vel_angular_yaw[mask_step_change] = torch.rand(true_count,device=self.device) *(-0.3-0.3) + 0.3
        self.vel_line_forward_y_level03[mask_step_change] = torch.rand(true_count,device=self.device) *(-0.2-0.2) + 0.2
        self.vel_line_forward_x[mask_step_change] = torch.rand(true_count,device=self.device) *(-0.2-0.2) + 0.2
        self.vel_angular_yaw[mask_step_change] = torch.rand(true_count,device=self.device) *(-0.3-0.3) + 0.3
        self.vel_line_z[mask_step_change] = torch.rand(true_count,device=self.device) *-0.0008/7*8 + 0.0004/7*8
        self.table_commands[:,0] = self.vel_line_forward_x
        self.table_commands[:,1] = self.vel_line_forward_y_level03
        self.table_commands[:,5] = self.vel_angular_yaw
        self.table_commands[:,2] = self.vel_line_z
 
    def compute_table_command_level_05(self):   ## vertical movement
        self.num_steps_to_change_mode[...] += 1
        mask_step_change = (self.num_steps_to_change_mode % (self.rand_change_step*6*self.increase_to_change_step)==0)
        self.increase_to_change_step[mask_step_change]+=1
        self.mask_step_change = mask_step_change
        true_count = mask_step_change.sum().item()
        self.rand_change_step[mask_step_change] = torch.randint(low=40, high=60, size=(true_count,), device=self.device)
        
        self.vel_line_forward_y[mask_step_change] = torch.rand(true_count,device=self.device) *(-0.2-0.2) + 0.2
        self.vel_line_forward_x[mask_step_change] = torch.rand(true_count,device=self.device) *(-0.2-0.2) + 0.2
        self.vel_angular_yaw[mask_step_change] = torch.rand(true_count,device=self.device) *(-0.3-0.3) + 0.3
        self.vel_line_z_fix[mask_step_change] = -self.vel_line_z_fix[mask_step_change]
        self.table_commands[:,0] = 0
        self.table_commands[:,1] = 0
        self.table_commands[:,5] = 0
        self.table_commands[:,2] = self.vel_line_z_fix

    def floating_table_control(self):
        if self.table_level == "Level01":
            self.compute_table_command_level_01()
        elif self.table_level == "Level02":
            self.compute_table_command_level_02()
        elif self.table_level == "Level03":
            self.compute_table_command_level_03()
        elif self.table_level == "Level04":
            self.compute_table_command_level_04()
        elif self.table_level == "Level05":
            self.compute_table_command_level_05()
        elif self.table_level == "Level00":
            self.compute_table_command_level_00()
        else:
            raise ValueError(f"table_level {self.table_level} is not supported")
        actor_root_state_copy = self._root_states.clone()
        table_root_pos = self._root_states[self._table_actor_ids, :2]
        table_root_quat = self._root_states[self._table_actor_ids, 3:7]
        #  [v_x, v_y, 0]
        v_local = torch.zeros((self.num_envs, 3), device=self._root_states.device, dtype=self._root_states.dtype)
        v_local[:, 0] = self.table_commands[:, 0]  
        v_local[:, 1] = self.table_commands[:, 1]  
        # Local line velocity to global line velocity
        v_global = quat_rotate(table_root_quat, v_local)  # [num_envs, 3]
        actor_root_state_copy[self._table_actor_ids, 7] = v_global[:, 0]  # global vx
        actor_root_state_copy[self._table_actor_ids, 8] = v_global[:, 1]  # global vy
        actor_root_state_copy[self._table_actor_ids, 9] = 0 ## set z_vel to zero, this is nessesary!
        # self.table_commands[:, 4] = self.table_commands[:, 4]
        self.table_commands[:, 4] = self.table_commands[:, 4]+self.table_commands[:,2]
        self.table_commands[:, 4] = torch.clamp(self.table_commands[:, 4],max=self.table_z_upper-self.table_dimz / 2.0, min=self.table_z_lower-self.table_dimz / 2.0)
        self.table_heights = self.table_commands[:,4] + self.table_dimz / 2.0
        actor_root_state_copy[self._table_actor_ids, 2] = self.table_commands[:, 4]
        actor_root_state_copy[self._table_actor_ids, 12] = self.table_commands[:,5]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        actor_root_state_copy[self._table_actor_ids, 3:5]=0.0
        actor_root_state_copy[self._table_actor_ids, 10:12]=0.0
        _table_actor_ids32 = self._table_actor_ids.to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(actor_root_state_copy),
                                                        gymtorch.unwrap_tensor(_table_actor_ids32), self.num_envs)


    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """
        
        if self.pred_success:
            if actions.shape[-1] == (self.action_space.shape[-1]+1):
                actions, _ = actions[...,:-1], actions[...,-1]
                
        if self.rand_control:
            self.control_freq_low = random.randint(self.control_freq_low_init-1, self.control_freq_low_init) 
        
        # randomize actions
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        
        # apply actions
        ee_goal_cart, ee_goal_orn_quat = self.pre_physics_step(action_tensor)
        
        img_delay_frame = self.img_delay_frame
        if self.img_delay_frame >= 1:
            img_delay_frame = random.randint(self.img_delay_frame-1, self.img_delay_frame) # random the delay
        # front_img_delay_frame = random.randint(self.img_delay_frame-1, self.img_delay_frame+1) 
        # wrist_img_delay_frame = random.randint(front_img_delay_frame, self.img_delay_frame+1)

        # -------------------------query low level and simulate--------------------------
        # Within a single high-level cycle, the physical simulation is executed a total of 8 * 4 = 32 times, or 7 * 4 = 28 steps.
        time1 = time.time()
        for low_step in range(self.control_freq_low):
            if not self.floating_base:
                self._compute_low_level_observations()
                with torch.no_grad():
                    low_actions = self.low_level_policy(self.low_obs_buf.detach(), hist_encoding=True)
                low_actions = reindex_all(low_actions)
            
                self.last_low_actions[:] = low_actions[:]
            
            if self.floating_base:
                all_pos_targets = self.get_all_pos_targets(ee_goal_cart, ee_goal_orn_quat)
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(all_pos_targets))
                self.floating_base_control()
                
            # step physics and render each frame
            for i in range(self.control_freq_inv):
                if self.floating_base:
                    pass
                    # self.floating_base_control()
                else:
                    env_torques = self.get_torques() # This is the chassis of the robotic dog, excluding the torque of the robotic arm (all degrees of freedom are set to 0), which is the final output given to the robot.
                    all_pos_targets = self.get_all_pos_targets(ee_goal_cart, ee_goal_orn_quat) # Obtain the target motors (degrees of freedom) on the robotic arm, excluding the chassis robot dog.
                    ## DQ add ##
                    self.floating_table_control()
                    self.table_heights = self.table_commands[:,4] + self.table_dimz / 2.0
                    ## DQ add ##
                    self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(all_pos_targets))
                    self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(env_torques))
                    
                self.gym.simulate(self.sim)
                self._refresh_sim_tensors()

            if not self.headless:
                self.render()
            
            if self.enable_camera or self.camera_test:
                if ((self.control_freq_low - low_step - 1) == img_delay_frame): # simulate the real delay, 0.1s
                    self.obtain_imgs()
                    # self.obtain_front_imgs() # original one
                    
        
        self.one_epoch_step+=1
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        ################ !!!!!!!!!!!!!!!!!!!!!!!! IMPORTANT !!!!!!!!!!!!!!!!!!!!!!! ########################
        self.post_physics_step()
        #### obs , rew , reset and so on all can find in this func

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)
        # randomize observations
        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras
    
    def pre_physics_step(self, action_tensor):
        if (not self.stu_distill) and self.global_step_counter <= (self.total_timesteps/5):
            action_delay = 0
        else:
            action_delay = 0 
            
        # -------------------------receive commands----------------------------
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:], action_tensor[:, None, :]], dim=1)
        action_tensor = self.action_history_buf[:, -action_delay - 1]
        if self.arm_delay and (action_delay > 0):
            action_tensor[:, :7] =  self.action_history_buf[:, -action_delay - 2, :7] # Arm uses one more step delay
        self.actions[:] = action_tensor[:]
        # -------------------------delta based on current ee pos----------------------------
        # arm_base_local = torch.tensor([0.3, 0.0, 0.09], device=self.device).repeat(self.num_envs, 1)
        # arm_base = quat_apply(self._robot_root_states[:, 3:7], arm_base_local) + self._robot_root_states[:, :3]
        # ee_cart = quat_rotate_inverse(self._robot_root_states[:, 3:7], self.ee_pos - arm_base)
        # ee_goal_cart_local = ee_cart + torch.clip(actions[:, :3], -0.1, 0.1)
        # ee_goal_cart = quat_apply(self._robot_root_states[:, 3:7], ee_goal_cart_local) + arm_base
        # self.curr_ee_goal_cart[:] = ee_goal_cart
        
        # ee_goal_orn_delta_rpy = torch.clip(actions[:, 3:6], -0.5, 0.5)
        # ee_local_orn = quat_mul(quat_conjugate(self._robot_root_states[:, 3:7]), self.ee_orn)
        # ee_local_orn_rpy = torch.stack(euler_from_quat(ee_local_orn), dim=-1)
        # ee_goal_orn_rpy_local = ee_local_orn_rpy + ee_goal_orn_delta_rpy
        # ee_goal_orn_quat_local = quat_from_euler_xyz(ee_goal_orn_rpy_local[:, 0], ee_goal_orn_rpy_local[:, 1], ee_goal_orn_rpy_local[:, 2])
        # ee_goal_orn_quat = quat_mul(self._robot_root_states[:, 3:7], ee_goal_orn_quat_local)
        # -------------------------delta based on current ee pos----------------------------
        
        # -------------------------delta based on current ee goal----------------------------
        if False: # self.cfg["env"].get("lastCommands", False) and (self.global_step_counter > 50000 or self.enable_camera):
            if self.cfg["env"]["useTanh"]:
                self.delta_goal_cart = self.actions[:, :3] * 0.015
            else:
                self.delta_goal_cart = torch.clip(self.actions[:, :3], -0.015, 0.015)
        else:
            if self.cfg["env"]["useTanh"]: #### 特别注意好像high预测出来的pos只是一个delta_pos值，下面代码找到的目标pos值是用现在的pos加上delta_pos
                self.delta_goal_cart = self.actions[:, :3] * 0.02 #my change 0.02->0,03# 
            else:
                self.delta_goal_cart = torch.clip(self.actions[:, :3], -0.02, 0.02)
        self.curr_ee_goal_cart[:] = self.curr_ee_goal_cart + self.delta_goal_cart
        # self.curr_ee_goal_cart[:, 0] = torch.clip(self.curr_ee_goal_cart[:, 0], 0.0, 0.9)

        self.curr_ee_goal_cart[:, 0] = torch.clip(self.curr_ee_goal_cart[:, 0], 0., 0.7)
        self.curr_ee_goal_cart[:, 1] = torch.clip(self.curr_ee_goal_cart[:, 1], -0.7, 0.7)
        self.curr_ee_goal_cart[:, 2] = torch.clip(self.curr_ee_goal_cart[:, 2], -0.6, 0.6)
        ee_goal_cart = quat_apply(self._robot_root_states[:, 3:7], self.curr_ee_goal_cart) + self.arm_base
        self.ee_goal_global_cart = ee_goal_cart.clone()

        
        if self.cfg["env"]["useTanh"]:
            self.delta_goal_orn = self.actions[:, 3:6] * 0.06 #my change 0.06->0.08# 
        else:
            self.delta_goal_orn = torch.clip(self.actions[:, 3:6], -0.06, 0.06)
        self.curr_ee_goal_orn_rpy[:] = self.curr_ee_goal_orn_rpy + self.delta_goal_orn
        # self.curr_ee_goal_orn_rpy[:] = self.curr_ee_goal_orn_rpy + torch.clip(actions[:, 3:6], -0.04, 0.04)
        # self.curr_ee_goal_orn_rpy[:] = self.curr_ee_goal_orn_rpy + torch.clip(actions[:, 3:6], -0.02, 0.02)
        ee_goal_local_orn = quat_from_euler_xyz(self.curr_ee_goal_orn_rpy[:, 0], self.curr_ee_goal_orn_rpy[:, 1], self.curr_ee_goal_orn_rpy[:, 2])
        ee_goal_orn_quat = quat_mul(self._robot_root_states[:, 3:7], ee_goal_local_orn)
        self.ee_goal_global_orn_quat = ee_goal_orn_quat.clone()
        # -------------------------delta based on current ee goal----------------------------
        self.set_gripper()
        self.clip_commands()
        if (not self.pitch_control) and (not self.floating_base):
            self.commands[:, 1] = 0
        if self.pitch_control:
            if self.cfg["env"]["useTanh"]:
                self.commands[:, 1] = self.actions[:, 9] * 0.6
            else:
                self.commands[:, 1] = torch.clip(self.actions[:, 9], -0.6, 0.6)
        
        if self.floating_base:
            if self.cfg["env"]["useTanh"]:
                self.commands[:, 1] = self.actions[:, 9] * 0.3
            else:
                self.commands[:, 1] = torch.clip(self.actions[:, 9], -0.3, 0.3)
        
        if self.cfg["env"]["useTanh"]:
            self.commands[:, 2] = self.actions[:, 8] * 0.6
        else:
            self.commands[:, 2] = torch.clip(self.actions[:, 8], -0.6, 0.6)
        # set small commands to zero
        if self.cfg["env"].get("smallValueSetZero", False):
            self.commands[:, :] *= (torch.logical_or(torch.abs(self.commands[:, 0]) > LIN_VEL_X_CLIP, torch.abs(self.commands[:, 2]) > ANG_VEL_YAW_CLIP)).unsqueeze(1)
        # -------------------------receive commands----------------------------
        self.clipped_actions[:, :3] = self.delta_goal_cart[:].clone()
        self.clipped_actions[:, 3:6] = self.delta_goal_orn[:].clone()
        self.clipped_actions[:, 6] = self.actions[:,6].clone()
        if self.pitch_control:
            self.clipped_actions[:, 7:] = self.commands[:].clone()
        else:
            self.clipped_actions[:, 7] = self.commands[:,0].clone()
            self.clipped_actions[:, 8] = self.commands[:,2].clone()
        self.command_history_buf = torch.cat([self.command_history_buf[:, 1:], self.clipped_actions[:, None, :]], dim=1)
        
        if self.stop_pick: 
            self.commands = torch.where(self.actions[:, 6].unsqueeze(-1) < 0, torch.zeros_like(self.commands), self.commands) # Start to close then we should stop
        
        if self.rand_depth_clip:
            self.depth_clip_lower = torch_rand_float(self.depth_clip_rand_range[0], self.depth_clip_rand_range[1], (self.num_envs,1), device=self.device).unsqueeze(-1)

        return ee_goal_cart, ee_goal_orn_quat
    
    def set_gripper(self):
        u_gripper = self.actions[:, 6].unsqueeze(-1)
        if self.num_gripper_dof == 1:
            self.gripper_dof_pos[:] = torch.where(u_gripper >= 0, self.dof_limits_lower[-1].item(), self.dof_limits_upper[-1].item()) # >= 0 open    
        elif self.num_gripper_dof == 2:
            # TODO: May check this; dof limits lower or higher corresponds to gripper open or close
            self.gripper_dof_pos[:, -2] = torch.where(u_gripper >= 0, self.dof_limits_lower[-2].item(), self.dof_limits_upper[-2].item())
            self.gripper_dof_pos[:, -1] = torch.where(u_gripper >= 0, self.dof_limits_lower[-1].item(), self.dof_limits_upper[-1].item())
        else:
            raise NotImplementedError
    
    # For random gripper kp
    def clip_commands(self):
        if not self.commands_curriculum:
            if self.cfg["env"]["useTanh"]:
                self.commands[:, 0] = self.actions[:, 7] * 0.4
            else:
                self.commands[:, 0] = torch.clip(self.actions[:, 7], -0.4, 0.4)
            return

        if self.cfg["env"]["useTanh"]:
            if (self.global_step_counter > (self.total_timesteps/5 * 3)) or self.stu_distill:
                self.commands[:, 0] = self.actions[:, 7] * 0.45 # my change #
            elif self.global_step_counter > (self.total_timesteps/2):
                self.commands[:, 0] = self.actions[:, 7] * 0.45 # my change #
            elif self.global_step_counter > (self.total_timesteps/5):
                self.commands[:, 0] = self.actions[:, 7] * 0.45 # my change #
            else:
                self.commands[:, 0] = self.actions[:, 7] * 0.45 # my change #
        else:
            if (self.global_step_counter > (self.total_timesteps/5 * 3)) or self.stu_distill:
                self.commands[:, 0] = torch.clip(self.actions[:, 7], -0.45, 0.45)
            elif self.global_step_counter > (self.total_timesteps/2):
                self.commands[:, 0] = torch.clip(self.actions[:, 7], -0.45, 0.45)
            elif self.global_step_counter > (self.total_timesteps/5):
                self.commands[:, 0] = torch.clip(self.actions[:, 7], -0.45, 0.45)
            else:
                self.commands[:, 0] = torch.clip(self.actions[:, 7], -0.45, 0.45)
    
    def _compute_states_buf(self): # if enabled_camera is true, then this function will be called
        self.states_buf[:] = torch.cat([self.camera_history_buf.view(self.num_envs, -1), self.obs_buf[:, (self.num_features + 6):-(self.num_actions + 3+180+2)], self.obs_buf[:, -self.num_actions:]], dim=-1)
        # self.states_buf[:] = torch.cat([self.camera_history_buf.view(self.num_envs, -1), self.obs_buf[:, (self.num_features + 12):-(self.num_actions + 3)], self.obs_buf[:, -self.num_actions:]], dim=-1) # my change
        
    def update_roboinfo(self):
        # Arm base
        base_quat = self._robot_root_states[:, 3:7]
        arm_base_local = torch.tensor([0.3, 0.0, 0.09], device=self.device).repeat(self.num_envs, 1)
        self.arm_base = quat_apply(base_quat, arm_base_local) + self._robot_root_states[:, :3]
        
    def obtain_imgs(self):
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        self.depth_image_flat, self.wrist_depth_image_flat, self.forward_mask, self.wrist_mask, self.forward_depth_seg, self.wrist_depth_seg = self._get_camera_obs()
        if self.camera_test==True:
            _,_,_ = self.get_fix_camera_obs()
        self.gym.end_access_image_tensors(self.sim)
        
    def make_img_obs(self):
        if self.camera_mode == "full" or self.camera_mode == "seperate":
            tensor_obs = torch.cat([self.forward_mask, self.wrist_mask, self.forward_depth_seg, self.wrist_depth_seg], dim=-1)
        elif self.camera_mode == "wrist_seg":
            tensor_obs = torch.cat([self.forward_mask, self.wrist_mask, self.forward_depth_seg], dim=-1)
        elif self.camera_mode == "front_only":
            tensor_obs = torch.cat([self.forward_mask, self.forward_depth_seg], dim=-1)
        self.camera_history_buf = torch.where(
            (self.progress_buf <= 1)[:, None, None],
            torch.stack([tensor_obs] * self.camera_history_len, dim=1),
            self.camera_history_buf
        )
        self.camera_history_buf = torch.cat([
            self.camera_history_buf[:, 1:],
            tensor_obs.unsqueeze(1)
        ], dim=1)
        
    def obtain_front_imgs(self):
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        self.depth_image_flat, self.forward_mask, self.forward_depth_seg = self._get_front_camera_obs()
        self.gym.end_access_image_tensors(self.sim)
    
    def obtain_wrist_imgs(self):
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        self.wrist_depth_image_flat, self.wrist_mask, self.wrist_depth_seg = self._get_wrist_camera_obs()
        self.gym.end_access_image_tensors(self.sim)
    
    def post_physics_step(self):
        self.progress_buf += 1
        self.global_step_counter += 1
        self.local_step_counter += 1
        self._refresh_sim_tensors()
        self.update_roboinfo()
        self.check_termination()
        self.compute_reward()
        self.last_actions[:] = self.actions[:]
        self._compute_observations()
        if self.enable_camera: 
            # self.obtain_imgs() # We do not obtain here, to simulate the camera delay
            # self.obtain_wrist_imgs()
            self.make_img_obs() # Here, we stitch together the observations from all the cameras to obtain a single image tensor.
            self._compute_states_buf()
        
        if self.debug_vis:
            env_id = 0
            if self.enable_camera or self.camera_test:
                forward_depth_window_name = "Forward Depth Image"
                wrist_depth_window_name = "Wrist Depth Image"
                forward_mask_window_name = "Forward Mask Image"
                wrist_mask_window_name = "Wrist Mask Image"
                forward_depth_seg_window_name = "Forward Depth Seg Image"
                wrist_depth_seg_window_name = "Wrist Depth Seg Image"
                cv2.namedWindow(forward_depth_window_name, cv2.WINDOW_NORMAL)
                cv2.namedWindow(wrist_depth_window_name, cv2.WINDOW_NORMAL)
                cv2.namedWindow(forward_mask_window_name, cv2.WINDOW_NORMAL)
                cv2.namedWindow(wrist_mask_window_name, cv2.WINDOW_NORMAL)
                cv2.namedWindow(forward_depth_seg_window_name, cv2.WINDOW_NORMAL)
                cv2.namedWindow(wrist_depth_seg_window_name, cv2.WINDOW_NORMAL)
                forward_depth, wrist_depth, forward_mask, wrist_mask, forward_depth_seg, wrist_depth_seg = self.depth_image_flat, self.wrist_depth_image_flat, self.forward_mask, self.wrist_mask, self.forward_depth_seg, self.wrist_depth_seg
                # forward_depth, wrist_depth, forward_mask, wrist_mask, forward_depth_seg, wrist_depth_seg = self._get_camera_obs()
                img_width = self.cfg["sensor"]["resized_resolution"][0]
                img_height = self.cfg["sensor"]["resized_resolution"][1]
                forward_depth_image = forward_depth[env_id].reshape(img_height, img_width)
                self.forward_depth_image_head = forward_depth.view(-1,img_height, img_width)
                wrist_depth_image = wrist_depth[env_id].reshape(img_height, img_width)
                self.wrist_depth_image_wirst = wrist_depth.view(-1,img_height, img_width)
                forward_mask_image = forward_mask[env_id].reshape(img_height, img_width)
                wrist_mask_image = wrist_mask[env_id].reshape(img_height, img_width)
                forward_depth_seg_image = forward_depth_seg[env_id].reshape(img_height, img_width)
                wrist_depth_seg_image = wrist_depth_seg[env_id].reshape(img_height, img_width)
                
                cv2.imshow("Forward Depth Image", forward_depth_image.cpu().numpy())
                cv2.imshow("Wrist Depth Image", wrist_depth_image.cpu().numpy())
                cv2.imshow("Forward Mask Image", forward_mask_image.cpu().numpy())
                cv2.imshow("Wrist Mask Image", wrist_mask_image.cpu().numpy())
                cv2.imshow("Forward Depth Seg Image", forward_depth_seg_image.cpu().numpy())
                cv2.imshow("Wrist Depth Seg Image", wrist_depth_seg_image.cpu().numpy())
                cv2.waitKey(1)
                
                # self._draw_ee_goal_target()
            else:
                self._draw_camera_sensors()
                # self._draw_ee_goal_target()
        
        self.extras["terminate"] = self._terminate_buf.to(self.rl_device)
        return
    
@torch.jit.script
def reindex_all(vec):
    # type: (Tensor) -> Tensor
    return torch.hstack((vec[:, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]], vec[:, 12:]))

@torch.jit.script
def reindex_feet(vec):
    return vec[:, [1, 0, 3, 2]]
