import numpy as np
import os
import torch
import cv2
from typing import Dict, Any, Tuple, List, Set
from collections import defaultdict
import wandb
from termcolor import cprint    

from .b1z1_base import B1Z1Base, reindex_all, reindex_feet, LIN_VEL_X_CLIP, ANG_VEL_YAW_CLIP, torch_rand_int
from utils.low_level_model import ActorCritic
from utils.config import load_cfg, get_params, copy_cfg
from PredictPiontTransform.predictpoint_mul import PredictPoint
import time

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from isaacgym.torch_utils import *
from torch import Tensor
import torchvision.transforms as transforms


class B1Z1PickMulti(B1Z1Base,PredictPoint):
    def __init__(self, cfg_terrain=None, table_height=None,camera_6p_tensor=None, grasp_cv_tensor=None, cube_init_tensor=None, eval=False, *args, **kwargs):
        self.cfg_terrain = cfg_terrain
        self.eval = eval
        self.num_actors = 3
        cprint(f"B1Z1PickMulti is called","cyan")

        B1Z1Base.__init__(self, *args, **kwargs)
        PredictPoint.__init__(self, camera_6p_tensor, grasp_cv_tensor, cube_init_tensor, device=self.device)
        args = get_params()
        self.near_goal_stop = self.cfg["env"].get("near_goal_stop", False)
        self.obj_move_prob = self.cfg["env"].get("obj_move_prob", 0.0)
        self.table_heights_fix = table_height
        self.total_timesteps = args.timesteps
        self.train_reward_strict = args.timesteps / 2

    def update_roboinfo(self):
        super().update_roboinfo()
        base_obj_dis = self._cube_root_states[:, :2] - self.arm_base[:, :2]
        self.base_obj_dis = torch.norm(base_obj_dis, dim=-1)
        
    def _setup_obs_and_action_info(self):
        super()._setup_obs_and_action_info(removed_dim=9+30*6+2, num_action=9, num_obs=38+30*6+2+self.num_features-1)
        # super()._setup_obs_and_action_info(removed_dim=9, num_action=9, num_obs=38+self.one_img_features-1) ## my change
        
        
    def _extra_env_settings(self):
        self.multi_obj = self.cfg["env"]["asset"]["asset_multi"]  #the cfg here refer to the b1z1_pickmulti.yaml
        self.obj_list = list(self.multi_obj.keys())
        self.obj_height = [self.multi_obj[obj]["height"] for obj in self.obj_list]
        self.obj_orn = [self.multi_obj[obj]["orientation"] for obj in self.obj_list]
        self.obj_scale = [self.multi_obj[obj]["scale"] for obj in self.obj_list]
        obj_dir = os.path.join(self.cfg["env"]["asset"]["assetRoot"], self.cfg["env"]["asset"]["assetFileObj"])
        if not self.no_feature:
            features = []
            for obj_name in self.obj_list:
                feature_path = os.path.join(obj_dir, obj_name, "features.npy")
                feature = np.load(feature_path, allow_pickle=True)
                features.append(feature)
            assert len(features) == len(self.obj_list)
            self.features = np.concatenate(features, axis=0) 
            self.num_features = self.features.shape[1]
        else:
            self.num_features = 0

    def _init_tensors(self):
        """Add extra tensors for cube and table
        """
        super()._init_tensors()
        
        self._table_root_states = self._root_states.view(self.num_envs, self.num_actors, self._actor_root_state.shape[-1])[..., 1, :]
        self._initial_table_root_states = self._table_root_states.clone()
        self._initial_table_root_states[:, 7:13] = 0.0
        
        self._cube_root_states = self._root_states.view(self.num_envs, self.num_actors, self._actor_root_state.shape[-1])[..., 2, :]
        self._initial_cube_root_states = self._cube_root_states.clone()
        self._initial_cube_root_states[:, 7:13] = 0.0
        
        self._table_actor_ids = self.num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.long) + 1
        self._cube_actor_ids = self.num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.long) + 2
        
        self.table_idx = self.gym.find_actor_rigid_body_index(self.envs[0], self.table_handles[0], "box", gymapi.DOMAIN_ENV)
        
        self.lifted_success_threshold = self.cfg["env"]["liftedSuccessThreshold"]
        self.lifted_init_threshold = self.cfg["env"]["liftedInitThreshold"]
        self.base_object_distace_threshold = self.cfg["env"]["baseObjectDisThreshold"] # default 0.6
        self.lifted_object = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        self.highest_object = -torch.ones(self.num_envs, device=self.device, dtype=torch.float)
        self.curr_height = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        self.random_angle = torch.as_tensor(np.array([0, 1.5708, -1.5708]), device=self.device, dtype=torch.float)
        self.lifted_now = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        if self.pred_success:
            self.predlift_success_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        

    def _create_extra(self, env_i):
        self.env_ptr = self.envs[env_i]
        col_group = env_i
        col_filter = 0
        i = env_i

        obj_idx = i % len(self.obj_list)
        obj_asset = self.ycb_asset_list[obj_idx]
        obj_height = self.obj_height[obj_idx]

        table_pos = [0.0, 2.0, self.table_dims.z / 2]
        self.table_heights[i] = table_pos[-1] + self.table_dims.z / 2

        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.table_start_pose = table_start_pose
        self.table_handle = self.gym.create_actor(self.env_ptr, self.table_asset, table_start_pose, "table", col_group, col_filter, 1)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 2.0, 0.5)  # initial position
        pose.r = gymapi.Quat(0, 0, 0.7071, -0.7071)  # initial orientation

        self.cube_start_pose = gymapi.Transform()
        self.cube_start_pose.p.x = table_start_pose.p.x 
        self.cube_start_pose.p.y = table_start_pose.p.y 
        self.cube_start_pose.p.z = self.table_heights[i] + obj_height  

        self.cube_handle = self.gym.create_actor(self.env_ptr, obj_asset, self.cube_start_pose, "cube", col_group, col_filter, 2)

        self.cube_indices[i] = self.gym.get_actor_index(self.env_ptr, self.cube_handle, gymapi.DOMAIN_SIM)

        self.cube_handles.append(self.cube_handle)
        self.table_handles.append(self.table_handle)

        if not self.no_feature:
            self.feature_obs[i, :] = self.features[obj_idx, :]
        self.init_height[i] = obj_height
        self.init_quat[i, :] = torch.tensor(self.obj_orn[obj_idx], device=self.device)

        ## create camera fixed in the floating table ## just when self.camera_test is True to debug  or to catch the image for grasp point prediction 
        if self.camera_test:
            fix_camera_handle = self._create_fix_cameras(self.env_ptr,self.table_handle,i)
            self.fix_camera_handles.append(fix_camera_handle)

            


    def _create_envs(self):
        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file_ycb = self.cfg["env"]["asset"]["assetFileObj"]

        self.table_heights = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # table
        self.table_dimz = 0.1
        self.table_dims = gymapi.Vec3(0.2, 0.2, self.table_dimz)
        table_options = gymapi.AssetOptions()
        table_options.fix_base_link = False
        table_options.density = 50000
        self.table_asset = self.gym.create_box(self.sim, self.table_dims.x, self.table_dims.y, self.table_dims.z, table_options)
        table_rigid_shape_props = self.gym.get_asset_rigid_shape_properties(self.table_asset)
        table_rigid_shape_props[0].friction = 2.0
        self.gym.set_asset_rigid_shape_properties(self.table_asset, table_rigid_shape_props)


        # cube 
        ycb_opts = gymapi.AssetOptions()
        ycb_opts.use_mesh_materials = True
        ycb_opts.vhacd_enabled = True
        ycb_opts.override_inertia = True
        ycb_opts.override_com = False
        ycb_opts.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        ycb_opts.disable_gravity = False  
        ycb_opts.fix_base_link = False    

        self.ycb_asset_list = []
        cprint(f"############## len(self.obj_list) is {len(self.obj_list)} ################" ,"yellow")
        for i in range(len(self.obj_list)):
            file_path = asset_file_ycb + self.obj_list[i] + "/model.urdf"
            ycb_asset = self.gym.load_asset(self.sim, asset_root, file_path, ycb_opts)

            ycb_asset_props = self.gym.get_asset_rigid_shape_properties(ycb_asset)
            ######## important: because that cube's ycb_asset_props's obj(index) don't just have one ,you may look up to the .obj not just .urdf ######
            for i in range(len(ycb_asset_props)):
                ycb_asset_props[i].friction = 2.0

            self.gym.set_asset_rigid_shape_properties(ycb_asset, ycb_asset_props)
            self.ycb_asset_list.append(ycb_asset)

        self.table_handles, self.cube_handles = [], []
        self.cube_positions = np.zeros((self.num_envs, 2))  # store cube position
        self.cube_angles = np.zeros(self.num_envs)  # store cube angle
        self.cube_speeds = np.zeros(self.num_envs)    

        self.cube_indices = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        if not self.no_feature:
            self.feature_obs = torch.zeros(self.num_envs, self.num_features, device=self.device, dtype=torch.float)
            self.features = torch.tensor(self.features, device=self.device, dtype=torch.float)
        
        self.init_height = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.init_quat = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float)
        

        super()._create_envs()
        
    def _record_one_time_success(self):
        num_group = self.num_envs // 30
        bowl_indices_np = np.array([[0+i*30, 9+i*30, 8+i*30, 23+i*30, 27+i*30] for i in range(num_group)]).reshape(1,-1).squeeze()
        bowl_indices = torch.from_numpy(bowl_indices_np).to(self.device)
        ball_indices_np = np.array([[3+i*30, 14+i*30, 15+i*30, 19+i*30] for i in range(num_group)]).reshape(1,-1).squeeze()
        ball_indices = torch.from_numpy(ball_indices_np).to(self.device)
        long_box_indices_np = np.array([[1+i*30] for i in range(num_group)]).reshape(1,-1).squeeze()
        long_box_indices = torch.from_numpy(long_box_indices_np).to(self.device)
        square_box_indices_np = np.array([[11+i*30, 20+i*30] for i in range(num_group)]).reshape(1,-1).squeeze()
        square_box_indices = torch.from_numpy(square_box_indices_np).to(self.device)
        bottle_indices_np = np.array([[2+i*30, 6+i*30, 18+i*30, 21+i*30,26+i*30, 29+i*30] for i in range(num_group)]).reshape(1,-1).squeeze()
        bottle_indices = torch.from_numpy(bottle_indices_np).to(self.device)
        cup_indices_np = np.array([[5+i*30, 12+i*30, 24+i*30,25+i*30] for i in range(num_group)]).reshape(1,-1).squeeze()
        cup_indices = torch.from_numpy(cup_indices_np).to(self.device)
        drill_indices_np = np.array([[7+i*30] for i in range(num_group)]).reshape(1,-1).squeeze()
        drill_indices = torch.from_numpy(drill_indices_np).to(self.device)
        lipton_tea_indices_np = np.array([[7+i*30] for i in range(num_group)]).reshape(1,-1).squeeze()
        lipton_tea_indices = torch.from_numpy(lipton_tea_indices_np).to(self.device)
        plate_holder_indices_np = np.array([[16+i*30] for i in range(num_group)]).reshape(1,-1).squeeze()
        plate_holder_tea_indices = torch.from_numpy(plate_holder_indices_np).to(self.device)
        book_holder_3_indices_np = np.array([[17+i*30] for i in range(num_group)]).reshape(1,-1).squeeze()
        book_holder_3_tea_indices = torch.from_numpy(book_holder_3_indices_np).to(self.device)
        phillips_screwdriver_indices_np = np.array([[22+i*30] for i in range(num_group)]).reshape(1,-1).squeeze()
        phillips_screwdriver_tea_indices = torch.from_numpy(phillips_screwdriver_indices_np).to(self.device)
        Plastic_banana_indices_np = np.array([[2+i*30] for i in range(num_group)]).reshape(1,-1).squeeze()
        Plastic_banana_tea_indices = torch.from_numpy(Plastic_banana_indices_np).to(self.device)
        clear_box_indices_np = np.array([[28+i*30] for i in range(num_group)]).reshape(1,-1).squeeze()
        clear_box_tea_indices = torch.from_numpy(clear_box_indices_np).to(self.device)
        
        # Ensure all index tensors are of long type
        bowl_indices = bowl_indices.long()  
        ball_indices = ball_indices.long()
        long_box_indices = long_box_indices.long()
        square_box_indices = square_box_indices.long()
        bottle_indices = bottle_indices.long()
        cup_indices = cup_indices.long()
        drill_indices = drill_indices.long()
        lipton_tea_indices = lipton_tea_indices.long()
        plate_holder_tea_indices = plate_holder_tea_indices.long()
        book_holder_3_tea_indices = book_holder_3_tea_indices.long()
        phillips_screwdriver_tea_indices = phillips_screwdriver_tea_indices.long()
        Plastic_banana_tea_indices = Plastic_banana_tea_indices.long()
        clear_box_tea_indices = clear_box_tea_indices.long()

        bowl_success_time = self.success_onetime_counter[bowl_indices].sum().item(), self.episode_counter[bowl_indices].sum().item()
        ball_success_time = self.success_onetime_counter[ball_indices].sum().item(), self.episode_counter[ball_indices].sum().item()
        longbox_success_time = self.success_onetime_counter[long_box_indices].sum().item(), self.episode_counter[long_box_indices].sum().item()
        squarebox_success_time = self.success_onetime_counter[square_box_indices].sum().item(), self.episode_counter[square_box_indices].sum().item()
        bottle_success_time = self.success_onetime_counter[bottle_indices].sum().item(), self.episode_counter[bottle_indices].sum().item()
        cup_success_time = self.success_onetime_counter[cup_indices].sum().item(), self.episode_counter[cup_indices].sum().item()
        drill_success_time = self.success_onetime_counter[drill_indices].sum().item(), self.episode_counter[drill_indices].sum().item()
        lipton_tea_success_time = self.success_onetime_counter[lipton_tea_indices].sum().item(), self.episode_counter[lipton_tea_indices].sum().item()
        plate_holder_tea_success_time = self.success_onetime_counter[plate_holder_tea_indices].sum().item(), self.episode_counter[plate_holder_tea_indices].sum().item()
        book_holder_3_tea_success_time = self.success_onetime_counter[book_holder_3_tea_indices].sum().item(), self.episode_counter[book_holder_3_tea_indices].sum().item()
        phillips_screwdriver_tea_success_time = self.success_onetime_counter[phillips_screwdriver_tea_indices].sum().item(), self.episode_counter[phillips_screwdriver_tea_indices].sum().item()
        Plastic_banana_tea_success_time = self.success_onetime_counter[Plastic_banana_tea_indices].sum().item(), self.episode_counter[Plastic_banana_tea_indices].sum().item()
        clear_box_tea_success_time = self.success_onetime_counter[clear_box_tea_indices].sum().item(), self.episode_counter[clear_box_tea_indices].sum().item()


        bowl_success_rate = min(bowl_success_time[0], bowl_success_time[1]) / max(bowl_success_time[1], 1)
        ball_success_rate = min(ball_success_time[0], ball_success_time[1]) / max(ball_success_time[1], 1)
        longbox_success_rate = min(longbox_success_time[0], longbox_success_time[1]) / max(longbox_success_time[1], 1)
        squarebox_success_rate = min(squarebox_success_time[0], squarebox_success_time[1]) / max(squarebox_success_time[1], 1)
        bottle_success_rate = min(bottle_success_time[0], bottle_success_time[1]) / max(bottle_success_time[1], 1)
        cup_success_rate = min(cup_success_time[0], cup_success_time[1]) / max(cup_success_time[1], 1)
        drill_success_rate = min(drill_success_time[0], drill_success_time[1]) / max(drill_success_time[1], 1)
        lipton_tea_success_rate = min(lipton_tea_success_time[0], lipton_tea_success_time[1]) / max(lipton_tea_success_time[1], 1)
        plate_holder_tea_success_rate = min(plate_holder_tea_success_time[0], plate_holder_tea_success_time[1]) / max(plate_holder_tea_success_time[1], 1)
        book_holder_3_tea_success_rate = min(book_holder_3_tea_success_time[0], book_holder_3_tea_success_time[1]) / max(book_holder_3_tea_success_time[1], 1)
        phillips_screwdriver_tea_success_rate = min(phillips_screwdriver_tea_success_time[0], phillips_screwdriver_tea_success_time[1]) / max(phillips_screwdriver_tea_success_time[1], 1)
        Plastic_banana_tea_success_rate = min(Plastic_banana_tea_success_time[0], Plastic_banana_tea_success_time[1]) / max(Plastic_banana_tea_success_time[1], 1)
        clear_box_tea_success_rate = min(clear_box_tea_success_time[0], clear_box_tea_success_time[1]) / max(clear_box_tea_success_time[1], 1)

        wandb_dict = {
            "success_rate": {
                "Onetime_SuccessRate / Bowl": bowl_success_rate,
                "Onetime_SuccessRate / Ball": ball_success_rate,
                "Onetime_SuccessRate / LongBox": longbox_success_rate,
                "Onetime_SuccessRate / SquareBox": squarebox_success_rate,
                "Onetime_SuccessRate / Bottle": bottle_success_rate,
                "Onetime_SuccessRate / Cup": cup_success_rate,
                "Onetime_SuccessRate / Drill": drill_success_rate,
                "Onetime_SuccessRate / Lipton_tea": lipton_tea_success_rate,
                "Onetime_SuccessRate / Plate_holder": plate_holder_tea_success_rate,
                "Onetime_SuccessRate / Book_holder_3": book_holder_3_tea_success_rate,
                "Onetime_SuccessRate / Phillips_screwdriver": phillips_screwdriver_tea_success_rate,
                "Onetime_SuccessRate / Plastic_banana": Plastic_banana_tea_success_rate,
                "Onetime_SuccessRate / Clear_box": clear_box_tea_success_rate,
            }
        }
        if self.pred_success:
            predlift_success_rate = 0 if self.global_step_counter==0 else (self.predlift_success_counter / self.local_step_counter).mean().item()
            wandb_dict["success_rate"]["SuccessRate / PredLifted"] = predlift_success_rate
        
        if self.cfg["env"]["wandb"]:
            self.extras.update(wandb_dict)
            # wandb.log(wandb_dict, step=self.global_step_counter)
        else:
            print("\n",wandb_dict)
            print("\n, Bowl count: {}\n, Ball count: {}\n, LongBox count: {}\n, SquareBox count: {}\n, Bottle count: {}\n, Cup count: {}\n, Drill count: {}\n, Lipton_tea count: {}\n, Plate_holder count: {}\n, Book_holder_3 count: {}\n, Phillips_screwdriver count: {}\n, Plastic_banana count: {}\n, Clear_box count: {}\n".format(bowl_success_time[1], ball_success_time[1], longbox_success_time[1], squarebox_success_time[1], 
                                                        bottle_success_time[1], cup_success_time[1], drill_success_time[1], lipton_tea_success_time[1], 
                                                        plate_holder_tea_success_time[1], book_holder_3_tea_success_time[1], phillips_screwdriver_tea_success_time[1], 
                                                        Plastic_banana_tea_success_time[1], clear_box_tea_success_time[1]))
            success_time = self.success_onetime_counter.sum().item(), self.episode_counter.sum().item()
            success_rate = min(success_time[0], success_time[1]) / max(success_time[1], 1)
            print("One time Total success rate", success_rate)
            
            mask1=self.closest_dist<0.3
            mask2 = (self.curr_dist > 0.5) & mask1
            self.surpass_dist_flag[mask2] = 0

            
    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if len(env_ids) > 0:

            num_group = self.num_envs // 30
            bowl_indices_np = np.array([[0+i*30, 9+i*30, 8+i*30, 23+i*30, 27+i*30] for i in range(num_group)]).reshape(1,-1).squeeze()
            bowl_indices = torch.from_numpy(bowl_indices_np).to(self.device)
            ball_indices_np = np.array([[3+i*30, 14+i*30, 15+i*30, 19+i*30] for i in range(num_group)]).reshape(1,-1).squeeze()
            ball_indices = torch.from_numpy(ball_indices_np).to(self.device)
            long_box_indices_np = np.array([[1+i*30] for i in range(num_group)]).reshape(1,-1).squeeze()
            long_box_indices = torch.from_numpy(long_box_indices_np).to(self.device)
            square_box_indices_np = np.array([[11+i*30, 20+i*30] for i in range(num_group)]).reshape(1,-1).squeeze()
            square_box_indices = torch.from_numpy(square_box_indices_np).to(self.device)
            bottle_indices_np = np.array([[2+i*30, 6+i*30, 18+i*30, 21+i*30,26+i*30, 29+i*30] for i in range(num_group)]).reshape(1,-1).squeeze()
            bottle_indices = torch.from_numpy(bottle_indices_np).to(self.device)
            cup_indices_np = np.array([[5+i*30, 12+i*30, 24+i*30,25+i*30] for i in range(num_group)]).reshape(1,-1).squeeze()
            cup_indices = torch.from_numpy(cup_indices_np).to(self.device)
            drill_indices_np = np.array([[7+i*30] for i in range(num_group)]).reshape(1,-1).squeeze()
            drill_indices = torch.from_numpy(drill_indices_np).to(self.device)
            lipton_tea_indices_np = np.array([[7+i*30] for i in range(num_group)]).reshape(1,-1).squeeze()
            lipton_tea_indices = torch.from_numpy(lipton_tea_indices_np).to(self.device)
            plate_holder_indices_np = np.array([[16+i*30] for i in range(num_group)]).reshape(1,-1).squeeze()
            plate_holder_tea_indices = torch.from_numpy(plate_holder_indices_np).to(self.device)
            book_holder_3_indices_np = np.array([[17+i*30] for i in range(num_group)]).reshape(1,-1).squeeze()
            book_holder_3_tea_indices = torch.from_numpy(book_holder_3_indices_np).to(self.device)
            phillips_screwdriver_indices_np = np.array([[22+i*30] for i in range(num_group)]).reshape(1,-1).squeeze()
            phillips_screwdriver_tea_indices = torch.from_numpy(phillips_screwdriver_indices_np).to(self.device)
            Plastic_banana_indices_np = np.array([[2+i*30] for i in range(num_group)]).reshape(1,-1).squeeze()
            Plastic_banana_tea_indices = torch.from_numpy(Plastic_banana_indices_np).to(self.device)
            clear_box_indices_np = np.array([[28+i*30] for i in range(num_group)]).reshape(1,-1).squeeze()
            clear_box_tea_indices = torch.from_numpy(clear_box_indices_np).to(self.device)
            
            # Ensure all index tensors are of long type
            bowl_indices = bowl_indices.long()  
            ball_indices = ball_indices.long()
            long_box_indices = long_box_indices.long()
            square_box_indices = square_box_indices.long()
            bottle_indices = bottle_indices.long()
            cup_indices = cup_indices.long()
            drill_indices = drill_indices.long()
            lipton_tea_indices = lipton_tea_indices.long()
            plate_holder_tea_indices = plate_holder_tea_indices.long()
            book_holder_3_tea_indices = book_holder_3_tea_indices.long()
            phillips_screwdriver_tea_indices = phillips_screwdriver_tea_indices.long()
            Plastic_banana_tea_indices = Plastic_banana_tea_indices.long()
            clear_box_tea_indices = clear_box_tea_indices.long()

            bowl_success_time = self.success_counter[bowl_indices].sum().item(), self.episode_counter[bowl_indices].sum().item()
            ball_success_time = self.success_counter[ball_indices].sum().item(), self.episode_counter[ball_indices].sum().item()
            longbox_success_time = self.success_counter[long_box_indices].sum().item(), self.episode_counter[long_box_indices].sum().item()
            squarebox_success_time = self.success_counter[square_box_indices].sum().item(), self.episode_counter[square_box_indices].sum().item()
            bottle_success_time = self.success_counter[bottle_indices].sum().item(), self.episode_counter[bottle_indices].sum().item()
            cup_success_time = self.success_counter[cup_indices].sum().item(), self.episode_counter[cup_indices].sum().item()
            drill_success_time = self.success_counter[drill_indices].sum().item(), self.episode_counter[drill_indices].sum().item()
            lipton_tea_success_time = self.success_counter[lipton_tea_indices].sum().item(), self.episode_counter[lipton_tea_indices].sum().item()
            plate_holder_tea_success_time = self.success_counter[plate_holder_tea_indices].sum().item(), self.episode_counter[plate_holder_tea_indices].sum().item()
            book_holder_3_tea_success_time = self.success_counter[book_holder_3_tea_indices].sum().item(), self.episode_counter[book_holder_3_tea_indices].sum().item()
            phillips_screwdriver_tea_success_time = self.success_counter[phillips_screwdriver_tea_indices].sum().item(), self.episode_counter[phillips_screwdriver_tea_indices].sum().item()
            Plastic_banana_tea_success_time = self.success_counter[Plastic_banana_tea_indices].sum().item(), self.episode_counter[Plastic_banana_tea_indices].sum().item()
            clear_box_tea_success_time = self.success_counter[clear_box_tea_indices].sum().item(), self.episode_counter[clear_box_tea_indices].sum().item()


            bowl_success_rate = min(bowl_success_time[0], bowl_success_time[1]) / max(bowl_success_time[1], 1)
            ball_success_rate = min(ball_success_time[0], ball_success_time[1]) / max(ball_success_time[1], 1)
            longbox_success_rate = min(longbox_success_time[0], longbox_success_time[1]) / max(longbox_success_time[1], 1)
            squarebox_success_rate = min(squarebox_success_time[0], squarebox_success_time[1]) / max(squarebox_success_time[1], 1)
            bottle_success_rate = min(bottle_success_time[0], bottle_success_time[1]) / max(bottle_success_time[1], 1)
            cup_success_rate = min(cup_success_time[0], cup_success_time[1]) / max(cup_success_time[1], 1)
            drill_success_rate = min(drill_success_time[0], drill_success_time[1]) / max(drill_success_time[1], 1)
            lipton_tea_success_rate = min(lipton_tea_success_time[0], lipton_tea_success_time[1]) / max(lipton_tea_success_time[1], 1)
            plate_holder_tea_success_rate = min(plate_holder_tea_success_time[0], plate_holder_tea_success_time[1]) / max(plate_holder_tea_success_time[1], 1)
            book_holder_3_tea_success_rate = min(book_holder_3_tea_success_time[0], book_holder_3_tea_success_time[1]) / max(book_holder_3_tea_success_time[1], 1)
            phillips_screwdriver_tea_success_rate = min(phillips_screwdriver_tea_success_time[0], phillips_screwdriver_tea_success_time[1]) / max(phillips_screwdriver_tea_success_time[1], 1)
            Plastic_banana_tea_success_rate = min(Plastic_banana_tea_success_time[0], Plastic_banana_tea_success_time[1]) / max(Plastic_banana_tea_success_time[1], 1)
            clear_box_tea_success_rate = min(clear_box_tea_success_time[0], clear_box_tea_success_time[1]) / max(clear_box_tea_success_time[1], 1)

            wandb_dict = {
                "success_rate": {
                    "SuccessRate / Bowl": bowl_success_rate,
                    "SuccessRate / Ball": ball_success_rate,
                    "SuccessRate / LongBox": longbox_success_rate,
                    "SuccessRate / SquareBox": squarebox_success_rate,
                    "SuccessRate / Bottle": bottle_success_rate,
                    "SuccessRate / Cup": cup_success_rate,
                    "SuccessRate / Drill": drill_success_rate,
                    "SuccessRate / Lipton_tea": lipton_tea_success_rate,
                    "SuccessRate / Plate_holder": plate_holder_tea_success_rate,
                    "SuccessRate / Book_holder_3": book_holder_3_tea_success_rate,
                    "SuccessRate / Phillips_screwdriver": phillips_screwdriver_tea_success_rate,
                    "SuccessRate / Plastic_banana": Plastic_banana_tea_success_rate,
                    "SuccessRate / Clear_box": clear_box_tea_success_rate,
                }
            }
            if self.pred_success:
                predlift_success_rate = 0 if self.global_step_counter==0 else (self.predlift_success_counter / self.local_step_counter).mean().item()
                wandb_dict["success_rate"]["SuccessRate / PredLifted"] = predlift_success_rate
            
            if self.cfg["env"]["wandb"]:
                self.extras.update(wandb_dict)
                # wandb.log(wandb_dict, step=self.global_step_counter)
            else:
                print(wandb_dict)
                print("\n, Bowl count: {}\n, Ball count: {}\n, LongBox count: {}\n, SquareBox count: {}\n, Bottle count: {}\n, Cup count: {}\n, Drill count: {}\n, Lipton_tea count: {}\n, Plate_holder count: {}\n, Book_holder_3 count: {}\n, Phillips_screwdriver count: {}\n, Plastic_banana count: {}\n, Clear_box count: {}\n".format(bowl_success_time[1], ball_success_time[1], longbox_success_time[1], squarebox_success_time[1], 
                                                           bottle_success_time[1], cup_success_time[1], drill_success_time[1], lipton_tea_success_time[1], 
                                                           plate_holder_tea_success_time[1], book_holder_3_tea_success_time[1], phillips_screwdriver_tea_success_time[1], 
                                                           Plastic_banana_tea_success_time[1], clear_box_tea_success_time[1]))
                # Accumulate historical steps and count
                self.episode_step_sum[env_ids] += self.one_epoch_step[env_ids]
                # Reset the step count for this round.
                self.one_epoch_step[env_ids] = 0.0
                # Calculate the average step of the current batch of environments.
                mean_step = (self.episode_step_sum[env_ids] / self.episode_counter[env_ids]).mean().item()
                global_avg_step = (self.episode_step_sum.sum() / self.episode_counter.sum()).item()
                print(f", global_avg_step: {global_avg_step}")
                print(f", mean_step: {mean_step}\n")
                success_time = self.success_counter.sum().item(), self.episode_counter.sum().item()
                success_rate = min(success_time[0], success_time[1]) / max(success_time[1], 1)
                print("Total success rate", success_rate)
                if self.eval:
                    self._record_one_time_success()
                cprint(f"\n########################################################################################################################################################################","yellow")
                

                
    def _reset_objs(self, env_ids):
        if len(env_ids)==0:
            return
        
        self._cube_root_states[env_ids] = self._initial_cube_root_states[env_ids]
        self._cube_root_states[env_ids,0] = self.cube_start_pose.p.x
        self._cube_root_states[env_ids,1] = self.cube_start_pose.p.y
        self._cube_root_states[env_ids,3] = self.cube_start_pose.r.x
        self._cube_root_states[env_ids,4] = self.cube_start_pose.r.y
        self._cube_root_states[env_ids,5] = self.cube_start_pose.r.z
        self._cube_root_states[env_ids,6] = self.cube_start_pose.r.w
        self._cube_root_states[env_ids,7:13] = 0.0
        self._cube_root_states[env_ids,2] = self.table_heights[env_ids] + self.init_height[env_ids]
        
        rand_yaw_box = torch_rand_float(-1.2, 1.2, (len(env_ids), 1), device=self.device).squeeze(1) 
        
        if True: 
            self._cube_root_states[env_ids, 3:7] = self.init_quat[env_ids] # Make sure to learn basic grasp
        else:
            rand_r_box = self.random_angle[torch_rand_int(0, 3, (len(env_ids),1), device=self.device).squeeze(1)]
            rand_p_box = self.random_angle[torch_rand_int(0, 3, (len(env_ids),1), device=self.device).squeeze(1)]
            self._cube_root_states[env_ids, 3:7] = quat_mul(quat_from_euler_xyz(rand_r_box, rand_p_box, rand_yaw_box), self.init_quat[env_ids])
        self._cube_root_states[env_ids, 7:13] = 0.
        
    def _reset_table(self, env_ids):
        if len(env_ids)==0:
            return
        
        self._table_root_states[env_ids] = self._initial_table_root_states[env_ids]
        self._table_root_states[env_ids,0] = self.table_start_pose.p.x
        self._table_root_states[env_ids,1] = self.table_start_pose.p.y
        self._table_root_states[env_ids,3] = self.table_start_pose.r.x
        self._table_root_states[env_ids,4] = self.table_start_pose.r.y
        self._table_root_states[env_ids,5] = self.table_start_pose.r.z
        self._table_root_states[env_ids,6] = self.table_start_pose.r.w
        # if self.table_heights_fix is None:
        #     rand_heights = torch_rand_float(0.5, 0.8, (len(env_ids), 1), device=self.device)
        # else:
        #     rand_heights = torch.ones((len(env_ids), 1), device=self.device, dtype=torch.float)*self.table_heights_fix - self.table_dimz / 2
        
        self.table_commands[env_ids,4] = torch_rand_float(0.2, 0.7, (len(env_ids), 1), device=self.device).squeeze(1) - self.table_dimz / 2.0
        self._table_root_states[env_ids, 2] = self.table_commands[env_ids,4]

        self.table_heights[env_ids] = self.table_commands[env_ids,4] + self.table_dimz / 2.0

    
    def _reset_actors(self, env_ids):
        self._reset_table(env_ids)
        self._reset_objs(env_ids)
        super()._reset_actors(env_ids)

        return
    
    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        robot_ids_int64 = self._robot_actor_ids[env_ids]
        table_ids_int64 = self._table_actor_ids[env_ids]
        cube_ids_int64 = self._cube_actor_ids[env_ids]
        multi_ids_int64 = torch.cat([robot_ids_int64, table_ids_int64, cube_ids_int64], dim=0)
        multi_ids_int32 = multi_ids_int64.to(torch.int32)
        robot_ids_int32 = robot_ids_int64.to(torch.int32)
        
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
                                                    gymtorch.unwrap_tensor(robot_ids_int32), len(robot_ids_int64))
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                        gymtorch.unwrap_tensor(multi_ids_int32), len(multi_ids_int64))
        self.gym.simulate(self.sim)
        self._refresh_sim_tensors()
        self.render()

        self.lifted_object[env_ids] = 0
        self.curr_height[env_ids] = 0.
        self.highest_object[env_ids] = -1.
        return
    
    def _refresh_sim_tensors(self):
        super()._refresh_sim_tensors()
        self._update_curr_dist()
    
    def _update_curr_dist(self):
        # d_s = self.ee_pos - self._cube_root_states[:, :3]
        d = torch.norm(self.ee_pos - self._cube_root_states[:, :3], dim=-1)
        self.curr_dist[:] = d
        self.closest_dist = torch.where(self.closest_dist < 0, self.curr_dist, self.closest_dist)
        
        self.curr_height[:] = self._cube_root_states[:, 2] - self.table_heights - self.init_height
        self.highest_object = torch.where(self.highest_object < 0, self.curr_height, self.highest_object)

    def _compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
            
        obs = super()._compute_observations(env_ids)
        
        if not self.no_feature:
            if self.cfg["env"].get("lastCommands", False):
                self.obs_buf[env_ids] = torch.cat([self.feature_obs[env_ids, :], obs, self.command_history_buf[env_ids, -1]], dim=-1)
            else:
                ####### grasp predict #######
                grasp_predict_world_pos,grasp_predict_world_rpy = self.get_grasp_world(obs[...,-6:], env_ids)

                ####### related pose ########
                grasp_global_rot = quat_from_euler_xyz(grasp_predict_world_rpy[...,0],grasp_predict_world_rpy[...,1],grasp_predict_world_rpy[...,2])
                base_quat = self._robot_root_states[env_ids, 3:7]  # [num_envs, 4], base quaternion
                base_quat = base_quat.unsqueeze(1).repeat(1, grasp_global_rot.shape[1], 1)  # [num_env, n, 4] ， base quaternion
                base_quat_conj = quat_conjugate(base_quat)   # Inverse of base quaternion
                #  Compute local quaternion: q_local = q_base^(-1) * q_global
                ee_grasp_local_orn_quat = quat_mul(base_quat_conj, grasp_global_rot)
                grasp_predict_local_rpy = quat_to_euler_zyx(ee_grasp_local_orn_quat)
                ####### related pose ########

                ####### related orientation ########
                # base_quat = self._robot_root_states[env_ids, 3:7]  # [num_envs, 4], base quaternion
                arm_base = self.arm_base[env_ids].unsqueeze(1).repeat(1, grasp_global_rot.shape[1], 1)
                relative_pos = grasp_predict_world_pos - arm_base
                grasp_predict_local_pos = quat_rotate_inverse_batched(base_quat, relative_pos)
                grasp_predict_local_pos[..., 2] = torch.clip(grasp_predict_local_pos[..., 2], -0.6, 0.6)
                ####### related orientation ########

                ####### grasp predict #######

                grasp_predict_world_pos_flat = grasp_predict_local_pos.view(grasp_predict_local_pos.shape[0], -1) ## [num_env,3*object_num]
                grasp_predict_world_rpy_flat = grasp_predict_local_rpy.view(grasp_predict_local_rpy.shape[0], -1) ## [num_env,3*object_num]
                obs_new = torch.cat([obs[...,:-6],grasp_predict_world_pos_flat,grasp_predict_world_rpy_flat],dim = -1)

                self.obs_buf[env_ids] = torch.cat([self.feature_obs[env_ids, :], obs_new, self.action_history_buf[env_ids, -1]], dim=-1) # self.action_history_buf: [num_env,3,9], obs: [num_env, 61]

        else:
            if self.cfg["env"].get("lastCommands", False):
                self.obs_buf[env_ids] = torch.cat([obs, self.command_history_buf[env_ids, -1]], dim=-1)
            else:
                self.obs_buf[env_ids] = torch.cat([obs, self.action_history_buf[env_ids, -1]], dim=-1)
    
    def _compute_robot_obs(self, env_ids=None):
        if env_ids is None:
            robot_root_state = self._robot_root_states
            table_root_state = self._table_root_states
            cube_root_state = self._cube_root_states
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel
            commands = self.commands
            table_dim = torch.tensor([0.6, 1.0, self.table_dimz]).repeat(self.num_envs, 1).to(self.device)
            base_quat_yaw = self.base_yaw_quat
            spherical_center = self.get_ee_goal_spherical_center()
            ee_goal_cart = self.curr_ee_goal_cart
            ee_goal_orn_rpy = self.curr_ee_goal_orn_rpy
        else:
            robot_root_state = self._robot_root_states[env_ids] 
            table_root_state = self._table_root_states[env_ids] 
            cube_root_state = self._cube_root_states[env_ids] 
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            commands = self.commands[env_ids]
            # table_dim = torch.tensor([0.6, 1.0, self.table_dimz]).repeat(len(env_ids), 1).to(self.device)
            table_dim = torch.tensor([self.table_dims.x, self.table_dims.y, self.table_dims.z]).repeat(len(env_ids), 1).to(self.device) ## my change _table_dim
            base_quat_yaw = self.base_yaw_quat[env_ids]
            spherical_center = self.get_ee_goal_spherical_center()[env_ids]
            ee_goal_cart = self.curr_ee_goal_cart[env_ids]
            ee_goal_orn_rpy = self.curr_ee_goal_orn_rpy[env_ids]
            cube_vel = cube_root_state[:, 7:10]
            cube_angular = cube_root_state[:, 10:]
            cube_vel_local = quat_rotate_inverse(base_quat_yaw, cube_vel)
            cube_angular_local = quat_rotate_inverse(base_quat_yaw, cube_angular)
            ee_goal_global_cart = self.ee_goal_global_cart[env_ids]
            ee_goal_orn_quat = self.ee_goal_global_orn_quat[env_ids]
        
        obs = compute_robot_observations(robot_root_state, table_root_state, cube_root_state, body_pos,
                                         body_rot, body_vel, body_ang_vel, dof_pos, dof_vel, base_quat_yaw, spherical_center, commands, self.gripper_idx, table_dim,
                                         ee_goal_cart, ee_goal_orn_rpy, ee_goal_global_cart, ee_goal_orn_quat, self.use_roboinfo, self.floating_base)
        
        return obs
    
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """
        cmd_dim = 2
        pred_dim = 1 if self.pred_success else 0
        if self.near_goal_stop:
            self.extras["replaced_action"] = torch.clone(actions)
            ############### when the end-effector and the object 's distance is lower than 0.6m , the action will not to be applied. ############
            self.extras["replaced_action"][self.base_obj_dis < 0.6, -(cmd_dim+pred_dim):-pred_dim] = 0.0 # enforced these cmd to be 0
            # if not self.enable_camera:
            actions = self.extras["replaced_action"]
        
        # Randomly change the object position in a small probability (like 0.1)
        obj_move_prob = torch_rand_float(0, 1, (self.num_envs, 1), device=self.device).squeeze()
        changed_env_ids = torch.range(0, self.num_envs-1, dtype=int, device=self.device)[obj_move_prob < self.obj_move_prob] # This is a tensor with the length of env_nums
        self.now_env_ids = changed_env_ids
        self._reset_objs(changed_env_ids)

        self.extras["lifted_now"] = self.lifted_now.unsqueeze(-1)*2-1 # This is for the lifted results from the last step, exactly what we want. Lifted = 1, unlifted = -1
        res = super().step(actions)

        pred_lift = actions[...,-1] > 0
        if self.pred_success and (actions.shape[-1] == (self.action_space.shape[-1]+1)):
            pred_true = (self.lifted_now == pred_lift)
            self.predlift_success_counter = torch.where(pred_true, self.predlift_success_counter + 1, self.predlift_success_counter)

        ####### obeserve the velocity ###############
        # _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # rb_states = gymtorch.wrap_tensor(_rb_states)
        ####################################################
        
        return res
    
    def check_termination(self):
        super().check_termination()

        # Check if lifted
        cube_height = self._cube_root_states[:, 2]
        box_pos = self._cube_root_states[:, :3]
        d1 = torch.norm(box_pos - self.ee_pos, dim=-1)
        self.lifted_now = torch.logical_and((cube_height - self.table_heights) > (0.03 / 2 + self.lifted_success_threshold), d1 < 0.2)
        self.reset_buf = torch.where(~self.lifted_now & self.lifted_object, torch.ones_like(self.reset_buf), self.reset_buf) # reset the dropped envs
        self.lifted_object = torch.logical_and((cube_height - self.table_heights - self.init_height) > (self.lifted_success_threshold), d1 < 0.2)

        z_cube = self._cube_root_states[:, 2]
        # cube_falls = (z_cube < (self.table_heights + 0.03 / 2 - 0.05))
        cube_falls = z_cube+0.015 < self.table_heights # Fall or model glitch
        self.reset_buf[:] = self.reset_buf | cube_falls
        
        if self.enable_camera:
            robot_head_dir = quat_apply(self.base_yaw_quat, torch.tensor([1., 0., 0.], device=self.device).repeat(self.num_envs, 1))
            cube_dir = self._cube_root_states[:, :3] - self._robot_root_states[:, :3]
            cube_dir[:, 2] = 0
            cube_dir = cube_dir / torch.norm(cube_dir, dim=-1).unsqueeze(-1)
            # check if dot product is negative
            deviate_much = torch.sum(robot_head_dir * cube_dir, dim=-1) < 0. 
            fov_camera_pos = self._robot_root_states[:, :3] + quat_apply(self._robot_root_states[:, 3:7], torch.tensor(self.cfg["sensor"]["onboard_camera"]["position"], device=self.device).repeat(self.num_envs, 1))
            too_close_table = (fov_camera_pos[:, 0] > 0.)
            
            self.reset_buf = self.reset_buf | deviate_much | too_close_table

    # --------------------------------- reward functions ---------------------------------
    
    def _reward_limit_yaw_rotation_penalty(self):
        reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.current_eular_angles = quaternion_tensor_to_euler(self._robot_root_states[:,3:7])
        # print("self.current_eular_angles is ",self.current_eular_angles)
        self.current_yaw = self.current_eular_angles[:,0]
        delta_yaw = torch.abs(self.current_yaw - self.initial_yaw_z)
        # print("delta_yaw",delta_yaw)
        delta_yaw_angle = torch.deg2rad(delta_yaw)
        # Convert 20 degrees to radians
        # if self.global_step_counter <= self.total_timesteps/4:
        #     angle_threshold = torch.deg2rad(torch.tensor(40.0))
        #     angle_reset_threshold = torch.deg2rad(torch.tensor(50.0))
        # elif self.global_step_counter <= self.total_timesteps/2:
        #     angle_threshold = torch.deg2rad(torch.tensor(50.0))
        #     angle_reset_threshold = torch.deg2rad(torch.tensor(60.0))
        # elif self.global_step_counter <= self.total_timesteps*3/4:
        #     angle_threshold = torch.deg2rad(torch.tensor(60.0))
        #     angle_reset_threshold = torch.deg2rad(torch.tensor(70.0))
        # else:
        #     angle_threshold = torch.deg2rad(torch.tensor(70.0))
        #     angle_reset_threshold = torch.deg2rad(torch.tensor(80.0))
        angle_threshold = torch.deg2rad(torch.tensor(60.0))
        angle_reset_threshold = torch.deg2rad(torch.tensor(70.0))
        # Update reward based on the condition
        mask = delta_yaw > angle_threshold
        reward[mask] = torch.tanh(delta_yaw[mask] )
        reward[mask] *= -1
        self.reset_buf[delta_yaw > angle_reset_threshold] = 1

        return reward,reward

    # def _reward_gripper_grasp_expect_reward(self):
    #     reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
    #     mask = self.curr_dist < 0.1
    #     torch.abs(self._cube_root_states[:, 8]) > 0.1
    #     return reward,reward

    # def _reward_standpick(self):    # for when the ee approach the cube and encourage the robot to stand statically to grasp, but may not be suitable for aynamic grasping
    #     reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
    #     reward[(self.base_obj_dis < self.base_object_distace_threshold) & (self.commands[:, 0] < LIN_VEL_X_CLIP)] = 1.0
    #     if self.global_step_counter < self.train_reward_strict:
    #         reward = 0.
    #     return reward, reward
    
    def _reward_grasp_base_height(self):
        cube_height = self._cube_root_states[:, 2]
        box_pos = self._cube_root_states[:, :3]
        d1 = torch.norm(box_pos - self.ee_pos, dim=-1)
        
        reward, _ = self._reward_base_height()
        reward *= self.lifted_now # no reward for not grasped
        
        return reward, reward
    
    def _reward_approaching(self):
        """Change the reward function to be effective only when the object is lifted
        """
        reward, _ = super()._reward_approaching()
        reward *= ~self.lifted_object
        return reward, reward
    
    def _reward_base_approaching(self):
        rew, _ = super()._reward_base_approaching(self._cube_root_states[:, :3])
        return rew, rew
    
    def _reward_command_reward(self):
        rew, _ = super()._reward_command_reward(self._cube_root_states[:, :3])
        return rew, rew
    
    # def _reward_command_penalty(self):
    #     rew, _ = super()._reward_command_penalty(self._cube_root_states[:, :3])
    #     return rew, rew
    
    def _reward_ee_orn(self):
        rew, _ = super()._reward_ee_orn(self._cube_root_states[:, :3])
        return rew, rew
    
    def _reward_base_dir(self):
        rew, _ = super()._reward_base_dir(self._cube_root_states[:, :3])
        return rew, rew
    
    # --------------------------------- reward functions ---------------------------------
    


# --------------------------------- jit functions ---------------------------------

@torch.jit.script
def compute_robot_observations(robot_root_state, table_root_state, cube_root_state, body_pos, 
                               body_rot, body_vel, body_ang_vel, dof_pos, dof_vel, base_quat_yaw, spherical_center, commands, gripper_idx, table_dim, 
                               ee_goal_cart, ee_goal_orn_rpy, ee_goal_global_cart, ee_goal_orn_quat, use_roboinfo, floating_base) :
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, Tensor, Tensor, Tensor, Tensor, Tensor,bool, bool) ->  torch.Tensor
    cube_pos = cube_root_state[:, :3]
    cube_orn = cube_root_state[:, 3:7]
    cube_rpy = torch.stack(euler_from_quat(cube_orn), dim=-1) # rot-squen: x-y-z
    cube_vel = cube_root_state[:, 7:10]
    cube_angular = cube_root_state[:, 10:]
    car_mainbody_vel = table_root_state[:, 7:10]
    car_mainbody_angular = table_root_state[:, 10:]
    
    ee_pos = body_pos[..., gripper_idx, :]
    ee_rot = body_rot[..., gripper_idx, :]
    ee_vel = body_vel[..., gripper_idx, :]
    ee_ang_vel = body_ang_vel[..., gripper_idx, :]
    # box pos and orientation  3+4=7
    # dof pos + vel  6+6=12
    # ee state  13
    if use_roboinfo:
        dof_pos = dof_pos[..., :]
        dof_vel = dof_vel[..., :-1] * 0.05
        if not floating_base:
            dof_pos = reindex_all(dof_pos)
            dof_vel = reindex_all(dof_vel)
    else:
        dof_pos = dof_pos[..., 12:-1] # arm_dof_pos
        dof_vel = dof_vel[..., 12:-1] # arm_dof_vel
    
    base_quat = robot_root_state[:, 3:7]
    arm_base_local = torch.tensor([0.3, 0.0, 0.09], device=robot_root_state.device).repeat(robot_root_state.shape[0], 1) ## have to change if change robot model
    arm_base = quat_apply(base_quat, arm_base_local) + robot_root_state[:, :3]
    
    # cube_pos_local = quat_rotate_inverse(base_quat, cube_pos - arm_base)
    # cube_orn_local = quat_mul(quat_conjugate(base_quat), cube_orn)
    # cube_pos_local = quat_rotate_inverse(base_quat_yaw, cube_pos - spherical_center)
    cube_pos_local = quat_rotate_inverse(base_quat_yaw, cube_pos - arm_base) # This represents the coordinates of the cube's position in the local coordinate system of the robot arm's base.
    cube_pos_local[:, 2] = cube_pos[:, 2]
    cube_orn_local = quat_mul(quat_conjugate(base_quat_yaw), cube_orn)
    cube_orn_local_rpy = torch.stack(euler_from_quat(cube_orn_local), dim=-1) # rot-squen: x-y-z
    cube_vel_local = quat_rotate_inverse(base_quat_yaw, cube_vel)
    cube_angular_local = quat_rotate_inverse(base_quat_yaw, cube_angular) 
    car_mainbody_vel_local = quat_rotate_inverse(base_quat_yaw, car_mainbody_vel)
    car_mainbody_angular_local = quat_rotate_inverse(base_quat_yaw, car_mainbody_angular)
    car_mainbody_vel_local[...,0] = -car_mainbody_vel[...,1]
    car_mainbody_vel_local[...,1] = car_mainbody_vel[...,0]
    car_mainbody_angular_local[...,0] = -car_mainbody_angular[...,1]
    car_mainbody_angular_local[...,0] = car_mainbody_angular[...,0]
    
    table_pos_local = quat_rotate_inverse(base_quat_yaw, table_root_state[:, :3] - spherical_center)
    table_orn_local = quat_mul(quat_conjugate(base_quat_yaw), table_root_state[:, 3:7])
    table_dim_local = quat_rotate_inverse(base_quat_yaw, table_dim)

    ee_pos_local = quat_rotate_inverse(base_quat, ee_pos - arm_base)
    ee_rot_local = quat_mul(quat_conjugate(base_quat), ee_rot)
    ee_rot_local_rpy = torch.stack(euler_from_quat(ee_rot_local), dim=-1)
    
    robot_vel_local = quat_rotate_inverse(base_quat_yaw, robot_root_state[:, 7:10])
    cube_vel_local = cube_vel_local - robot_vel_local
    cube_vel_local = cube_vel_local[...,:2]
    
    obs = torch.cat((cube_pos_local, cube_orn_local_rpy, ee_pos_local, ee_rot_local_rpy, dof_pos, dof_vel,
                     commands, ee_goal_cart, ee_goal_orn_rpy, robot_vel_local ,cube_vel_local ,cube_pos, cube_rpy), dim=-1) ## 61 + 6
    obs_have_cube_vel = torch.cat((cube_pos_local, cube_orn_local_rpy, car_mainbody_vel_local, car_mainbody_angular_local, ee_pos_local, ee_rot_local_rpy, dof_pos, dof_vel,
                     commands, ee_goal_cart, ee_goal_orn_rpy, robot_vel_local), dim=-1) # if have robo_info: have total 67 dim, if not, have 67-24 dim

    return obs
