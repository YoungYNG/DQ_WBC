import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from predictpoint_mul import PredictPoint

# HGGD_cube_predict_info_path = "HGGD_29obj_img4_Results2_cancel"
# cube_root_states_info_path = "22all_cube_root_states4.pt"
contact_grasp_info_path = "contact_grasp_info_mul"
cube_root_states_info_path = "30all_nomove_cube_root_states5.pt"

num_env = 50
intervel = 20
delta_height = 0.1

curr_dir = os.getcwd()
data_dir = os.path.join(curr_dir, "transform_info")
cube_root_states_info_path = os.path.join(data_dir, cube_root_states_info_path)
save_dir = os.path.join(data_dir, "grasp_videos")
os.makedirs(save_dir, exist_ok=True)

# 加载物体状态（position + quaternion + velocity...）
cube_states = torch.load(cube_root_states_info_path)[0].cpu()
# print(f"cube_states.shape is {cube_states.shape}")
# print(f"cube_states.type is {cube_states.dtype}")

cube_pos = cube_states[:, :3]
cube_quat = cube_states[:, 3:7]
cube_rpy = torch.tensor(R.from_quat(cube_quat.numpy()).as_euler('xyz'))  # 转为 RPY
cube_init_tensor = torch.cat([cube_pos, cube_rpy], dim=1)
cube_init_tensor = cube_init_tensor.to(torch.float32)
# print(f"cube_init_tensor.dtype is {cube_init_tensor.dtype}")
# print(f"cube_pos is {cube_pos[0]}, cube_rpy is {cube_rpy[0]}")

base_n = cube_init_tensor.shape[0]  # 原始的n个样本数
T_grasp_cv_all = []

# 读取 base_n 个 sample 的 n 个抓取点 transform
for i in range(base_n):
    sample_dir = os.path.join(data_dir, contact_grasp_info_path)
    json_path = os.path.join(sample_dir, f"predictions_image_{(i):02d}.json")
    with open(json_path, 'r') as f:
        data = json.load(f)
    grasp_matrices = np.array(data['transform'])  # 假设形状 [n, 4, 4]
    T_grasp_cv_all.append(grasp_matrices)

# 转换为张量，形状 [base_n, n, 4, 4]
T_grasp_cv_all = np.stack(T_grasp_cv_all, axis=0)
n_grasps = T_grasp_cv_all.shape[1]  # 获取 n（每个样本的 transform 数量）
T_grasp_cv_all = torch.tensor(T_grasp_cv_all, dtype=torch.float32)

# 按需扩展到 num_env
T_grasp_cv_list = [T_grasp_cv_all[i % base_n] for i in range(num_env)]
grasp_cv_tensor = torch.stack(T_grasp_cv_list, dim=0)  # 形状 [num_env, n, 4, 4]
# print(f"grasp_cv_tensor is ",grasp_cv_tensor[:,0,0])

# 相机和物体状态初始化
camera_tensor = []
cube_init_list = []
cube_init_ori_list = []

# position: [0.0, -0.30, 0.80]
# rotation: [0.0, 0.70, 1.57]

for i in range(num_env):
    base_idx = i % base_n
    height_offset = (delta_height / intervel) * ((i % intervel) + 1)
    # print(f"i is {i} ; height_offset is {height_offset}")
    # height_offset = 0

    # 相机位置（z方向高度偏移）
    camera_x = -0.3
    camera_y = 2.0
    camera_z = 0.80 + height_offset
    camera_info = [camera_x, camera_y, camera_z, 0.0, 0.70, 0.0]
    # 为每个 n 复制相机信息
    camera_tensor.append([camera_info] * n_grasps)

    # 物体初始状态位置（z方向高度偏移）
    cube_init = cube_init_tensor[base_idx].clone()
    cube_init[2] += height_offset  # z 增加
    # 为每个 n 复制物体信息
    cube_init_list.append([cube_init] * n_grasps)
    cube_init_ori_list.append(cube_init)

# 转换为张量
camera_tensor = torch.tensor(camera_tensor, dtype=torch.float32)  # 形状 [num_env, n, 6]
cube_init_tensor = torch.stack([torch.stack(cube_init, dim=0) for cube_init in cube_init_list], dim=0)  # 形状 [num_env, n, 6]
# print(f"camera_tensor is {camera_tensor} , cube_init_tensor is {cube_init_tensor}")

# print(f"grasp_cv_tensor is {grasp_cv_tensor.shape} ; camera_tensor is {camera_tensor.shape} ; cube_init_tensor is {cube_init_tensor.shape}")
# print(f"grasp_cv_tensor is {grasp_cv_tensor[:,0,0]} ; camera_tensor is {camera_tensor[:,2]} ; cube_init_tensor is {cube_init_tensor[:,2]}")
# print(f"grasp_cv_tensor.dtype is {grasp_cv_tensor.dtype} ; camera_tensor.dtype is {camera_tensor.dtype} ; cube_init_tensor.dtype is {cube_init_tensor.dtype}")
print(f"grasp_cv_tensor,shape is {grasp_cv_tensor.shape},grasp_cv_tensor is {grasp_cv_tensor[21]}")

predictor = PredictPoint(camera_tensor, grasp_cv_tensor, cube_init_tensor)

cube_init_ori_tensor = torch.stack(cube_init_ori_list, dim=0)
env_ids = torch.ones(50,dtype=torch.bool)
xyz,rpy = predictor.get_grasp_world(cube_init_ori_tensor,env_ids)
# print(f"xyz is {xyz.shape} , rpy0 is {rpy[0,0]} , rpy1 is {rpy[0,1]} , rpy3 is {rpy[0,2]} , rpy3 is {rpy[0,3]} , rpy3 is {rpy[0,4]} ,  ")
print(f"xyz is {xyz.shape} , rpy0 is {rpy[21]} ")