import gym
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *
from isaacgym import gymtorch
import torch
import math
import torch
import time

# 初始化仿真
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.dt = 1 / 60  # 仿真时间步长
sim_params.substeps = 2  # 子步数
sim_params.up_axis = gymapi.UP_AXIS_Z  # Z轴为向上方向
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)  # 重力
sim_params.use_gpu_pipeline = True  # 使用GPU加速
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# 设置环境网格
spacing = 5.0  # 环境间距
lower = gymapi.Vec3(-spacing, 0.0, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)
num_per_row = 1  # 每行的环境数量
num_env = 1
# 创建环境
env = gym.create_env(sim, lower, upper, num_per_row)

# 加载资产
asset_root = "/home/young/RL_project/visual_wholebody-main/high-level/data/asset/car/urdf"  # 资产文件夹路径
asset_file = "car.urdf"  # 资产文件名（URDF或SDF文件）
asset_options = gymapi.AssetOptions()
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_VEL
asset_options.collapse_fixed_joints = True
asset_options.replace_cylinder_with_capsule = True
asset_options.flip_visual_attachments = False
asset_options.fix_base_link = False
asset_options.density = 300000.0
asset_options.angular_damping = 0.
asset_options.linear_damping = 0.
asset_options.max_angular_velocity = 1000.
asset_options.max_linear_velocity = 1000.
asset_options.armature = 0.
asset_options.thickness = 0.1
asset_options.disable_gravity = False
asset_options.use_mesh_materials = True
asset_options.override_com = True
asset_options.override_inertia = True
# asset_options.vhacd_enabled = True
# asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
num_dofs = gym.get_asset_dof_count(robot_asset)
num_bodies = gym.get_asset_rigid_body_count(robot_asset)
dof_props_asset = gym.get_asset_dof_properties(robot_asset)

rigid_shape_props_asset = gym.get_asset_rigid_shape_properties(robot_asset)
for i in range(len(rigid_shape_props_asset)):
    rigid_shape_props_asset[i].friction = 1.0
gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props_asset)

# 设置物体的初始位置和姿态
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0, 0.0)  # 初始位置
pose.r = gymapi.Quat(0, 0, 0, 1)  # 初始姿态（无旋转）

# 创建动态物体
car_actor_handle = gym.create_actor(env, robot_asset, pose, "dynamic_object", 0, 0)

props = gym.get_actor_dof_properties(env, car_actor_handle)
props["driveMode"] = gymapi.DOF_MODE_VEL
props["stiffness"][:].fill(0)
props["damping"][:].fill(150)
gym.set_actor_dof_properties(env, car_actor_handle, props)

box_dims = gymapi.Vec3(0.2, 0.2, 0.2)
box_options = gymapi.AssetOptions()
box_options.fix_base_link = False
box_options.disable_gravity = False
box_options.thickness = 0.1
box_asset = gym.create_box(sim, box_dims.x,box_dims.y, box_dims.z, box_options)
rigid_shape_props_asset = gym.get_asset_rigid_shape_properties(box_asset)
for i in range(len(rigid_shape_props_asset)):
    rigid_shape_props_asset[i].friction = 5.0
gym.set_asset_rigid_shape_properties(box_asset, rigid_shape_props_asset)
box_start_pose = gymapi.Transform()
box_start_pose.p = gymapi.Vec3(0.0,0.0,0.4)
box_start_pose.r = gymapi.Quat(0, 0, 0, 1)
box_handle = gym.create_actor(env,box_asset, box_start_pose, "box", 0, 0)
color_3_vec = gymapi.Vec3(100,50,80)
gym.set_rigid_body_color(env,box_handle,0,gymapi.MESH_VISUAL,color_3_vec)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
plane_params.static_friction = 1.0
plane_params.dynamic_friction = 1.0
plane_params.restitution = 0.0
# gym.add_ground(sim, plane_params)
gym.add_ground(sim, plane_params)


gym.prepare_sim(sim)
dof_state = gym.acquire_dof_state_tensor(sim)
dof_state = gymtorch.wrap_tensor(dof_state)
dof_vel = dof_state.view(1,4,2)[:,:,1]
_root_tensor = gym.acquire_actor_root_state_tensor(sim)
_rigid_body_state = gym.acquire_rigid_body_state_tensor(sim)
rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state)
# rigid_body_state_reshaped = rigid_body_state.view(2,-1)
vel = rigid_body_state[5,7:10]
# wheel_vel[:,:,1] = 0

wheel_vel = -2.5*torch.ones((num_env,num_dofs),device="cuda:0")
wheel_vel[...,:2] = -1.8

start_time = time.time()

# gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(wheel_vel))
# forces = torch.zeros((1, 5, 3), device="cuda:0", dtype=torch.float)
# torques = torch.zeros((1, 5, 3), device="cuda:0", dtype=torch.float)
# forces[:, 0, 2] = 1000           # 27 is the cube_index , 26 is the box_index
# torques[:, 4, 2] = 0.0
# gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)


while not gym.query_viewer_has_closed(viewer):
    # wheel_vel = -10*torch.ones((num_env,num_dofs,1),device="cuda:0")
    gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(wheel_vel))
    # print("vel is",vel)
    time_now = time.time()
    dt = time_now - start_time
    if(dt >= 20 and dt <=21):
        print("vel is",vel)

    
    # print(dof_vel)
    # gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)

    
    # Step the simulation
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)

    # Wait for dt to elapse in real time (synchronizes the physics simulation with the rendering rate)
    gym.sync_frame_time(sim)

# Cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

# 仿真循环
# while True:
#     gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(wheel_vel))

#     # 更新仿真状态
#     gym.simulate(sim)
#     gym.fetch_results(sim, True)

#     # 可选：更新物体的目标位置或速度
#     # 示例：逐步更新位置目标
#     # pos_targets += 0.01
#     # gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_targets))

# # 清理资源
# gym.destroy_actor(actor_handle)
# gym.destroy_env(env)
# gym.destroy_sim(sim)
