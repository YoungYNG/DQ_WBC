import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple

class PredictPoint:
    def __init__(self, camera_tensor, grasp_tensor, obj_init_tensor, device="cpu"):
        """
        camera_tensor: Tensor of shape (N, n, 6) - [pos(3), rpy(3)] in world frame
        grasp_tensor: Tensor of shape (N, n, 4, 4) - grasp transform relative to CV camera
        obj_init_tensor: Tensor of shape (N, n, 6) - [pos(3), rpy(3)] in world frame
        """
        if torch.isnan(grasp_tensor).any() or torch.isinf(grasp_tensor).any():
            raise ValueError("NaN or inf in grasp_tensor")

        # 检查 grasp_tensor 的旋转部分
        for i in range(grasp_tensor.shape[0]):
            for j in range(grasp_tensor.shape[1]):
                R_mat = grasp_tensor[i, j, :3, :3]
                identity = torch.matmul(R_mat.T, R_mat)
                if not torch.allclose(identity, torch.eye(3, device=R_mat.device), atol=1e-3):
                    print("Non-orthogonal grasp_cv rotation:", R_mat)
                    raise ValueError("grasp_cv rotation is not orthogonal")

        self.device = device
        self.N = camera_tensor.shape[0]
        self.n = camera_tensor.shape[1]  # n 个抓取点

        # 将输入转为设备上的 tensor
        self.camera_pos = camera_tensor[:, :, :3].to(device)  # [N, n, 3]
        self.camera_rpy = camera_tensor[:, :, 3:].to(device)  # [N, n, 3]
        self.grasp_cv = grasp_tensor.to(device)  # [N, n, 4, 4]
        self.obj_pos_init = obj_init_tensor[:, :, :3].to(device)  # [N, n, 3]
        self.obj_rpy_init = obj_init_tensor[:, :, 3:].to(device)  # [N, n, 3]

        # CV to Isaac 坐标系变换矩阵
        self.T_cv_to_isaac = torch.tensor([
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32, device=self.device)

        self.rpy_cv_to_isaac_change = torch.tensor([
            [0, -1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)

        self.Rx_rot = torch.tensor([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ], dtype=torch.float32, device=self.device)

        self.Rz_rot = torch.tensor([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)

        # 预计算 grasp 相对于 object 的固定变换
        print(f"!!!!!!!! start to initialize predictpoint !!!!!!!!!!!! ")
        self.T_grasp_in_obj = []
        for i in range(self.N):
            T_grasp_in_obj_i = []
            for j in range(self.n):
                T_cam_world = self._make_transform(self.camera_pos[i, j], self.camera_rpy[i, j])
                T_grasp_cam = self.T_cv_to_isaac @ self.grasp_cv[i, j]
                T_grasp_world = T_cam_world @ T_grasp_cam

                T_obj_world = self._make_transform(self.obj_pos_init[i, j], self.obj_rpy_init[i, j])
                T_obj_world_inv = self._invert_transform(T_obj_world)

                T_grasp_obj = T_obj_world_inv @ T_grasp_world

                # 检查 T_grasp_obj 的旋转部分
                R_mat = T_grasp_obj[:3, :3]
                identity = torch.matmul(R_mat.T, R_mat)
                if not torch.allclose(identity, torch.eye(3, device=R_mat.device), atol=1e-3):
                    print("Non-orthogonal T_grasp_obj rotation:", R_mat)
                    print("camera_rpy:", self.camera_rpy[i, j], "obj_rpy_init:", self.obj_rpy_init[i, j])
                    raise ValueError("T_grasp_obj rotation is not orthogonal")

                T_grasp_in_obj_i.append(T_grasp_obj)
            self.T_grasp_in_obj.append(torch.stack(T_grasp_in_obj_i, dim=0))

        self.T_grasp_in_obj = torch.stack(self.T_grasp_in_obj, dim=0)  # [N, n, 4, 4]
        print(f"!!!!!!!! finish initialize predictpoint !!!!!!!!!!!! ")


    def _invert_transform(self, T):
        R_mat = T[:3, :3]
        t = T[:3, 3]
        T_inv = torch.eye(4, dtype=torch.float32, device=self.device)
        T_inv[:3, :3] = R_mat.T
        T_inv[:3, 3] = -R_mat.T @ t
        return T_inv

    def _make_transform(self, pos, rpy):
        R_mat = self._rpy_to_matrix(rpy)
        T = torch.eye(4, dtype=torch.float32, device=self.device)
        T[:3, :3] = R_mat
        T[:3, 3] = pos
        return T

    def _batch_make_transform(self, pos_tensor, rpy_tensor):
        """
        将 batch 的 [pos(3), rpy(3)] 转换为 batch 的变换矩阵 T (B, n, 4, 4)
        pos_tensor: (B, 3) or (B, n, 3)
        rpy_tensor: (B, 3) or (B, n, 3)
        返回: (B, 4, 4) or (B, n, 4, 4)
        """
        assert pos_tensor.shape[0] == rpy_tensor.shape[0], "Position and RPY batch size must match"
        B = pos_tensor.shape[0]

        # 处理 n 维度
        if pos_tensor.ndim == 3:  # [B, n, 3]
            n = pos_tensor.shape[1]
            pos_tensor = pos_tensor.view(B * n, 3)
            rpy_tensor = rpy_tensor.view(B * n, 3)
            rot_mats = self._rpy_to_matrix(rpy_tensor)  # [B*n, 3, 3]
            rot_mats = rot_mats.view(B, n, 3, 3)
            T = torch.eye(4, device=pos_tensor.device, dtype=pos_tensor.dtype).unsqueeze(0).unsqueeze(0).repeat(B, n, 1, 1)
            T[:, :, :3, :3] = rot_mats
            T[:, :, :3, 3] = pos_tensor.view(B, n, 3)
        else:  # [B, 3]
            rot_mats = self._rpy_to_matrix(rpy_tensor)
            T = torch.eye(4, device=pos_tensor.device, dtype=pos_tensor.dtype).unsqueeze(0).repeat(B, 1, 1)
            T[:, :3, :3] = rot_mats
            T[:, :3, 3] = pos_tensor
        return T

    def _rpy_to_matrix(self, rpy_tensor):
        if rpy_tensor.ndim == 1:
            rpy_tensor = rpy_tensor.unsqueeze(0)
        roll, pitch, yaw = rpy_tensor[:, 0], rpy_tensor[:, 1], rpy_tensor[:, 2]

        cos_r = torch.cos(roll)
        sin_r = torch.sin(roll)
        cos_p = torch.cos(pitch)
        sin_p = torch.sin(pitch)
        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)

        Rz = torch.stack([
            torch.stack([cos_y, -sin_y, torch.zeros_like(yaw)], dim=-1),
            torch.stack([sin_y, cos_y, torch.zeros_like(yaw)], dim=-1),
            torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)], dim=-1)
        ], dim=-2)

        Ry = torch.stack([
            torch.stack([cos_p, torch.zeros_like(pitch), sin_p], dim=-1),
            torch.stack([torch.zeros_like(pitch), torch.ones_like(pitch), torch.zeros_like(pitch)], dim=-1),
            torch.stack([-sin_p, torch.zeros_like(pitch), cos_p], dim=-1)
        ], dim=-2)

        Rx = torch.stack([
            torch.stack([torch.ones_like(roll), torch.zeros_like(roll), torch.zeros_like(roll)], dim=-1),
            torch.stack([torch.zeros_like(roll), cos_r, -sin_r], dim=-1),
            torch.stack([torch.zeros_like(roll), sin_r, cos_r], dim=-1)
        ], dim=-2)

        rot_mats = Rz @ Ry @ Rx
        return rot_mats

    def get_grasp_world(self, obj_tensor, env_ids):
        """
        输入:
            obj_tensor: Tensor of shape [len(env_ids), 6] - [pos(3), rpy(3)] in world frame
            env_ids: List or tensor of environment indices
        返回:
            xyz: Tensor of shape [len(env_ids), n, 3] - grasp positions
            rpy: Tensor of shape [len(env_ids), n, 3] - grasp orientations (xyz order)
        """
        xyz_tensor, rpy_tensor = obj_tensor[:, :3], obj_tensor[:, 3:]  # [len(env_ids), 3]
        T_obj_world = self._batch_make_transform(xyz_tensor, rpy_tensor)  # [len(env_ids), 4, 4]
        T_grasp_in_obj = self.T_grasp_in_obj[env_ids]  # [len(env_ids), n, 4, 4]
        T_grasp_world = T_obj_world.unsqueeze(1) @ T_grasp_in_obj  # [len(env_ids), n, 4, 4]
        xyz, rpy = self.decompose_pose_to_xyz_rpy(T_grasp_world, order="xyz")  # [len(env_ids), n, 3]
        return xyz, rpy

    def decompose_pose_to_xyz_rpy(self, T, order="xyz"):
        """
        输入:
            T: Tensor [..., n, 4, 4]，位姿矩阵
            order: "xyz" 或 "zyx"，表示旋转顺序
        返回:
            xyz: 平移 [..., n, 3]
            rpy: 旋转欧拉角 [..., n, 3]
        """
        batch_shape = T.shape[:-2]  # [..., n]
        T = T.view(-1, 4, 4)  # [..., n, 4, 4] -> [N*n, 4, 4]
        xyz = T[:, :3, 3]  # [N*n, 3]
        # R = T[:, :3, :3]  # [N*n, 3, 3]
        R =  T[:, :3, :3] @ self.Rx_rot @ self.Rz_rot @ self.Rx_rot


        if order == "xyz":
            input_asin = torch.clamp(-R[:, 2, 0], -1.0, 1.0)
            pitch = torch.asin(input_asin)
            cos_pitch = torch.cos(pitch)
            gimbal_lock = torch.abs(cos_pitch) < 1e-6

            rpy = torch.zeros((R.shape[0], 3), device=R.device, dtype=R.dtype)

            not_gimbal_lock = ~gimbal_lock
            if not_gimbal_lock.any():
                roll = torch.atan2(R[not_gimbal_lock, 2, 1], R[not_gimbal_lock, 2, 2])
                yaw = torch.atan2(R[not_gimbal_lock, 1, 0], R[not_gimbal_lock, 0, 0])
                rpy[not_gimbal_lock, 0] = roll
                rpy[not_gimbal_lock, 1] = pitch[not_gimbal_lock]
                rpy[not_gimbal_lock, 2] = yaw

            if gimbal_lock.any():
                roll = torch.atan2(-R[gimbal_lock, 0, 1], R[gimbal_lock, 0, 2])
                rpy[gimbal_lock, 0] = roll
                rpy[gimbal_lock, 1] = pitch[gimbal_lock]
                rpy[gimbal_lock, 2] = 0.0

        elif order == "zyx":
            input_asin = torch.clamp(R[:, 0, 2], -1.0, 1.0)
            pitch = torch.asin(-input_asin)
            cos_pitch = torch.cos(pitch)
            gimbal_lock = torch.abs(cos_pitch) < 1e-6

            rpy = torch.zeros((R.shape[0], 3), device=R.device, dtype=R.dtype)

            not_gimbal_lock = ~gimbal_lock
            if not_gimbal_lock.any():
                yaw = torch.atan2(R[not_gimbal_lock, 1, 0], R[not_gimbal_lock, 0, 0])
                roll = torch.atan2(R[not_gimbal_lock, 2, 1], R[not_gimbal_lock, 2, 2])
                rpy[not_gimbal_lock, 0] = roll
                rpy[not_gimbal_lock, 1] = pitch[not_gimbal_lock]
                rpy[not_gimbal_lock, 2] = yaw

            if gimbal_lock.any():
                yaw = torch.atan2(-R[gimbal_lock, 1, 2], R[gimbal_lock, 1, 1])
                rpy[gimbal_lock, 0] = 0.0
                rpy[gimbal_lock, 1] = pitch[gimbal_lock]
                rpy[gimbal_lock, 2] = yaw

        else:
            raise ValueError(f"Unsupported rotation order: {order}")

        xyz = xyz.view(*batch_shape, 3)  # [..., n, 3]
        rpy = rpy.view(*batch_shape, 3)  # [..., n, 3]
        return xyz, rpy