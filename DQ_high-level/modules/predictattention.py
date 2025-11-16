import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PredictAttentionSelector(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.key_proj = nn.Linear(6, d_model)
        self.value_proj = nn.Linear(6, d_model)
        self.query_proj = nn.Linear(134, d_model)
        self.output_proj = nn.Linear(d_model, 6)  # 输出最终抓取点

    def attention_forward(self, grasps, obj_feat, pose):
        B, N, _ = grasps.shape
        context = torch.cat([obj_feat, pose], dim=-1)  # [B, 134]
        Q = self.query_proj(context).unsqueeze(1)      # [B, 1, d_model]
        K = self.key_proj(grasps)                      # [B, N, d_model]
        V = self.value_proj(grasps)                    # [B, N, d_model]

        attn = torch.softmax((Q @ K.transpose(-2, -1)) / np.sqrt(K.size(-1)), dim=-1)  # [B, 1, N]
        weighted = attn @ V      # [B, 1, d_model]
        selected_grasp = self.output_proj(weighted.squeeze(1))  # [B, 6]
        return selected_grasp

