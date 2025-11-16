import torch
import torch.nn.functional as F
import numpy as np
import cv2
import random

def apply_depth_perturbation(depth, severity=0.5, device='cpu'):
    """
    深度扰动: 高斯噪声 + 随机缺测
    depth: (B, H, W)
    severity: 0~1 控制扰动强度
    """
    B, H, W = depth.shape

    # === 1. 加性高斯噪声 ===
    # 噪声标准差最大设为 10cm
    sigma = 0.1 * severity  # 单位米
    noise = torch.randn_like(depth, device=device) * sigma
    depth_noisy = depth + noise

    # === 2. 随机缺测(dropout) ===
    # 丢失概率 0~0.2
    drop_prob = 0.2 * severity
    dropout_mask = (torch.rand_like(depth, device=device) > drop_prob).float()
    depth_drop = depth_noisy * dropout_mask

    return depth_drop


def apply_segmentation_perturbation(seg, severity=0.5):
    """
    分割扰动: 边界膨胀/腐蚀 + 随机挖空
    seg: (B, H, W) 二值分割图 [0,1]
    severity: 0~1 控制扰动强度
    """
    seg_np = seg.cpu().numpy().astype(np.uint8)
    B, H, W = seg_np.shape

    perturbed = []
    for b in range(B):
        img = seg_np[b]

        # === 1. 边界类误差 ===
        kernel_size = int(1 + 4 * severity)  # 1~5 px
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        if np.random.rand() > 0.5:
            img = cv2.dilate(img, kernel, iterations=1)
        else:
            img = cv2.erode(img, kernel, iterations=1)

        # === 2. 随机挖空 ===
        hole_count = int(5 * severity + 1)
        for _ in range(hole_count):
            h = int(np.random.uniform(0.05, 0.2) * H * severity)
            w = int(np.random.uniform(0.05, 0.2) * W * severity)
            y = np.random.randint(0, H - h)
            x = np.random.randint(0, W - w)
            img[y:y+h, x:x+w] = 0

        perturbed.append(img)

    perturbed = torch.from_numpy(np.stack(perturbed)).float().to(seg.device)
    perturbed = (perturbed > 0.5).float()  # 重新二值化
    return perturbed


def perturb_depth_with_seg(depth, seg, mode='depth', severity=0.5, device='cpu'):
    """
    主函数：
    mode: 'depth' 或 'segmentation'
    输出: 分割区域内的深度图 (B, H, W)
    """
    depth = depth.clone().to(device)
    seg = seg.clone().to(device)

    if mode == 'depth':
        depth_perturbed = apply_depth_perturbation(depth, severity, device)
        seg_used = seg
    elif mode == 'segmentation':
        seg_used = apply_segmentation_perturbation(seg, severity)
        depth_perturbed = depth
    elif mode == 'seg_depth':
        depth_perturbed = apply_depth_perturbation(depth, severity, device)
        seg_used = apply_segmentation_perturbation(seg, severity)
    else:
        raise ValueError("mode must be 'depth' or 'segmentation'")

    # 最终输出：分割区域的深度信息
    output = depth_perturbed * seg_used
    return output, seg_used, depth_perturbed


def sample_perturbation_params():
    # 1. 随机选模式
    modes = ["segmentation", "depth", "seg_depth"]
    probs = [0.4, 0.4, 0.2]
    mode = random.choices(modes, probs)[0]
    
    # 2. 从 Beta 分布中采样 severity（范围 0~0.3，前小后大概率小）
    # severity = np.random.beta(a=2.0, b=8.0) * 0.3
    severity = np.random.beta(a=4.0, b=6.0) * 0.5
    return mode, severity

if __name__ == "__main__":
    B, H, W = 4, 256, 256
    depth = torch.rand(B, H, W) * 2.0   # 假设单位是米
    seg = (torch.rand(B, H, W) > 0.7).float()  # 二值mask

    # 1. 深度扰动（中等强度）
    out_depth, seg_used, depth_used = perturb_depth_with_seg(
        depth, seg, mode='depth', severity=0.5
    )

    # 2. 分割扰动（较强）
    out_seg, seg_used, depth_used = perturb_depth_with_seg(
        depth, seg, mode='segmentation', severity=0.8
    )

    print(out_depth.shape)  # torch.Size([B, H, W])
