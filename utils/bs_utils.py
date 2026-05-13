import torch
import numpy as np

# ARKit 标准 52 个 Blendshape 名称（按索引顺序）
ARKIT_BS_NAMES = [
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft",
    "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft", "eyeBlinkRight",
    "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight",
    "eyeLookUpRight", "eyeSquintRight", "eyeWideRight", "jawForward",
    "jawLeft", "jawRight", "jawOpen", "mouthClose", "mouthFunnel",
    "mouthPucker", "mouthLeft", "mouthRight", "mouthSmileLeft",
    "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper", "mouthPressLeft",
    "mouthPressRight", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "browDownLeft",
    "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft",
    "cheekSquintRight", "noseSneerLeft", "noseSneerRight",
    "tongueOut",
]


def bs_to_dict(bs_vector):
    """将 BS 向量转为带名称的字典。"""
    return {name: float(bs_vector[i]) for i, name in enumerate(ARKIT_BS_NAMES)}


def dict_to_bs(bs_dict, dim=52):
    """将带名称的字典转为 BS 向量。"""
    bs = np.zeros(dim, dtype=np.float32)
    for i, name in enumerate(ARKIT_BS_NAMES[:dim]):
        if name in bs_dict:
            bs[i] = bs_dict[name]
    return bs


def random_bs(batch_size=1, dim=52, device="cpu"):
    """
    生成随机 BS 系数（用于合成数据）。
    采样策略：合理范围内随机，使表情自然。
    """
    bs = torch.zeros(batch_size, dim, device=device)
    # 大部分 BS 在 [0, 1] 范围，少数在 [-1, 1]
    for i in range(dim):
        if i < 14:  # 眼部：偏向闭合/睁开
            bs[:, i] = torch.rand(batch_size, device=device) * 0.8
        elif i < 22:  # 嘴部基本动作
            bs[:, i] = (torch.rand(batch_size, device=device) - 0.5) * 0.6
        elif i < 42:  # 嘴部精细动作
            bs[:, i] = torch.rand(batch_size, device=device) * 0.5
        elif i < 47:  # 眉毛
            bs[:, i] = (torch.rand(batch_size, device=device) - 0.3) * 0.6
        else:  # 脸颊、鼻子等
            bs[:, i] = torch.rand(batch_size, device=device) * 0.4
    return bs.clamp(0, 1)


def empathy_transform(user_bs, intensity=1.0):
    """
    将用户 BS 转换为共情回应 BS 的规则基变换。
    Args:
        user_bs: [B, 52] 用户表情 BS
        intensity: 回应强度 (0~1)
    Returns:
        response_bs: [B, 52] 共情回应 BS
    """
    B = user_bs.shape[0]
    response = torch.zeros_like(user_bs)

    # 检测用户情绪线索
    smile = (user_bs[:, 23] + user_bs[:, 24]) / 2
    brow_frown = (user_bs[:, 41] + user_bs[:, 42]) / 2
    mouth_frown = (user_bs[:, 25] + user_bs[:, 26]) / 2
    surprise = (user_bs[:, 17] + user_bs[:, 6] + user_bs[:, 13]) / 3

    # ─── 开心共情 ───
    happy_mask = (smile > 0.2) & (brow_frown < 0.2)
    if happy_mask.any():
        idx = happy_mask
        response[idx, 23] = 0.3 * intensity  # mouthSmileLeft
        response[idx, 24] = 0.3 * intensity  # mouthSmileRight
        response[idx, 44] = 0.1 * intensity  # browOuterUpLeft
        response[idx, 45] = 0.1 * intensity  # browOuterUpRight

    # ─── 难过/悲伤共情 ───
    sad_mask = ((mouth_frown > 0.15) | (brow_frown > 0.25)) & (smile < 0.1)
    if sad_mask.any():
        idx = sad_mask
        response[idx, 25] = 0.1 * intensity  # mouthFrownLeft
        response[idx, 26] = 0.1 * intensity  # mouthFrownRight
        response[idx, 35] = 0.1 * intensity  # mouthPressLeft
        response[idx, 36] = 0.1 * intensity  # mouthPressRight
        response[idx, 41] = 0.2 * intensity  # browDownLeft
        response[idx, 42] = 0.2 * intensity  # browDownRight
        response[idx, 43] = 0.1 * intensity  # browInnerUp

    # ─── 惊讶共情 ───
    surprise_mask = surprise > 0.3
    if surprise_mask.any():
        idx = surprise_mask
        response[idx, 17] = 0.1 * intensity  # jawOpen
        response[idx, 44] = 0.2 * intensity  # browOuterUpLeft
        response[idx, 45] = 0.2 * intensity  # browOuterUpRight

    # ─── 中性：轻微微笑 ───
    neutral_mask = ~(happy_mask | sad_mask | surprise_mask)
    if neutral_mask.any():
        idx = neutral_mask
        response[idx, 23] = 0.075 * intensity  # mouthSmileLeft
        response[idx, 24] = 0.075 * intensity  # mouthSmileRight

    return response.clamp(0, 1)
