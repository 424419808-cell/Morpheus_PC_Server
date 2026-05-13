import torch
import torch.nn as nn
import torch.nn.functional as F


class EmpathyLoss(nn.Module):
    """
    共情模型损失函数组合：

    1. L1 Loss — BS 系数的绝对误差（主要监督）
    2. 余弦相似度 Loss — 保持 BS 整体轮廓形状
    3. 表情感知 Loss — 鼓励正确的表情语义方向
    """
    def __init__(self, l1_weight=1.0, cos_weight=0.5, emotion_weight=0.3):
        super().__init__()
        self.l1_weight = l1_weight
        self.cos_weight = cos_weight
        self.emotion_weight = emotion_weight

    def forward(self, pred_noise, target_noise, pred_bs=None, target_bs=None):
        """
        Args:
            pred_noise: [B, 52] 模型预测的噪声
            target_noise: [B, 52] 真实的噪声（训练 diffusion 时）
            pred_bs: [B, 52] 去噪后的 BS 预测（可选）
            target_bs: [B, 52] 目标 BS（可选）
        """
        # 扩散损失：MSE 噪声预测
        diffusion_loss = F.mse_loss(pred_noise, target_noise)

        total = self.l1_weight * diffusion_loss

        if pred_bs is not None and target_bs is not None:
            # L1
            l1 = F.l1_loss(pred_bs, target_bs)
            total += self.l1_weight * l1

            # 余弦相似度（BS 分布形状）
            cos_sim = F.cosine_similarity(pred_bs, target_bs, dim=1).mean()
            cos_loss = 1.0 - cos_sim
            total += self.cos_weight * cos_loss

        return total, {
            "diffusion": diffusion_loss.item(),
            "total": total.item(),
        }
