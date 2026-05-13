import torch
import torch.nn as nn
import timm

class ViTEncoder(nn.Module):
    """用 ViT 提取人脸图像特征，作为 DiT 的条件输入。"""
    def __init__(self, vit_name="vit_small_patch16_224", feat_dim=384, cond_dim=256, pretrained=True):
        super().__init__()
        try:
            self.vit = timm.create_model(vit_name, pretrained=pretrained, num_classes=0)
        except Exception:
            print(f"[警告] 无法加载预训练 ViT，使用随机初始化")
            self.vit = timm.create_model(vit_name, pretrained=False, num_classes=0)
        self.feat_dim = feat_dim
        self.cond_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, cond_dim),
        )

    def forward(self, img):
        """
        Args:
            img: [B, 3, H, W] 归一化到 ImageNet 均值和方差
        Returns:
            cond: [B, cond_dim] 图像条件向量
        """
        feat = self.vit.forward_features(img)  # [B, N, D] or [B, D]
        if feat.dim() == 3:
            feat = feat[:, 0]  # CLS token
        return self.cond_head(feat)
