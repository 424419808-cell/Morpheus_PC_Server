import torch
import torch.nn as nn
from .vit_encoder import ViTEncoder
from .dit_decoder import DiTDecoder


class EmpathyModel(nn.Module):
    """
    端到端共情模型。
    输入：人脸图像
    输出：共情回应的 BS 系数

    训练时：ViT → cond → diffusion forward process (加噪→去噪)
    推理时：ViT → cond → diffusion reverse process (采样)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = ViTEncoder(
            vit_name=config.vit_name,
            feat_dim=config.vit_feat_dim,
            cond_dim=config.cond_dim,
            pretrained=config.vit_pretrained,
        )
        self.decoder = DiTDecoder(
            bs_dim=config.bs_dim,
            cond_dim=config.cond_dim,
            hidden_dim=config.dit_hidden_dim,
            n_layers=config.dit_n_layers,
            n_heads=config.dit_n_heads,
        )

        # 扩散超参数
        self.register_buffer("betas", torch.linspace(config.beta_start, config.beta_end, config.noise_steps))
        self.register_buffer("alphas", 1 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.noise_steps = config.noise_steps

        # Stage 1: BS→cond 投影（decoder_only_mode 用）
        self.bs_to_feat = nn.Sequential(
            nn.Linear(config.bs_dim, config.vit_feat_dim),
            nn.LayerNorm(config.vit_feat_dim),
        )

    def q_sample(self, bs, noise, t):
        """前向加噪过程：bs_t = sqrt(ᾱ_t) * bs_0 + sqrt(1-ᾱ_t) * noise"""
        sqrt_ac = self.alphas_cumprod[t].sqrt()[:, None]
        sqrt_one_minus_ac = (1 - self.alphas_cumprod[t]).sqrt()[:, None]
        return sqrt_ac * bs + sqrt_one_minus_ac * noise

    def forward(self, img, target_bs=None, return_loss=True):
        """
        训练模式：编码图像，对 target_bs 加噪后预测噪声
        推理模式：从噪声采样
        """
        cond = self.encoder(img)  # [B, cond_dim]

        if not return_loss:
            # 推理 - 从 cond 采样
            return self.sample(cond)

        # 训练 - DDPM 去噪
        B = target_bs.shape[0]
        t = torch.randint(0, self.noise_steps, (B,), device=img.device)
        noise = torch.randn_like(target_bs)
        noisy_bs = self.q_sample(target_bs, noise, t)
        pred_noise = self.decoder(noisy_bs, t, cond)
        return pred_noise, noise

    @torch.no_grad()
    def sample(self, cond, steps=None, ddim=True):
        """
        DDIM 采样：从条件 cond 生成 BS 系数
        Args:
            cond: [B, cond_dim] 图像条件
            steps: 采样步数 (默认用 config.sample_steps)
            ddim: 使用 DDIM 加速采样
        """
        B = cond.shape[0]
        steps = steps or self.config.sample_steps

        if ddim:
            return self._ddim_sample(cond, B, steps)
        else:
            return self._ddpm_sample(cond, B)

    def _ddpm_sample(self, cond, B):
        """标准 DDPM 采样"""
        bs = torch.randn(B, self.config.bs_dim, device=cond.device)
        for t in reversed(range(self.noise_steps)):
            t_batch = torch.full((B,), t, device=cond.device, dtype=torch.long)
            pred_noise = self.decoder(bs, t_batch, cond)
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            coef1 = 1 / alpha.sqrt()
            coef2 = beta / (1 - alpha_cumprod).sqrt()
            bs = coef1 * (bs - coef2 * pred_noise)
            if t > 0:
                noise = torch.randn_like(bs) * beta.sqrt()
                bs = bs + noise
        return bs

    def _ddim_sample(self, cond, B, steps):
        """DDIM 加速采样"""
        step_size = self.noise_steps // steps
        times = torch.linspace(0, self.noise_steps - 1, steps, device=cond.device).long()

        bs = torch.randn(B, self.config.bs_dim, device=cond.device)
        for i in reversed(range(steps)):
            t = times[i]
            t_prev = times[i - 1] if i > 0 else torch.tensor(0, device=cond.device)
            t_batch = torch.full((B,), t, device=cond.device, dtype=torch.long)
            pred_noise = self.decoder(bs, t_batch, cond)

            ac = self.alphas_cumprod[t]
            ac_prev = self.alphas_cumprod[t_prev]
            sigma = 0.0  # DDIM: sigma=0 是确定性采样

            pred_bs = (bs - (1 - ac).sqrt() * pred_noise) / ac.sqrt()
            coef = (1 - ac_prev - sigma ** 2).sqrt()
            bs = ac_prev.sqrt() * pred_bs + coef * pred_noise
        return bs

    def decoder_only_mode(self, cond_bs, target_bs):
        """
        第一阶段训练：跳过 ViT encoder，直接用 user_bs 作为条件。
        训练 diffusion decoder 学习 BS→BS 映射能力。
        """
        bs_feat = self.bs_to_feat(cond_bs)  # BS (52) → ViT_feat_dim (384)
        cond = self.encoder.cond_head(bs_feat)  # → cond_dim (256)
        B = target_bs.shape[0]
        t = torch.randint(0, self.noise_steps, (B,), device=target_bs.device)
        noise = torch.randn_like(target_bs)
        noisy_bs = self.q_sample(target_bs, noise, t)
        pred_noise = self.decoder(noisy_bs, t, cond)
        return pred_noise, noise

    def encode_face(self, img):
        """只编码图像，返回条件向量（可用于分析/可视化）"""
        return self.encoder(img)
