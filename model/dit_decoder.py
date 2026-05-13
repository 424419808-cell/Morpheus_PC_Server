import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class DiTDecoder(nn.Module):
    """
    DiT-style decoder for BS coefficient generation.
    Treats each BS dimension as a token (seq_len=bs_dim).
    """
    def __init__(self, bs_dim=52, cond_dim=256, hidden_dim=256, n_layers=4, n_heads=4):
        super().__init__()
        self.bs_dim = bs_dim
        self.hidden_dim = hidden_dim

        # BS coefficient embedding
        self.bs_embed = nn.Linear(1, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, bs_dim, hidden_dim))

        # 时间步 embedding
        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 条件 embedding（来自 ViT 的图像特征）
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 将所有条件融合为一个 token，拼到序列前面（classifier-free guidance 风格）
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, noisy_bs, t, cond):
        """
        Args:
            noisy_bs: [B, bs_dim] 加噪后的 BS 系数
            t: [B] 时间步 (long)
            cond: [B, cond_dim] 条件向量（来自 ViT 的图像特征）
        Returns:
            pred_noise: [B, bs_dim] 预测的噪声
        """
        B = noisy_bs.shape[0]
        # [B, bs_dim, 1] -> [B, bs_dim, hidden_dim]
        x = self.bs_embed(noisy_bs.unsqueeze(-1))
        x = x + self.pos_embed

        # 时间 embedding
        t_float = t.float() / 1000.0  # normalize to [0,1]
        t_emb = self.time_proj(self.time_embed(t_float))  # [B, hidden_dim]
        x = x + t_emb.unsqueeze(1)

        # 条件 token
        cond_tok = self.cond_proj(cond).unsqueeze(1)  # [B, 1, hidden_dim]
        cls_tok = self.cls_token.expand(B, -1, -1)
        c = cls_tok + cond_tok

        # 拼接: [CLS, BS1, BS2, ...] -> [B, 1+bs_dim, hidden_dim]
        x = torch.cat([c, x], dim=1)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        # 去掉 CLS token，取 BS 部分的输出
        bs_out = x[:, 1:, :]  # [B, bs_dim, hidden_dim]
        return self.out(bs_out).squeeze(-1)  # [B, bs_dim]
