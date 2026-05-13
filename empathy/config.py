import torch

class Config:
    # ─── model ───
    img_size = 224
    vit_name = "vit_small_patch16_224"  # timm ViT
    vit_feat_dim = 384                  # vit_small 的 CLS 维度
    vit_pretrained = False              # WSL2 无网络，设为 False 跳过下载
    cond_dim = 256                      # 条件向量维度
    bs_dim = 52                         # BS 系数维度
    dit_hidden_dim = 256
    dit_n_layers = 4
    dit_n_heads = 4

    # ─── diffusion ───
    noise_steps = 1000
    beta_start = 1e-4
    beta_end = 0.02
    sample_steps = 50                   # 推理时 DDIM 采样步数

    # ─── training ───
    batch_size = 64
    lr = 1e-4
    weight_decay = 1e-6
    num_epochs = 200
    log_interval = 50
    save_interval = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ─── data ───
    num_synthetic_samples = 50000       # 初始合成数据量
    real_data_path = "data/real"        # 真实数据目录

    # ─── paths ───
    ckpt_dir = "checkpoints"
    log_dir = "logs"
