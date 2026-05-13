"""
合成数据生成器。

生成策略：
  1. 随机采样用户 BS 系数（模拟各种表情）
  2. 用 empathy_transform 规则生成对应的共情回应 BS
  3. 对少量样本用 Blender/FLAME 渲染人脸图像（需安装 blender）
  4. 保存成 PyTorch Dataset 格式

第一阶段：纯 BS 数据（训练 diffusion decoder 的 BS→BS 映射）
第二阶段：BS + 渲染图像（训练端到端 ViT + DiT）
"""
import sys
sys.path.insert(0, ".")
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils.bs_utils import random_bs, empathy_transform


def generate_bs_pairs(num_samples=50000, dim=52, intensity_range=(0.5, 1.0)):
    """
    生成 (user_bs, response_bs) 配对合成数据。

    Returns:
        user_bs: [num_samples, dim]
        response_bs: [num_samples, dim]
    """
    all_user = []
    all_response = []

    for _ in tqdm(range(num_samples), desc="生成合成 BS 数据"):
        user = random_bs(1, dim).squeeze(0)
        intensity = torch.empty(1).uniform_(*intensity_range).item()
        response = empathy_transform(user.unsqueeze(0), intensity).squeeze(0)
        all_user.append(user)
        all_response.append(response)

    return torch.stack(all_user), torch.stack(all_response)


def save_bs_dataset(user_bs, response_bs, save_dir="data/synthetic_bs"):
    """保存 BS 配对数据为 .pt 文件。"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "user_bs": user_bs,
        "response_bs": response_bs,
    }, save_dir / "bs_pairs.pt")
    print(f"已保存 {len(user_bs)} 条合成 BS 数据到 {save_dir / 'bs_pairs.pt'}")
    print(f"  用户 BS 范围: [{user_bs.min():.3f}, {user_bs.max():.3f}]")
    print(f"  回应 BS 范围: [{response_bs.min():.3f}, {response_bs.max():.3f}]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=50000)
    parser.add_argument("--intensity", type=float, default=0.8)
    parser.add_argument("--out", type=str, default="data/synthetic_bs")
    args = parser.parse_args()

    print(f"生成 {args.num} 条合成 BS 数据...")
    user_bs, resp_bs = generate_bs_pairs(args.num, intensity_range=(args.intensity*0.6, args.intensity))
    save_bs_dataset(user_bs, resp_bs, args.out)
