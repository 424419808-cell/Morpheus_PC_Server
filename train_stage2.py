"""
第二阶段训练：端到端 image->BS（ViT encoder + DiT decoder）。
加载第一阶段预训练的权重，在真实图像数据上微调。

用法：
    conda run -n morpheus python train_stage2.py --data_dir data/real
"""
import sys
sys.path.insert(0, ".")

import argparse
import torch
from config import Config
from model.empathy_model import EmpathyModel
from train.dataset import FaceToBSPairDataset, create_dataloader
from train.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/real")
    parser.add_argument("--pretrained", type=str, default="checkpoints/empathy_best.pth")
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    cfg = Config()
    cfg.num_epochs = args.epochs

    # 端到端数据集（人脸图像 -> 回应 BS）
    dataset = FaceToBSPairDataset(args.data_dir, img_size=cfg.img_size)
    if len(dataset) == 0:
        print(f"[错误] 数据目录 {args.data_dir} 为空或不存在")
        print("请准备图像+BS 配对数据")
        return

    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = create_dataloader(train_ds, cfg.batch_size // 4)
    val_loader = create_dataloader(val_ds, cfg.batch_size // 4, shuffle=False)

    model = EmpathyModel(cfg)

    # 加载第一阶段预训练权重（只加载 decoder 部分）
    if args.pretrained:
        pretrained = torch.load(args.pretrained, map_location="cpu")
        if "model_state_dict" in pretrained:
            pretrained = pretrained["model_state_dict"]
        # 只加载 decoder 权重，encoder 保持 pretrained ViT
        model_state = model.state_dict()
        for k, v in pretrained.items():
            if k.startswith("decoder.") and k in model_state:
                model_state[k] = v
        model.load_state_dict(model_state)
        print(f"已加载预训练权重（decoder 部分）: {args.pretrained}")

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  encoder: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"  decoder: {sum(p.numel() for p in model.decoder.parameters()):,}")

    trainer = Trainer(model, cfg)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
