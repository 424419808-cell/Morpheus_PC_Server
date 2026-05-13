"""
第一阶段训练：纯 BS 数据上训 diffusion decoder（跳过 ViT encoder）。
用 (user_bs, response_bs) 配对数据训练 decoder 学习 BS->BS 映射。

用法：
    conda run -n morpheus python train_stage1.py
"""
import sys
sys.path.insert(0, ".")

from config import Config
from model.empathy_model import EmpathyModel
from train.dataset import BSPairDataset, create_dataloader
from train.trainer import Trainer


def main():
    cfg = Config()
    cfg.num_epochs = 100
    cfg.batch_size = 256  # BS 数据很小，可以大 batch

    # 检查数据
    data_path = "data/synthetic_bs/bs_pairs.pt"
    try:
        train_ds = BSPairDataset(data_path, "train")
        val_ds = BSPairDataset(data_path, "val")
    except FileNotFoundError:
        print(f"[错误] 找不到合成数据: {data_path}")
        print("请先运行: python data/prepare_synthetic.py")
        return

    train_loader = create_dataloader(train_ds, cfg.batch_size)
    val_loader = create_dataloader(val_ds, cfg.batch_size, shuffle=False)

    model = EmpathyModel(cfg)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(model, cfg)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
