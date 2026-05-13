import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import time

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

from .loss import EmpathyLoss


class Trainer:
    def __init__(self, model, config):
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.criterion = EmpathyLoss()
        self.writer = SummaryWriter(config.log_dir) if HAS_TB else None
        self.ckpt_dir = Path(config.ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float("inf")
        self.start_epoch = 0

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        total_diff_loss = 0
        num_batches = 0

        pbar = tqdm(loader, desc="Train")
        for batch in pbar:
            target_bs = batch["response_bs"].to(self.config.device)

            if "user_bs" in batch:
                # 第一阶段：BS->BS（user_bs 模拟 image encoder 的输出）
                cond_bs = batch["user_bs"].to(self.config.device)
                pred_noise, target_noise = self.model.decoder_only_mode(cond_bs, target_bs)
            else:
                # 第二阶段：端到端 image->BS
                img = batch["img"].to(self.config.device)
                pred_noise, target_noise = self.model(img, target_bs)

            loss, logs = self.criterion(pred_noise, target_noise)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += logs["total"]
            total_diff_loss += logs["diffusion"]
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{logs['total']:.4f}",
                "diff": f"{logs['diffusion']:.4f}",
            })

        avg_loss = total_loss / num_batches
        avg_diff = total_diff_loss / num_batches
        return avg_loss, avg_diff

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(loader, desc="Val"):
            target_bs = batch["response_bs"].to(self.config.device)

            if "user_bs" in batch:
                cond_bs = batch["user_bs"].to(self.config.device)
                pred_noise, target_noise = self.model.decoder_only_mode(cond_bs, target_bs)
            else:
                img = batch["img"].to(self.config.device)
                pred_noise, target_noise = self.model(img, target_bs)

            loss, _ = self.criterion(pred_noise, target_noise)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def fit(self, train_loader, val_loader=None):
        print(f"\n{'='*50}")
        print(f"开始训练共情模型")
        print(f"  设备: {self.config.device}")
        print(f"  训练样本: {len(train_loader.dataset)}")
        print(f"  批次: {self.config.batch_size}")
        print(f"  轮数: {self.config.num_epochs}")
        print(f"{'='*50}\n")

        for epoch in range(self.start_epoch, self.config.num_epochs):
            t0 = time.time()

            train_loss, train_diff = self.train_epoch(train_loader)
            self.scheduler.step()

            if self.writer:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/diffusion", train_diff, epoch)
                self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)

            val_str = ""
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                if self.writer:
                    self.writer.add_scalar("Loss/val", val_loss, epoch)
                val_str = f" | val_loss={val_loss:.4f}"

            elapsed = time.time() - t0
            print(f"Epoch {epoch:3d}/{self.config.num_epochs} | "
                  f"loss={train_loss:.4f} | diff={train_diff:.4f}{val_str} | "
                  f"lr={self.optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s")

            if (epoch + 1) % self.config.save_interval == 0:
                ckpt_path = self.ckpt_dir / f"empathy_epoch_{epoch+1}.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": train_loss,
                }, ckpt_path)

            if train_loss < self.best_loss:
                self.best_loss = train_loss
                torch.save(self.model.state_dict(), self.ckpt_dir / "empathy_best.pth")

        torch.save(self.model.state_dict(), self.ckpt_dir / "empathy_final.pth")
        print(f"\n训练完成！最佳 loss: {self.best_loss:.4f}")
        print(f"模型已保存到 {self.ckpt_dir}")
