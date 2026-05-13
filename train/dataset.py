import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np


class BSPairDataset(Dataset):
    """
    纯 BS 配对数据：直接加载 (user_bs, response_bs) 对。
    第一阶段训练 diffusion decoder 用。
    """
    def __init__(self, data_path, split="train", ratio=0.9):
        data = torch.load(data_path)
        n = len(data["user_bs"])
        n_train = int(n * ratio)

        if split == "train":
            self.user_bs = data["user_bs"][:n_train]
            self.response_bs = data["response_bs"][:n_train]
        elif split == "val":
            self.user_bs = data["user_bs"][n_train:]
            self.response_bs = data["response_bs"][n_train:]
        else:  # all
            self.user_bs = data["user_bs"]
            self.response_bs = data["response_bs"]

    def __len__(self):
        return len(self.user_bs)

    def __getitem__(self, idx):
        return {
            "user_bs": self.user_bs[idx],
            "response_bs": self.response_bs[idx],
        }


class FaceToBSPairDataset(Dataset):
    """
    端到端数据：人脸图像 → 共情回应 BS。
    目录结构：
        data/real/
            subject_001/
                img_000.jpg, ...   (用户表情图像)
                bs_000.pt          (对应的共情回应 BS)
            ...
    """
    def __init__(self, data_dir, img_size=224, transform=None):
        self.data_dir = Path(data_dir)
        self.samples = []
        self.img_size = img_size
        self.transform = transform

        if not self.data_dir.exists():
            print(f"[警告] 数据目录 {data_dir} 不存在，使用空数据集")
            return

        for subj_dir in sorted(self.data_dir.iterdir()):
            if not subj_dir.is_dir():
                continue
            # 加载 BS 文件
            bs_file = subj_dir / "bs.pt"
            if not bs_file.exists():
                img_files = sorted(subj_dir.glob("*.jpg")) + sorted(subj_dir.glob("*.png"))
                bs_files = sorted(subj_dir.glob("bs_*.pt"))
                for img_f, bs_f in zip(img_files, bs_files):
                    self.samples.append((str(img_f), str(bs_f)))
            else:
                # 所有图共用同一个 BS（整段视频同一种共情回应）
                for img_f in sorted(subj_dir.glob("*.jpg")) + sorted(subj_dir.glob("*.png")):
                    self.samples.append((str(img_f), str(bs_file)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, bs_path = self.samples[idx]

        # 读图
        img = cv2.imread(img_path)
        if img is None:
            return self.__getitem__((idx + 1) % len(self.samples))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        # ImageNet normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        # 读 BS
        bs = torch.load(bs_path)
        if isinstance(bs, dict):
            bs = bs["response_bs"] if "response_bs" in bs else bs["bs"]
        if isinstance(bs, np.ndarray):
            bs = torch.from_numpy(bs).float()

        return {
            "img": img,
            "response_bs": bs,
        }


def create_dataloader(dataset, batch_size=64, shuffle=True, num_workers=2):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
