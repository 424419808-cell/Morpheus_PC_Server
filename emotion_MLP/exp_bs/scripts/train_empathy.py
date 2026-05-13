"""
训练 BS2BS_Empathy 模型 — 用户 52 BS → 共情 52 BS

三阶段训练:
  1. 预训练（恒等映射初始化）— 在 motor_babbling 数据上
  2. 共情微调 — 在 gen_empathy_data 生成的配对数据上
  3. 循环对齐 — 提高 cycle-consistency 权重

用法:
  python exp_bs/scripts/train_empathy.py
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 路径设置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.join(BASE_DIR, "..", "..")
DATA_DIR = os.path.join(PROJ_ROOT, "data")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "..", "models")
BS2ANGLE_MODEL_DIR = os.path.join(PROJ_ROOT, "bs2angle", "models")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ==================== 模型定义 ====================
class BS2BS_Empathy(nn.Module):
    """输入用户 52 BS → 输出共情 52 BS"""
    def __init__(self, input_dim=52, output_dim=52):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(52, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 52), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ==================== 数据集 ====================
class EmpathyDataset(Dataset):
    def __init__(self, data, is_train=True, split_ratio=0.8):
        n = len(data["inputs"])
        indices = np.arange(n)
        np.random.shuffle(indices)
        split = int(n * split_ratio)

        if is_train:
            idx = indices[:split]
        else:
            idx = indices[split:]

        self.inputs = torch.tensor(data["inputs"][idx], dtype=torch.float32)
        self.targets = torch.tensor(data["targets"][idx], dtype=torch.float32)
        self.user_emos = data["user_emotions"][idx]
        self.tgt_emos = data["target_emotions"][idx]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return self.inputs[i], self.targets[i]


# ==================== 循环一致性检查 ====================
def load_cycle_models(device):
    """加载用于 cycle-consistency 的 FaceBS2Angle + Angle2BS 模型"""
    try:
        # 加载正向模型
        forward_path = os.path.join(BS2ANGLE_MODEL_DIR, "angle2bs_full.pth")
        fwd_ckpt = torch.load(forward_path, map_location=device)
        bs_keys = fwd_ckpt["bs_keys"]
        used_motors = fwd_ckpt["used_motors"]
        input_dim = fwd_ckpt["input_dim"]
        output_dim = fwd_ckpt["output_dim"]

        # 重建 Angle2BS 网络
        fwd_net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, output_dim),
        )
        fwd_sd = {k.replace("net.", ""): v for k, v in fwd_ckpt["model_state_dict"].items()}
        fwd_net.load_state_dict(fwd_sd)
        fwd_net.to(device).eval()

        # 加载上下半脸逆向模型
        upper_path = os.path.join(BS2ANGLE_MODEL_DIR, "upper_face_bs2angle.pth")
        lower_path = os.path.join(BS2ANGLE_MODEL_DIR, "lower_face_bs2angle.pth")
        upper_ckpt = torch.load(upper_path, map_location=device)
        lower_ckpt = torch.load(lower_path, map_location=device)

        upper_dim_in = len(upper_ckpt["upper_bs_keys"])
        upper_dim_out = len(upper_ckpt["upper_motor_ids"])
        lower_dim_in = len(lower_ckpt["lower_bs_idx"])
        lower_dim_out = len(lower_ckpt["lower_motor_ids"])

        def make_inverse(dim_in, dim_out):
            net = nn.Sequential(
                nn.Linear(dim_in, 256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, dim_out), nn.Sigmoid(),
            )
            return net

        upper_net = make_inverse(upper_dim_in, upper_dim_out)
        lower_net = make_inverse(lower_dim_in, lower_dim_out)
        upper_sd = {k.replace("net.", ""): v for k, v in upper_ckpt["model_state_dict"].items()}
        lower_sd = {k.replace("net.", ""): v for k, v in lower_ckpt["model_state_dict"].items()}
        upper_net.load_state_dict(upper_sd)
        lower_net.load_state_dict(lower_sd)
        upper_net.to(device).eval()
        lower_net.to(device).eval()

        return {
            "forward": fwd_net,
            "upper": upper_net,
            "lower": lower_net,
            "bs_keys": bs_keys,
            "used_motors": used_motors,
            "upper_bs_keys": upper_ckpt["upper_bs_keys"],
            "lower_bs_idx": lower_ckpt["lower_bs_idx"],
            "upper_motor_ids": upper_ckpt["upper_motor_ids"],
            "lower_motor_ids": lower_ckpt["lower_motor_ids"],
            "motor_ranges": {**upper_ckpt["motor_ranges"], **lower_ckpt["motor_ranges"]},
        }
    except Exception as e:
        print(f"[警告] 无法加载循环一致性模型: {e}")
        print("[警告] 将跳过 cycle-consistency 损失（仅使用 L1 损失）")
        return None


def cycle_consistency_loss(pred_bs, cycle_models, device):
    """
    计算循环一致性损失：
    pred_BS → (upper/lower inverse) → angles → forward_model → reconstructed BS
    """
    if cycle_models is None:
        return torch.tensor(0.0, device=device)

    bs_keys = cycle_models["bs_keys"]
    upper_bs_keys = cycle_models["upper_bs_keys"]
    lower_bs_idx = cycle_models["lower_bs_idx"]
    upper_motor_ids = cycle_models["upper_motor_ids"]
    lower_motor_ids = cycle_models["lower_motor_ids"]
    motor_ranges = cycle_models["motor_ranges"]

    batch_size = pred_bs.shape[0]

    # BS 字典映射（与 test_emotion_to_face.py 一致）
    bs_dict = {"_neutral": torch.zeros(batch_size, device=device)}
    for arkit_idx in range(51):
        key = bs_keys[arkit_idx + 1]
        bs_dict[key] = pred_bs[:, arkit_idx]

    with torch.no_grad():
        # 上半脸
        upper_input = torch.stack([bs_dict.get(k, torch.zeros(batch_size, device=device))
                                   for k in upper_bs_keys], dim=1)
        upper_pred = cycle_models["upper"](upper_input)

        # 下半脸
        lower_bs_keys = [bs_keys[i] for i in lower_bs_idx]
        lower_input = torch.stack([bs_dict.get(k, torch.zeros(batch_size, device=device))
                                   for k in lower_bs_keys], dim=1)
        lower_pred = cycle_models["lower"](lower_input)

        # 合并角度
        angle_vec = torch.zeros(batch_size, len(cycle_models["used_motors"]), device=device)
        for i, mid in enumerate(upper_motor_ids):
            idx = cycle_models["used_motors"].index(mid)
            angle_vec[:, idx] = upper_pred[:, i]
        for i, mid in enumerate(lower_motor_ids):
            idx = cycle_models["used_motors"].index(mid)
            angle_vec[:, idx] = lower_pred[:, i]

        # 通过正向模型重建 BS
        recon_bs = cycle_models["forward"](angle_vec)
    return nn.functional.l1_loss(recon_bs, pred_bs)


# ==================== 训练 ====================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载共情数据
    data_path = os.path.join(DATA_DIR, "empathy_training_data.npz")
    if not os.path.exists(data_path):
        print(f"未找到共情数据，正在生成...")
        sys.path.insert(0, BASE_DIR)
        from gen_empathy_data import save_empathy_data
        save_empathy_data(data_path)

    data = np.load(data_path)
    print(f"加载数据: {data_path}")
    print(f"  样本数: {len(data['inputs'])}")

    train_set = EmpathyDataset(data, is_train=True)
    val_set = EmpathyDataset(data, is_train=False)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)

    model = BS2BS_Empathy().to(device)
    cycle_models = load_cycle_models(device)

    # ===== 阶段 1: 预训练（恒等映射） =====
    print("\n===== 阶段 1: 恒等映射预训练 =====")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()

    for epoch in range(100):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 20 == 0:
            print(f"  epoch {epoch}: train_loss={total_loss/len(train_loader):.6f}")

    # ===== 阶段 2: 共情微调 =====
    print("\n===== 阶段 2: 共情微调 =====")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(200):
        model.train()
        total_loss = 0
        total_emp_loss = 0
        total_cycle_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)

            # 1) 共情 L1 损失
            emp_loss = criterion(pred, y)

            # 2) 循环一致性损失
            cyc_loss = cycle_consistency_loss(pred, cycle_models, device)

            # 3) 中性保持损失
            neutral_mask = (y == 0).all(dim=1)
            if neutral_mask.any():
                neutral_loss = criterion(pred[neutral_mask], y[neutral_mask])
            else:
                neutral_loss = torch.tensor(0.0, device=device)

            loss = emp_loss + 0.5 * cyc_loss + 0.01 * neutral_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_emp_loss += emp_loss.item()
            total_cycle_loss += cyc_loss.item()

        if epoch % 20 == 0:
            print(f"  epoch {epoch}: total={total_loss/len(train_loader):.6f} "
                  f"emp={total_emp_loss/len(train_loader):.6f} "
                  f"cyc={total_cycle_loss/len(train_loader):.6f}")

    # ===== 阶段 3: 循环对齐 =====
    if cycle_models is not None:
        print("\n===== 阶段 3: 循环对齐 =====")
        optimizer = optim.Adam(model.parameters(), lr=5e-5)

        for epoch in range(50):
            model.train()
            total_loss = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(x)

                emp_loss = criterion(pred, y)
                cyc_loss = cycle_consistency_loss(pred, cycle_models, device)
                loss = emp_loss + 2.0 * cyc_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0:
                print(f"  epoch {epoch}: loss={total_loss/len(train_loader):.6f}")

    # ===== 验证 =====
    print("\n===== 验证 =====")
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            val_loss += criterion(pred, y).item()
    print(f"  验证 L1 损失: {val_loss/len(val_loader):.6f}")

    # ===== 保存模型 =====
    save_path = os.path.join(MODEL_SAVE_DIR, "bs2bs_empathy.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n模型已保存: {save_path}")


if __name__ == "__main__":
    train()
