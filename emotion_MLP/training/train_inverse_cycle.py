import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ==================== 1. 定义网络结构（与正向模型一致） ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
class Angle2BSNet(nn.Module):
    """正向模型: angle -> blendshape"""
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128, 64]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class BS2AngleNet(nn.Module):
    """逆向模型: blendshape -> angle (输出归一化角度)"""
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128, 64]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        layers.append(nn.Sigmoid())   # 输出归一化到 [0,1]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# ==================== 2. 加载正向模型 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

checkpoint = torch.load(os.path.join(BASE_DIR, '..', 'models', 'angle2bs_full.pth'), map_location=device)
used_motors = checkpoint['used_motors']   # 电机 ID 列表，顺序固定
bs_keys = checkpoint['bs_keys']           # 52 个 blendshape 键顺序
motor_ranges = checkpoint['motor_ranges'] # 每个电机的 (min, max)

forward_model = Angle2BSNet(checkpoint['input_dim'], checkpoint['output_dim'])
forward_model.load_state_dict(checkpoint['model_state_dict'])
forward_model = forward_model.to(device)
forward_model.eval()
for param in forward_model.parameters():
    param.requires_grad = False
print(f"正向模型加载成功，电机数量: {checkpoint['input_dim']}, blendshape 数量: {checkpoint['output_dim']}")

# ==================== 3. 加载数据 ====================
json_path = os.path.join(BASE_DIR, '..', 'data', 'motor_babbling_data_PC.json')
with open(json_path, 'r') as f:
    raw_data = json.load(f)

all_bs = []
all_angles_norm = []
for item in raw_data:
    bs_dict = item.get("blendshapes", {})
    bs_vals = [bs_dict.get(k, 0.0) for k in bs_keys]
    all_bs.append(np.array(bs_vals, dtype=np.float32))
    
    motor_dict = item.get("motor_commands", {})
    angle_raw = [motor_dict.get(str(mid), 0.0) for mid in used_motors]
    angle_norm = []
    for i, mid in enumerate(used_motors):
        minv, maxv = motor_ranges[mid]
        if maxv - minv < 1e-6:
            norm = 0.5
        else:
            norm = (angle_raw[i] - minv) / (maxv - minv)
        angle_norm.append(norm)
    all_angles_norm.append(np.array(angle_norm, dtype=np.float32))

X_bs = np.stack(all_bs)
Y_angle = np.stack(all_angles_norm)
print(f"数据总数: {len(X_bs)}")

X_bs_train, X_bs_temp, Y_angle_train, Y_angle_temp = train_test_split(
    X_bs, Y_angle, test_size=0.3, random_state=42)
X_bs_val, X_bs_test, Y_angle_val, Y_angle_test = train_test_split(
    X_bs_temp, Y_angle_temp, test_size=0.5, random_state=42)
print(f"训练集: {X_bs_train.shape[0]}, 验证集: {X_bs_val.shape[0]}, 测试集: {X_bs_test.shape[0]}")

# ==================== 4. 数据集 ====================
class BSDataset(Dataset):
    def __init__(self, bs, angles=None):
        self.bs = torch.tensor(bs, dtype=torch.float32)
        self.angles = torch.tensor(angles, dtype=torch.float32) if angles is not None else None
    def __len__(self):
        return len(self.bs)
    def __getitem__(self, idx):
        if self.angles is not None:
            return self.bs[idx], self.angles[idx]
        else:
            return self.bs[idx]

batch_size = 64
train_loader = DataLoader(BSDataset(X_bs_train, Y_angle_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(BSDataset(X_bs_val, Y_angle_val), batch_size=batch_size)
test_loader = DataLoader(BSDataset(X_bs_test, Y_angle_test), batch_size=batch_size)

# ==================== 5. 定义辅助损失函数 ====================
def smoothness_loss(angles, motor_pairs):
    """
    angles: (batch, num_motors)
    motor_pairs: list of (i, j) 相邻舵机索引对
    """
    if len(motor_pairs) == 0:
        return torch.tensor(0.0, device=angles.device)
    loss = 0.0
    for i, j in motor_pairs:
        loss += torch.mean((angles[:, i] - angles[:, j]) ** 2)
    return loss / len(motor_pairs)

def centering_loss(angles, target=0.5):
    return torch.mean((angles - target) ** 2)

def boundary_loss(angles, margin=0.05):
    lower = torch.relu(margin - angles)
    upper = torch.relu(angles - (1.0 - margin))
    return torch.mean(lower ** 2 + upper ** 2)

def mmd_loss(x, y, sigma=0.5):
    """
    最大均值差异（RBF核）
    """
    def gaussian_kernel(x, y, sigma):
        x_norm = (x**2).sum(1).view(-1,1)
        y_norm = (y**2).sum(1).view(1,-1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
        return torch.exp(-dist / (2 * sigma**2))
    xx = gaussian_kernel(x, x, sigma)
    yy = gaussian_kernel(y, y, sigma)
    xy = gaussian_kernel(x, y, sigma)
    return xx.mean() + yy.mean() - 2 * xy.mean()

# ==================== 6. 定义舵机邻接关系 ====================
# 电机ID到索引的映射
motor_id_to_idx = {mid: idx for idx, mid in enumerate(used_motors)}

# 定义对称对和区域对（使用电机ID，后续转换为索引）
motor_pairs_raw = [
    # 左右对称
    (0, 29),   # 左右鼻子
    (1, 28),   # 左右脸颊
    (2, 27),   # 左右眉心
    (3, 26),   # 左右眉峰
    (4, 25),   # 左右上眼睑
    (5, 24),   # 左右下眼睑
    (8, 21),   # 左右上嘴角
    (13, 18),  # 左右下嘴角
    (12, 17),  # 左右下巴前后
    (19, 22),  # 嘴巴张合左右
    (15, 16),  # 左右下唇
    # 同侧相邻区域
    (0, 1), (29, 28),      # 鼻子与脸颊
    (1, 2), (28, 27),      # 脸颊与眉心
    (2, 3), (27, 26),      # 眉心与眉峰
    (3, 4), (26, 25),      # 眉峰与上眼睑
    (4, 5), (25, 24),      # 上眼睑与下眼睑
    (8, 13), (21, 18),     # 上嘴角与下嘴角
    (12, 14), (17, 14),    # 下巴与下唇中心
    (14, 15), (14, 16),    # 下唇中心与左右下唇
    (19, 20), (22, 20),    # 嘴巴张合与上唇
    (7, 14),               # 上唇中心与下唇中心
]

# 转换为索引对，只保留两个电机都存在的对
motor_pairs_idx = []
for a, b in motor_pairs_raw:
    if a in motor_id_to_idx and b in motor_id_to_idx:
        motor_pairs_idx.append((motor_id_to_idx[a], motor_id_to_idx[b]))
print(f"定义的相邻舵机对数量: {len(motor_pairs_idx)}")

# ==================== 7. 初始化逆向模型 ====================
inverse_model = BS2AngleNet(len(bs_keys), len(used_motors)).to(device)
print(f"逆向模型输入维度: {len(bs_keys)}, 输出维度: {len(used_motors)}")

criterion_cycle = nn.MSELoss()
criterion_angle = nn.MSELoss()

# 损失权重配置（可根据训练情况微调）
lambda_cycle = 1.0          # 主损失
lambda_sup = 0.01           # 监督权重调小
lambda_smooth = 0.005
lambda_center = 0.001
lambda_boundary = 0.005
lambda_mmd = 0.05
lambda_inv_consist = 0.1

optimizer = optim.Adam(inverse_model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# ==================== 8. 训练循环 ====================
num_epochs = 150
best_val_loss = float('inf')
best_model_state = None

for epoch in range(num_epochs):
    inverse_model.train()
    train_loss_total = 0.0
    train_loss_cycle = 0.0
    train_loss_sup = 0.0
    train_loss_smooth = 0.0
    train_loss_center = 0.0
    train_loss_boundary = 0.0
    train_loss_mmd = 0.0
    train_loss_inv = 0.0

    for bs_batch, angle_batch in train_loader:
        bs_batch = bs_batch.to(device)
        angle_batch = angle_batch.to(device)

        pred_angle = inverse_model(bs_batch)

        # 1. 循环一致性损失
        reconstructed_bs = forward_model(pred_angle)
        loss_cycle = criterion_cycle(reconstructed_bs, bs_batch)

        # 2. 监督损失（权重很小）
        loss_sup = criterion_angle(pred_angle, angle_batch)

        # 3. 平滑性损失
        loss_smooth = smoothness_loss(pred_angle, motor_pairs_idx)

        # 4. 中心性损失
        loss_center = centering_loss(pred_angle, 0.5)

        # 5. 边界惩罚
        loss_boundary = boundary_loss(pred_angle, margin=0.05)

        # 6. MMD 分布匹配（仅在batch size较大时计算）
        if pred_angle.size(0) > 10:
            loss_mmd = mmd_loss(pred_angle, angle_batch, sigma=0.5)
        else:
            loss_mmd = torch.tensor(0.0, device=device)

        # 7. 逆一致性损失
        with torch.no_grad():
            pred_bs_from_angle = forward_model(angle_batch)
        recon_angle = inverse_model(pred_bs_from_angle)
        loss_inv = criterion_angle(recon_angle, angle_batch)

        loss_total = (lambda_cycle * loss_cycle +
                      lambda_sup * loss_sup +
                      lambda_smooth * loss_smooth +
                      lambda_center * loss_center +
                      lambda_boundary * loss_boundary +
                      lambda_mmd * loss_mmd +
                      lambda_inv_consist * loss_inv)

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        train_loss_total += loss_total.item() * bs_batch.size(0)
        train_loss_cycle += loss_cycle.item() * bs_batch.size(0)
        train_loss_sup += loss_sup.item() * bs_batch.size(0)
        train_loss_smooth += loss_smooth.item() * bs_batch.size(0)
        train_loss_center += loss_center.item() * bs_batch.size(0)
        train_loss_boundary += loss_boundary.item() * bs_batch.size(0)
        train_loss_mmd += loss_mmd.item() * bs_batch.size(0)
        train_loss_inv += loss_inv.item() * bs_batch.size(0)

    n_train = len(train_loader.dataset)
    train_loss_total /= n_train
    train_loss_cycle /= n_train
    train_loss_sup /= n_train
    train_loss_smooth /= n_train
    train_loss_center /= n_train
    train_loss_boundary /= n_train
    train_loss_mmd /= n_train
    train_loss_inv /= n_train

    # 验证
    inverse_model.eval()
    val_loss_total = 0.0
    with torch.no_grad():
        for bs_batch, angle_batch in val_loader:
            bs_batch = bs_batch.to(device)
            angle_batch = angle_batch.to(device)
            pred_angle = inverse_model(bs_batch)
            reconstructed_bs = forward_model(pred_angle)
            loss_cycle = criterion_cycle(reconstructed_bs, bs_batch)
            loss_sup = criterion_angle(pred_angle, angle_batch)
            loss_smooth = smoothness_loss(pred_angle, motor_pairs_idx)
            loss_center = centering_loss(pred_angle, 0.5)
            loss_boundary = boundary_loss(pred_angle, margin=0.05)
            if pred_angle.size(0) > 10:
                loss_mmd = mmd_loss(pred_angle, angle_batch, sigma=0.5)
            else:
                loss_mmd = torch.tensor(0.0, device=device)
            with torch.no_grad():
                pred_bs_from_angle = forward_model(angle_batch)
            recon_angle = inverse_model(pred_bs_from_angle)
            loss_inv = criterion_angle(recon_angle, angle_batch)
            loss_total = (lambda_cycle * loss_cycle +
                          lambda_sup * loss_sup +
                          lambda_smooth * loss_smooth +
                          lambda_center * loss_center +
                          lambda_boundary * loss_boundary +
                          lambda_mmd * loss_mmd +
                          lambda_inv_consist * loss_inv)
            val_loss_total += loss_total.item() * bs_batch.size(0)
    val_loss_total /= len(val_loader.dataset)

    scheduler.step(val_loss_total)

    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1:3d} | Train Total: {train_loss_total:.6f} "
              f"(Cyc:{train_loss_cycle:.4f}, Sup:{train_loss_sup:.4f}, "
              f"Sm:{train_loss_smooth:.4f}, Cen:{train_loss_center:.4f}, "
              f"Bd:{train_loss_boundary:.4f}, MMD:{train_loss_mmd:.4f}, Inv:{train_loss_inv:.4f}) "
              f"| Val Total: {val_loss_total:.6f}")

    if val_loss_total < best_val_loss:
        best_val_loss = val_loss_total
        best_model_state = inverse_model.state_dict().copy()

# 测试
inverse_model.load_state_dict(best_model_state)
inverse_model.eval()
test_loss_total = 0.0
with torch.no_grad():
    for bs_batch, angle_batch in test_loader:
        bs_batch = bs_batch.to(device)
        angle_batch = angle_batch.to(device)
        pred_angle = inverse_model(bs_batch)
        reconstructed_bs = forward_model(pred_angle)
        loss_cycle = criterion_cycle(reconstructed_bs, bs_batch)
        loss_sup = criterion_angle(pred_angle, angle_batch)
        loss_total = lambda_cycle * loss_cycle + lambda_sup * loss_sup  # 评估用主损失
        test_loss_total += loss_total.item() * bs_batch.size(0)
test_loss_total /= len(test_loader.dataset)
print(f"训练完成！最佳验证损失: {best_val_loss:.6f}, 测试损失: {test_loss_total:.6f}")

# 保存模型
save_dict = {
    'model_state_dict': best_model_state,
    'used_motors': used_motors,
    'bs_keys': bs_keys,
    'motor_ranges': motor_ranges,
    'input_dim': len(bs_keys),
    'output_dim': len(used_motors),
}
torch.save(save_dict, os.path.join(BASE_DIR, '..', 'models', 'bs2angle_cycle.pth'))
print("增强版逆向模型已保存为 bs2angle_cycle.pth")
