#!/usr/bin/env python3
"""
上半脸专用逆向模型训练 (BS → 舵机)
输入：19维上半脸 BS（眉毛+眼睛）
输出：8维上半脸舵机归一化角度（电机2,3,4,5,24,25,26,27）
训练方式：循环一致性 + 监督学习 + 简单物理约束
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ==================== 区域定义 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPPER_BS_KEYS = [
    'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
    'eyeBlinkLeft', 'eyeBlinkRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight',
    'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight',
    'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight'
]
UPPER_MOTOR_IDS = [2, 3, 4, 5, 24, 25, 26, 27]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ==================== 加载正向模型 ====================
checkpoint = torch.load(os.path.join(BASE_DIR, '..', 'models', 'angle2bs_full.pth'), map_location=device)
used_motors_full = checkpoint['used_motors']
bs_keys_full = checkpoint['bs_keys']
motor_ranges = checkpoint['motor_ranges']

class Angle2BSNet(nn.Module):
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

forward_model = Angle2BSNet(checkpoint['input_dim'], checkpoint['output_dim'])
forward_model.load_state_dict(checkpoint['model_state_dict'])
forward_model = forward_model.to(device)
forward_model.eval()
for p in forward_model.parameters():
    p.requires_grad = False
print("正向模型加载成功")

# ==================== 构建索引 ====================
upper_bs_idx = [bs_keys_full.index(k) for k in UPPER_BS_KEYS if k in bs_keys_full]
upper_motor_idx = [used_motors_full.index(mid) for mid in UPPER_MOTOR_IDS if mid in used_motors_full]
print(f"上半脸 BS 数量: {len(upper_bs_idx)}, 舵机数量: {len(upper_motor_idx)}")

# ==================== 加载数据 ====================
json_path = os.path.join(BASE_DIR, '..', 'data', 'motor_babbling_data_PC.json')
with open(json_path, 'r') as f:
    raw_data = json.load(f)

all_bs_upper, all_angles_upper_norm = [], []
for item in raw_data:
    bs_dict = item["blendshapes"]
    bs_upper = [bs_dict.get(bs_keys_full[i], 0.0) for i in upper_bs_idx]
    all_bs_upper.append(np.array(bs_upper, dtype=np.float32))
    
    motor_dict = item["motor_commands"]
    angle_norm = []
    for mid in UPPER_MOTOR_IDS:
        raw = motor_dict.get(str(mid), 0.0)
        minv, maxv = motor_ranges[mid]
        norm = (raw - minv) / (maxv - minv) if maxv > minv else 0.5
        angle_norm.append(norm)
    all_angles_upper_norm.append(np.array(angle_norm, dtype=np.float32))

X = np.stack(all_bs_upper)
Y = np.stack(all_angles_upper_norm)
print(f"样本数: {len(X)}")

X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

class UpperFaceDataset(Dataset):
    def __init__(self, bs, angles):
        self.bs = torch.tensor(bs, dtype=torch.float32)
        self.angles = torch.tensor(angles, dtype=torch.float32)
    def __len__(self): return len(self.bs)
    def __getitem__(self, idx): return self.bs[idx], self.angles[idx]

batch_size = 64
train_loader = DataLoader(UpperFaceDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(UpperFaceDataset(X_val, Y_val), batch_size=batch_size)

# ==================== 上半脸逆向模型 ====================
class UpperFaceBS2Angle(nn.Module):
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
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

model = UpperFaceBS2Angle(len(upper_bs_idx), len(upper_motor_idx)).to(device)

# ==================== 损失函数 ====================
criterion_cycle = nn.SmoothL1Loss()
criterion_angle = nn.SmoothL1Loss()

def centering_loss(angles, target=0.5):
    return torch.mean((angles - target).abs())

def boundary_loss(angles, margin=0.05):
    lower = torch.relu(margin - angles)
    upper = torch.relu(angles - (1.0 - margin))
    return torch.mean(lower + upper)

lambda_cycle = 1.0
lambda_sup = 0.3
lambda_center = 0.0001
lambda_boundary = 0.005

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# ==================== 辅助函数 ====================
def fill_full_angle(upper_angles):
    batch = upper_angles.size(0)
    full = torch.zeros(batch, len(used_motors_full), device=device)
    full[:, upper_motor_idx] = upper_angles
    return full

# ==================== 训练循环 ====================
num_epochs = 150
best_val_loss = float('inf')
best_state = None

for epoch in range(num_epochs):
    model.train()
    train_loss_total = 0.0
    for bs_batch, angle_batch in train_loader:
        bs_batch, angle_batch = bs_batch.to(device), angle_batch.to(device)
        
        pred_upper = model(bs_batch)
        full_pred = fill_full_angle(pred_upper)
        recon_bs_full = forward_model(full_pred)
        recon_bs_upper = recon_bs_full[:, upper_bs_idx]
        
        loss_cycle = criterion_cycle(recon_bs_upper, bs_batch)
        loss_sup = criterion_angle(pred_upper, angle_batch)
        loss_center = centering_loss(pred_upper)
        loss_boundary = boundary_loss(pred_upper)
        
        loss = (lambda_cycle * loss_cycle +
                lambda_sup * loss_sup +
                lambda_center * loss_center +
                lambda_boundary * loss_boundary)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_total += loss.item() * bs_batch.size(0)
    
    train_loss_total /= len(train_loader.dataset)
    
    # 验证
    model.eval()
    val_loss_total = 0.0
    with torch.no_grad():
        for bs_batch, angle_batch in val_loader:
            bs_batch, angle_batch = bs_batch.to(device), angle_batch.to(device)
            pred_upper = model(bs_batch)
            full_pred = fill_full_angle(pred_upper)
            recon_bs = forward_model(full_pred)[:, upper_bs_idx]
            loss_cycle = criterion_cycle(recon_bs, bs_batch)
            val_loss_total += loss_cycle.item() * bs_batch.size(0)
    val_loss_total /= len(val_loader.dataset)
    
    scheduler.step(val_loss_total)
    
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss_total:.6f} | Val Cycle Loss: {val_loss_total:.6f}")
    
    if val_loss_total < best_val_loss:
        best_val_loss = val_loss_total
        best_state = model.state_dict().copy()

# 测试
model.load_state_dict(best_state)
model.eval()
test_loss = 0.0
test_loader = DataLoader(UpperFaceDataset(X_test, Y_test), batch_size=batch_size)
with torch.no_grad():
    for bs_batch, angle_batch in test_loader:
        bs_batch = bs_batch.to(device)
        pred_upper = model(bs_batch)
        full_pred = fill_full_angle(pred_upper)
        recon_bs = forward_model(full_pred)[:, upper_bs_idx]
        test_loss += criterion_cycle(recon_bs, bs_batch).item() * bs_batch.size(0)
test_loss /= len(X_test)
print(f"测试循环损失: {test_loss:.6f}")

# 保存模型
torch.save({
    'model_state_dict': best_state,
    'upper_bs_keys': UPPER_BS_KEYS,
    'upper_bs_idx': upper_bs_idx,
    'upper_motor_ids': UPPER_MOTOR_IDS,
    'upper_motor_idx': upper_motor_idx,
    'motor_ranges': motor_ranges,
    'used_motors_full': used_motors_full,   # 用于测试时填充完整角度
    'input_dim': len(upper_bs_idx),
    'output_dim': len(upper_motor_idx),
}, os.path.join(BASE_DIR, '..', 'models', 'upper_face_bs2angle.pth'))
print("上半脸模型已保存为 upper_face_bs2angle.pth")
