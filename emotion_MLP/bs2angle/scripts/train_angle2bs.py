import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ==================== 1. 舵机范围定义（仅定义数据中出现的舵机） ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 根据你提供的表格，只取数据中实际有
motor_ranges = {
    0:  (10.0,  90.0),   # 左鼻子
    1:  (50.0,  100.0),  # 左脸颊
    2:  (70.0,  105.0),  # 左眉心
    3:  (70.0,  100.0),  # 左眉峰
    4:  (20.0,  120.0),  # 左上眼睑
    5:  (70.0,  100.0),  # 左下眼睑 (已修改)
    7:  (54.0,  90.0),   # 上唇中心
    8:  (45.0,  135.0),  # 左上嘴角
    9:  (70.0,  115.0),  # 右上唇 (已修改)
    13: (45.0,  135.0),  # 左下嘴角
    14: (105.0, 150.0),  # 下唇中心
    15: (60.0,  90.0),   # 左下唇 (已修改)
    16: (125.0, 150.0),  # 右下唇
    18: (45.0,  135.0),  # 右下嘴角
    19: (90.0,  150.0),  # 嘴巴张合（左） (已修改)
    20: (20.0,  62.0),   # 左上唇
    21: (45.0,  135.0),  # 右上嘴角
    22: (30.0,  100.0),  # 嘴巴张合（右） (已修改)
    24: (20.0,  50.0),   # 右下眼睑
    25: (60.0,  150.0),  # 右上眼睑
    26: (65.0,  100.0),  # 右眉峰
    27: (60.0,  115.0),  # 右眉心
    28: (0.0,   40.0),   # 右脸颊
    29: (0.0,   90.0),   # 右鼻子
}
   

def normalize_motor(value, motor_id):
    minv, maxv = motor_ranges[motor_id]
    if maxv - minv < 1e-6:
        return 0.5
    return (value - minv) / (maxv - minv)

# ==================== 2. 加载 JSON，自动提取实际出现的电机 ID 和 blendshape 键 ====================
json_path = os.path.join(BASE_DIR, '..', '..', 'data_coll', 'raw_data', 'motor_babbling_data_PC.json')
with open(json_path, 'r') as f:
    raw_data = json.load(f)

# 收集所有样本中出现的电机 ID（你的数据中就是那24个）
motor_ids_set = set()
for item in raw_data:
    for k in item.get("motor_commands", {}).keys():
        motor_ids_set.add(int(k))
USED_MOTORS = sorted(motor_ids_set)
print(f"实际使用的电机 ID: {USED_MOTORS} (共 {len(USED_MOTORS)} 个)")

# 收集所有 blendshape 键名
bs_keys_set = set()
for item in raw_data:
    for k in item.get("blendshapes", {}).keys():
        bs_keys_set.add(k)
bs_keys = sorted(bs_keys_set)
print(f"Blendshape 键数量: {len(bs_keys)}")

# 构建样本（电机值按 USED_MOTORS 顺序提取）
samples = []
for item in raw_data:
    motor_dict = item.get("motor_commands", {})
    motor_vals = [motor_dict.get(str(mid), 0.0) for mid in USED_MOTORS]  # 缺失补0（实际上你的数据不缺失）
    bs_dict = item.get("blendshapes", {})
    bs_vals = [bs_dict.get(k, 0.0) for k in bs_keys]
    samples.append({
        "motor_raw": np.array(motor_vals, dtype=np.float32),
        "blendshape": np.array(bs_vals, dtype=np.float32)
    })
print(f"加载样本数: {len(samples)}")

# 归一化电机角度
for s in samples:
    norm_vals = []
    for i, mid in enumerate(USED_MOTORS):
        norm_vals.append(normalize_motor(s["motor_raw"][i], mid))
    s["motor_norm"] = np.array(norm_vals, dtype=np.float32)

X = np.stack([s["motor_norm"] for s in samples])
y = np.stack([s["blendshape"] for s in samples])

# 划分数据集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(f"训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")

# ==================== 3. 定义 Dataset 和 DataLoader ====================
class Motor2BSDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 64
train_loader = DataLoader(Motor2BSDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(Motor2BSDataset(X_val, y_val), batch_size=batch_size)
test_loader = DataLoader(Motor2BSDataset(X_test, y_test), batch_size=batch_size)

# ==================== 4. 定义 MLP 模型 ====================
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Angle2BSNet(len(USED_MOTORS), len(bs_keys)).to(device)
print(f"使用设备: {device}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# ==================== 5. 训练循环 ====================
num_epochs = 200
best_val_loss = float('inf')
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(Xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * Xb.size(0)
    train_loss /= len(train_loader.dataset)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            pred = model(Xb)
            loss = criterion(pred, yb)
            val_loss += loss.item() * Xb.size(0)
    val_loss /= len(val_loader.dataset)
    
    scheduler.step(val_loss)
    
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()

# 测试集评估
model.load_state_dict(best_model_state)
model.eval()
test_loss = 0.0
with torch.no_grad():
    for Xb, yb in test_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        pred = model(Xb)
        loss = criterion(pred, yb)
        test_loss += loss.item() * Xb.size(0)
test_loss /= len(test_loader.dataset)
print(f"最佳验证损失: {best_val_loss:.6f}, 测试损失: {test_loss:.6f}")

# ==================== 6. 保存单个文件（模型 + 元数据） ====================
save_dict = {
    'model_state_dict': best_model_state,
    'motor_ranges': motor_ranges,       # 舵机范围（只包含出现的）
    'used_motors': USED_MOTORS,         # 实际使用的电机 ID 列表（顺序固定）
    'bs_keys': bs_keys,                 # blendshape 键顺序
    'input_dim': len(USED_MOTORS),
    'output_dim': len(bs_keys),
}
torch.save(save_dict, os.path.join(BASE_DIR, '..', 'models', 'angle2bs_full.pth'))
print("模型已保存为 angle2bs_full.pth")

# 简单测试：用第一个验证样本推理
sample_motor = torch.tensor(X_val[0:1], dtype=torch.float32).to(device)
with torch.no_grad():
    pred_bs = model(sample_motor).cpu().numpy()[0]
print("\n验证集第一个样本：")
print("真实 blendshape 前5个值:", y_val[0][:5])
print("预测 blendshape 前5个值:", pred_bs[:5])
