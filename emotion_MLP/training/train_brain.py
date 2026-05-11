import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from gen_batch_data import get_base_bs

class EmotionBrain(nn.Module):
    def __init__(self, input_dim=19, output_dim=52):
        super(EmotionBrain, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 52),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

emotions = ["Neutral", "Happy", "Excitement", "Humor", "Pride", "Trust", "Love", "Relief", "Hope", "Anger", "Disgust", "Fear", "Vigilance", "Sad", "Loneliness", "Guilt", "Surprise", "Confusion", "Shyness"]

print(">>> 正在生成带噪声的增强数据...")
x_train, y_train = [], []
for i, emo in enumerate(emotions):
    base_vec = get_base_bs(emo)
    for _ in range(200):  # 增加到200个变体
        noise = np.random.normal(0, 0.02, 52)
        scale = np.random.uniform(0.85, 1.15)
        variant = np.clip(base_vec * scale + noise, 0, 1)
        
        one_hot = np.zeros(len(emotions))
        one_hot[i] = 1.0
        x_train.append(one_hot)
        y_train.append(variant)

x_train = torch.tensor(np.array(x_train)).float()
y_train = torch.tensor(np.array(y_train)).float()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionBrain().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print(f">>> 开始训练 (使用设备: {device})...")
for epoch in range(3001):
    optimizer.zero_grad()
    pred = model(x_train.to(device))
    loss = criterion(pred, y_train.to(device))
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"Epoch [{epoch}/3000], Loss: {loss.item():.6f}")

os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), "../models/emotion_brain.pth")
print(">>> 训练完成！模型已存为 models/emotion_brain.pth")
