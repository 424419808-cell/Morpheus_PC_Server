import torch
import numpy as np
import sys
import os
import subprocess

# 1. 模型定义
class EmotionBrain(torch.nn.Module):
    def __init__(self, input_dim=19, output_dim=52):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 52), torch.nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# 2. 二阶平滑器
class SmoothFilter:
    def __init__(self, val, f=3.0, z=0.6, r=0.0):
        self.xp = val
        self.y = val
        self.yd = np.zeros_like(val)
        self.k1 = z / (np.pi * f)
        self.k2 = 1.0 / ((2 * np.pi * f)**2)
        self.k3 = r * z / (2 * np.pi * f)

    def update(self, x, dt):
        xd = (x - self.xp) / dt
        self.xp = x
        self.y = self.y + dt * self.yd
        self.yd = self.yd + dt * (x + self.k3*xd - self.y - self.k1*self.yd) / self.k2
        return self.y

def generate_multi_transition(indices):
    emotions = ["Neutral", "Happy", "Excitement", "Humor", "Pride", "Trust", "Love", "Relief", "Hope", "Anger", "Disgust", "Fear", "Vigilance", "Sad", "Loneliness", "Guilt", "Surprise", "Confusion", "Shyness"]
    
    # 加载模型
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model = EmotionBrain()
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, "..", "exp_bs", "models", "emotion_brain.pth")))
    model.eval()

    def get_vec(idx):
        one_hot = torch.zeros(1, 19)
        one_hot[0, idx] = 1.0
        with torch.no_grad():
            return model(one_hot).numpy()[0]

    # 获取所有目标向量
    targets = [get_vec(i) for i in indices]
    
    # 模拟设置
    fps = 30
    dt = 1/fps
    # 每个阶段分配的时长 (秒)
    stage_duration = 1.5 
    frames_per_stage = int(stage_duration * fps)
    
    # 初始化平滑器 (从第一个点开始)
    smoother = SmoothFilter(targets[0], f=2.5, z=0.8, r=0.0)
    
    all_frames = []
    
    # 逐阶段生成动画 (X->Y, 然后 Y->Z)
    for s in range(len(targets) - 1):
        target_vec = targets[s+1]
        print(f">>> 计算阶段 {s+1}: {emotions[indices[s]]} -> {emotions[indices[s+1]]}")
        for f in range(frames_per_stage):
            current = smoother.update(target_vec, dt)
            all_frames.append(current.copy())
    
    # 保存 NPY
    path_str = "_to_".join([emotions[i] for i in indices])
    output_name = f"multi_{path_str}"
    os.makedirs("./result/transitions", exist_ok=True)
    np.save(f"./result/{output_name}.npy", np.array(all_frames).astype(np.float32))

    # 渲染
    print(f">>> 正在 Blender 渲染动画序列...")
    subprocess.run(f"./blender/blender -t 64 -b ./render.blend -P ./render.py -- ./result/ {output_name}", shell=True)
    
    # 合成视频
    ffmpeg_cmd = f"ffmpeg -y -r 30 -i ./result/{output_name}/%d.png -c:v libx264 -pix_fmt yuv420p ./result/{output_name}.mp4 -loglevel error"
    subprocess.run(ffmpeg_cmd, shell=True)
    print(f">>> [完成] 连续过渡视频已生成: ./result/{output_name}.mp4")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("用法: python3 render_transition.py [X序号] [Y序号] [Z序号]")
    else:
        # 将输入的参数全部转化为序号列表
        indices = [int(arg) for arg in sys.argv[1:]]
        generate_multi_transition(indices)
