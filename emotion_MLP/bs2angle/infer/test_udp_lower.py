#!/usr/bin/env python3
"""
WSL 端：实时下半脸推理 + UDP 发送（不按空格，连续驱动）
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import json
import socket
from collections import deque

# ================= 配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 本地摄像头用整数索引（0=默认摄像头），网络流用 URL 字符串
VIDEO_SOURCE = 1  # 如果还是 ivcam，试试 2 或 3
RPI_IP = "172.16.0.166"
RPI_PORT = 8888
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "lower_face_bs2angle.pth")
FORWARD_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "angle2bs_full.pth")
LANDMARKER_MODEL = os.path.join(BASE_DIR, "..", "..", "face_landmarker.task")

# ================= 模型结构定义 =================
class LowerFaceBS2Angle(nn.Module):
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 使用设备: {device}")

# ================= 加载模型 =================
# 加载下半脸反向模型
checkpoint = torch.load(MODEL_PATH, map_location=device)
lower_motor_ids = checkpoint['lower_motor_ids']
motor_ranges = checkpoint['motor_ranges']
used_motors_full = checkpoint['used_motors_full']

# 从正向模型获取 full bs_keys，再通过 lower_bs_idx 还原下半脸 BS 键名
forward_ckpt = torch.load(FORWARD_MODEL_PATH, map_location=device)
bs_keys_full = forward_ckpt['bs_keys']
lower_bs_idx = checkpoint['lower_bs_idx']
lower_bs_keys = [bs_keys_full[i] for i in lower_bs_idx]

model = LowerFaceBS2Angle(
    input_dim=len(lower_bs_keys),
    output_dim=len(lower_motor_ids)
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✅ 下半脸模型加载成功，BS 数量: {len(lower_bs_keys)}, 输出舵机: {lower_motor_ids}")

# ================= 舵机安全限幅表 (从中提取默认角度) =================
TABLE_V_CONFIG = {
    0:  (0.0,   90.0,  90.0),   1:  (27.0,  100.0, 100.0),  2:  (70.0,  105.0, 105.0),
    3:  (70.0,  100.0,  100.0), 4:  (20.0,  120.0, 30.0),   5:  (70.0,  100.0, 70.0),
    6:  (30.0,  110.0,  60.0),  7:  (54.0,  90.0,  90.0),   8:  (45.0,  135.0, 90.0),
    9:  (80.0,  90.0,  90.0),   10: (70.0,  90.0, 70.0),    11: (70.0,  100.0, 70.0),
    12: (90.0,  140.0, 135.0),  13: (45.0,  135.0, 90.0),   14: (110.0, 180.0, 180.0),
    15: (60.0,  105.0, 105.0),  16: (130.0, 165.0, 130.0),  17: (40.0,  90.0,  50.0),
    18: (45.0,  135.0, 90.0),   19: (70.0,  130.0, 70.0),   20: (20.0,  62.0,  62.0),
    21: (45.0,  135.0, 90.0),   22: (40.0,  100.0, 100.0),  23: (55.0,  95.0,  75.0),
    24: (20.0,  50.0,  40.0),   25: (60.0,  150.0, 140.0),  26: (65.0,  100.0, 90.0),
    27: (60.0,  110.0, 90.0),   28: (0.0,   50.0,  0.0),    29: (0.0,   90.0,  0.0),
    30: (50.0,  165.0, 150.0),  31: (20.0,  100.0, 34.0),   32: (0,     180.0, 90)
}

# 中性默认角度
default_angles = {}
for mid in used_motors_full:
    if mid in TABLE_V_CONFIG:
        # 使用表中的第三个值（索引为2）作为绝对安全的中性起点
        default_angles[str(mid)] = float(TABLE_V_CONFIG[mid][2])
    else:
        # 兜底逻辑：如果表中没有，再取中值
        minv, maxv = motor_ranges[mid]
        default_angles[str(mid)] = float((minv + maxv) / 2.0)

# ================= MediaPipe 初始化 =================
base_options = python.BaseOptions(model_asset_path=LANDMARKER_MODEL)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)
landmarker = vision.FaceLandmarker.create_from_options(options)

# ================= UDP Socket =================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(f"✅ UDP Socket 就绪，目标 {RPI_IP}:{RPI_PORT}")

def denormalize_angle(norm, motor_id):
    minv, maxv = motor_ranges[motor_id]
    return norm * (maxv - minv) + minv

# ================= 主循环 =================
def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"❌ 无法打开视频源: {VIDEO_SOURCE}")
        sock.close()
        return

    fps_deque = deque(maxlen=30)
    prev_time = time.time()
    frame_timestamp_ms = 0

    print("\n🎯 实时模式：无需按键，连续驱动下半脸舵机。按 Q 退出。\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 视频流中断，尝试重连...")
            time.sleep(1)
            continue

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        frame_timestamp_ms = int(time.time() * 1000)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # FPS 计算
        curr_time = time.time()
        fps_deque.append(1.0 / (curr_time - prev_time + 1e-6))
        prev_time = curr_time
        avg_fps = sum(fps_deque) / len(fps_deque)

        cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if detection_result.face_blendshapes:
            cv2.putText(display_frame, "Face Detected", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 提取下半脸 BS 并推理
            blendshapes = detection_result.face_blendshapes[0]
            bs_dict_raw = {bs.category_name: bs.score for bs in blendshapes}
            bs_lower_input = [bs_dict_raw.get(key, 0.0) for key in lower_bs_keys]

            input_tensor = torch.tensor([bs_lower_input], dtype=torch.float32).to(device)
            with torch.no_grad():
                pred_lower_norm = model(input_tensor).cpu().numpy()[0]

            # 构造完整角度指令
            angles_dict = default_angles.copy()
            for i, mid in enumerate(lower_motor_ids):
                raw = denormalize_angle(pred_lower_norm[i], mid)
                
                # 【修改这里：对输出进行 0.5 度的量化取整】
                # 例如：90.1 和 90.2 都会变成 90.0；90.4 会变成 90.5
                quantized_angle = float(round(float(raw) * 2) / 2.0)
                
                angles_dict[str(mid)] = quantized_angle

            # UDP 发送
            data = json.dumps(angles_dict) + '\n'
            sock.sendto(data.encode('utf-8'), (RPI_IP, RPI_PORT))

        cv2.putText(display_frame, "Live Mode (UDP - Lower Face)", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow('WSL Lower Face Realtime', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    sock.close()
    landmarker.close()
    print("程序结束。")

if __name__ == "__main__":
    main()
