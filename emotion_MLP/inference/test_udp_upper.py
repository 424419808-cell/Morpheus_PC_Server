#!/usr/bin/env python3
"""
WSL 端：实时上半脸推理 + UDP 发送（不按空格，连续驱动）
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
VIDEO_STREAM_URL = "http://172.16.1.55:5000/video_feed"
RPI_IP = "172.16.0.166"
RPI_PORT = 8888
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "upper_face_bs2angle.pth")
LANDMARKER_MODEL = os.path.join(BASE_DIR, "..", "face_landmarker.task")

# ================= 模型结构定义 =================
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 使用设备: {device}")

# ================= 加载模型 =================
checkpoint = torch.load(MODEL_PATH, map_location=device)
upper_bs_keys = checkpoint['upper_bs_keys']
upper_motor_ids = checkpoint['upper_motor_ids']
motor_ranges = checkpoint['motor_ranges']
used_motors_full = checkpoint['used_motors_full']

model = UpperFaceBS2Angle(
    input_dim=len(upper_bs_keys),
    output_dim=len(upper_motor_ids)
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✅ 上半脸模型加载成功，输出舵机: {upper_motor_ids}")

# 中性默认角度
default_angles = {}
for mid in used_motors_full:
    minv, maxv = motor_ranges[mid]
    default_angles[str(mid)] = (minv + maxv) / 2.0

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
    cap = cv2.VideoCapture(VIDEO_STREAM_URL)
    if not cap.isOpened():
        print(f"❌ 无法打开视频流: {VIDEO_STREAM_URL}")
        sock.close()
        return

    fps_deque = deque(maxlen=30)
    prev_time = time.time()
    frame_timestamp_ms = 0

    print("\n🎯 实时模式：无需按键，连续驱动上半脸舵机。按 Q 退出。\n")

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

            # 提取上半脸 BS 并推理
            blendshapes = detection_result.face_blendshapes[0]
            bs_dict_raw = {bs.category_name: bs.score for bs in blendshapes}
            bs_upper_input = [bs_dict_raw.get(key, 0.0) for key in upper_bs_keys]

            input_tensor = torch.tensor([bs_upper_input], dtype=torch.float32).to(device)
            with torch.no_grad():
                pred_upper_norm = model(input_tensor).cpu().numpy()[0]

            # 构造完整角度指令
            angles_dict = default_angles.copy()
            for i, mid in enumerate(upper_motor_ids):
                raw = denormalize_angle(pred_upper_norm[i], mid)
                angles_dict[str(mid)] = round(raw, 2)

            # UDP 发送
            data = json.dumps(angles_dict) + '\n'
            sock.sendto(data.encode('utf-8'), (RPI_IP, RPI_PORT))

        cv2.putText(display_frame, "Live Mode (UDP)", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow('WSL Upper Face Realtime', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    sock.close()
    landmarker.close()
    print("程序结束。")

if __name__ == "__main__":
    main()
