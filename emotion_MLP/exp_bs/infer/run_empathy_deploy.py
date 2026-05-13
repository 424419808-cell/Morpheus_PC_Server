"""
共情模型部署 — BS2BS_Empathy → FaceBS2Angle → UDP:8888 → RK3588 → 舵机

用法:
  python exp_bs/infer/run_empathy_deploy.py                   # 摄像头 → 共情 → 舵机
  python exp_bs/infer/run_empathy_deploy.py --no-udp         # 仅打印，不发送
  python exp_bs/infer/run_empathy_deploy.py --cam 1          # 指定摄像头

等机器人脸皮修改完成后，此脚本替代 test_emotion_to_face.py 的管线。
"""
import argparse
import json
import os
import socket
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.join(BASE_DIR, "..", "..")

EMPATHY_MODEL_PATH = os.path.join(PROJ_ROOT, "exp_bs", "models", "bs2bs_empathy.pth")
UPPER_MODEL_PATH = os.path.join(PROJ_ROOT, "bs2angle", "models", "upper_face_bs2angle.pth")
LOWER_MODEL_PATH = os.path.join(PROJ_ROOT, "bs2angle", "models", "lower_face_bs2angle.pth")
FORWARD_MODEL_PATH = os.path.join(PROJ_ROOT, "bs2angle", "models", "angle2bs_full.pth")
LANDMARKER_PATH = os.path.join(PROJ_ROOT, "face_landmarker.task")

RPI_IP = "172.16.0.166"
RPI_PORT = 8888
FPS = 30


class BS2BS_Empathy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(52, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 52), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class FaceBS2Angle(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(256, 128, 64)):
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


class SmoothFilter:
    def __init__(self, val, f=2.5, z=0.8, r=0.0):
        self.xp = val.copy()
        self.y = val.copy()
        self.yd = np.zeros_like(val)
        self.k1 = z / (np.pi * f)
        self.k2 = 1.0 / ((2 * np.pi * f) ** 2)
        self.k3 = r * z / (2 * np.pi * f)

    def update(self, x, dt):
        xd = (x - self.xp) / dt
        self.xp = x.copy()
        self.y = self.y + dt * self.yd
        self.yd = self.yd + dt * (x + self.k3 * xd - self.y - self.k1 * self.yd) / self.k2
        return self.y


def load_models(device):
    """加载共情模型 + 上下半脸逆向模型"""
    # 共情模型
    empathy_model = BS2BS_Empathy().to(device)
    if os.path.exists(EMPATHY_MODEL_PATH):
        empathy_model.load_state_dict(torch.load(EMPATHY_MODEL_PATH, map_location=device))
        print(f"[OK] bs2bs_empathy.pth")
    else:
        print("[警告] 共情模型未训练，将使用恒等映射")
    empathy_model.eval()

    # 正向模型（提取元数据）
    forward_ckpt = torch.load(FORWARD_MODEL_PATH, map_location=device)
    bs_keys_full = forward_ckpt["bs_keys"]
    used_motors_full = forward_ckpt["used_motors"]
    print(f"[OK] angle2bs_full.pth: {len(bs_keys_full)} BS, {len(used_motors_full)} motors")

    # 上半脸逆向模型
    upper_ckpt = torch.load(UPPER_MODEL_PATH, map_location=device)
    upper_bs_keys = upper_ckpt["upper_bs_keys"]
    upper_motor_ids = upper_ckpt["upper_motor_ids"]
    upper_ranges = upper_ckpt["motor_ranges"]
    upper_model = FaceBS2Angle(len(upper_bs_keys), len(upper_motor_ids)).to(device)
    upper_model.load_state_dict(upper_ckpt["model_state_dict"])
    upper_model.eval()
    print(f"[OK] upper_face_bs2angle.pth")

    # 下半脸逆向模型
    lower_ckpt = torch.load(LOWER_MODEL_PATH, map_location=device)
    lower_bs_idx = lower_ckpt["lower_bs_idx"]
    lower_bs_keys = [bs_keys_full[i] for i in lower_bs_idx]
    lower_motor_ids = lower_ckpt["lower_motor_ids"]
    lower_ranges = lower_ckpt["motor_ranges"]
    lower_model = FaceBS2Angle(len(lower_bs_keys), len(lower_motor_ids)).to(device)
    lower_model.load_state_dict(lower_ckpt["model_state_dict"])
    lower_model.eval()
    print(f"[OK] lower_face_bs2angle.pth")

    motor_ranges = {}
    motor_ranges.update(upper_ranges)
    motor_ranges.update(lower_ranges)

    return {
        "empathy_model": empathy_model,
        "upper_model": upper_model,
        "lower_model": lower_model,
        "bs_keys_full": bs_keys_full,
        "upper_bs_keys": upper_bs_keys,
        "lower_bs_keys": lower_bs_keys,
        "upper_motor_ids": upper_motor_ids,
        "lower_motor_ids": lower_motor_ids,
        "motor_ranges": motor_ranges,
        "used_motors_full": used_motors_full,
        "device": device,
    }


def bs_to_servo(bs_arkit_52, models_dict):
    """52 BS (ARKit 顺序) → 24 舵机角度字典"""
    bs_keys = models_dict["bs_keys_full"]
    device = models_dict["device"]

    bs_dict = {"_neutral": 0.0}
    for arkit_idx in range(51):
        key = bs_keys[arkit_idx + 1]
        bs_dict[key] = float(bs_arkit_52[arkit_idx])

    upper_input = [bs_dict.get(k, 0.0) for k in models_dict["upper_bs_keys"]]
    upper_tensor = torch.tensor([upper_input], dtype=torch.float32, device=device)
    with torch.no_grad():
        upper_pred = models_dict["upper_model"](upper_tensor).cpu().numpy()[0]

    lower_input = [bs_dict.get(k, 0.0) for k in models_dict["lower_bs_keys"]]
    lower_tensor = torch.tensor([lower_input], dtype=torch.float32, device=device)
    with torch.no_grad():
        lower_pred = models_dict["lower_model"](lower_tensor).cpu().numpy()[0]

    motor_ranges = models_dict["motor_ranges"]
    default_angles = {}
    for mid in models_dict["used_motors_full"]:
        minv, maxv = motor_ranges[mid]
        default_angles[str(mid)] = float((minv + maxv) / 2.0)

    for i, mid in enumerate(models_dict["upper_motor_ids"]):
        norm_val = float(upper_pred[i])
        minv, maxv = motor_ranges[mid]
        default_angles[str(mid)] = round(norm_val * (maxv - minv) + minv, 2)

    for i, mid in enumerate(models_dict["lower_motor_ids"]):
        norm_val = float(lower_pred[i])
        minv, maxv = motor_ranges[mid]
        default_angles[str(mid)] = round(norm_val * (maxv - minv) + minv, 2)

    return default_angles


def main():
    parser = argparse.ArgumentParser(description="共情模型部署 — 机器人驱动")
    parser.add_argument("--no-udp", action="store_true", help="不发送UDP")
    parser.add_argument("--cam", type=int, default=0, help="摄像头ID")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    models = load_models(device)
    used_motors = models["used_motors_full"]

    # 初始化 MediaPipe
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=LANDMARKER_PATH),
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[错误] 无法打开摄像头 {args.cam}")
        sys.exit(1)

    # UDP 发送
    udp_sock = None
    if not args.no_udp:
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"UDP 目标: {RPI_IP}:{RPI_PORT}")

    # 平滑器
    init_vec = np.zeros(len(used_motors), dtype=np.float32)
    smoother = SmoothFilter(init_vec, f=2.5, z=0.8, r=0.0)

    tick = 1.0 / FPS
    print(f"\n共情推理部署中... (Ctrl+C 退出)\n")

    try:
        while cap.isOpened():
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                continue

            timestamp = int(time.time() * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, timestamp)

            if result.face_blendshapes:
                user_bs = np.array([b.score for b in result.face_blendshapes[0]], dtype=np.float32)

                # 共情 → 舵机
                with torch.no_grad():
                    tensor = torch.tensor([user_bs], dtype=torch.float32, device=device)
                    empathy_bs = models["empathy_model"](tensor).cpu().numpy()[0]

                raw_angles = bs_to_servo(empathy_bs, models)
                raw_vec = np.array([raw_angles[str(mid)] for mid in used_motors], dtype=np.float32)
                smooth_vec = smoother.update(raw_vec, tick)

                final_angles = {}
                for i, mid in enumerate(used_motors):
                    final_angles[str(mid)] = round(float(smooth_vec[i]), 2)

                if udp_sock:
                    data = json.dumps(final_angles) + "\n"
                    udp_sock.sendto(data.encode("utf-8"), (RPI_IP, RPI_PORT))

                # 调试输出
                frame_count = int(time.time() * FPS)
                if frame_count % 30 == 0:
                    diff = np.mean(np.abs(empathy_bs - user_bs))
                    print(f"\rBS差异={diff:.4f} | motor[19]={final_angles.get('19', 0):.1f} "
                          f"motor[22]={final_angles.get('22', 0):.1f}" + " " * 10, end="", flush=True)
            else:
                print(f"\r[无人脸] 保持中性" + " " * 30, end="", flush=True)

            elapsed = time.time() - loop_start
            sleep_time = tick - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        cap.release()
        landmarker.close()
        if udp_sock:
            udp_sock.close()


if __name__ == "__main__":
    main()
