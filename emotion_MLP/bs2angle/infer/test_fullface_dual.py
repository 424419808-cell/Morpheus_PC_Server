#!/usr/bin/env python3
"""
全脸实时驱动 — 整合上半脸 + 下半脸双模型
同时驱动全部 24 个舵机，UDP 发送到树莓派。

用法:
  python test_fullface_dual.py                     # 本地摄像头 0
  python test_fullface_dual.py --camera 1          # 指定摄像头索引
  python test_fullface_dual.py --stream http://x.x.x.x:5000/video_feed  # HTTP 视频流
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import socket
import sys
import time

# torch 必须在 cv2/mediapipe 之前导入，
# 确保 libiomp5md.dll 先于 libomp.dll 加载，避免 OpenMP 冲突导致 fbgemm.dll 崩溃
import torch
import torch.nn as nn
import numpy as np
import cv2
import mediapipe as mp

# ================= 配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CAMERA_ID = 1
RPI_IP = "172.16.0.166"
RPI_PORT = 8888

UPPER_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "upper_face_bs2angle.pth")
LOWER_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "lower_face_bs2angle.pth")
FORWARD_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "angle2bs_full.pth")
LANDMARKER_MODEL = os.path.join(BASE_DIR, "..", "..", "face_landmarker.task")


# ================= 模型结构（双模型共用） =================
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


def load_dual_models(device):
    """加载上半脸 + 下半脸两个独立模型。
    每个模型使用各自训练时的 motor_ranges 进行反归一化。"""
    # 正向模型仅用于解析下半脸 BS 键名（lower_bs_idx → bs_key 名称）
    forward_ckpt = torch.load(FORWARD_MODEL_PATH, map_location=device)
    bs_keys_full = forward_ckpt["bs_keys"]

    # --- 上半脸模型 ---
    upper_ckpt = torch.load(UPPER_MODEL_PATH, map_location=device)
    upper_bs_keys = upper_ckpt["upper_bs_keys"]
    upper_motor_ids = upper_ckpt["upper_motor_ids"]
    upper_ranges = upper_ckpt["motor_ranges"]
    used_motors = upper_ckpt["used_motors_full"]

    upper_model = FaceBS2Angle(len(upper_bs_keys), len(upper_motor_ids)).to(device)
    upper_model.load_state_dict(upper_ckpt["model_state_dict"])
    upper_model.eval()

    # --- 下半脸模型 ---
    lower_ckpt = torch.load(LOWER_MODEL_PATH, map_location=device)
    lower_bs_idx = lower_ckpt["lower_bs_idx"]
    lower_bs_keys = [bs_keys_full[i] for i in lower_bs_idx]
    lower_motor_ids = lower_ckpt["lower_motor_ids"]
    lower_ranges = lower_ckpt["motor_ranges"]

    lower_model = FaceBS2Angle(len(lower_bs_keys), len(lower_motor_ids)).to(device)
    lower_model.load_state_dict(lower_ckpt["model_state_dict"])
    lower_model.eval()

    return (
        upper_model, upper_bs_keys, upper_motor_ids, upper_ranges,
        lower_model, lower_bs_keys, lower_motor_ids, lower_ranges,
        used_motors,
    )


def denormalize(norm_val, motor_id, motor_ranges):
    minv, maxv = motor_ranges[motor_id]
    return float(norm_val * (maxv - minv) + minv)


def open_video(camera_id=0, stream_url=None):
    if stream_url:
        cap = cv2.VideoCapture(stream_url)
    else:
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {stream_url or f'摄像头 {camera_id}'}")
    return cap


def main():
    parser = argparse.ArgumentParser(description="全脸双模型实时驱动")
    parser.add_argument("--camera", type=int, default=CAMERA_ID, help="本地摄像头索引")
    parser.add_argument("--stream", type=str, default=None, help="HTTP 视频流 URL")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载模型
    print("加载模型...")
    (
        upper_model, upper_bs_keys, upper_motor_ids, upper_ranges,
        lower_model, lower_bs_keys, lower_motor_ids, lower_ranges,
        used_motors,
    ) = load_dual_models(device)

    print(f"  上半脸: {len(upper_bs_keys)} BS → {len(upper_motor_ids)} 舵机 {upper_motor_ids}")
    print(f"  下半脸: {len(lower_bs_keys)} BS → {len(lower_motor_ids)} 舵机 {lower_motor_ids}")
    print(f"  合计覆盖: {len(used_motors)} 舵机 (全脸)")

    # 构建统一的 motor_ranges（上半/下半各自使用自己的训练范围）
    motor_ranges = {}
    motor_ranges.update(upper_ranges)
    motor_ranges.update(lower_ranges)  # 有冲突时以下半脸为准（不影响上半脸舵机）

    # 默认角度 = 各舵机中位值
    default_angles = {}
    for mid in used_motors:
        minv, maxv = motor_ranges[mid]
        default_angles[str(mid)] = float((minv + maxv) / 2.0)

    # MediaPipe
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=LANDMARKER_MODEL),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
        num_faces=1,
    )
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    # UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"UDP → {RPI_IP}:{RPI_PORT}")

    cap = open_video(args.camera, args.stream)
    print("全脸驱动运行中... 按 Q 退出\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp)

            if not result.face_blendshapes:
                cv2.imshow("Full Face Dual-Model", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # BS 字典 {名称: 值}
            bs_dict = {b.category_name: b.score for b in result.face_blendshapes[0]}

            # 上半脸推理
            upper_input = [bs_dict.get(k, 0.0) for k in upper_bs_keys]
            upper_tensor = torch.tensor([upper_input], dtype=torch.float32).to(device)
            with torch.no_grad():
                upper_pred = upper_model(upper_tensor).cpu().numpy()[0]

            # 下半脸推理
            lower_input = [bs_dict.get(k, 0.0) for k in lower_bs_keys]
            lower_tensor = torch.tensor([lower_input], dtype=torch.float32).to(device)
            with torch.no_grad():
                lower_pred = lower_model(lower_tensor).cpu().numpy()[0]

            # 合并角度 → 覆盖全部 24 个舵机（各自使用训练时的 ranges）
            angles = default_angles.copy()
            for i, mid in enumerate(upper_motor_ids):
                angles[str(mid)] = round(denormalize(upper_pred[i], mid, upper_ranges), 2)
            for i, mid in enumerate(lower_motor_ids):
                angles[str(mid)] = round(denormalize(lower_pred[i], mid, lower_ranges), 2)

            # UDP 发送
            data = json.dumps(angles) + "\n"
            sock.sendto(data.encode("utf-8"), (RPI_IP, RPI_PORT))

            cv2.imshow("Full Face Dual-Model", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        sock.close()
        print("程序结束。")


if __name__ == "__main__":
    main()
