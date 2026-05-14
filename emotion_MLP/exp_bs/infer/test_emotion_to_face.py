#!/usr/bin/env python3
"""
情绪标签 → 52 BS (EmotionBrain) → 上下脸逆向模型 → 24 舵机角度 → UDP 8888 → 机器人面部

用法:
  # 命令行模式：指定情绪序号，支持多个实现循环过渡
  python emotion_MLP/inference/test_emotion_to_face.py 0              # Neutral
  python emotion_MLP/inference/test_emotion_to_face.py 1              # Happy
  python emotion_MLP/inference/test_emotion_to_face.py 0 1 9 13       # Neutral→Happy→Anger→Sad 循环
  python emotion_MLP/inference/test_emotion_to_face.py 1 --no-udp      # Happy，仅打印不发送UDP

  # UDP 监听模式：接收 deepseek_response.py 发来的情绪 ID（端口 5009）
  python emotion_MLP/inference/test_emotion_to_face.py --listen
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import socket
import sys
import time
import threading

import numpy as np
import torch
import torch.nn as nn


# ================= 配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

RPI_IP = "172.16.0.166"
RPI_PORT = 8888
UDP_LISTEN_PORT = 5009
FPS = 60
TICK = 1.0 / FPS

EMOTION_BRAIN_PATH = os.path.join(PROJ_ROOT, "exp_bs", "models", "emotion_brain.pth")
FORWARD_MODEL_PATH = os.path.join(PROJ_ROOT, "bs2angle", "models", "angle2bs_full.pth")
UPPER_MODEL_PATH = os.path.join(PROJ_ROOT, "bs2angle", "models", "upper_face_bs2angle.pth")
LOWER_MODEL_PATH = os.path.join(PROJ_ROOT, "bs2angle", "models", "lower_face_bs2angle.pth")

# 24 种情绪（与 train_brain.py / gen_batch_data.py 保持一致）
EMOTIONS = [
    "Neutral", "Happy", "Excitement", "Humor", "Pride",
    "Trust", "Love", "Relief", "Hope",
    "Anger", "Disgust", "Fear", "Vigilance",
    "Sad", "Loneliness", "Guilt",
    "Surprise", "Confusion", "Shyness",
    "Comfort", "Playful", "Impressed", "Concerned", "Awkward",
]


# ================= 模型结构 =================
class EmotionBrain(nn.Module):
    """24 维 one-hot → 52 维 BS（ARKit 顺序）"""
    def __init__(self, input_dim=24, output_dim=52):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, output_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class FaceBS2Angle(nn.Module):
    """双模型共用结构：BS → 舵机角度"""
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


# ================= 二阶低通平滑器（从 render_transition.py 复用） =================
class SmoothFilter:
    def __init__(self, val, f=3.0, z=0.6, r=0.0):
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


# ================= 模型加载 =================
def load_models(device):
    """加载全部 4 个模型 + 元数据"""
    # 1. EmotionBrain（19 情绪 → 52 BS）
    emotion_model = EmotionBrain().to(device)
    emotion_model.load_state_dict(torch.load(EMOTION_BRAIN_PATH, map_location=device))
    emotion_model.eval()
    print(f"  [OK] EmotionBrain: models/emotion_brain.pth")

    # 2. 正向模型（仅用于提取 bs_keys 字母序列表）
    forward_ckpt = torch.load(FORWARD_MODEL_PATH, map_location=device)
    bs_keys_full = forward_ckpt["bs_keys"]  # 52 个字母序键名
    used_motors_full = forward_ckpt["used_motors"]
    print(f"  [OK] angle2bs_full.pth: {len(bs_keys_full)} BS keys, {len(used_motors_full)} motors")

    # 3. 上半脸逆向模型
    upper_ckpt = torch.load(UPPER_MODEL_PATH, map_location=device)
    upper_bs_keys = upper_ckpt["upper_bs_keys"]
    upper_motor_ids = upper_ckpt["upper_motor_ids"]
    upper_ranges = upper_ckpt["motor_ranges"]
    upper_model = FaceBS2Angle(len(upper_bs_keys), len(upper_motor_ids)).to(device)
    upper_model.load_state_dict(upper_ckpt["model_state_dict"])
    upper_model.eval()
    print(f"  [OK] upper_face_bs2angle.pth: {len(upper_bs_keys)} BS → {len(upper_motor_ids)} motors")

    # 4. 下半脸逆向模型
    lower_ckpt = torch.load(LOWER_MODEL_PATH, map_location=device)
    lower_bs_idx = lower_ckpt["lower_bs_idx"]
    lower_bs_keys = [bs_keys_full[i] for i in lower_bs_idx]
    lower_motor_ids = lower_ckpt["lower_motor_ids"]
    lower_ranges = lower_ckpt["motor_ranges"]
    lower_model = FaceBS2Angle(len(lower_bs_keys), len(lower_motor_ids)).to(device)
    lower_model.load_state_dict(lower_ckpt["model_state_dict"])
    lower_model.eval()
    print(f"  [OK] lower_face_bs2angle.pth: {len(lower_bs_keys)} BS → {len(lower_motor_ids)} motors")

    # 合并 motor_ranges（上下脸各自的训练范围）
    motor_ranges = {}
    motor_ranges.update(upper_ranges)
    motor_ranges.update(lower_ranges)

    return {
        "emotion_model": emotion_model,
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


# ================= 核心函数 =================
def emotion_to_bs(emotion_id, model, device):
    """情绪 ID → 52-dim BS (ARKit 顺序)"""
    one_hot = torch.zeros(1, 24, device=device)
    one_hot[0, emotion_id] = 1.0
    with torch.no_grad():
        bs = model(one_hot).cpu().numpy()[0]  # (52,)
    return bs


def bs_to_servo(bs_arkit_52, models_dict):
    """
    52-dim BS (ARKit 顺序) → 24 舵机角度字典

    关键：ARKit 索引 vs 字母序 bs_keys 的映射
    bs_keys[0] = '_neutral' (ARKit index 51)
    bs_keys[1] = 'browDownLeft' (ARKit index 0)
    ...
    bs_keys[44] = 'mouthSmileLeft' (ARKit index 43)
    所以 bs_arkit_52[arkit_idx] → bs_keys[arkit_idx + 1]
    """
    bs_keys = models_dict["bs_keys_full"]
    device = models_dict["device"]

    # 构建 bs_dict: key_name → value
    # _neutral 固定为 0.0
    bs_dict = {"_neutral": 0.0}
    for arkit_idx in range(51):
        key = bs_keys[arkit_idx + 1]
        bs_dict[key] = float(bs_arkit_52[arkit_idx])

    # 上半脸推理
    upper_input = [bs_dict.get(k, 0.0) for k in models_dict["upper_bs_keys"]]
    upper_tensor = torch.tensor([upper_input], dtype=torch.float32, device=device)
    with torch.no_grad():
        upper_pred = models_dict["upper_model"](upper_tensor).cpu().numpy()[0]

    # 下半脸推理
    lower_input = [bs_dict.get(k, 0.0) for k in models_dict["lower_bs_keys"]]
    lower_tensor = torch.tensor([lower_input], dtype=torch.float32, device=device)
    with torch.no_grad():
        lower_pred = models_dict["lower_model"](lower_tensor).cpu().numpy()[0]

    # 默认角度 = 各舵机中位值
    motor_ranges = models_dict["motor_ranges"]
    default_angles = {}
    for mid in models_dict["used_motors_full"]:
        minv, maxv = motor_ranges[mid]
        default_angles[str(mid)] = float((minv + maxv) / 2.0)

    # 覆盖预测角度
    for i, mid in enumerate(models_dict["upper_motor_ids"]):
        norm_val = float(upper_pred[i])
        minv, maxv = motor_ranges[mid]
        default_angles[str(mid)] = round(norm_val * (maxv - minv) + minv, 2)

    for i, mid in enumerate(models_dict["lower_motor_ids"]):
        norm_val = float(lower_pred[i])
        minv, maxv = motor_ranges[mid]
        default_angles[str(mid)] = round(norm_val * (maxv - minv) + minv, 2)

    return default_angles


def print_bs_debug(bs_arkit, models_dict):
    """打印 BS 关键维度值，用于调试"""
    bs_keys = models_dict["bs_keys_full"]
    print("  BS key values (top activations):")
    items = []
    for arkit_idx in range(51):
        key = bs_keys[arkit_idx + 1]
        val = bs_arkit[arkit_idx]
        if val > 0.05:
            items.append((key, val))
    items.sort(key=lambda x: -x[1])
    for key, val in items[:10]:
        print(f"    {key}: {val:.3f}")


def print_angles_debug(angles_dict):
    """打印角度字典关键值"""
    print("  Servo angles:")
    items = sorted(angles_dict.items(), key=lambda x: int(x[0]))
    for mid, val in items:
        print(f"    motor {mid}: {val:.1f}")


# ================= UDP 监听器（线程） =================
class UDPListener:
    """监听 UDP 端口，接收情绪 ID 文本"""

    def __init__(self, port, callback):
        self.port = port
        self.callback = callback
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()
        print(f"  UDP 监听器已启动: 0.0.0.0:{self.port}")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)

    def _listen(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.5)
        sock.bind(("0.0.0.0", self.port))
        while self.running:
            try:
                data, addr = sock.recvfrom(1024)
                text = data.decode().strip()
                # 支持 "5" 或 "5\n" 或 "5 some_extra"
                parts = text.split()
                if parts:
                    try:
                        emotion_id = int(parts[0])
                        if 0 <= emotion_id < len(EMOTIONS):
                            self.callback(emotion_id)
                        else:
                            print(f"  [UDP] 收到无效情绪ID: {emotion_id}")
                    except ValueError:
                        print(f"  [UDP] 无法解析: {text}")
            except socket.timeout:
                continue
            except Exception as e:
                print(f"  [UDP] 错误: {e}")
        sock.close()


# ================= 主函数 =================
def main():
    parser = argparse.ArgumentParser(description="情绪标签 → 面部舵机角度 → UDP")
    parser.add_argument("emotions", type=int, nargs="*", default=None,
                        help="情绪序号（0-23），多个则循环播放")
    parser.add_argument("--listen", action="store_true",
                        help="UDP 监听模式，接收 deepseek 的情绪 ID")
    parser.add_argument("--no-udp", action="store_true",
                        help="不发送 UDP，仅打印调试信息")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="每个情绪持续秒数（默认 3.0）")
    parser.add_argument("--fps", type=int, default=FPS,
                        help="推理帧率（默认 60）")
    args = parser.parse_args()

    if not args.emotions and not args.listen:
        parser.print_help()
        print("\n错误：请指定情绪序号或使用 --listen 模式")
        sys.exit(1)

    tick = 1.0 / args.fps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print("加载模型（共 4 个）...")

    models = load_models(device)
    used_motors = models["used_motors_full"]
    print(f"合计覆盖: {len(used_motors)} 舵机 ({used_motors})")

    # UDP 发送 socket
    udp_sock = None
    if not args.no_udp:
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"UDP 目标: {RPI_IP}:{RPI_PORT}")

    # 初始化平滑器（24 维角度向量）
    init_bs = emotion_to_bs(args.emotions[0] if args.emotions else 0, models["emotion_model"], device)
    init_angles = bs_to_servo(init_bs, models)
    init_vec = np.array([init_angles[str(mid)] for mid in used_motors], dtype=np.float32)
    smoother = SmoothFilter(init_vec, f=2.5, z=0.8, r=0.0)

    # UDP 监听器
    current_emotion_id = args.emotions[0] if args.emotions else 0
    emotion_lock = threading.Lock()

    def on_emotion_received(emotion_id):
        nonlocal current_emotion_id
        with emotion_lock:
            old_id = current_emotion_id
            current_emotion_id = emotion_id
        print(f"\n>> [UDP] 情绪切换: {EMOTIONS[old_id]} → {EMOTIONS[emotion_id]} (ID:{emotion_id})")

    if args.listen:
        listener = UDPListener(UDP_LISTEN_PORT, on_emotion_received)
        listener.start()

    # ================= 主循环 =================
    print("\n开始运行... 按 Ctrl+C 退出\n")
    emotion_idx = 0
    emotion_hold_start = time.time()

    try:
        while True:
            loop_start = time.time()

            # ---- 获取当前目标情绪 ----
            if args.listen:
                with emotion_lock:
                    target_id = current_emotion_id
            else:
                # CLI 模式：按 duration 切换下一个情绪
                if time.time() - emotion_hold_start >= args.duration:
                    emotion_idx = (emotion_idx + 1) % len(args.emotions)
                    target_id = args.emotions[emotion_idx]
                    with emotion_lock:
                        current_emotion_id = target_id
                    emotion_hold_start = time.time()
                    print(f"\n>> 切换到: {EMOTIONS[target_id]} (ID:{target_id})")
                else:
                    with emotion_lock:
                        target_id = current_emotion_id

            # ---- EmotionBrain 推理 ----
            bs_arkit = emotion_to_bs(target_id, models["emotion_model"], device)

            # ---- BS → 舵机角度 ----
            raw_angles = bs_to_servo(bs_arkit, models)
            raw_vec = np.array([raw_angles[str(mid)] for mid in used_motors], dtype=np.float32)

            # ---- 平滑滤波 ----
            smooth_vec = smoother.update(raw_vec, tick)

            # ---- 构建最终角度字典 ----
            final_angles = {}
            for i, mid in enumerate(used_motors):
                final_angles[str(mid)] = round(float(smooth_vec[i]), 2)

            # ---- UDP 发送 ----
            if udp_sock:
                data = json.dumps(final_angles) + "\n"
                udp_sock.sendto(data.encode("utf-8"), (RPI_IP, RPI_PORT))

            # ---- 调试打印（每 30 帧 ≈ 0.5 秒一次） ----
            frame_num = int(time.time() * args.fps)
            if frame_num % 30 == 0:
                emotion_name = EMOTIONS[target_id]
                print(f"\r[{emotion_name:12s}] BS={bs_arkit[25]:.2f} jawOpen | "
                      f"motor[19]={final_angles.get('19', 0):.1f} motor[22]={final_angles.get('22', 0):.1f}",
                      end="", flush=True)

            # ---- 帧率控制 ----
            elapsed = time.time() - loop_start
            sleep_time = tick - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        if args.listen:
            listener.stop()
        if udp_sock:
            udp_sock.close()
        print("程序结束。")


if __name__ == "__main__":
    main()
