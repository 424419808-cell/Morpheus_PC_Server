#!/usr/bin/env python3
"""
情感协处理器 — 语音情感双脑架构的 PC 端组件

功能：
  1. UDP 接收 ASR 语音文本（默认端口 5012）
  2. DeepSeek Chat API 推理：文本 → 共情情绪 ID (0-23)
  3. 情绪 ID → EmotionBrain → 52 BS → BS2Angle → 24 舵机角度 → UDP:8888
  4. 转发 ASR 文本到 RK3588（UDP:5011），供其 LLM 生成回答

用法：
  # UDP 监听模式（与 ASR Ear 配合使用）
  python emotion_MLP/brain/emotion_coprocessor.py

  # 手动测试模式（单次文本输入，不启动 UDP）
  python emotion_MLP/brain/emotion_coprocessor.py --test-text "今天好开心啊"
  python emotion_MLP/brain/emotion_coprocessor.py --test-text "今天被老板骂了"

  # 调试：打印 BS 和角度详情
  python emotion_MLP/brain/emotion_coprocessor.py --test-text "你好" --debug
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import re
import socket
import sys
import threading
import time

import numpy as np
import torch
import torch.nn as nn
from openai import OpenAI


# ================= 配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(BASE_DIR)  # emotion_MLP/

# --- DeepSeek LLM ---
DEEPSEEK_API_KEY = "sk-ca5be62dcb3f4d1f912a576e8742f4fd"
LLM_MODEL = "deepseek-chat"
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# --- UDP 端口 ---
ASR_LISTEN_PORT = 5012       # 接收 ASR 文本（避开 deepseek_response 的 5008）
SERVO_TARGET_IP = "172.16.0.166"  # 舵机目标（RK3588 实际 IP，与 test_emotion_to_face 保持一致）
SERVO_TARGET_PORT = 8888
RK3588_IP = "10.192.53.8"   # RK3588 IP（用于转发 ASR 文本）
RK3588_TEXT_PORT = 5011      # RK3588 LLM 接收端口

# --- 模型路径 ---
EMOTION_BRAIN_PATH = os.path.join(PROJ_ROOT, "exp_bs", "models", "emotion_brain.pth")
FORWARD_MODEL_PATH = os.path.join(PROJ_ROOT, "bs2angle", "models", "angle2bs_full.pth")
UPPER_MODEL_PATH = os.path.join(PROJ_ROOT, "bs2angle", "models", "upper_face_bs2angle.pth")
LOWER_MODEL_PATH = os.path.join(PROJ_ROOT, "bs2angle", "models", "lower_face_bs2angle.pth")

# 24 种情绪（与 gen_batch_data.py / test_emotion_to_face.py 一致）
EMOTIONS = [
    "Neutral", "Happy", "Excitement", "Humor", "Pride",
    "Trust", "Love", "Relief", "Hope",
    "Anger", "Disgust", "Fear", "Vigilance",
    "Sad", "Loneliness", "Guilt",
    "Surprise", "Confusion", "Shyness",
    "Comfort", "Playful", "Impressed", "Concerned", "Awkward",
]

# === 情绪 LLM 系统提示词 ===
EMOTION_SYSTEM_PROMPT = (
    "你是一个情感共情分析器。你的任务是根据用户说的话，推断用户当前的情绪，"
    "并决定机器人应该表达什么情绪来回应用户。\n\n"
    "核心原则：根据场景判断是镜像用户的情绪还是表达共情。\n"
    "- 如果用户表达强烈负面情绪（如极度悲伤），适当镜像可以表达理解\n"
    "- 如果用户需要安慰，可以混合表达（如悲伤中带着关爱）\n"
    "- 一般日常对话保持温和积极的回应\n\n"
    "输出格式（二选一）：\n"
    "  单情绪： (ID: N)\n"
    "  混合情绪：(ID: N1:W1, N2:W2, ...) 其中W是权重(0-1)，所有权重建议和为1\n\n"
    "示例：\n"
    "  用户：\"今天好开心啊\" → (ID: 1)  # 单情绪：Happy\n"
    "  用户：\"好难过，感觉要撑不下去了\" → (ID: 13:0.5, 6:0.5)  # 混合：Sad + Love\n"
    "  用户：\"气死我了！\" → (ID: 9:0.6, 17:0.4)  # 混合：Anger + Confusion\n\n"
    "情绪ID对应表：\n"
    "0=Neutral, 1=Happy, 2=Excitement, 3=Humor, 4=Pride, "
    "5=Trust, 6=Love, 7=Relief, 8=Hope, "
    "9=Anger, 10=Disgust, 11=Fear, 12=Vigilance, "
    "13=Sad, 14=Loneliness, 15=Guilt, "
    "16=Surprise, 17=Confusion, 18=Shyness, "
    "19=Comfort, 20=Playful, 21=Impressed, 22=Concerned, 23=Awkward\n\n"
    "只输出(ID: ...)，不要输出任何其他内容。"
)


# ================= 模型结构（复用 test_emotion_to_face.py） =================
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
    """BS → 舵机角度 MLP"""
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
    """二阶低通滤波器（复用 test_emotion_to_face.py）"""
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


# ================= 模型加载（复用 test_emotion_to_face.py） =================
def load_models(device):
    """加载全部 4 个模型 + 元数据"""
    # 1. EmotionBrain（24 情绪 → 52 BS）
    emotion_model = EmotionBrain().to(device)
    emotion_model.load_state_dict(torch.load(EMOTION_BRAIN_PATH, map_location=device))
    emotion_model.eval()
    print(f"  [OK] EmotionBrain: {EMOTION_BRAIN_PATH}")

    # 2. 正向模型（提取 bs_keys 字母序列表 + motor_ranges）
    forward_ckpt = torch.load(FORWARD_MODEL_PATH, map_location=device)
    bs_keys_full = forward_ckpt["bs_keys"]
    used_motors_full = forward_ckpt["used_motors"]
    print(f"  [OK] angle2bs_full: {len(bs_keys_full)} BS keys, {len(used_motors_full)} motors")

    # 3. 上半脸逆向模型
    upper_ckpt = torch.load(UPPER_MODEL_PATH, map_location=device)
    upper_bs_keys = upper_ckpt["upper_bs_keys"]
    upper_motor_ids = upper_ckpt["upper_motor_ids"]
    upper_ranges = upper_ckpt["motor_ranges"]
    upper_model = FaceBS2Angle(len(upper_bs_keys), len(upper_motor_ids)).to(device)
    upper_model.load_state_dict(upper_ckpt["model_state_dict"])
    upper_model.eval()
    print(f"  [OK] upper_face: {len(upper_bs_keys)} BS → {len(upper_motor_ids)} motors")

    # 4. 下半脸逆向模型
    lower_ckpt = torch.load(LOWER_MODEL_PATH, map_location=device)
    lower_bs_idx = lower_ckpt["lower_bs_idx"]
    lower_bs_keys = [bs_keys_full[i] for i in lower_bs_idx]
    lower_motor_ids = lower_ckpt["lower_motor_ids"]
    lower_ranges = lower_ckpt["motor_ranges"]
    lower_model = FaceBS2Angle(len(lower_bs_keys), len(lower_motor_ids)).to(device)
    lower_model.load_state_dict(lower_ckpt["model_state_dict"])
    lower_model.eval()
    print(f"  [OK] lower_face: {len(lower_bs_keys)} BS → {len(lower_motor_ids)} motors")

    # 合并 motor_ranges
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


# ================= 核心函数（复用 test_emotion_to_face.py） =================
def emotion_to_bs(emotion_id, model, device):
    """情绪 ID → 52-dim BS (ARKit 顺序)"""
    one_hot = torch.zeros(1, 24, device=device)
    one_hot[0, emotion_id] = 1.0
    with torch.no_grad():
        bs = model(one_hot).cpu().numpy()[0]
    return bs


def blend_emotions(weighted_emotions, model, device):
    """
    混合多种情绪 → 52-dim BS

    参数:
        weighted_emotions: list of [(emotion_id, weight), ...]
        例如 [(13, 0.6), (6, 0.4)] = 60% Sad + 40% Love
    返回:
        52-dim BS 数组，值域 [0, 1]
    """
    if not weighted_emotions:
        return emotion_to_bs(0, model, device)

    bs_total = np.zeros(52)
    weight_sum = 0.0
    for eid, w in weighted_emotions:
        bs_total += w * emotion_to_bs(eid, model, device)
        weight_sum += w

    if weight_sum > 0:
        bs_total /= weight_sum
    return np.clip(bs_total, 0, 1)


def bs_to_servo(bs_arkit_52, models_dict):
    """52-dim BS → 24 舵机角度字典"""
    bs_keys = models_dict["bs_keys_full"]
    device = models_dict["device"]

    bs_dict = {"_neutral": 0.0}
    for arkit_idx in range(51):
        key = bs_keys[arkit_idx + 1]
        bs_dict[key] = float(bs_arkit_52[arkit_idx])

    # 上半脸
    upper_input = [bs_dict.get(k, 0.0) for k in models_dict["upper_bs_keys"]]
    upper_tensor = torch.tensor([upper_input], dtype=torch.float32, device=device)
    with torch.no_grad():
        upper_pred = models_dict["upper_model"](upper_tensor).cpu().numpy()[0]

    # 下半脸
    lower_input = [bs_dict.get(k, 0.0) for k in models_dict["lower_bs_keys"]]
    lower_tensor = torch.tensor([lower_input], dtype=torch.float32, device=device)
    with torch.no_grad():
        lower_pred = models_dict["lower_model"](lower_tensor).cpu().numpy()[0]

    # 默认角度 = 中位值
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


# ================= 情绪 LLM 推理 =================
def infer_emotion_id(text):
    """
    调用 DeepSeek Chat API，从用户文本推断共情情绪 ID

    返回: list of [(emotion_id, weight), ...]
    例如: [(13, 0.6), (6, 0.4)] 或 [(1, 1.0)]
    失败时返回 [(0, 1.0)] (Neutral)
    """
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": EMOTION_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.3,
            max_tokens=50,
        )
        reply = response.choices[0].message.content.strip()

        # 匹配混合格式: (ID: 13:0.6, 6:0.4)
        blend_match = re.findall(r"\(ID:\s*(\d+:\s*[\d.]+(?:\s*,\s*\d+:\s*[\d.]+)*)\)", reply)
        if blend_match:
            pairs = []
            for pair_str in blend_match[0].split(","):
                pair_str = pair_str.strip()
                # 解析 "13:0.6"
                id_match = re.match(r"(\d+)\s*:\s*([\d.]+)", pair_str)
                if id_match:
                    eid = int(id_match.group(1))
                    w = float(id_match.group(2))
                    if 0 <= eid < len(EMOTIONS) and w > 0:
                        pairs.append((eid, w))
            if pairs:
                return pairs

        # 匹配单情绪格式: (ID: 5)
        single_match = re.search(r"\(ID:\s*(\d+)\)", reply)
        if single_match:
            emotion_id = int(single_match.group(1))
            if 0 <= emotion_id < len(EMOTIONS):
                return [(emotion_id, 1.0)]

        print(f"  情绪 LLM 返回格式异常: {reply}")
    except Exception as e:
        print(f"  情绪 LLM 调用失败: {e}")

    return [(0, 1.0)]


# ================= UDP 通信 =================
def create_udp_socket():
    """创建 UDP 发送 socket"""
    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


class ASRTextListener:
    """UDP 监听 ASR 文本，线程安全地更新最新文本"""

    def __init__(self, port, on_text_received=None):
        self.port = port
        self.on_text_received = on_text_received
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()
        print(f"  ASR 文本监听器已启动: 0.0.0.0:{self.port}")

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
                data, addr = sock.recvfrom(4096)
                text = data.decode("utf-8").strip()
                if text and self.on_text_received:
                    self.on_text_received(text)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"  [UDP 监听错误] {e}")
        sock.close()


# ================= 主程序 =================
def main():
    parser = argparse.ArgumentParser(description="情感协处理器 — 语音→情绪→舵机")
    parser.add_argument("--test-text", type=str, default=None,
                        help="手动输入测试文本，不启动 UDP 监听")
    parser.add_argument("--no-udp", action="store_true",
                        help="不发送 UDP 舵机指令（仅打印调试）")
    parser.add_argument("--debug", action="store_true",
                        help="打印详细的 BS 和角度信息")
    args = parser.parse_args()

    # ---- 设备 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ---- 加载模型 ----
    print("加载舵机映射模型（共 4 个）...")
    models = load_models(device)
    used_motors = models["used_motors_full"]
    print(f"合计覆盖: {len(used_motors)} 舵机")

    # ---- UDP socket（舵机发送） ----
    servo_sock = None
    if not args.no_udp:
        servo_sock = create_udp_socket()
        print(f"舵机 UDP 目标: {SERVO_TARGET_IP}:{SERVO_TARGET_PORT}")

    # ---- UDP socket（RK3588 文本转发） ----
    forward_sock = None
    if not args.no_udp:
        forward_sock = create_udp_socket()
        print(f"文本转发目标: {RK3588_IP}:{RK3588_TEXT_PORT}")

    # ---- 初始化平滑器 ----
    init_bs = emotion_to_bs(0, models["emotion_model"], device)
    init_angles = bs_to_servo(init_bs, models)
    init_vec = np.array([init_angles[str(mid)] for mid in used_motors], dtype=np.float32)
    smoother = SmoothFilter(init_vec, f=2.5, z=0.8, r=0.0)

    # ---- 最新待处理文本（线程安全） ----
    pending_text = None
    text_lock = threading.Lock()
    processing = False

    def on_asr_text(text):
        nonlocal pending_text
        with text_lock:
            pending_text = text

    # ---- 处理单条文本 ----
    def process_text(text):
        nonlocal processing, pending_text
        processing = True
        print(f"\n>> [ASR] 收到文本: {text}")

        # 1. 转发文本到 RK3588
        if forward_sock:
            try:
                forward_sock.sendto(text.encode("utf-8"), (RK3588_IP, RK3588_TEXT_PORT))
                print(f"  [转发] → RK3588:{RK3588_TEXT_PORT}")
            except Exception as e:
                print(f"  [转发失败] {e}")

        # 2. DeepSeek API 情绪推理
        print("  [LLM] 正在分析情绪...", end="", flush=True)
        weighted = infer_emotion_id(text)
        display_parts = [f"{EMOTIONS[eid]}:{w}" for eid, w in weighted]
        print(f" → {', '.join(display_parts)}")

        # 3. EmotionBrain 推理 (ID → 52 BS)，支持混合
        if len(weighted) == 1:
            bs_arkit = emotion_to_bs(weighted[0][0], models["emotion_model"], device)
            display_name = EMOTIONS[weighted[0][0]]
        else:
            bs_arkit = blend_emotions(weighted, models["emotion_model"], device)
            display_name = "+".join([EMOTIONS[eid] for eid, _ in weighted])

        if args.debug:
            print(f"  52 BS (前10活跃):")
            bs_keys = models["bs_keys_full"]
            items = []
            for arkit_idx in range(51):
                key = bs_keys[arkit_idx + 1]
                val = bs_arkit[arkit_idx]
                if val > 0.05:
                    items.append((key, val))
            items.sort(key=lambda x: -x[1])
            for key, val in items[:10]:
                print(f"    {key}: {val:.3f}")

        # 4. BS → 舵机角度
        raw_angles = bs_to_servo(bs_arkit, models)
        raw_vec = np.array([raw_angles[str(mid)] for mid in used_motors], dtype=np.float32)

        # 5. 平滑滤波
        smooth_vec = smoother.update(raw_vec, 1.0 / 60)

        # 6. 构建最终角度字典
        final_angles = {}
        for i, mid in enumerate(used_motors):
            final_angles[str(mid)] = round(float(smooth_vec[i]), 2)

        # 7. UDP 发送舵机角度
        if servo_sock:
            data = json.dumps(final_angles) + "\n"
            servo_sock.sendto(data.encode("utf-8"), (SERVO_TARGET_IP, SERVO_TARGET_PORT))
            print(f"  [舵机] 发送 {len(final_angles)} 个角度到 {SERVO_TARGET_IP}:{SERVO_TARGET_PORT}")

        if args.debug:
            items = sorted(final_angles.items(), key=lambda x: int(x[0]))
            print(f"  舵机角度:")
            for mid, val in items:
                print(f"    motor {mid}: {val:.1f}")
        else:
            motor_19 = final_angles.get("19", 0)
            motor_22 = final_angles.get("22", 0)
            print(f"  [{display_name:12s}] motor[19]={motor_19:.1f} motor[22]={motor_22:.1f}")

        processing = False

    # ---- 手动测试模式 ----
    if args.test_text:
        process_text(args.test_text)
        print("\n测试完成。")
        return

    # ---- UDP 监听模式 ----
    listener = ASRTextListener(ASR_LISTEN_PORT, on_asr_text)
    listener.start()

    print(f"\n情绪协处理器已启动（UDP 监听 :{ASR_LISTEN_PORT}）")
    print(f"等待 ASR 文本... (按 Ctrl+C 退出)\n")

    try:
        while True:
            with text_lock:
                text = pending_text
                if text is not None and not processing:
                    pending_text = None

            if text is not None and not processing:
                process_text(text)

            time.sleep(0.05)  # 50ms 轮询间隔

    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        listener.stop()
        if servo_sock:
            servo_sock.close()
        if forward_sock:
            forward_sock.close()
        print("程序结束。")


if __name__ == "__main__":
    main()
