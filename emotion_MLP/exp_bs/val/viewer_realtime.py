"""
实时 3D 预览 — 共情模型输入/输出 BS 可视化

用法:
  python exp_bs/val/viewer_realtime.py                    # 实时预览（摄像头）
  python exp_bs/val/viewer_realtime.py --demo             # 演示模式（遍历情绪模板）
  python exp_bs/val/viewer_realtime.py --no-cam           # 无摄像头，手动输入 BS
"""
import argparse
import os
import sys
import time

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.join(BASE_DIR, "..", "..")
EMPATHY_MODEL_PATH = os.path.join(PROJ_ROOT, "exp_bs", "models", "bs2bs_empathy.pth")
LANDMARKER_PATH = os.path.join(PROJ_ROOT, "face_landmarker.task")

EMOTION_NAMES = [
    "Neutral", "Happy", "Excitement", "Humor", "Pride",
    "Trust", "Love", "Relief", "Hope",
    "Anger", "Disgust", "Fear", "Vigilance",
    "Sad", "Loneliness", "Guilt",
    "Surprise", "Confusion", "Shyness",
]

ARKIT_BS_NAMES = [
    "_neutral",
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]

BROWS = [1, 2, 3, 4, 5]
EYES = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
JAW = [23, 24, 25, 26]
MOUTH = [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
NOSE = [51, 52]


class BS2BS_Empathy:
    """轻量共情模型 — 纯 numpy 推理（无需 torch）"""
    def __init__(self):
        self.w = None
        self.b = None

    def load(self, path):
        try:
            import torch
            ckpt = torch.load(path, map_location="cpu")
            state = ckpt if isinstance(ckpt, dict) else ckpt.state_dict()
            self.w = [v.numpy() for k, v in state.items() if "weight" in k]
            self.b = [v.numpy() for k, v in state.items() if "bias" in k]
            return True
        except Exception as e:
            print(f"[警告] 加载模型失败: {e}")
            return False

    def forward(self, x):
        if self.w is None:
            return x
        h = x
        for i in range(len(self.w)):
            h = h @ self.w[i].T + self.b[i]
            if i < len(self.w) - 2:
                h = np.maximum(h, 0)
            elif i == len(self.w) - 2:
                h = np.maximum(h, 0)
            else:
                h = 1.0 / (1.0 + np.exp(-h))
        return h


def bar_chart(bs_values, title, ax, color="steelblue"):
    """在指定轴上绘制 BS 条形图，按区域着色"""
    ax.clear()
    bars = ax.bar(range(52), bs_values, width=0.8)

    # 按区域着色
    region_colors = {
        "brows": "#4ECDC4", "eyes": "#FFE66D", "jaw": "#FF6B6B",
        "mouth": "#95E1D3", "nose": "#F38181",
    }

    for i in range(52):
        if i in BROWS:
            bars[i].set_color(region_colors["brows"])
        elif i in EYES:
            bars[i].set_color(region_colors["eyes"])
        elif i in JAW:
            bars[i].set_color(region_colors["jaw"])
        elif i in MOUTH:
            bars[i].set_color(region_colors["mouth"])
        elif i in NOSE:
            bars[i].set_color(region_colors["nose"])

    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("BS Index")
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.3)

    # 区域标签
    ax.text(3, 0.95, "brows", fontsize=7, color=region_colors["brows"],
            fontweight="bold", ha="center")
    ax.text(14, 0.95, "eyes", fontsize=7, color=region_colors["eyes"],
            fontweight="bold", ha="center")
    ax.text(25, 0.95, "jaw", fontsize=7, color=region_colors["jaw"],
            fontweight="bold", ha="center")
    ax.text(39, 0.95, "mouth", fontsize=7, color=region_colors["mouth"],
            fontweight="bold", ha="center")
    ax.text(51.5, 0.95, "nose", fontsize=7, color=region_colors["nose"],
            fontweight="bold", ha="center")


def face_wireframe(bs_values, ax, title="Face"):
    """简单的 2D 面部线框图 — 用关键点位置反映 BS 值变化"""
    ax.clear()

    # 面部关键点位置（简化的 2D 坐标）
    n_pts = 52
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    radii = 0.3 + 0.4 * bs_values  # BS 值越大，点越向外
    x = 0.5 + radii * np.cos(angles)
    y = 0.5 + radii * np.sin(angles)

    # 绘制连接线（相邻点 + 对称点）
    for i in range(n_pts):
        j = (i + 1) % n_pts
        ax.plot([x[i], x[j]], [y[i], y[j]], color="steelblue", alpha=0.3, linewidth=1)

    # 绘制关键点
    scatter = ax.scatter(x, y, c=bs_values, cmap="RdYlGn", s=80, vmin=0, vmax=1, edgecolors="gray")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")

    return scatter


def demo_mode(model):
    """演示模式：遍历 19 种情绪模板"""
    print("\n=== 演示模式 ===")
    sys.path.insert(0, os.path.join(PROJ_ROOT, "exp_bs", "scripts"))
    from gen_batch_data import get_base_bs

    import matplotlib.pyplot as plt
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("BS2BS_Empathy — 共情模型实时预览 (DEMO)", fontsize=14, fontweight="bold")

    try:
        while True:
            for emo_idx, emo_name in enumerate(EMOTION_NAMES):
                input_bs = get_base_bs(emo_name)
                empathy_bs = model.forward(input_bs)

                bar_chart(input_bs, f"Input: {emo_name}", axes[0, 0], color="#4ECDC4")
                bar_chart(empathy_bs, f"Empathy Output", axes[0, 1], color="#FF6B6B")

                face_wireframe(input_bs, axes[1, 0], title=f"Input Face: {emo_name}")
                face_wireframe(empathy_bs, axes[1, 1], title="Empathy Face")

                fig.tight_layout()
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(1.5)

    except KeyboardInterrupt:
        print("\n用户中断")


def live_cam_mode(model, cam_id=0):
    """摄像头实时模式"""
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError:
        print("[错误] 需要 opencv-python 和 mediapipe")
        print("  pip install opencv-python mediapipe")
        return

    if not os.path.exists(LANDMARKER_PATH):
        print(f"[错误] MediaPipe 模型不存在: {LANDMARKER_PATH}")
        return

    import matplotlib.pyplot as plt
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("BS2BS_Empathy — 实时共情预览", fontsize=14, fontweight="bold")

    # 初始化 MediaPipe
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=LANDMARKER_PATH),
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[错误] 无法打开摄像头 {cam_id}")
        return

    print("\n实时共情预览运行中... (Ctrl+C 退出)")
    smooth_input = np.zeros(52, dtype=np.float32)
    smooth_output = np.zeros(52, dtype=np.float32)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            timestamp = int(time.time() * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, timestamp)

            if result.face_blendshapes:
                user_bs = np.array([b.score for b in result.face_blendshapes[0]], dtype=np.float32)
                empathy_bs = model.forward(user_bs)

                # 平滑
                smooth_input = smooth_input * 0.7 + user_bs * 0.3
                smooth_output = smooth_output * 0.7 + empathy_bs * 0.3

                bar_chart(smooth_input, "User BS", axes[0], color="#4ECDC4")
                bar_chart(smooth_output, "Empathy BS", axes[1], color="#FF6B6B")

                fig.tight_layout()
                fig.canvas.draw()
                fig.canvas.flush_events()
            else:
                # 无检测到时逐渐衰减
                smooth_input *= 0.95
                smooth_output *= 0.95
                bar_chart(smooth_input, "User BS (no face)", axes[0], color="#4ECDC4")
                bar_chart(smooth_output, "Empathy BS", axes[1], color="#FF6B6B")
                fig.canvas.draw()
                fig.canvas.flush_events()

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        cap.release()
        landmarker.close()
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="共情模型实时 3D 预览")
    parser.add_argument("--demo", action="store_true", help="演示模式（遍历情绪模板）")
    parser.add_argument("--no-cam", action="store_true", help="无摄像头模式")
    parser.add_argument("--cam", type=int, default=0, help="摄像头 ID")
    args = parser.parse_args()

    # 加载模型
    model = BS2BS_Empathy()
    if os.path.exists(EMPATHY_MODEL_PATH):
        model.load(EMPATHY_MODEL_PATH)
        print(f"[OK] 共情模型: {EMPATHY_MODEL_PATH}")
    else:
        print(f"[警告] 模型不存在，使用恒等映射")
        model.w = None

    if args.demo:
        demo_mode(model)
    elif args.no_cam:
        print("无摄像头模式：请使用 --demo 或提供摄像头")
    else:
        live_cam_mode(model, cam_id=args.cam)


if __name__ == "__main__":
    main()
