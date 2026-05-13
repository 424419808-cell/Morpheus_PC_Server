"""
实时共情推理 — MediaPipe BS → BS2BS_Empathy → 数字人渲染验证

用法:
  python exp_bs/infer/run_empathy_live.py                    # 实时共情 + 控制台打印
  python exp_bs/infer/run_empathy_live.py --viewer           # 实时共情 + 3D 预览
  python exp_bs/infer/run_empathy_live.py --blender-render   # 实时共情 + Blender EEVEE
"""
import argparse
import os
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.join(BASE_DIR, "..", "..")
MODEL_PATH = os.path.join(PROJ_ROOT, "exp_bs", "models", "bs2bs_empathy.pth")
LANDMARKER_PATH = os.path.join(PROJ_ROOT, "face_landmarker.task")
EMOTION_MODEL_PATH = os.path.join(PROJ_ROOT, "emo_clf", "models", "emotion_model.pkl")

# 情绪名称
EMOTIONS = [
    "Neutral", "Happy", "Excitement", "Humor", "Pride",
    "Trust", "Love", "Relief", "Hope",
    "Anger", "Disgust", "Fear", "Vigilance",
    "Sad", "Loneliness", "Guilt",
    "Surprise", "Confusion", "Shyness",
]

# 共情映射（用于参考）
EMPATHY_LABELS = {
    1: "Happy", 2: "Excitement", 3: "Humor", 6: "Love",
    5: "Trust", 7: "Relief", 8: "Hope", 0: "Neutral",
}


class BS2BS_Empathy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(52, 128), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 256), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 52), torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def load_models(device):
    model = BS2BS_Empathy().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"[OK] 加载共情模型: {MODEL_PATH}")
    else:
        print(f"[警告] 未找到共情模型: {MODEL_PATH}")
        print("[警告] 将使用恒等映射（无共情效果）进行测试")

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="实时共情推理")
    parser.add_argument("--viewer", action="store_true", help="启动 3D 实时预览")
    parser.add_argument("--blender-render", action="store_true", help="Blender EEVEE 渲染")
    parser.add_argument("--no-udp", action="store_true", help="不发送UDP")
    parser.add_argument("--cam", type=int, default=0, help="摄像头ID (默认 0)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载模型
    model = load_models(device)

    # 初始化 MediaPipe
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=LANDMARKER_PATH),
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    # 可选加载情绪分类器（用于辅助信息）
    rf_classifier = None
    try:
        import joblib
        import pandas as pd
        rf_classifier = joblib.load(EMOTION_MODEL_PATH)
        print(f"[OK] 加载情绪分类器")
    except Exception as e:
        print(f"[信息] 未加载情绪分类器: {e}")

    # 摄像头
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[错误] 无法打开摄像头 {args.cam}")
        sys.exit(1)

    print("\n=== 实时共情推理运行中 ===")
    print("  摄像头 → MediaPipe 52 BS → BS2BS_Empathy → 共情 BS")
    print("  按 Ctrl+C 退出\n")

    # 平滑滤波
    smooth_bs = np.zeros(52, dtype=np.float32)

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
                # 提取用户 BS
                user_bs = np.array([b.score for b in result.face_blendshapes[0]], dtype=np.float32)

                # 共情推理
                with torch.no_grad():
                    tensor = torch.tensor([user_bs], dtype=torch.float32, device=device)
                    empathy_bs = model(tensor).cpu().numpy()[0]

                # 平滑
                smooth_bs = smooth_bs * 0.7 + empathy_bs * 0.3

                # 情绪分类（辅助）
                emotion_label = "?"
                if rf_classifier is not None:
                    import pandas as pd
                    features = pd.DataFrame([user_bs], columns=[f"bs_{i}" for i in range(52)])
                    emotion_label = rf_classifier.predict(features)[0]

                # 显示差异
                diff = np.mean(np.abs(empathy_bs - user_bs))
                active_in = np.mean(user_bs[user_bs > 0.05]) if np.any(user_bs > 0.05) else 0
                active_out = np.mean(empathy_bs[empathy_bs > 0.05]) if np.any(empathy_bs > 0.05) else 0

                print(f"\r用户[{emotion_label:8}] | 输入强度={active_in:.3f} "
                      f"输出强度={active_out:.3f} | BS差异={diff:.4f}" + " " * 10, end="", flush=True)

            else:
                print(f"\r[未检测到人脸]" + " " * 40, end="", flush=True)
                smooth_bs *= 0.9  # 衰减到中性

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        cap.release()
        landmarker.close()
        print("程序结束。")


if __name__ == "__main__":
    main()
