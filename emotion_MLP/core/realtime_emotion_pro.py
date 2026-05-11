
import mediapipe as mp
import joblib
import numpy as np
import pandas as pd
import cv2
import time
import os
import socket  # 新增：用于端口通信

# --- 配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "models", "emotion_model.pkl")
FACE_TASK = os.path.join(ROOT_DIR, "face_landmarker.task")
GEMMA_IP = "127.0.0.1"  # 如果在同一台机器或WSL，请根据实际调整
GEMMA_PORT = 5007       # 预留的 UDP 端口

# 初始化 UDP Socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 颜色代码 (让控制台输出更清晰)
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


# 加载模型
if not os.path.exists(MODEL_PATH):
    print(f"错误: 找不到模型文件 {MODEL_PATH}")
    exit()

clf = joblib.load(MODEL_PATH)
feature_names = [f'bs_{i}' for i in range(52)]

# 初始化 MediaPipe
options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=FACE_TASK),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    output_face_blendshapes=True
)
landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)


def get_color(label):
    if label == "Happy": return Colors.GREEN
    if label in ["Angry", "Fear"]: return Colors.RED
    if label == "Surprise": return Colors.BLUE
    if label == "Neutral": return Colors.RESET
    return Colors.YELLOW


cap = cv2.VideoCapture(0)

print(f"{Colors.BOLD}>>> 情绪监测引擎已启动 (控制台模式){Colors.RESET}")
print("提示: 保持面部在摄像头中心 | 按 Ctrl+C 停止\n")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 图像处理 (仅用于特征提取，不显示)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp)

        if result.face_blendshapes:
            # 1. 提取并准备数据
            current_bs = [b.score for b in result.face_blendshapes[0]]
            X_live = pd.DataFrame([current_bs], columns=feature_names)

            # 2. 预测
            prediction = clf.predict(X_live)[0]
            prob = np.max(clf.predict_proba(X_live))

            # --- 新增：发送数据给 Gemma ---
            if prob > 0.3:
                msg = f"{prediction}:{prob:.2f}"
                sock.sendto(msg.encode(), (GEMMA_IP, GEMMA_PORT))

            # 3. 控制台美化输出
            color = get_color(prediction)
            bar_length = int(prob * 20)
            bar = "█" * bar_length + "-" * (20 - bar_length)

            # 实时刷新当前行 (\r)
            output = f"\r{color}[{prediction:8}]{Colors.RESET} 置信度: |{bar}| {prob:.2%}"
            print(output, end="", flush=True)
        else:

            print(f"\r{Colors.YELLOW}[ 未检测到人脸 ]{Colors.RESET}" + " " * 40, end="", flush=True)

except KeyboardInterrupt:
    print(f"\n\n{Colors.BOLD}程序已手动停止。{Colors.RESET}")

finally:
    cap.release()
    landmarker.close()
    sock.close()