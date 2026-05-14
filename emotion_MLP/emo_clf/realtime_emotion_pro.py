
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
MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion_model.pkl")
FACE_TASK = os.path.join(BASE_DIR, "..", "face_landmarker.task")
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


CAM_ID = int(os.environ.get("CAM", "1"))
cap = cv2.VideoCapture(CAM_ID)
if not cap.isOpened():
    print(f"摄像头 {CAM_ID} 无法打开，尝试 CAM=1 或 CAM=2")
    exit()
WINDOW_NAME = "Emotion Recognition"
cv2.namedWindow(WINDOW_NAME)

print(f"{Colors.BOLD}>>> 情绪监测引擎已启动 (摄像头窗口){Colors.RESET}")
print("提示: 保持面部在摄像头中心 | 按 q 或 ESC 退出\n")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        h, w = frame.shape[:2]
        display = frame.copy()

        # 图像处理
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp)

        if result.face_blendshapes and result.face_landmarks:
            # 提取 BS
            current_bs = [b.score for b in result.face_blendshapes[0]]
            X_live = pd.DataFrame([current_bs], columns=feature_names)

            # 预测
            prediction = clf.predict(X_live)[0]
            prob = np.max(clf.predict_proba(X_live))

            # 发送 UDP
            if prob > 0.3:
                msg = f"{prediction}:{prob:.2f}"
                sock.sendto(msg.encode(), (GEMMA_IP, GEMMA_PORT))

            # 画人脸外框（从 landmarks 计算）
            landmarks = result.face_landmarks[0]
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 确定显示颜色
            label_color = {
                "Happy": (0, 255, 0), "Angry": (0, 0, 255),
                "Fear": (0, 0, 255), "Surprise": (255, 200, 0),
                "Sad": (255, 0, 0), "Neutral": (200, 200, 200),
                "Disgust": (0, 150, 255),
            }.get(prediction, (255, 255, 255))

            # 显示情绪标签 + 置信度
            label_text = f"{prediction} ({prob:.0%})"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
            cv2.rectangle(display, (x1, y1 - th - 20), (x1 + tw + 20, y1), label_color, -1)
            cv2.putText(display, label_text, (x1 + 10, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

            # 显示 top-3 概率条
            probs = clf.predict_proba(X_live)[0]
            top3_idx = np.argsort(probs)[-3:][::-1]
            bar_y = y2 + 25
            for i, idx in enumerate(top3_idx):
                label = clf.classes_[idx]
                p = probs[idx]
                bar_w = int(p * 200)
                cv2.putText(display, f"{label}:", (x1, bar_y + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.rectangle(display, (x1 + 60, bar_y + i * 25 - 12),
                              (x1 + 60 + bar_w, bar_y + i * 25 + 2),
                              label_color if i == 0 else (100, 100, 100), -1)
                cv2.putText(display, f"{p:.0%}", (x1 + 70 + bar_w, bar_y + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # 控制台输出
            color = get_color(prediction)
            bar_len = int(prob * 20)
            bar_str = "█" * bar_len + "-" * (20 - bar_len)
            output = f"\r{color}[{prediction:8}]{Colors.RESET} 置信度: |{bar_str}| {prob:.2%}"
            print(output, end="", flush=True)
        else:
            cv2.putText(display, "No face detected", (w//2-120, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            print(f"\r{Colors.YELLOW}[ 未检测到人脸 ]{Colors.RESET}" + " " * 40, end="", flush=True)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q 或 ESC
            break

except KeyboardInterrupt:
    print(f"\n\n{Colors.BOLD}程序已手动停止。{Colors.RESET}")

finally:
    cap.release()
    landmarker.close()
    sock.close()
    cv2.destroyAllWindows()