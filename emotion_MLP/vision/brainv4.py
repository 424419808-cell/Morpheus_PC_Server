import os
import cv2
import numpy as np
import pickle
import sys
import socket
import threading
import time
import struct
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 0. 环境设置 ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 强行指定 CUDA 库路径（针对 WSL2 优化，确保 GPU 被唤醒）
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

# 延迟导入以确保环境变量生效
import face_recognition
import dlib

# 网络配置
LISTEN_PORT = 5006
RPI_IP, RPI_PORT = "172.16.0.166", 5005

WIN_IP = "172.16.1.55"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(ROOT_DIR, "face_landmarker.task")
LBPH_MODEL = os.path.join(BASE_DIR, "model.yml")
NAMES_PKL = os.path.join(ROOT_DIR, "names.pkl")
EMOTION_MODEL_PATH = os.path.join(ROOT_DIR, "models", "emotion_model.pkl")

SIO_UDP_CONNRESET = 0x98000001


# --- 1. 视频流取流类 ---
class VideoStream:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if self.cap.isOpened():
                try:
                    ret, frame = self.cap.read()
                    if ret:
                        self.frame = frame
                    else:
                        time.sleep(0.01)
                except:
                    break

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        time.sleep(0.2)
        if self.cap.isOpened():
            self.cap.release()


# --- 2. 视觉处理引擎（支持二维盲寻） ---
class BrainEngine:
    def __init__(self):
        # 加载名单
        with open(NAMES_PKL, "rb") as f:
            self.names = pickle.load(f)

        # 状态控制
        self.mode = "IDLE"
        self.target_id = -1
        self.init_direction_h = 0.0   # 水平初始方向
        self.init_direction_v = 0.0   # 垂直初始方向
        self.has_found_vip_once = False
        self.current_signal = "0.0,0.0"

        # 模型加载
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            output_face_blendshapes=True
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # --- GPU 加速底库加载 ---
        print(f"🚀 [GPU 初始化] CUDA 状态: {dlib.DLIB_USE_CUDA}")
        self.known_face_encodings = []
        self.known_face_ids = []
        
        DATASET_PATH = os.path.join(BASE_DIR, "dataset")
        if os.path.exists(DATASET_PATH):
            for person_name in os.listdir(DATASET_PATH):
                p_path = os.path.join(DATASET_PATH, person_name)
                if not os.path.isdir(p_path): continue
                
                # 获取对应的 target_id
                current_id = -1
                for tid, name in self.names.items():
                    if name == person_name:
                        current_id = tid
                        break
                
                if current_id == -1: continue

                for img_n in os.listdir(p_path):
                    if img_n.lower().endswith(('.jpg', '.png', '.jpeg')):
                        img = face_recognition.load_image_file(os.path.join(p_path, img_n))
                        encs = face_recognition.face_encodings(img)
                        if encs:
                            self.known_face_encodings.append(encs[0])
                            self.known_face_ids.append(current_id)
                            print(f"✅ [底库就绪]: {person_name} (ID:{current_id})")
                            break

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if sys.platform == "win32":
            try:
                self.sock.ioctl(SIO_UDP_CONNRESET, struct.pack("L", 0))
            except:
                pass

    def activate(self, tid, direction_h, direction_v):
        """听到'启动'时激活识别，传入水平和垂直初始方向"""
        self.target_id = tid
        self.init_direction_h = direction_h
        self.init_direction_v = direction_v
        self.has_found_vip_once = False
        self.mode = "TRACKING"
        print(f">>> [识别启动] 目标: {self.names.get(tid)} | 水平: {direction_h}, 垂直: {direction_v}")

    def deactivate(self):
        """听到'关机'时进入待机"""
        self.mode = "IDLE"
        self.current_signal = "0.0,0.0"
        print(">>> [识别挂起] 系统进入 IDLE 模式")

    def update_direction(self, direction_h, direction_v):
        """在追踪过程中接收听觉大脑传来的实时声源方向，用于盲寻"""
        if self.mode == "TRACKING":
            self.init_direction_h = direction_h
            self.init_direction_v = direction_v
            # 如果当前已经丢失目标（曾经锁定过但丢失），重置标志，重新开始扫描
            if self.has_found_vip_once:
                self.has_found_vip_once = False

    def process_frame(self, frame):
        """每一帧的逻辑处理"""
        if frame is None:
            return None

        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2

        # --- IDLE 模式：仅显示画面，不做高负载计算 ---
        if self.mode == "IDLE":
            cv2.putText(frame, "SYSTEM IDLE", (20, 40), 1, 1.5, (100, 100, 100), 2)
            self.current_signal = "0.0,0.0"
            return frame

        # --- TRACKING 模式：运行识别算法 ---
        # 缩小图片提升 GPU 处理速度
        small = cv2.resize(frame, (w // 2, h // 2))
        rgb_small = np.ascontiguousarray(small[:, :, ::-1])

        # 使用 GPU CNN 模型检测人脸位置
        face_locations = face_recognition.face_locations(rgb_small, model="cnn")
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        best_vip_now = None
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # 比对目标 ID
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.45)
            if True in matches:
                first_match_index = matches.index(True)
                if self.known_face_ids[first_match_index] == self.target_id:
                    # 匹配成功，将坐标还原回原图尺寸 (small 是 1/2，所以乘以 2)
                    best_vip_now = {"box": (left * 2, top * 2, (right - left) * 2, (bottom - top) * 2)}
                    break

        found_this_frame = False
        if best_vip_now:
            bx, by, bw, bh = best_vip_now["box"]
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

            # Mediapipe 处理
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            result = self.detector.detect(mp_image)

            if result.face_landmarks:
                for i, landmarks in enumerate(result.face_landmarks):
                    pupil = landmarks[473]
                    px, py = int(pupil.x * w), int(pupil.y * h)
                    if bx < px < bx + bw and by < py < by + bh:
                        # 计算瞳孔相对于画面中心的偏移
                        self.current_signal = f"{px - center_x},{py - center_y}"
                        if not self.has_found_vip_once:
                            self.has_found_vip_once = True

                        cv2.circle(frame, (px, py), 7, (0, 255, 255), -1)
                        cv2.putText(frame, "LOCKED", (bx, by - 10), 1, 1.2, (0, 255, 255), 2)
                        found_this_frame = True
                        break

        # 处理盲寻逻辑（使用水平和垂直方向）
        if not found_this_frame:
            if not self.has_found_vip_once:
                # 根据水平和垂直方向生成扫描信号（固定幅度 250）
                h_signal = 250.0 if self.init_direction_h > 0 else (-250.0 if self.init_direction_h < 0 else 0.0)
                v_signal = 250.0 if self.init_direction_v > 0 else (-250.0 if self.init_direction_v < 0 else 0.0)
                self.current_signal = f"{h_signal},{v_signal}"
                cv2.putText(frame, "SCANNING...", (20, 40), 1, 1.5, (0, 165, 255), 2)
            else:
                # 曾经锁定过但当前丢失，停止转动
                self.current_signal = "0.0,0.0"
                cv2.putText(frame, "LOST TARGET", (20, 40), 1, 1.5, (0, 0, 255), 2)

        # 装饰 UI
        cv2.putText(frame, f"TARGET: {self.names.get(self.target_id)}", (20, h - 30), 1, 1.5, (0, 255, 0), 2)
        cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), (255, 255, 255), 1)
        cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), (255, 255, 255), 1)

        return frame


# --- 3. 主进程控制 ---
def start_brain():
    engine = BrainEngine()
    url = f"http://{WIN_IP}:5000/video_feed"
    vs = VideoStream(url).start()

    cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cmd_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    cmd_sock.bind(("0.0.0.0", LISTEN_PORT))
    cmd_sock.setblocking(False)

    # 发送线程（向树莓派发送云台控制信号）
    def sender_loop():
        while True:
            if engine.current_signal:
                try:
                    engine.sock.sendto(engine.current_signal.encode(), (RPI_IP, RPI_PORT))
                except:
                    pass
            time.sleep(0.05)

    threading.Thread(target=sender_loop, daemon=True).start()

    print("\n" + "=" * 40)
    print("   Morpheus Brain 视觉中心 V3.1 (二维盲寻版)")
    print("   状态: 窗口已开启，等待指令...")
    print("=" * 40)

    while True:
        try:
            data, _ = cmd_sock.recvfrom(1024)
            msg = data.decode().strip()
            print(f">>> 指令: {msg}")

            if msg.startswith("START:"):
                parts = msg.split(":")
                tid = int(parts[1])
                # 兼容新旧协议：如果只有两个部分，垂直默认为0
                h = float(parts[2]) if len(parts) > 2 else 0.0
                v = float(parts[3]) if len(parts) > 3 else 0.0
                engine.activate(tid, h, v)
            elif msg == "STOP":
                engine.deactivate()
            elif msg.startswith("DIR:"):
                parts = msg.split(":")
                h = float(parts[1]) if len(parts) > 1 else 0.0
                v = float(parts[2]) if len(parts) > 2 else 0.0
                engine.update_direction(h, v)
        except BlockingIOError:
            pass
        except Exception as e:
            print(f"Error: {e}")

        frame = vs.read()
        if frame is not None:
            display_frame = frame.copy()
            processed = engine.process_frame(display_frame)
            cv2.imshow('Morpheus Brain Vision', processed)


        if cv2.waitKey(1) == 27:
            break

    vs.stop()
    cv2.destroyAllWindows()
    cmd_sock.close()


if __name__ == "__main__":
    start_brain()
