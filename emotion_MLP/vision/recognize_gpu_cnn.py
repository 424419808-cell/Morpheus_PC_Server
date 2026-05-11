import os
import sys
import numpy as np
import cv2

# 解决 WSL2 GPU 路径问题
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

import face_recognition
import dlib

print(f"🚀 CUDA 状态: {dlib.DLIB_USE_CUDA} | 显卡: {dlib.cuda.get_num_devices()}")

WIN_IP = "172.16.1.55"
DATASET_PATH = "./dataset"

known_face_encodings = []
known_face_names = []

# --- 1. 稳健加载底库 ---
print("正在扫描底库...")
for person_name in os.listdir(DATASET_PATH):
    p_path = os.path.join(DATASET_PATH, person_name)
    if not os.path.isdir(p_path): continue
    
    for img_n in os.listdir(p_path):
        if img_n.lower().endswith(('.jpg', '.png')):
            # 使用 face_recognition 自带加载函数更稳定
            img = face_recognition.load_image_file(os.path.join(p_path, img_n))
            
            # 自动定位人脸（不传 known_face_locations）
            encs = face_recognition.face_encodings(img)
            if encs:
                known_face_encodings.append(encs[0])
                known_face_names.append(person_name)
                print(f"✅ 成员就绪: {person_name}")
                break

# --- 2. 视频识别 (只有上面成功才会运行到这里) ---
cap = cv2.VideoCapture(f"http://{WIN_IP}:5000/video_feed")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True:
    ret, frame = cap.read()
    if not ret: break

    # GPU 模式下，fx=0.25 到 0.4 都可以
    small_frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)
    rgb_small = np.ascontiguousarray(small_frame[:, :, ::-1])

    # 关键：model="cnn" 启用显卡
    face_locs = face_recognition.face_locations(rgb_small, model="cnn")
    face_encs = face_recognition.face_encodings(rgb_small, face_locs)

    for (top, right, bottom, left), f_enc in zip(face_locs, face_encs):
        matches = face_recognition.compare_faces(known_face_encodings, f_enc, tolerance=0.45)
        name = "Unknown"
        if True in matches:
            name = known_face_names[matches.index(True)]

        # 坐标还原
        top, right, bottom, left = [int(x * 2.5) for x in [top, right, bottom, left]]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top-10), 1, 1.5, (255, 255, 255), 2)

    cv2.imshow('Morpheus GPU', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

