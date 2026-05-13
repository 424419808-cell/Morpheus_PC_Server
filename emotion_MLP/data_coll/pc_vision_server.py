import cv2
import json
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from flask import Flask, request, jsonify
import threading

app = Flask(__name__)

# ================= 核心配置区 =================
IMAGE_SAVE_DIR = r"I:\captured_faces"
DATA_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_coll", "raw_data", "motor_babbling_data_PC.json")
TARGET_VALID_SAMPLES = 11000
# ==============================================

if not os.path.exists(IMAGE_SAVE_DIR):
    os.makedirs(IMAGE_SAVE_DIR)

dataset = []
if os.path.exists(DATA_JSON_PATH):
    try:
        with open(DATA_JSON_PATH, "r", encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"已检测到历史数据！已成功加载 {len(dataset)} 条记录。")
        print(f"将从 ID {len(dataset)} 开始继续采集，目标总数: {TARGET_VALID_SAMPLES}")
    except Exception as e:
        print(f"读取历史数据失败，将作为新任务开始: {e}")
else:
    print("未检测到历史数据，将开启全新采集任务。")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "face_landmarker.task")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "face_landmarker.task"

try:
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    print("MediaPipe 模型加载成功！")
except Exception as e:
    print(f"模型加载失败: {e}")
    exit()

@app.route('/capture', methods=['POST'])
def capture_and_process():
    data = request.json
    motor_commands = data.get("motor_commands")

    success, frame = cap.read()
    if not success:
        return jsonify({"status": "error", "message": "无法读取画面"}), 500

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    if detection_result.face_blendshapes:
        current_id = len(dataset)

        img_filename = f"face_{current_id:04d}.jpg"
        img_path = os.path.join(IMAGE_SAVE_DIR, img_filename)
        cv2.imwrite(img_path, frame)

        blendshapes_categories = detection_result.face_blendshapes[0]
        bs_dict = {cat.category_name: cat.score for cat in blendshapes_categories}

        sample = {
            "sample_id": current_id,
            "image_file": img_filename,
            "motor_commands": motor_commands,
            "blendshapes": bs_dict
        }
        dataset.append(sample)

        if len(dataset) % 10 == 0:
            def save_data_async(data_copy):
                with open(DATA_JSON_PATH, "w", encoding='utf-8') as f:
                    json.dump(data_copy, f, ensure_ascii=False)

            threading.Thread(target=save_data_async, args=(list(dataset),)).start()

        print(f"采集成功: {len(dataset)} / {TARGET_VALID_SAMPLES}")
    else:
        print(f"未检测到人脸，跳过。当前进度: {len(dataset)} / {TARGET_VALID_SAMPLES}")

    if len(dataset) >= TARGET_VALID_SAMPLES:
        return jsonify({"status": "ok", "stop": True})
    else:
        return jsonify({"status": "ok", "stop": False})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
