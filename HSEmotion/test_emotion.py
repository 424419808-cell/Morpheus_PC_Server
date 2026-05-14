#!/usr/bin/env python3
"""
Test fine-tuned MobileNetV2 emotion model via ONNX Runtime.
Connects to HTTP video stream, detects faces, classifies emotions.
"""
import cv2
import numpy as np
import os
import onnxruntime

STREAM_URL = "http://10.192.48.233:5000/video_feed"
ONNX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "mobilenetv2_emotion_finetuned.onnx")
EMO_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Load ONNX model
sess = onnxruntime.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
print(f"Model loaded: {ONNX_PATH}")
print(f"Input: {sess.get_inputs()[0].shape} ({sess.get_inputs()[0].type})")
print(f"Output: {sess.get_outputs()[0].shape} -> {EMO_LABELS}")

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    print("ERROR: cannot open stream")
    exit(1)
print("Stream connected\n")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

    display = frame.copy()

    if len(faces) > 0:
        (fx, fy, fw, fh) = max(faces, key=lambda r: r[2] * r[3])

        # Expand by 30% margin
        cx, cy = fx + fw // 2, fy + fh // 2
        margin = 0.3
        nw, nh = int(fw * (1 + margin)), int(fh * (1 + margin))
        x1 = max(0, cx - nw // 2)
        y1 = max(0, cy - nh // 2)
        x2 = min(frame.shape[1] - 1, cx + nw // 2)
        y2 = min(frame.shape[0] - 1, cy + nh // 2)

        face = frame[y1:y2, x1:x2]
        if face.size > 0:
            face128 = cv2.resize(face, (128, 128))
            rgb = face128[..., ::-1].astype(np.float32)  # BGR→RGB
            rgb = rgb / 127.5 - 1.0  # normalize to [-1, 1]
            inp = np.expand_dims(rgb, 0)  # (1, 128, 128, 3)

            outputs = sess.run(None, {input_name: inp})
            probs = outputs[0][0]
            pred = np.argmax(probs)
            conf = float(probs[pred])

            label = f"{EMO_LABELS[pred]} ({conf:.2f})"
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show probabilities bar
            bar_x, bar_y = 10, 60
            for i, (emo, p) in enumerate(zip(EMO_LABELS, probs)):
                color = (0, 255, 0) if i == pred else (200, 200, 200)
                text = f"{emo}: {p:.2f}"
                cv2.putText(display, text, (bar_x, bar_y + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    cv2.putText(display, f"Frame: {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    cv2.imshow("MobileNetV2 Emotion Test", display)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")
