#!/usr/bin/env python3
"""
Face data collection for emotion fine-tuning.
- Connects to HTTP video stream
- OpenCV Haar cascade face detection
- Press 0-6 to save face crop to labeled folder
- Collects 128x128 RGB images (same format as MobileNetV2 model input)

Keys:  0=Angry  1=Disgust  2=Fear  3=Happy  4=Neutral  5=Sad  6=Surprise
       SPACE = skip (don't save)   Q = quit
"""

import cv2
import numpy as np
import os
import time
import signal

STREAM_URL = "http://10.192.48.233:5000/video_feed"
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data")
EMO_DIRS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Ensure dirs exist
for d in EMO_DIRS:
    os.makedirs(os.path.join(SAVE_DIR, d), exist_ok=True)

# Haar cascade
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    print("ERROR: cannot load Haar cascade")
    exit(1)

# Ctrl+C graceful stop
running = True
def handler(*a):
    global running
    running = False
signal.signal(signal.SIGINT, handler)

cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    print(f"ERROR: cannot open stream {STREAM_URL}")
    exit(1)

print(f"Data will be saved to: {SAVE_DIR}")
print()
print("  0 = Angry    1 = Disgust   2 = Fear    3 = Happy")
print("  4 = Neutral  5 = Sad       6 = Surprise")
print("  SPACE = snapshot (no label)   ESC/q = quit")
print()
print("Instructions:")
print("  1. Position your face in frame")
print("  2. Make an expression")
print("  3. Press the number key for that emotion")
print("  -> The face crop (128x128) is saved immediately")
print()

counts = {d: 0 for d in EMO_DIRS}
total = 0

while running:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.1)
        continue

    display = frame.copy()
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )

    face_crop = None
    if len(faces) > 0:
        # Take the largest face
        (fx, fy, fw, fh) = max(faces, key=lambda r: r[2] * r[3])

        # Expand by 30% margin (same as crop_face_for_emotion)
        margin = 0.3
        cx, cy = fx + fw // 2, fy + fh // 2
        nw = int(fw * (1 + margin))
        nh = int(fh * (1 + margin))
        x1 = max(0, cx - nw // 2)
        y1 = max(0, cy - nh // 2)
        x2 = min(w - 1, cx + nw // 2)
        y2 = min(h - 1, cy + nh // 2)

        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Crop and resize to 128x128
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size > 0:
            face_crop = cv2.resize(face_crop, (128, 128))
            # Show crop in corner
            display[5:133, w - 133 : w - 5] = face_crop
            cv2.rectangle(display, (w - 133, 5), (w - 5, 133), (255, 255, 0), 1)

    # Display info
    info = f"Saved: {total}  |  Keys: 0-6 to save"
    for i, d in enumerate(EMO_DIRS):
        info += f" | {i}={counts[d]}"
    cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 2)

    if len(faces) == 0:
        cv2.putText(display, "NO FACE", (w // 2 - 60, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Face Data Collection", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:  # q or ESC
        break
    elif ord("0") <= key <= ord("6"):
        idx = key - ord("0")
        if face_crop is not None:
            emo_dir = EMO_DIRS[idx]
            ts = int(time.time() * 1000000)
            filename = f"{emo_dir}_{ts}.jpg"
            filepath = os.path.join(SAVE_DIR, emo_dir, filename)
            # Save as RGB (same format as model input)
            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            cv2.imwrite(filepath, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            counts[emo_dir] += 1
            total += 1
            print(f"  Saved: {emo_dir}/{filename}")
        else:
            print("  No face detected, nothing saved")
    elif key == ord(" "):
        # Space = save snapshot without label
        if face_crop is not None:
            ts = int(time.time() * 1000000)
            filename = f"unlabeled_{ts}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, face_crop)
            print(f"  Saved: {filename}")
        else:
            print("  No face detected, nothing saved")

cap.release()
cv2.destroyAllWindows()
print(f"\nDone. Collected {total} images total.")
for d in EMO_DIRS:
    n = len(os.listdir(os.path.join(SAVE_DIR, d)))
    print(f"  {d}: {n}")
