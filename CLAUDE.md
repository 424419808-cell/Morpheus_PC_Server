# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Always respond in Chinese (中文).** All communication, explanations, code comments, and documentation must be in Chinese unless the user explicitly requests English.

## Project Overview

Morpheus is a multi-modal emotional interaction system for a humanoid robot face. It uses MLP-based models to map between facial blendshapes and servo angles, drives a physical robot face via RK3588 + PCA9685, and integrates emotion recognition, voice conversation, lip synchronization, and face recognition.

## Codebase Architecture

Two physical hosts, one RK3588 (ARM Linux):

**Windows (PC)** — core runtime:
- `emotion_MLP/core/deepseek_response.py` — Central brain: receives emotion labels + voice text, calls DeepSeek LLM, drives DashScope TTS
- `emotion_MLP/core/realtime_emotion_pro.py` — Real-time emotion classification (52 BS → 7 emotions), sends via UDP:5007
- `emotion_MLP/core/xunfei_ear_realtime.py` — Xunfei ASR voice recognition, sends text via UDP:5006/5008
- `emotion_MLP/core/sensevoice_ear_realtime.py` — Local SenseVoice ASR + CAM++ voiceprint alternative
- `emotion_MLP/inference/test_emotion_to_face.py` — Emotion brain pipeline: emotion label → 52 BS → upper/lower inverse models → 24 servo angles → UDP:8888
- `emotion_MLP/training/pc_vision_server.py` — Flask data collection server (receives servo commands, captures camera + MediaPipe blendshape)

**WSL2 (Ubuntu)** — vision + GPU inference:
- `emotion_MLP/vision/brainv4.py` — Vision engine: GPU face recognition, target search, sends servo cmds via UDP:5005
- `emotion_MLP/inference/test_udp_fullface.py` — Full-face real-time driving (camera→MediaPipe BS→inverse model→UDP)

**RK3588 (ARM Linux)** — servo control (replaces RPi):
- `pi/Morpheus/pi_servo_udp.py` — UDP servo daemon (listens UDP:8888, drives PCA9685 via I2C bus 4)
- `pi/Morpheus/pi_servo_service.py` — TCP servo pipeline service
- `pi/Morpheus/mor_servo_dev.py` — PCA9685 I2C low-level driver

All inter-process communication uses UDP (ports 5005-5010, 8888). RK3588 IP: `10.192.53.8`, Windows IP: `172.16.1.55`.

## Directory Structure

```
emotion_MLP/
├── core/                 # Runtime: emotion recognition, speech, LLM
├── training/             # MLP training + data collection server
├── inference/            # Real-time driving scripts (UDP/TCP/camera)
│   └── test_emotion_to_face.py  # Emotion brain: label→BS→servo angles→UDP
├── models/               # Pretrained .pth / .pkl weights
├── data/                 # Training data (servo-BS paired JSON)
├── utils/                # Data cleaning, rendering, comparison tools
├── vision/               # Face recognition, GPU-accelerated
│   ├── dataset/          # 4 persons × 300 face images
│   └── requirements_vision.txt
├── face_landmarker.task  # MediaPipe model
├── names.pkl             # Person name mapping
└── requirements.txt
pi/Morpheus/              # RK3588 servo control daemons (PCA9685 over I2C bus 4)
blender_test/             # Offline rendering validation (Blender Cycles)
```

## Key ML Models

| Model | Type | File |
|-------|------|------|
| Forward (Angle→BS) | 256-128-64 MLP | `models/angle2bs_full.pth` |
| Full-face inverse (BS→Angle) | 256-128-64 MLP | `models/bs2angle_cycle.pth` |
| Upper-face inverse (19 BS→8 angles) | 256-128-64 MLP | `models/upper_face_bs2angle.pth` |
| Lower-face inverse (32 BS→16 angles) | 256-128-64 MLP | `models/lower_face_bs2angle.pth` |
| Emotion classifier (52 BS→7 emotions) | RandomForest | `models/emotion_model.pkl` |
| Emotional brain (label→52 BS) | MLP | `models/emotion_brain.pth` |

## Common Commands

```bash
# Environment
conda activate emotion_mlp
pip install -r emotion_MLP/requirements.txt

# Training
python emotion_MLP/training/train_angle2bs.py
python emotion_MLP/training/train_inverse_cycle.py
python emotion_MLP/training/train_upper_face_mlp.py
python emotion_MLP/training/train_lower_face_mlp.py
python emotion_MLP/training/train_brain.py

# Data pipeline
python emotion_MLP/training/pc_vision_server.py           # start collection server
python emotion_MLP/utils/data_cleaner.py                   # interactive data cleaning
python emotion_MLP/utils/compare_lips.py                    # inspect lip BS values

# Real-time inference (WSL2 — camera→MediaPipe BS→servo)
python emotion_MLP/inference/test_udp_fullface.py          # dual-model full face
python emotion_MLP/inference/test_udp_upper.py             # upper face only
python emotion_MLP/inference/test_udp_lower.py             # lower face only

# Real-time inference (PC — emotion brain pipeline)
python emotion_MLP/inference/test_emotion_to_face.py 0 1 9 # CLI: Neutral→Happy→Anger 循环
python emotion_MLP/inference/test_emotion_to_face.py --listen # UDP监听 deepseek 情绪ID
python emotion_MLP/core/realtime_emotion_pro.py            # emotion recognition
python emotion_MLP/core/deepseek_response.py               # central brain (LLM + TTS)

# RK3588 servo daemon
python3 pi/Morpheus/pi_servo_udp.py

# Blender rendering validation
python blender_test/render_blendshape.py --sample_id 42

# GPU check
python -c "import torch; print(torch.cuda.is_available())"
```

## Inferences vs. Training Distinction

- Training scripts (`emotion_MLP/training/`) are run on **Windows** and produce `.pth` model files in `models/`
- WSL2 inference (`emotion_MLP/inference/test_udp_*.py`, `vision/brainv4.py`) uses camera + MediaPipe BS, sends UDP servo commands to RK3588
- PC inference (`emotion_MLP/inference/test_emotion_to_face.py`) runs emotion brain pipeline (no camera needed), sends UDP servo commands to RK3588
- The `vision/brainv4.py` handles face recognition + gaze tracking on WSL2

## Important Configuration

- Each inference/test script has a config block at the top for camera URL, RPi IP/port, and model paths
- `emotion_MLP/core/deepseek_response.py` requires `DEEPSEEK_API_KEY` and `dashscope.api_key`
- `emotion_MLP/core/xunfei_ear_realtime.py` requires Xunfei `APPID`, `APIKey`, `APISecret`
- RK3588 servo limits are defined in `TABLE_V_CONFIG` in both `pi_servo_udp.py` and `set_start_bound.py` — must stay in sync
- `test_emotion_to_face.py` 中 `RPI_IP` 常量应指向 RK3588 的实际 IP
- `blender_test/config.py` must set `BLENDER_EXE` and `REFERENCE_IMAGE_DIR` before use

## Hardware

- RK3588 (ARM Linux 5.10) with I2C enabled, 3× PCA9685 boards at addresses 0x40/0x41/0x42 on I2C bus 4
- 33 servos total (0-15 on 0x40, 16-31 on 0x41, channel 32 on 0x42)
- External servo power supply required
