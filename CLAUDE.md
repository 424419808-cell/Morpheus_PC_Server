# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Always respond in Chinese (中文).** All communication, explanations, code comments, and documentation must be in Chinese unless the user explicitly requests English.

## Project Overview

Morpheus is a multi-modal emotional interaction system for a humanoid robot face. It uses MLP-based models to map between facial blendshapes and servo angles, drives a physical robot face via RK3588 + PCA9685, and integrates emotion recognition, voice conversation, lip synchronization, and face recognition.

## Codebase Architecture

Two physical hosts, one RK3588 (ARM Linux):

**Windows (PC)** — core runtime:
- `emotion_MLP/brain/deepseek_response.py` — Central brain: receives emotion labels + voice text, calls DeepSeek LLM, drives DashScope TTS
- `emotion_MLP/emo_clf/realtime_emotion_pro.py` — Real-time emotion classification (52 BS → 7 emotions), sends via UDP:5007
- `emotion_MLP/audio/xunfei_ear_realtime.py` — Xunfei ASR voice recognition, sends text via UDP:5006/5008
- `emotion_MLP/audio/sensevoice_ear_realtime.py` — Local SenseVoice ASR + CAM++ voiceprint alternative
- `emotion_MLP/exp_bs/infer/test_emotion_to_face.py` — Emotion brain pipeline: emotion label → 52 BS → upper/lower inverse models → 24 servo angles → UDP:8888
- `emotion_MLP/exp_bs/infer/run_empathy_live.py` — Real-time empathy: camera → 52 BS → BS2BS_Empathy → digital human validation
- `emotion_MLP/exp_bs/infer/run_empathy_deploy.py` — Full empathy deployment: camera → empathy → BS → inverse → servo angles → UDP:8888
- `emotion_MLP/data_coll/pc_vision_server.py` — Flask data collection server (receives servo commands, captures camera + MediaPipe blendshape)

**WSL2 (Ubuntu)** — vision + GPU inference:
- `emotion_MLP/vision/brainv4.py` — Vision engine: GPU face recognition, target search, sends servo cmds via UDP:5005
- `emotion_MLP/bs2angle/infer/test_udp_fullface.py` — Full-face real-time driving (camera→MediaPipe BS→inverse model→UDP)

**RK3588 (ARM Linux)** — servo control (replaces RPi):
- `pi/Morpheus/pi_servo_udp.py` — UDP servo daemon (listens UDP:8888, drives PCA9685 via I2C bus 4)
- `pi/Morpheus/pi_servo_service.py` — TCP servo pipeline service
- `pi/Morpheus/mor_servo_dev.py` — PCA9685 I2C low-level driver

All inter-process communication uses UDP (ports 5005-5010, 8888). RK3588 IP: `10.192.53.8`, Windows IP: `172.16.1.55`.

## Directory Structure

```
emotion_MLP/
├── brain/                # 大脑引擎：LLM 对话、TTS 语音合成
│   └── deepseek_response.py
├── audio/                # 语音识别：ASR（讯飞、SenseVoice）
│   ├── xunfei_ear_realtime.py
│   ├── sensevoice_ear_realtime.py
│   └── xunfei_register.py
├── emo_clf/              # 情绪分类：52 BS → 7 种情绪
│   ├── realtime_emotion_pro.py
│   └── models/
│       └── emotion_model.pkl
├── exp_bs/               # 表情引擎：情绪/共情 → 52 BS
│   ├── scripts/          #   训练/数据生成脚本
│   │   ├── gen_batch_data.py
│   │   ├── train_brain.py
│   │   ├── empathy_map.py
│   │   ├── gen_empathy_data.py
│   │   └── train_empathy.py
│   ├── infer/            #   实时推理脚本
│   │   ├── test_emotion_to_face.py
│   │   ├── run_empathy_live.py
│   │   └── run_empathy_deploy.py
│   ├── val/              #   验证/渲染工具
│   │   └── test_offline.py
│   └── models/           #   表情模块模型权重
│       ├── emotion_brain.pth
│       └── bs2bs_empathy.pth (训练后生成)
├── bs2angle/             # 舵机转换：BS ↔ 角度映射
│   ├── scripts/          #   训练脚本 (BS↔Angle MLP)
│   │   ├── train_angle2bs.py
│   │   ├── train_inverse_cycle.py
│   │   ├── train_upper_face_mlp.py
│   │   └── train_lower_face_mlp.py
│   ├── infer/            #   实时舵机驱动脚本
│   │   ├── test_udp_fullface.py
│   │   ├── test_udp_upper.py
│   │   ├── test_udp_lower.py
│   │   ├── test_fullface.py
│   │   ├── test_fullface_dual.py
│   │   ├── test_tcp_upperface.py
│   │   └── test_speech_lower.py
│   └── models/           #   舵机模块模型权重
│       ├── angle2bs_full.pth
│       ├── bs2angle_cycle.pth
│       ├── upper_face_bs2angle.pth
│       ├── lower_face_bs2angle.pth
│       └── lower_face_bs2angle1.pth
├── vision/               # 视觉：人脸识别、追踪
│   ├── brainv4.py
│   ├── recognize_gpu_cnn.py
│   ├── web_test.py
│   ├── dataset/          # 4 persons × 300 face images
│   └── requirements_vision.txt
├── data_coll/            # 数据采集
│   ├── pc_vision_server.py
│   └── raw_data/         # 原始采集数据
│       ├── motor_babbling_data_PC.json
│       └── motor_babbling_data_clean.json
├── data/                 # 共享数据集
│   └── empathy_training_data.npz (生成后)
├── utils/                # 工具脚本
│   ├── data_cleaner.py
│   ├── clean_json.py
│   ├── compare_lips.py
│   └── render_transition.py
├── face_landmarker.task  # MediaPipe 模型（根目录，多模块共享）
├── names.pkl             # Person name mapping
└── requirements.txt
pi/Morpheus/              # RK3588 servo control daemons (PCA9685 over I2C bus 4)
blender_test/             # Offline rendering validation (Blender Cycles)
```

## Key ML Models

| Model | Type | File |
|-------|------|------|
| Forward (Angle→BS) | 256-128-64 MLP | `emotion_MLP/bs2angle/models/angle2bs_full.pth` |
| Full-face inverse (BS→Angle) | 256-128-64 MLP | `emotion_MLP/bs2angle/models/bs2angle_cycle.pth` |
| Upper-face inverse (19 BS→8 angles) | 256-128-64 MLP | `emotion_MLP/bs2angle/models/upper_face_bs2angle.pth` |
| Lower-face inverse (32 BS→16 angles) | 256-128-64 MLP | `emotion_MLP/bs2angle/models/lower_face_bs2angle.pth` |
| Emotion classifier (52 BS→7 emotions) | RandomForest | `emotion_MLP/emo_clf/models/emotion_model.pkl` |
| Emotional brain (label→52 BS) | MLP | `emotion_MLP/exp_bs/models/emotion_brain.pth` |
| BS2BS Empathy (user BS→empathy BS) | 128-256-128 MLP | `emotion_MLP/exp_bs/models/bs2bs_empathy.pth` |

## Common Commands

```bash
# Environment
conda activate emotion_mlp
pip install -r emotion_MLP/requirements.txt

# BS2Angle 舵机映射模型训练
python emotion_MLP/bs2angle/scripts/train_angle2bs.py
python emotion_MLP/bs2angle/scripts/train_inverse_cycle.py
python emotion_MLP/bs2angle/scripts/train_upper_face_mlp.py
python emotion_MLP/bs2angle/scripts/train_lower_face_mlp.py

# 表情引擎训练
python emotion_MLP/exp_bs/scripts/train_brain.py           # 情感大脑 (label→52 BS)
python emotion_MLP/exp_bs/scripts/gen_empathy_data.py      # 生成共情配对数据
python emotion_MLP/exp_bs/scripts/train_empathy.py         # 训练共情模型 (BS→BS)

# 数据采集与清洗
python emotion_MLP/data_coll/pc_vision_server.py           # start collection server
python emotion_MLP/utils/data_cleaner.py                    # interactive data cleaning
python emotion_MLP/utils/compare_lips.py                     # inspect lip BS values

# 实时舵机驱动 (WSL2 — camera→MediaPipe BS→servo)
python emotion_MLP/bs2angle/infer/test_udp_fullface.py     # dual-model full face
python emotion_MLP/bs2angle/infer/test_udp_upper.py        # upper face only
python emotion_MLP/bs2angle/infer/test_udp_lower.py        # lower face only

# 表情引擎推理 (PC — emotion brain pipeline)
python emotion_MLP/exp_bs/infer/test_emotion_to_face.py 0 1 9       # CLI: Neutral→Happy→Anger
python emotion_MLP/exp_bs/infer/test_emotion_to_face.py --listen    # UDP 监听 deepseek 情绪ID
python emotion_MLP/exp_bs/infer/run_empathy_live.py                # 实时共情 (数字人验证)
python emotion_MLP/exp_bs/infer/run_empathy_deploy.py              # 实时共情 (舵机部署)

# 情绪识别与大脑
python emotion_MLP/emo_clf/realtime_emotion_pro.py         # emotion recognition
python emotion_MLP/brain/deepseek_response.py              # central brain (LLM + TTS)

# 共情模型离线验证
python emotion_MLP/exp_bs/val/test_offline.py

# RK3588 servo daemon
python3 pi/Morpheus/pi_servo_udp.py

# Blender rendering validation
python blender_test/render_blendshape.py --sample_id 42
python blender_test/render_emotion_transition.py 0 1 9 --duration 2 --fps 15 --engine EEVEE

# GPU check
python -c "import torch; print(torch.cuda.is_available())"
```

## Inferences vs. Training Distinction

- 舵机映射模型训练 (`emotion_MLP/bs2angle/scripts/`) 在 **Windows** 上运行，在 `bs2angle/models/` 下产生 `.pth` 模型文件
- 表情引擎训练 (`emotion_MLP/exp_bs/scripts/`) 在 **Windows** 上运行，在 `exp_bs/models/` 下产生模型
- WSL2 推理 (`emotion_MLP/bs2angle/infer/test_udp_*.py`, `vision/brainv4.py`) 使用摄像头 + MediaPipe BS，通过 UDP 发送舵机指令到 RK3588
- PC 推理 (`emotion_MLP/exp_bs/infer/test_emotion_to_face.py`) 运行情感大脑管线（无需摄像头），通过 UDP 发送舵机指令到 RK3588
- PC 共情推理 (`emotion_MLP/exp_bs/infer/run_empathy_live.py`, `run_empathy_deploy.py`) 使用摄像头 + MediaPipe BS → 共情模型 → 数字人/舵机
- `vision/brainv4.py` 在 WSL2 上处理人脸识别 + 视线追踪

## Important Configuration

- Each inference/test script has a config block at the top for camera URL, RPi IP/port, and model paths
- `emotion_MLP/brain/deepseek_response.py` requires `DEEPSEEK_API_KEY` and `dashscope.api_key`
- `emotion_MLP/audio/xunfei_ear_realtime.py` requires Xunfei `APPID`, `APIKey`, `APISecret`
- RK3588 servo limits are defined in `TABLE_V_CONFIG` in both `pi_servo_udp.py` and `set_start_bound.py` — must stay in sync
- `test_emotion_to_face.py` 中 `RPI_IP` 常量应指向 RK3588 的实际 IP
- `blender_test/config.py` must set `BLENDER_EXE` and `REFERENCE_IMAGE_DIR` before use

## Development Workflow

### 功能模块划分规范

所有文件按功能模块存放，每个模块自包含脚本和模型：

- `brain/` — 大脑引擎：LLM 对话、TTS 语音合成
- `audio/` — 语音识别：ASR（讯飞、SenseVoice）、声纹注册
- `emo_clf/` — 情绪分类：52 BS → 7 种情绪分类
- `exp_bs/` — 表情引擎：情绪/共情 → 52 BS 系数
  - `scripts/` — 训练/数据生成脚本
  - `infer/` — 实时推理脚本
  - `val/` — 验证/渲染工具
  - `models/` — 本模块模型权重
- `bs2angle/` — 舵机转换：BS ↔ 舵机角度映射
  - `scripts/` — MLP 训练脚本
  - `infer/` — 实时舵机驱动
  - `models/` — 本模块模型权重
- `vision/` — 视觉识别：人脸识别、追踪
- `data_coll/` — 数据采集：采集服务器 + 原始 JSON 数据
- `data/` — 共享数据集（训练产生的 .npz 等）
- `utils/` — 工具脚本（数据处理、渲染）

模块命名采用英文缩写：`emo`=emotion, `clf`=classification, `exp`=expression, `bs`=blendshape, `coll`=collection, `MLP` 在目录名中简写为 `mlp`。

新增文件后，应询问是否同步更新 `README.md` 和 `requirements.txt`。

### 本地调试脚本（禁止上传）
`blender_test/debug/` 目录用于存放临时排查/诊断脚本：
- 仅用于本地调试和问题诊断
- **禁止提交到 GitHub**
- 不更新 README.md 和 requirements.txt
- 不记入 CLAUDE.md（本条规则除外）

## Hardware

- RK3588 (ARM Linux 5.10) with I2C enabled, 3× PCA9685 boards at addresses 0x40/0x41/0x42 on I2C bus 4
- 33 servos total (0-15 on 0x40, 16-31 on 0x41, channel 32 on 0x42)
- External servo power supply required
