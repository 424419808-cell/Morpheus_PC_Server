# Morpheus — 多模态情感交互系统

基于 MLP（多层感知机）的仿人机器人面部表情驱动系统，整合**情绪识别、语音对话、唇形同步、人脸识别、共情表情**等多模态交互能力。

## 概述

系统核心功能：

- **表情驱动**：摄像头采集面部 52 维 blendshape（MediaPipe），通过逆向 MLP 模型映射为舵机角度，实时驱动物理机器人面部
- **共情表情**：识别用户表情 BS → BS2BS_Empathy 模型生成共情 BS（非镜像，而是关心/抚慰等）→ 数字人渲染验证 / 舵机驱动
- **情绪识别**：从 blendshape 分类 7 种情绪（开心/生气/害怕/惊讶/悲伤/厌恶/中性），通过 UDP 发送给中央大脑
- **语音对话**：讯飞 ASR 实时语音识别 → DeepSeek 大模型生成回复（带情感标签）→ 阿里 DashScope TTS 语音合成
- **唇形同步**：文字→拼音→韵母→唇形 BS 参数→下半脸舵机角度，配合语音播放实时驱动嘴唇
- **人脸识别**：GPU 加速人脸识别 + 目标搜索，自动定位并注视对话对象

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               Windows (PC)                                       │
│                                                                                  │
│  ┌──────────────────┐   ┌──────────────────┐   ┌─────────────────┐              │
│  │ audio/xunfei_ear_│   │ emo_clf/realtime_│   │ brain/deepseek_  │              │
│  │ realtime.py      │   │ emotion_pro.py   │   │ response.py     │              │
│  │                  │   │                  │   │                 │              │
│  │ 4声道麦克风采集   │   │ 摄像头 → MediaPipe│   │ DeepSeek 大模型  │              │
│  │ 讯飞ASR语音识别   │   │ → 情绪分类器      │   │ DashScope TTS   │              │
│  │ 声源定位(GCC)    │   │ → UDP:5007       │   │ 任务优先级调度   │              │
│  │ AEC回声消除      │   │                  │   │                 │              │
│  └────────┬─────────┘   └────────┬─────────┘   └────────┬────────┘              │
│           │ UDP:5006             │ UDP:5007              │ UDP:5010              │
│           │ 语音文本             │ 情绪标签              │ AEC参考               │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ exp_bs/ (表情引擎)                                                       │   │
│  │  test_emotion_to_face.py  — 情感大脑 (label→52 BS → servo)              │   │
│  │  run_empathy_live.py      — 实时共情 (camera→empathy→数字人)              │   │
│  │  run_empathy_deploy.py    — 实时共情 (camera→empathy→舵机)               │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────┼───────────────────────────────────────────┘
                                    │ UDP:8888
                                    ▼
                        ┌─────────────────────────┐
                        │   RK3588 (ARM Linux)      │
                        │                           │
                        │   pi_servo_udp.py         │
                        │   → PCA9685 (I2C bus 4)   │
                        │   → 33 路舵机              │
                        └─────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                            WSL2 (Ubuntu)                                      │
│                                                                               │
│  ┌──────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐   │
│  │ vision/brainv4.py│  │ bs2angle/infer/      │  │ bs2angle/infer/      │   │
│  │                  │  │ test_udp_fullface.py  │  │ test_udp_upper.py    │   │
│  │ 人脸识别(dlib)   │  │ 全脸实时驱动          │  │ 上半脸驱动           │   │
│  │ 目标搜索         │  │ BS→Angle→UDP         │  │ BS→Angle→UDP         │   │
│  └────────┬─────────┘  └──────────┬───────────┘  └──────────┬───────────┘   │
│           │ UDP:5005              │ UDP:8888                 │ UDP:8888       │
└───────────┼───────────────────────┼──────────────────────────┼───────────────┘
            │                       │                          │
            └───────────────────────┼──────────────────────────┘
                                    │
                                    ▼
                           RK3588 舵机控制器
```

## 目录结构

```
emotion_MLP/
├── brain/                        # 大脑引擎：LLM 对话、TTS 语音合成
│   └── deepseek_response.py
│
├── audio/                        # 语音识别：ASR（讯飞、SenseVoice）
│   ├── xunfei_ear_realtime.py
│   ├── sensevoice_ear_realtime.py
│   └── xunfei_register.py
│
├── emo_clf/                      # 情绪分类：52 BS → 7 种情绪
│   ├── realtime_emotion_pro.py
│   └── models/
│       └── emotion_model.pkl
│
├── exp_bs/                       # 表情引擎：情绪/共情 → 52 BS
│   ├── scripts/                  #   训练/数据生成脚本
│   │   ├── gen_batch_data.py     #   批量生成情绪 BS 序列
│   │   ├── train_brain.py        #   训练情感大脑 (label→52 BS)
│   │   ├── empathy_map.py        #   共情映射规则定义
│   │   ├── gen_empathy_data.py   #   自举生成共情配对数据
│   │   └── train_empathy.py      #   训练 BS2BS_Empathy 模型
│   ├── infer/                    #   实时推理脚本
│   │   ├── test_emotion_to_face.py   # 情感大脑 → 舵机驱动
│   │   ├── run_empathy_live.py       # 实时共情（数字人验证）
│   │   └── run_empathy_deploy.py     # 实时共情（机器人部署）
│   ├── val/                      #   验证/渲染工具
│   │   └── test_offline.py       #   离线共情验证（自洽性/可实现性/分类/多样性）
│   └── models/                   #   表情模块模型权重
│       ├── emotion_brain.pth
│       └── bs2bs_empathy.pth
│
├── bs2angle/                     # 舵机转换：BS ↔ 角度映射
│   ├── scripts/                  #   MLP 训练脚本
│   │   ├── train_angle2bs.py     #   正向模型：角度→BS
│   │   ├── train_inverse_cycle.py #   全脸逆向：循环一致训练
│   │   ├── train_upper_face_mlp.py # 上半脸逆向 (19 BS→8角度)
│   │   └── train_lower_face_mlp.py # 下半脸逆向 (32 BS→16角度)
│   ├── infer/                    #   实时舵机驱动脚本
│   │   ├── test_udp_fullface.py  #   全脸实时驱动（UDP）
│   │   ├── test_udp_upper.py     #   上半脸（UDP）
│   │   ├── test_udp_lower.py     #   下半脸（UDP）
│   │   ├── test_fullface.py      #   全脸驱动
│   │   ├── test_fullface_dual.py #   上下脸双模型
│   │   ├── test_tcp_upperface.py #   TCP 上半脸
│   │   └── test_speech_lower.py  #   语音唇形同步
│   └── models/                   #   舵机模块模型权重
│       ├── angle2bs_full.pth
│       ├── bs2angle_cycle.pth
│       ├── upper_face_bs2angle.pth
│       ├── lower_face_bs2angle.pth
│       └── lower_face_bs2angle1.pth
│
├── vision/                       # 视觉：人脸识别、追踪
│   ├── brainv4.py                # 视觉引擎（识别+目标搜索）
│   ├── recognize_gpu_cnn.py      # GPU 人脸识别验证
│   ├── web_test.py               # 视频流查看器
│   ├── requirements_vision.txt   # 独立依赖
│   ├── dataset/                  # 4人 × 300 张人脸图像
│   └── model.yml / names.pkl    # 识别模型
│
├── data_coll/                    # 数据采集
│   ├── pc_vision_server.py       # Flask 采集服务器
│   └── raw_data/                 # 原始采集数据
│       ├── motor_babbling_data_PC.json
│       └── motor_babbling_data_clean.json
│
├── data/                         # 共享数据集
│   └── empathy_training_data.npz # 共情训练配对数据
│
├── utils/                        # 工具脚本
│   ├── data_cleaner.py           # 交互式数据清洗
│   ├── clean_json.py             # JSON 批量删除 & 重编号
│   ├── compare_lips.py           # 唇部 BS 数值对比
│   └── render_transition.py      # 情感过渡动画 + Blender 渲染
│
├── face_landmarker.task          # MediaPipe 面部关键点模型
├── names.pkl                     # 人名→ID 映射
├── requirements.txt
└── README.md

pi/Morpheus/                      # RK3588 舵机控制守护程序
blender_test/                     # Blender 离线渲染验证
```

## 文件清单

### 大脑引擎 (brain/)

| 文件 | 运行平台 | 说明 |
|------|----------|------|
| `brain/deepseek_response.py` | Windows | 中央大脑：接收情绪/语音输入，调用 DeepSeek 大模型生成带情感标签的回复，通过 DashScope TTS 合成语音播放 |

### 语音识别 (audio/)

| 文件 | 运行平台 | 说明 |
|------|----------|------|
| `audio/xunfei_ear_realtime.py` | Windows | 4声道麦克风采集，讯飞 WebSocket ASR，GCC 声源定位，AEC 回声消除 |
| `audio/sensevoice_ear_realtime.py` | Windows | SenseVoice 本地 ASR + CAM++ 声纹 + Silero VAD |
| `audio/xunfei_register.py` | Windows | 讯飞声纹注册工具 |

### 情绪分类 (emo_clf/)

| 文件 | 运行平台 | 说明 |
|------|----------|------|
| `emo_clf/realtime_emotion_pro.py` | Windows | 实时摄像头情绪识别，52 BS → 7 种情绪，UDP:5007 发送 |

### 表情引擎 (exp_bs/)

| 文件 | 运行平台 | 说明 |
|------|----------|------|
| `exp_bs/scripts/gen_batch_data.py` | Windows | 批量生成情绪 BS 序列（19 种手工模板） |
| `exp_bs/scripts/train_brain.py` | Windows | 训练情感大脑：情绪 one-hot → 52 BS |
| `exp_bs/scripts/empathy_map.py` | Windows | 共情映射规则：7 种用户情绪 → 共情目标分布 |
| `exp_bs/scripts/gen_empathy_data.py` | Windows | 自举生成共情配对训练数据 (.npz) |
| `exp_bs/scripts/train_empathy.py` | Windows | 训练 BS2BS_Empathy 共情模型（3 阶段：预训练→微调→循环对齐） |
| `exp_bs/infer/test_emotion_to_face.py` | Windows | 情感大脑管线：label→BS→逆向模型→UDP:8888 |
| `exp_bs/infer/run_empathy_live.py` | Windows | 实时共情推理（摄像头→MediaPipe→BS2BS_Empathy→控制台/3D预览） |
| `exp_bs/infer/run_empathy_deploy.py` | Windows | 共情部署管线（摄像头→MediaPipe→共情→逆向模型→UDP:8888→舵机） |
| `exp_bs/val/test_offline.py` | Windows | 离线共情验证：自洽性、物理可实现性、情绪分类、多样性 |

### 舵机转换 (bs2angle/)

| 文件 | 运行平台 | 说明 |
|------|----------|------|
| `bs2angle/scripts/train_angle2bs.py` | Windows | 训练正向模型：舵机角度 → 52 BS |
| `bs2angle/scripts/train_inverse_cycle.py` | Windows | 训练全脸逆向：52 BS → 角度，循环一致损失 |
| `bs2angle/scripts/train_upper_face_mlp.py` | Windows | 训练上半脸逆向：19 BS → 8 角度 |
| `bs2angle/scripts/train_lower_face_mlp.py` | Windows | 训练下半脸逆向：32 BS → 16 角度 |
| `bs2angle/infer/test_udp_fullface.py` | WSL | 全脸实时驱动（UDP） |
| `bs2angle/infer/test_udp_upper.py` | WSL | 上半脸实时驱动（UDP） |
| `bs2angle/infer/test_udp_lower.py` | WSL | 下半脸实时驱动（UDP） |
| `bs2angle/infer/test_fullface.py` | WSL | 全脸驱动 |
| `bs2angle/infer/test_fullface_dual.py` | WSL | 上下脸双模型 |
| `bs2angle/infer/test_tcp_upperface.py` | WSL | TCP 上半脸驱动 |
| `bs2angle/infer/test_speech_lower.py` | PC | 语音唇形同步 |

### 数据采集 (data_coll/)

| 文件 | 说明 |
|------|------|
| `data_coll/pc_vision_server.py` | Flask 采集服务器：接收舵机指令 + 同步采集视频+MediaPipe BS |

### 视觉 (vision/)

| 文件 | 说明 |
|------|------|
| `vision/brainv4.py` | 视觉引擎：GPU人脸识别、MediaPipe BS、目标2D搜索、UDP:5005 |
| `vision/recognize_gpu_cnn.py` | GPU人脸识别验证 |
| `vision/web_test.py` | 视频流查看器 |

### 工具脚本 (utils/)

| 文件 | 说明 |
|------|------|
| `utils/data_cleaner.py` | 交互式数据审核清洗 |
| `utils/clean_json.py` | JSON 批量删除 & 重新编号 |
| `utils/compare_lips.py` | 唇部 BS 数值对比 |
| `utils/render_transition.py` | 情感过渡动画 + Blender 渲染 + ffmpeg 视频合成 |

## 预训练模型

| 模型 | 输入 → 输出 | 网络结构 | 文件 |
|------|-------------|----------|------|
| 正向（Angle→BS） | 30维角度 → 52维 BS | 256-128-64 MLP | `bs2angle/models/angle2bs_full.pth` |
| 全脸逆向（BS→Angle） | 52维 BS → 30维角度 | 256-128-64 MLP | `bs2angle/models/bs2angle_cycle.pth` |
| 上半脸逆向 | 19维 BS（眉+眼）→ 8维角度 | 256-128-64 MLP | `bs2angle/models/upper_face_bs2angle.pth` |
| 下半脸逆向 | 32维 BS（嘴+鼻+颊）→ 16维角度 | 256-128-64 MLP | `bs2angle/models/lower_face_bs2angle.pth` |
| 下半脸逆向 v1 | 32维 BS → 16维角度（备用） | 256-128-64 MLP | `bs2angle/models/lower_face_bs2angle1.pth` |
| 情绪分类器 | 52维 BS → 7种情绪 | 随机森林 | `emo_clf/models/emotion_model.pkl` |
| 情感大脑 | 情绪标签 → 52 BS | MLP | `exp_bs/models/emotion_brain.pth` |
| 共情模型 | 用户52 BS → 共情52 BS | 128-256-128 MLP | `exp_bs/models/bs2bs_empathy.pth` |

## 系统通信架构

所有模块通过 UDP 进行跨平台通信（Windows ↔ WSL2）：

| 端口 | 方向 | 内容 | 发送方 | 接收方 |
|------|------|------|--------|--------|
| 5005 | → 树莓派 | 舵机角度指令 | vision/brainv4.py / bs2angle/infer/test_*.py | RK3588 |
| 5006 | → WSL | 语音识别文本 | audio/xunfei_ear_realtime.py | brainv4.py |
| 5007 | → 大脑 | 情绪标签 | emo_clf/realtime_emotion_pro.py | brain/deepseek_response.py |
| 5008 | → 大脑 | 语音对话文本 | audio/xunfei_ear_realtime.py | brain/deepseek_response.py |
| 5009 | → 下游 | 处理后情绪输出 | brain/deepseek_response.py | 下游模块 |
| 5010 | → 下游 | AEC 参考音频 | brain/deepseek_response.py | audio/xunfei_ear_realtime.py |
| 8888 | → 树莓派 | 舵机指令 | exp_bs/infer/test_emotion_to_face.py / bs2angle/infer/test_*.py | RK3588 |

RK3588 IP：`10.192.53.8`，Windows 主机 IP：`172.16.1.55`。

## 环境配置

**推荐使用 conda 环境**：

```bash
conda activate emotion_mlp
pip install -r requirements.txt
```

### CUDA / GPU 配置

训练和推理均可利用 GPU 加速（`torch.cuda.is_available()` 自动检测）。

**1. 确认 NVIDIA 驱动已安装**
```bash
nvidia-smi
```

**2. 安装 GPU 版 PyTorch**

根据你的 CUDA 版本选择：
```bash
# CUDA 12.4 / 12.5 / 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**3. 验证 GPU 可用**
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### API 密钥配置

| 脚本 | 服务 | 配置变量 |
|------|------|----------|
| `brain/deepseek_response.py` | DeepSeek 大模型 | `DEEPSEEK_API_KEY` |
| `brain/deepseek_response.py` | DashScope TTS | `dashscope.api_key` |
| `audio/xunfei_ear_realtime.py` | 讯飞语音识别 | `APPID`, `APIKey`, `APISecret` |

### 部署说明

- **Windows 端**运行：`brain/deepseek_response.py`、`emo_clf/realtime_emotion_pro.py`、`audio/xunfei_ear_realtime.py`（或 `audio/sensevoice_ear_realtime.py`）、`bs2angle/infer/test_speech_lower.py`、`exp_bs/infer/run_empathy_deploy.py`
- **WSL2 端**运行：`vision/brainv4.py`、`bs2angle/infer/test_udp_fullface.py`、`bs2angle/infer/test_udp_upper.py`、`bs2angle/infer/test_udp_lower.py`
- **RK3588 端**运行：`pi/Morpheus/pi_servo_udp.py`（UDP:8888 接收指令）
- Windows 和 WSL2 之间通过 UDP 通信，确保 WSL2 的 IP 在 Windows 端正确配置

## 数据流水线

训练数据的采集 → 清洗 → 对比验证工作流：

```bash
# 步骤 1：启动数据采集服务器（需配合舵机随机摆动程序）
python emotion_MLP/data_coll/pc_vision_server.py

# 步骤 2：人工审核清洗数据（OpenCV 弹窗，K 保留 / D 删除 / Q 退出）
python emotion_MLP/utils/data_cleaner.py

# 步骤 3：批量删除指定索引范围的数据
python emotion_MLP/utils/clean_json.py

# 步骤 4：对比检查特定样本的唇部 blendshape 数值
python emotion_MLP/utils/compare_lips.py
```

数据采集服务器（`pc_vision_server.py`）在 `http://0.0.0.0:5000/capture` 接收 POST 请求，同步采集摄像头画面并提取 52 维 blendshape。

## 训练

### 舵机映射模型 (bs2angle)

```bash
# 正向模型（角度 → blendshape）
python emotion_MLP/bs2angle/scripts/train_angle2bs.py

# 全脸逆向模型（blendshape → 角度），循环一致训练
python emotion_MLP/bs2angle/scripts/train_inverse_cycle.py

# 分区逆向模型
python emotion_MLP/bs2angle/scripts/train_upper_face_mlp.py
python emotion_MLP/bs2angle/scripts/train_lower_face_mlp.py
```

训练数据来自 `data_coll/raw_data/motor_babbling_data_PC.json`（机械脸随机摆动采集的舵机角度-blendshape 配对数据）。

### 表情引擎 (exp_bs)

```bash
# 情感大脑模型（情绪标签 → 52 BS）
python emotion_MLP/exp_bs/scripts/train_brain.py

# 共情模型（用户 BS → 共情 BS）— 3 阶段训练
python emotion_MLP/exp_bs/scripts/gen_empathy_data.py   # 首先生成配对数据
python emotion_MLP/exp_bs/scripts/train_empathy.py       # 然后训练
```

训练数据来自 `data/empathy_training_data.npz`（自举生成的共情配对数据）。

## 实时推理

```bash
# === 舵机驱动 (bs2angle) ===
# 全脸实时驱动（UDP）
python emotion_MLP/bs2angle/infer/test_udp_fullface.py

# 上下脸双模型全脸驱动（UDP）
python emotion_MLP/bs2angle/infer/test_fullface_dual.py

# 上半脸实时驱动（UDP）
python emotion_MLP/bs2angle/infer/test_udp_upper.py

# 上半脸手动触发（TCP，空格键）
python emotion_MLP/bs2angle/infer/test_tcp_upperface.py

# 下半脸实时驱动（UDP，本地摄像头）
python emotion_MLP/bs2angle/infer/test_udp_lower.py

# === 表情引擎 (exp_bs) ===
# 情感大脑 → 舵机（CLI 切换情绪）
python emotion_MLP/exp_bs/infer/test_emotion_to_face.py 0 1 9

# 情感大脑 → 舵机（UDP 监听 deepseek 情绪ID）
python emotion_MLP/exp_bs/infer/test_emotion_to_face.py --listen

# 实时共情（控制台验证/数字人）
python emotion_MLP/exp_bs/infer/run_empathy_live.py
python emotion_MLP/exp_bs/infer/run_empathy_live.py --viewer     # 3D 预览

# 实时共情（舵机部署）
python emotion_MLP/exp_bs/infer/run_empathy_deploy.py

# === 语音唇形同步 ===
python emotion_MLP/bs2angle/infer/test_speech_lower.py

# === 情绪识别 ===
python emotion_MLP/emo_clf/realtime_emotion_pro.py

# === 中央大脑（对话 + TTS） ===
python emotion_MLP/brain/deepseek_response.py

# === 语音识别 ===
python emotion_MLP/audio/sensevoice_ear_realtime.py  # 本地 SenseVoice
python emotion_MLP/audio/xunfei_ear_realtime.py      # 讯飞云端 ASR

# === 离线共情验证 ===
python emotion_MLP/exp_bs/val/test_offline.py
```

每个测试脚本顶部有配置区，可按需修改：
- 摄像头/视频流 URL
- 树莓派/RK3588 IP 和端口
- 模型文件路径

## 训练策略

### 逆向模型训练

逆向模型采用**循环一致（Cycle-Consistent）**训练方法：

1. **循环损失**：`BS → angle → forward_model → 重建BS`，最小化重建误差
2. **监督损失**：与真实舵机角度的 MSE（低权重）
3. **平滑损失**：相邻舵机动作连贯
4. **居中损失**：无表情时偏向中性位置（0.5）
5. **边界损失**：惩罚超出 [0, 1] 范围的角度值
6. **MMD 损失**：预测角度分布匹配训练数据分布

### 共情模型训练

BS2BS_Empathy 采用**三阶段训练**：

1. **阶段 1 — 恒等映射预训练**：在 motor_babbling 数据上预训练，使模型学会恒等映射（输出≈输入）
2. **阶段 2 — 共情微调**：在配对数据上微调，引入循环一致性损失（`BS → angle → BS` 重建），同时保持中性
3. **阶段 3 — 循环对齐**：提高循环一致性损失权重，使共情输出在物理上可实现（即能被舵机精确复现）

损失函数：
- **共情 L1 损失**：输出与目标共情 BS 的差异
- **循环一致性损失**：`BS → FaceBS2Angle → Angle2BS → 重建BS` 的 L1 误差
- **中性保持损失**：对中性样本施加额外约束，确保无表情时保持中性

## 视觉子系统 (vision/)

视觉子系统运行在 WSL2 端，主要功能：

- **人脸识别**：基于 dlib + face_recognition，GPU 加速，支持多人识别
- **目标搜索**：2D 盲搜索——估计目标在视野中的水平/垂直偏移方向
- **情绪驱动**：从 MediaPipe BS 提取 + 情绪分类 + 逆向模型 → 舵机角度

数据集位于 `vision/dataset/`，包含 4 人各 300 张人脸图像。

如需单独安装视觉模块依赖：
```bash
pip install -r vision/requirements_vision.txt
```
