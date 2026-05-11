# Morpheus — 多模态情感交互系统

基于 MLP（多层感知机）的仿人机器人面部表情驱动系统，整合**情绪识别、语音对话、唇形同步、人脸识别**等多模态交互能力。

## 概述

系统核心功能：

- **表情驱动**：摄像头采集面部 52 维 blendshape（MediaPipe），通过逆向 MLP 模型映射为舵机角度，实时驱动物理机器人面部
- **情绪识别**：从 blendshape 分类 7 种情绪（开心/生气/害怕/惊讶/悲伤/厌恶/中性），通过 UDP 发送给中央大脑
- **语音对话**：讯飞 ASR 实时语音识别 → DeepSeek 大模型生成回复（带情感标签）→ 阿里 DashScope TTS 语音合成
- **唇形同步**：文字→拼音→韵母→唇形 BS 参数→下半脸舵机角度，配合语音播放实时驱动嘴唇
- **人脸识别**：GPU 加速人脸识别 + 目标搜索，自动定位并注视对话对象

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Windows (PC)                               │
│                                                                      │
│  ┌──────────────────┐   ┌──────────────────┐   ┌─────────────────┐  │
│  │ core/xunfei_ear_ │   │ core/realtime_    │   │ core/deepseek_   │  │
│  │ realtime.py      │   │ emotion_pro.py    │   │ response.py     │  │
│  │                  │   │                  │   │                 │  │
│  │ 4声道麦克风采集   │   │ 摄像头 → MediaPipe │   │ DeepSeek 大模型  │  │
│  │ 讯飞ASR语音识别   │   │ → 情绪分类器      │   │ DashScope TTS   │  │
│  │ 声源定位(GCC)    │   │ → UDP:5007       │   │ 任务优先级调度   │  │
│  │ AEC回声消除      │   │                  │   │                 │  │
│  └────────┬─────────┘   └────────┬─────────┘   └────────┬────────┘  │
│           │ UDP:5006             │ UDP:5007              │ UDP:5010  │
│           │ 语音文本             │ 情绪标签              │ AEC参考    │
└───────────┼──────────────────────┼──────────────────────┼───────────┘
            │                      │                      │
     ┌──────┴──────────────────────┴──────────────────────┴──────────┐
     │                        WSL2 (Ubuntu)                           │
     │                                                                │
     │  ┌──────────────────┐  ┌──────────────────┐                   │
     │  │ vision/brainv4.py│  │ test_*.py         │                  │
     │  │                  │  │                   │                  │
     │  │ 人脸识别(dlib)   │  │ BS→Angle 逆向模型  │                  │
     │  │ 目标搜索         │  │ 实时面部驱动       │                  │
     │  └────────┬─────────┘  └────────┬──────────┘                  │
     │           │ UDP:5005            │ UDP:5005                     │
     └───────────┼─────────────────────┼─────────────────────────────┘
                 │                     │
                 ▼                     ▼
        ┌────────────────────────────────────┐
        │         Raspberry Pi (树莓派)        │
        │                                     │
        │   UDP → 串口 → PCA9685 → 30路舵机    │
        └────────────────────────────────────┘
```

## 目录结构

```
emotion_MLP/
├── README.md
├── requirements.txt
├── names.pkl                        # 人名→ID 映射
├── face_landmarker.task             # MediaPipe 面部关键点模型
│
├── core/                            # 核心运行时入口
│   ├── deepseek_response.py         # 中央大脑（大模型对话 + TTS + 任务调度）
│   ├── realtime_emotion_pro.py      # 实时情绪识别
│   ├── sensevoice_ear_realtime.py   # SenseVoice 本地语音识别 + 声纹
│   └── xunfei_ear_realtime.py       # 讯飞语音识别 + 声源定位
│
├── training/                        # 训练脚本 & 数据采集
│   ├── pc_vision_server.py          # 数据采集服务器（Flask + MediaPipe）
│   ├── gen_batch_data.py             # 批量生成情绪 BS 序列（供渲染用）
│   ├── train_angle2bs.py            # 训练：正向模型（角度→BS）
│   ├── train_inverse_cycle.py       # 训练：全脸逆向模型（循环一致）
│   ├── train_upper_face_mlp.py      # 训练：上半脸逆向模型
│   ├── train_lower_face_mlp.py      # 训练：下半脸逆向模型
│   └── train_brain.py               # 训练：情感大脑模型（情绪标签→52 BS）
│
├── inference/                       # 推理/部署测试脚本
│   ├── test_fullface.py             # 测试：全脸实时驱动（UDP）
│   ├── test_fullface_dual.py        # 测试：上下脸双模型全脸驱动
│   ├── test_tcp_upperface.py        # 测试：上半脸驱动（TCP，空格触发）
│   ├── test_udp_upper.py            # 测试：上半脸实时驱动（UDP）
│   ├── test_udp_lower.py            # 测试：下半脸实时驱动（UDP）
│   ├── test_udp_fullface.py         # 测试：上下脸双模型实时驱动
│   └── test_speech_lower.py         # 测试：语音驱动唇形同步
│
├── utils/                           # 工具脚本
│   ├── xunfei_register.py           # 讯飞声纹注册工具
│   ├── data_cleaner.py              # 交互式数据审核清洗工具
│   ├── clean_json.py                # JSON 批量删除 & 重新编号
│   ├── compare_lips.py              # 唇部 blendshape 数值对比
│   ├── batch_render_emotions.sh     # 批量 Blender 渲染所有情绪动画
│   └── render_transition.py         # 情感过渡动画生成+Blender渲染+视频合成
│
├── models/                          # 预训练模型权重
│   ├── angle2bs_full.pth            # 正向模型：30角度 → 52 BS
│   ├── bs2angle_cycle.pth           # 全脸逆向：52 BS → 30角度
│   ├── upper_face_bs2angle.pth      # 上半脸逆向：19 BS → 8角度
│   ├── lower_face_bs2angle.pth      # 下半脸逆向：32 BS → 16角度
│   ├── lower_face_bs2angle1.pth     # 下半脸逆向（备用版本）
│   ├── emotion_model.pkl            # 情绪分类器：52 BS → 7情绪
│   └── emotion_brain.pth            # 情感大脑模型
│
├── data/                            # 训练数据
│   ├── motor_babbling_data_PC.json  # 舵机角度 + blendshape 配对数据
│   └── motor_babbling_data_clean.json  # 人工清洗后的纯净数据集
│
├── speaker_samples/                 # 声纹样本
│   ├── kangkang/                    # 24 条 .wav
│   ├── yinyin/                      # 20 条 .wav
│   └── zhengzheng/                  # 20 条 .wav
│
└── vision/                          # 视觉子系统
    ├── brainv4.py                   # 视觉引擎（人脸识别+目标搜索+舵机指令）
    ├── recognize_gpu_cnn.py         # GPU人脸识别验证
    ├── web_test.py                  # 视频流查看器
    ├── requirements_vision.txt      # 视觉模块独立依赖
    ├── face_landmarker.task         # MediaPipe 模型（副本）
    ├── model.yml                    # LBPH 人脸识别模型
    ├── names.pkl                    # 人名映射（副本）
    ├── haarcascade_frontalface_default.xml  # Haar 级联检测器
    └── dataset/                     # 人脸数据集
        ├── jiajie/      (300张)
        ├── kangkang/    (300张)
        ├── yinyin/      (300张)
        └── zhengzheng/  (300张)
```

## 文件清单

### 核心系统

| 文件 | 运行平台 | 说明 |
|------|----------|------|
| `core/deepseek_response.py` | Windows | 中央大脑：接收情绪/语音输入，调用 DeepSeek 大模型生成带情感标签的回复，通过 DashScope TTS 合成语音播放，按优先级调度任务（P1语音对话 > P2主动行为 > P3表情反应） |
| `core/realtime_emotion_pro.py` | Windows | 实时摄像头情绪识别，52维 BS → 7种情绪分类，通过 UDP:5007 发送给大脑 |
| `core/sensevoice_ear_realtime.py` | Windows | SenseVoice 本地 ASR + CAM++ 声纹识别 + Silero VAD + 声源定位，替代讯飞版本 |
| `core/xunfei_ear_realtime.py` | Windows | 4声道麦克风采集，讯飞 WebSocket ASR 语音识别，GCC 声源定位（水平+垂直角），AEC 回声消除，语音文本通过 UDP:5006/5008 发送 |

### 训练脚本 & 数据采集

| 文件 | 说明 |
|------|------|
| `training/pc_vision_server.py` | Flask 数据采集服务器：接收舵机指令，同步采集摄像头画面（MediaPipe 提取 52 维 BS），构建训练数据集 |
| `training/gen_batch_data.py` | 批量生成情绪 BS 序列数据：根据情绪名称生成 60 帧线性插值 52 维 BS，存为 .npy 供 Blender 渲染 |
| `training/train_angle2bs.py` | 训练正向模型：舵机角度 → 52维 blendshape（256-128-64 MLP + ReLU + Dropout） |
| `training/train_inverse_cycle.py` | 训练全脸逆向模型：52维 BS → 全部舵机角度，使用循环一致损失 |
| `training/train_upper_face_mlp.py` | 训练上半脸逆向：19维 BS（眉毛+眼睛）→ 8个舵机角度 |
| `training/train_lower_face_mlp.py` | 训练下半脸逆向：32维 BS（嘴巴+鼻子+脸颊）→ 16个舵机角度 |
| `training/train_brain.py` | 训练情感大脑模型：情绪 one-hot 标签 → 52 维 BS，生成 `models/emotion_brain.pth` |

### 工具脚本

| 文件 | 说明 |
|------|------|
| `utils/data_cleaner.py` | 交互式数据审核工具：OpenCV 弹窗展示人脸图片 + 违和感评分（不对称/假笑/物理冲突），人工按键 K/D 保留或删除 |
| `utils/clean_json.py` | 批量清洗工具：按索引范围删除 JSON 数据集中的样本并重新编号 |
| `utils/compare_lips.py` | 数据对比工具：打印指定样本 ID 的 22 维唇部 blendshape 参数对比表 |
| `utils/xunfei_register.py` | 讯飞声纹注册工具 |
| `utils/render_transition.py` | 情感过渡动画生成：加载 emotion_brain 模型，计算二阶平滑过渡 BS → Blender 渲染 → ffmpeg 合成视频 |
| `utils/batch_render_emotions.sh` | 批量渲染脚本：遍历 19 种情绪，自动执行数据生成→Blender 渲染→ffmpeg 视频合成 |

### 推理/部署脚本

| 文件 | 运行平台 | 说明 |
|------|----------|------|
| `inference/test_fullface.py` | WSL | 全脸实时驱动，通过网络摄像头流→BS→全脸逆向模型→UDP 舵机指令 |
| `inference/test_fullface_dual.py` | WSL | 上下脸双模型全脸实时驱动，UDP 持续发送 |
| `inference/test_tcp_upperface.py` | WSL | 上半脸驱动，空格键触发，TCP 发送舵机指令 |
| `inference/test_udp_upper.py` | WSL | 上半脸实时驱动，UDP 持续发送，带时序平滑 |
| `inference/test_udp_lower.py` | WSL | 下半脸实时驱动，本地摄像头，UDP 持续发送 |
| `inference/test_udp_fullface.py` | WSL | 上下脸双模型全脸实时驱动（备用版本） |
| `inference/test_speech_lower.py` | PC | 语音驱动唇形同步：中文文字→拼音→韵母→唇形BS→下半脸舵机，配合 TTS 播放 |

### 视觉子系统 (vision/)

| 文件 | 说明 |
|------|------|
| `vision/brainv4.py` | 视觉引擎：GPU人脸识别、MediaPipe BS提取、目标2D搜索、UDP:5005 发送舵机指令 |
| `vision/recognize_gpu_cnn.py` | GPU人脸识别验证脚本 |
| `vision/web_test.py` | 网络视频流查看器（调试用） |

## 预训练模型

| 模型 | 输入 → 输出 | 网络结构 | 文件 |
|------|-------------|----------|------|
| 正向（Angle→BS） | 30维舵机角度 → 52维 BS | 256-128-64 MLP | `models/angle2bs_full.pth` |
| 全脸逆向（BS→Angle） | 52维 BS → 30维舵机角度 | 256-128-64 MLP | `models/bs2angle_cycle.pth` |
| 上半脸逆向 | 19维 BS（眉+眼）→ 8维舵机角度 | 256-128-64 MLP | `models/upper_face_bs2angle.pth` |
| 下半脸逆向 | 32维 BS（嘴+鼻+颊）→ 16维舵机角度 | 256-128-64 MLP | `models/lower_face_bs2angle.pth` |
| 下半脸逆向 v1 | 32维 BS → 16维舵机角度（备用版本） | 256-128-64 MLP | `models/lower_face_bs2angle1.pth` |
| 情绪分类器 | 52维 BS → 7种情绪 | 随机森林（sklearn） | `models/emotion_model.pkl` |
| 情感大脑 | 情绪标签 → 反馈行为参数 | MLP | `models/emotion_brain.pth` |

## 系统通信架构

所有模块通过 UDP 进行跨平台通信（Windows ↔ WSL2）：

| 端口 | 方向 | 内容 | 发送方 | 接收方 |
|------|------|------|--------|--------|
| 5005 | → 树莓派 | 舵机角度指令 | brainv4.py / test_*.py | RPi 舵机控制器 |
| 5006 | → WSL | 语音识别文本 | core/xunfei_ear_realtime.py | brainv4.py |
| 5007 | → 大脑 | 情绪标签 | core/realtime_emotion_pro.py | core/deepseek_response.py |
| 5008 | → 大脑 | 语音对话文本 | core/xunfei_ear_realtime.py | core/deepseek_response.py |
| 5009 | → 下游 | 处理后情绪输出 | core/deepseek_response.py | 下游消费模块 |
| 5010 | → 下游 | AEC 参考音频 | core/deepseek_response.py | core/xunfei_ear_realtime.py |
| 8888 | → 树莓派 | 舵机指令（TCP） | test_tcp_upperface.py / test_speech_lower.py | RPi |

树莓派 IP：`172.16.0.166`，Windows 主机 IP：`172.16.1.55`。

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

### 特殊依赖安装

**pyaudio**（Windows 安装失败时）：
```bash
pip install pipwin && pipwin install pyaudio
```

**dlib**（Windows 安装失败时）：
从 https://github.com/z-mahmud22/Dlib_Windows_Python3 下载对应 Python 版本的 .whl 手动安装。

### API 密钥配置

以下模块需要 API 密钥，需要在对应脚本中修改：

| 脚本 | 服务 | 配置变量 |
|------|------|----------|
| `deepseek_response.py` | DeepSeek 大模型 | `DEEPSEEK_API_KEY` |
| `deepseek_response.py` | DashScope TTS | `dashscope.api_key` |
| `core/xunfei_ear_realtime.py` | 讯飞语音识别 | `APPID`, `APIKey`, `APISecret` |

### 部署说明

- **Windows 端**运行：`core/deepseek_response.py`、`core/realtime_emotion_pro.py`、`core/xunfei_ear_realtime.py`（或 `core/sensevoice_ear_realtime.py`）、`inference/test_speech_lower.py`
- **WSL2 端**运行：`vision/brainv4.py`、`inference/test_fullface.py`、`inference/test_udp_upper.py`、`inference/test_udp_lower.py`
- **树莓派端**运行：舵机控制服务（UDP:5005 接收指令）
- Windows 和 WSL2 之间通过 UDP 通信，确保 WSL2 的 IP 在 Windows 端正确配置（通过 `hostname -I` 查看）

## 数据流水线

训练数据的采集 → 清洗 → 对比验证工作流：

```bash
# 步骤 1：启动数据采集服务器（需配合舵机随机摆动程序）
python training/pc_vision_server.py

# 步骤 2：人工审核清洗数据（OpenCV 弹窗，K 保留 / D 删除 / Q 退出）
python utils/data_cleaner.py

# 步骤 3：批量删除指定索引范围的数据（在脚本中配置区间）
python utils/clean_json.py

# 步骤 4：对比检查特定样本的唇部 blendshape 数值
python utils/compare_lips.py
```

数据采集服务器（`pc_vision_server.py`）在 `http://0.0.0.0:5000/capture` 接收 POST 请求，body 格式为 `{"motor_commands": {...}}`，同步采集摄像头画面并提取 52 维 blendshape，每 10 条自动保存一次。

## 训练

```bash
# 正向模型（角度 → blendshape）
python training/train_angle2bs.py

# 全脸逆向模型（blendshape → 角度），循环一致训练
python training/train_inverse_cycle.py

# 分区逆向模型
python training/train_upper_face_mlp.py
python training/train_lower_face_mlp.py

# 情感大脑模型（情绪标签 → 52 BS）
python training/train_brain.py
```

训练数据来自 `motor_babbling_data_PC.json`（机械脸随机摆动采集的舵机角度-blendshape 配对数据）。

## 实时推理

```bash
# === 面部驱动 ===
# 全脸实时驱动（UDP）
python inference/test_fullface.py

# 上下脸双模型全脸驱动（UDP）
python inference/test_fullface_dual.py

# 上半脸实时驱动（UDP）
python inference/test_udp_upper.py

# 上半脸手动触发（TCP，空格键）
python inference/test_tcp_upperface.py

# 下半脸实时驱动（UDP，本地摄像头）
python inference/test_udp_lower.py

# === 语音唇形同步 ===
# 输入文字 → TTS 播放 + 唇形舵机同步
python inference/test_speech_lower.py

# === 情绪识别 ===
python core/realtime_emotion_pro.py

# === 中央大脑（对话 + TTS） ===
python core/deepseek_response.py

# === 语音识别（本地 SenseVoice + 声纹） ===
python core/sensevoice_ear_realtime.py

# === 语音识别（讯飞云端 ASR） ===
python core/xunfei_ear_realtime.py
```

每个测试脚本顶部有配置区，可按需修改：
- 摄像头/视频流 URL
- 树莓派 IP 和端口
- 模型文件路径

## 训练策略

逆向模型采用**循环一致（Cycle-Consistent）**训练方法：

1. **循环损失**：`BS → angle → forward_model → 重建BS`，最小化重建误差
2. **监督损失**：与真实舵机角度的 MSE（低权重）
3. **平滑损失**：相邻舵机动作连贯
4. **居中损失**：无表情时偏向中性位置（0.5）
5. **边界损失**：惩罚超出 [0, 1] 范围的角度值
6. **MMD 损失**：预测角度分布匹配训练数据分布

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
