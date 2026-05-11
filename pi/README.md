# Morpheus 树莓派舵机控制子系统

运行在 Raspberry Pi 上，通过 I2C 控制 3 块 PCA9685 舵机驱动板（33 路舵机），接收 PC/WSL 端通过 UDP/TCP 发来的舵机角度指令，驱动物理机器人面部表情。

## 硬件要求

- Raspberry Pi（树莓派 4B/5，启用 I2C）
- 3 块 PCA9685 舵机驱动板（地址 0x40、0x41、0x42，级联）
- 33 路舵机（对应人脸的眉毛、眼睛、嘴巴、下巴、脖子等部位）
- 外部电源（舵机供电，不可从 Pi 的 5V 取电）

## 依赖安装

```bash
pip install -r requirements.txt
```

启用树莓派 I2C：
```bash
sudo raspi-config  # Interface Options → I2C → Enable
sudo reboot
```

验证 I2C 设备是否在线：
```bash
sudo i2cdetect -y 1
# 应看到 0x40、0x41、0x42 三个地址
```

## 文件说明

### 核心服务（主入口）

| 文件 | 说明 |
|------|------|
| `pi_servo_udp.py` | **UDP 舵机服务**（推荐）：监听 UDP:8888，接收 JSON `{"通道": 角度}`，并行设置 33 路舵机立即到位，无平滑插值，最低延迟 |
| `pi_servo_service.py` | **TCP 舵机流水线服务**：监听 TCP:8888，支持多客户端连接，内置双缓冲执行器（current/next），带性能计时日志 |

### 舵机工具

| 文件 | 说明 |
|------|------|
| `servo_set_position.py` | 舵机位置手动设置工具：命令行指定通道和角度，支持平滑移动和进度显示，用于单点调试 |
| `set_start_bound.py` | **舵机安全限幅配置**：定义 33 个舵机的 `(最小角度, 最大角度, 中性位置)`，防止舵机超出物理安全范围。同时提供 `--sync` 一键同步复位功能 |

### 数据采集

| 文件 | 说明 |
|------|------|
| `data_collect.py` | 电机随机摆动采集 v1：随机生成目标表情 → 平滑移动舵机 → 每 10° 触发 PC 端摄像头采集 |
| `data_collect2.py` | 电机随机摆动采集 v2：配合 PC Flask 采集服务器，对称概率可调 |
| `data_collect3.py` | 电机随机摆动采集 v3：非对称版本，眉毛/眼睑对独立控制 |

### 测试脚本

| 文件 | 说明 |
|------|------|
| `morpheus_trackerv2.py` | 人脸追踪测试 v2：简易卡尔曼滤波，控制眼睛上下/左右 + 脖子点头/旋转 |
| `morpheus_trackerv3.py` | 人脸追踪测试 v3：改进版追踪逻辑，增加眼睑控制 |

### 驱动库

| 文件 | 说明 |
|------|------|
| `mor_servo_dev.py` | PCA9685 I2C 底层驱动：PWM 脉冲生成、角度/脉冲互转、多板级联支持 |

## 舵机通道映射

所有通道使用全局编号 0-32，通过 `get_hardware_target()` 自动路由到对应 PCA9685 板：

| 全局通道 | PCA 板 | 本地通道 | 面部区域 |
|----------|--------|----------|----------|
| 0-15 | 0x40 | 0-15 | 下半脸 + 部分上半脸 |
| 16-31 | 0x41 | 0-15 | 上半脸 + 下半脸 |
| 32 | 0x42 | 0 | 脖子旋转 |

具体每个通道对应的面部部位详见 `set_start_bound.py` 中的 `TABLE_V_CONFIG` 注释。

## 启动服务

```bash
# UDP 服务（配合 PC 端 test_udp_*.py 或 brainv4.py）
python3 pi_servo_udp.py

# TCP 服务（配合 PC 端 test_tcp_upperface.py 或 test_speech_lower.py）
python3 pi_servo_service.py
```

## 数据采集工作流

配合 PC 端 `emotion_MLP/training/pc_vision_server.py`（Flask 数据采集服务器）使用：

```bash
# 1. PC 端启动 Flask 采集服务器
python training/pc_vision_server.py

# 2. Pi 端启动随机摆动采集（修改 pc_ip 为 PC 的实际 IP）
python3 data_collect2.py --pc-ip 172.16.1.55
```

Pi 端随机生成舵机目标角度 → 平滑移动舵机 → 每 10° 发 HTTP 请求到 PC 端的 `/capture` → PC 端同步采集摄像头画面 + MediaPipe blendshape → 保存为训练数据对。

## 舵机安全限幅

`set_start_bound.py` 和 `pi_servo_udp.py` 中各有一份 `TABLE_V_CONFIG`，定义了每个舵机的安全活动范围 `(min, max, neutral)`。所有舵机指令在发送 PWM 前都会经过 `clamp_angle()` 限幅，防止舵机撞击机械限位。

如果物理结构有变动，请同步更新两份 `TABLE_V_CONFIG`。
