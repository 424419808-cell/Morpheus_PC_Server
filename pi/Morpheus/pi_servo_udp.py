#!/usr/bin/env python3
"""
树莓派舵机 UDP 服务 - 极速版
接收 UDP 数据包，解析 JSON，立即设置舵机角度（无平滑插值）
"""

import socket
import json
import threading
import time
from mor_servo_dev import PCA9685, angle_to_pulse_us

# ================= 舵机安全限幅表 =================
TABLE_V_CONFIG = {
    0:  (0.0,   90.0,  90.0),
    1:  (27.5,  100.0, 100.0),
    2:  (70.0,  105.0, 105.0),
    3:  (70.0,  100.0,  100.0),
    4:  (20.0,  120.0, 30.0),
    5:  (80.0,  110.0, 90.0),
    6:  (30.0,  110.0,  60.0),
    7:  (54.0,  90.0,  90.0),
    8:  (45.0,  135.0, 90.0),
    9:  (80.0,  90.0,  90.0),
    10: (70.0,  90.0, 70.0),
    11: (70.0,  100.0,  70.0),
    12: (90.0,  140.0, 135.0),
    13: (45.0,  135.0, 90.0),
    14: (110.0, 180.0, 180.0),
    15: (60.0,  105.0, 105.0),
    16: (120.0, 165.0, 120.0),
    17: (40.0,  90.0,  50.0),
    18: (45.0,  135.0, 90.0),
    19: (70.0,  130.0, 70.0),
    20: (20.0,  62.0,  62.0),
    21: (45.0,  135.0, 90.0),
    22: (40.0,  100.0, 100.0),
    23: (55.0,  95.0,  75.0),
    24: (20.0,  50.0,  40.0),
    25: (60.0,  150.0, 130.0),
    26: (65.0,  100.0, 90.0),
    27: (60.0,  110.0, 90.0),
    28: (0.0,   50.0,  0.0),
    29: (0.0,   90.0, 0.0),
    30: (50.0,  165.0, 150.0),
    31: (20.0,  100.0, 34.0),
    32: (0,     180.0, 90),
}

PCA_ADDRS = [0x40, 0x41, 0x42]

pcas = {}
i2c_lock = threading.Lock()

def init_pcas():
    for addr in PCA_ADDRS:
        pcas[addr] = PCA9685(bus_id=1, address=addr, freq_hz=50)
    print("✅ PCA9685 初始化完成")

def get_hardware_target(global_ch):
    if 0 <= global_ch <= 15:
        return pcas.get(0x40), global_ch
    elif 16 <= global_ch <= 31:
        return pcas.get(0x41), global_ch - 16
    elif global_ch == 32:
        return pcas.get(0x42), 0
    return None, None

def clamp_angle(channel, angle):
    if channel in TABLE_V_CONFIG:
        min_a, max_a, _ = TABLE_V_CONFIG[channel]
        return max(min_a, min(max_a, angle))
    return max(0.0, min(180.0, angle))

def set_servo_angle_immediate(global_ch, target_angle):
    pca, local_ch = get_hardware_target(global_ch)
    if pca is None:
        return
    target_angle = clamp_angle(global_ch, target_angle)
    pulse = angle_to_pulse_us(target_angle, 0, 180, 600, 2400)
    with i2c_lock:
        pca.set_servo_pulse_us(local_ch, pulse)

def execute_command(angles_dict):
    threads = []
    for ch_str, ang in angles_dict.items():
        ch = int(ch_str)
        t = threading.Thread(target=set_servo_angle_immediate, args=(ch, ang))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

# ================= UDP 服务器 =================
HOST = '0.0.0.0'
PORT = 8888

def main():
    init_pcas()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, PORT))
    print(f"🚀 UDP 舵机服务监听 {HOST}:{PORT}")

    while True:
        data, addr = sock.recvfrom(65535)
        try:
            # 假设 JSON 以换行分隔，也可以直接解析整个数据报
            text = data.decode().strip()
            if not text:
                continue
            angles = json.loads(text)
            execute_command(angles)
            # 可选：打印接收到的指令长度
            # print(f"收到指令，电机数: {len(angles)}")
        except json.JSONDecodeError as e:
            print(f"JSON 解析错误: {e}")
        except Exception as e:
            print(f"处理错误: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 服务终止")
    finally:
        for pca in pcas.values():
            pca.close()