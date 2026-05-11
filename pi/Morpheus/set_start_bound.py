#!/usr/bin/env python3
"""
Morpheus 机器人面部控制系统
集成物理同步、多线程并行、平滑移动以及位置存档功能
支持三板级联 (0x40, 0x41, 0x42)
"""

import sys
import time
import json
import os
import argparse
import threading
# 请确保您的 mor_servo_dev.py 处于同一目录下
from mor_servo_dev import PCA9685, pulse_us_to_angle, angle_to_pulse_us


# 物理同步操作指南（新通道号）：
# 手动拨正：断开电源，手动将 33 个部位拨动到中性起始位置。

# 一键同步（全通道 0-32）： python3 set_start_bound.py --channel $(seq 0 32) --sync

# 单点测试（例如颌部左右）： python3 set_start_bound.py --channel 19 --sync

# 0-180 逆时针为正  
# --- 1. 基于论文 Table V 的安全阈值与起始点配置 (使用 0-32 索引) ---
TABLE_V_CONFIG = {
    0:  (0.0,   90.0,  90.0),  # 左鼻子 (肌腱) 往小是上拉
    1:  (27.0,  100.0, 100.0), # 左脸颊 (肌腱) 往小是上拉
    2:  (70.0,  105.0, 105.0),  # 左眉心 (75.0,  100.0, 75.0)
    3:  (70.0,  100.0,  100.0),  # 左眉峰
    4:  (20.0,  120.0, 30.0), # 左上眼睑，往大闭眼
    5:  (70.0,  100.0, 70.0), # 左下眼睑，往大闭眼
    6:  (30.0,  110.0,  60.0),   # 眼睛上下视，往大是向上看
    7:  (54.0,  90.0,  90.0),  # 上唇中心 往小是上拉
    8:  (45.0,  135.0, 90.0),  # 左上嘴角 往大是上拉
    9:  (80.0,  90.0,  90.0),  # 右上唇 往大提拉
    10: (70.0,  90.0, 70.0),  # 舌头伸缩                     无
    11: (70.0,  100.0,  70.0),  # 舌头上下                   无
    12: (90.0,  140.0, 135.0), # 左下巴 (前后) 变小是往前     无
    13: (45.0,  135.0, 90.0),  # 左下嘴角 往大提拉
    14: (110.0,  180.0, 180.0),  # 下唇中心，往小是向后拉
    15: (60.0,  105.0, 105.0),  # 左下唇，往小是向后拉
    16: (130.0,  165.0, 130.0), # 右下唇，往大是向后拉
    17: (40.0,  90.0,  50.0),  # 右下巴 (前后) 变大是往前     无
    18: (45.0,  135.0, 90.0),  # 右下嘴角
    19: (70.0,  130.0, 70.0),  # 嘴巴张合（左） 变大张嘴
    20: (20.0,  62.0,  62.0),  # 左上唇 往小提拉
    21: (45.0,  135.0, 90.0),  # 右上嘴角
    22: (40.0,  100.0,  100.0),  # 嘴巴张合（右）  变小张嘴
    23: (55.0,  95.0,  75.0),   # 眼睛左右视，往大是右边看
    24: (20.0,   50.0,  40.0),    # 右下眼睑，往小闭眼
    25: (60.0,  150.0,  140.0),   # 右上眼睑，往小闭眼
    26: (65.0,  100.0, 90.0), # 右眉峰
    27: (60.0,  110.0, 90.0), # 右眉心 往大上挑
    28: (0.0,   50.0,  0.0),   # 右脸颊 (肌腱) 往大上拉
    29: (0.0,   90.0, 0.0),    # 右鼻子 (肌腱) 往大上拉
    30: (50.0,  165.0, 150.0), # 脖子点头/侧摆 (左)，往小低头
    31: (20.0,  100.0, 34.0),  # 脖子点头/侧摆 (右)，往大低头
    32: (0, 180.0, 90), # 脖子旋转 ，往大是向左
}

# --- 2. 位置存档管理器 ---
class ServoStateManager:
    def __init__(self, filename="morpheus_dual_state.json"):
        self.filename = filename
        self.state = self.load_state()

    def load_state(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    data = json.load(f)
                    return {int(k): float(v) for k, v in data.items()}
            except: pass
        return {}

    def save_state(self, channel, angle):
        self.state[channel] = angle
        with open(self.filename, 'w') as f:
            json.dump(self.state, f, indent=4)

    def get_last_angle(self, channel):
        return self.state.get(channel, TABLE_V_CONFIG.get(channel, (0, 180, 90.0))[2])

# --- 3. 硬件路由与运动逻辑 ---

def get_hardware_target(global_ch, pcas):
    """
    硬件路由逻辑：
    - 0x40 板：全局 0-15
    - 0x41 板：全局 16-31 (映射到 0-15)
    - 0x42 板：全局 32 (映射到 0)
    """
    if 0 <= global_ch <= 15:
        return pcas.get(0x40), global_ch
    elif 16 <= global_ch <= 31:
        return pcas.get(0x41), global_ch - 16
    elif global_ch == 32:
        return pcas.get(0x42), global_ch - 32
    return None, None

def clamp_angle(channel, angle):
    if channel in TABLE_V_CONFIG:
        min_a, max_a, _ = TABLE_V_CONFIG[channel]
        return max(min_a, min(max_a, angle))
    return max(0.0, min(180.0, angle))

def set_servo_angle_smooth(pca, local_ch, target_angle, current_angle, args, lock):
    if not pca: return
    
    if args.speed <= 0 or target_angle == current_angle:
        pulse = angle_to_pulse_us(target_angle, args.min_angle, args.max_angle, args.min_us, args.max_us)
        with lock: 
            pca.set_servo_pulse_us(local_ch, pulse)
        return

    step_dt = max(0.01, args.step_ms / 1000.0)
    max_step = args.speed * step_dt
    angle = current_angle
    direction = 1.0 if target_angle >= angle else -1.0

    while True:
        delta = target_angle - angle
        if abs(delta) <= max(args.tolerance, max_step):
            angle = target_angle
        else:
            angle += direction * max_step
        
        pulse = angle_to_pulse_us(angle, args.min_angle, args.max_angle, args.min_us, args.max_us)
        with lock: 
            pca.set_servo_pulse_us(local_ch, pulse)
        
        if angle == target_angle: break
        time.sleep(step_dt)

# --- 4. 主程序入口 ---

def main():
    parser = argparse.ArgumentParser(description='Morpheus 面部三板级联控制 (0x40, 0x41 & 0x42)')
    parser.add_argument('--channel', type=int, nargs='+', required=True, help='全局通道 (0-32)')
    parser.add_argument('--angle', type=float, nargs='+', help='目标角度')
    parser.add_argument('--speed', type=float, default=0.0, help='移动速度 (deg/s)')
    parser.add_argument('--sync', action='store_true', help='同步模式')
    parser.add_argument('--release', action='store_true', help='完成后释放力矩')
    parser.add_argument('--step-ms', type=float, default=20.0, help='步进间隔 (ms)')
    parser.add_argument('--tolerance', type=float, default=0.5, help='角度容差')
    parser.add_argument('--min-us', type=float, default=600.0, help='最小脉冲宽度')
    parser.add_argument('--max-us', type=float, default=2400.0, help='最大脉冲宽度')
    parser.add_argument('--min-angle', type=float, default=0.0, help='最小限制角度')
    parser.add_argument('--max-angle', type=float, default=180.0, help='最大限制角度')

    args = parser.parse_args()
    state_manager = ServoStateManager()
    
    # 所有控制板共用一条 I2C 总线，使用同一个锁
    i2c_bus_lock = threading.Lock()

    try:
        # 初始化三片控制板
        pcas = {
            0x40: PCA9685(bus_id=1, address=0x40, freq_hz=50),
            0x41: PCA9685(bus_id=1, address=0x41, freq_hz=50),
            0x42: PCA9685(bus_id=1, address=0x42, freq_hz=50)
        }
        
        target_angles = [TABLE_V_CONFIG[ch][2] for ch in args.channel] if args.sync else \
                        (args.angle if len(args.angle) == len(args.channel) else [args.angle[0]] * len(args.channel))

        threads = []
        def task(global_ch, tgt):
            pca, local_ch = get_hardware_target(global_ch, pcas)
            if not pca: return

            tgt = clamp_angle(global_ch, tgt)
            
            with i2c_bus_lock:
                try:
                    p = pca.get_servo_pulse_us(local_ch)
                    curr = pulse_us_to_angle(p, 0, 180, 600, 2400) if p > 1.0 else state_manager.get_last_angle(global_ch)
                except: 
                    curr = state_manager.get_last_angle(global_ch)
            
            start_pos = tgt if args.sync else curr
            set_servo_angle_smooth(pca, local_ch, tgt, start_pos, args, i2c_bus_lock)
            
            state_manager.save_state(global_ch, tgt)
            if args.release:
                with i2c_bus_lock: 
                    pca.set_channel_full_off(local_ch, True)

        for ch, tg in zip(args.channel, target_angles):
            t = threading.Thread(target=task, args=(ch, tg))
            threads.append(t)
            t.start()

        for t in threads: t.join()
        
        for p in pcas.values(): p.close()
        print("\n[✓] 指令执行完毕（三板级联），数据已持久化。")

    except Exception as e:
        print(f"运行失败: {e}")

if __name__ == "__main__":
    main()
