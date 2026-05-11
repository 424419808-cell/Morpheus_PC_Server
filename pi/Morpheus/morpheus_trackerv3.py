import socket
import time
import sys
import select
import numpy as np
import threading
import random

try:
    from mor_servo_dev import PCA9685, angle_to_pulse_us_with_mid, angle_to_pulse_us
except ImportError:
    print("错误: 未找到 mor_servo_dev.py 驱动文件。")
    sys.exit(1)

# --- 硬件与范围配置 ---
CONFIG = {
    6:  (30.0,  80.0,  60.0),   # 眼睛上下 (min, max, mid)
    23: (55.0,  95.0,  75.0),   # 眼睛左右
    30: (70.0,  130.0, 105.0),  # 脖子左推杆
    31: (55.0,  115.0, 80.0),   # 脖子右推杆
    32: (0, 180.0, 90)          # 脖子旋转
}

# --- 眼睑通道配置 ---
EYE_LIDS = {
    "TL": 4,   # 左上眼睑
    "BL": 5,   # 左下眼睑
    "TR": 25,  # 右上眼睑
    "BR": 24   # 右下眼睑
}
# 眨眼时的角度（闭合）
BLINK_POS = { "TL": 150.0, "BL": 162.0, "TR": 20.0, "BR": 0.0 }
# 平时睁开的角度
OPEN_POS = { "TL": 110.0, "BL": 108.0, "TR": 60.0, "BR": 50.0 }

class SimpleKalman:
    def __init__(self):
        self.state = np.zeros(4)
        self.P = np.eye(4) * 10
        self.F = np.eye(4)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.eye(4) * 0.1
        self.R = np.eye(2) * 5

    def predict(self, dt):
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.state)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.state = self.state + np.dot(K, y)
        self.P = np.dot((np.eye(4) - np.dot(K, self.H)), self.P)

class MorpheusTracker:
    def __init__(self):
        try:
            self.pcas = {
                0x40: PCA9685(bus_id=1, address=0x40),
                0x41: PCA9685(bus_id=1, address=0x41),
                0x42: PCA9685(bus_id=1, address=0x42)
            }
        except Exception as e:
            print(f"I2C 失败: {e}"); sys.exit(1)

        self.lock = threading.Lock() 
        self.curr_angles = {ch: cfg[2] for ch, cfg in CONFIG.items()}
        
        # 追踪控制参数
        self.kp_x, self.kp_y = 0.02, 0.022
        self.smooth = 0.2
        self.max_step = 2.5
        self.deadzone = 8
        self.micro_zone_x, self.micro_zone_y = 115, 50
        
        self.eye_x_mid = CONFIG[23][2] 
        self.align_p = 0.15 
        self.align_deadzone = 3.0 
        
        self.kf = SimpleKalman()
        self.last_time = time.time()
        self.has_detected_first_time = False 
        self.prediction_count = 0 
        self.MAX_PREDICTIONS = 100 
        
        self.is_sleeping = False
        self.is_blinking = False 
        
        # 随机眨眼计时器
        self.last_blink_time = time.time()
        self.blink_interval = random.uniform(1.0, 5.0) 
        
        self.reset_pose()

    def get_target(self, global_ch):
        if 0 <= global_ch <= 15: return self.pcas[0x40], global_ch
        if 16 <= global_ch <= 31: return self.pcas[0x41], global_ch - 16
        if global_ch == 32: return self.pcas[0x42], 0
        return None, None

    def update_servo(self, ch, target_angle):
        current_a = self.curr_angles[ch]
        diff = target_angle - current_a
        if abs(diff) > self.max_step:
            diff = self.max_step if diff > 0 else -self.max_step
        new_angle = current_a + (diff * self.smooth)
        
        min_a, max_a, _ = CONFIG[ch]
        new_angle = max(min_a, min(max_a, new_angle))
        
        pca, local_ch = self.get_target(ch)
        if pca:
            pulse = angle_to_pulse_us_with_mid(new_angle, 0, 90, 180, 600, 1500, 2400)
            with self.lock:
                pca.set_servo_pulse_us(local_ch, pulse)
                self.curr_angles[ch] = new_angle

    def update_eyelids_follow(self):
        """让眼睑随着眼球上下转动而联动"""
        if self.is_blinking or self.is_sleeping:
            return
        eye_y = self.curr_angles[6]
        y_min, y_max, y_mid = CONFIG[6]
        offset_ratio = (eye_y - y_mid) / (y_max - y_mid if eye_y > y_mid else y_mid - y_min)
        follow_amplitude = 15.0 
        l_up = OPEN_POS["TL"] + (offset_ratio * follow_amplitude)
        r_up = OPEN_POS["TR"] - (offset_ratio * follow_amplitude)

        with self.lock:
            pca, ch = self.get_target(EYE_LIDS["TL"])
            pca.set_servo_pulse_us(ch, angle_to_pulse_us(l_up, 0, 180, 600, 2400))
            pca, ch = self.get_target(EYE_LIDS["TR"])
            pca.set_servo_pulse_us(ch, angle_to_pulse_us(r_up, 0, 180, 600, 2400))

    def auto_blink(self):
        """执行一次物理眨眼动作"""
        self.is_blinking = True
        # 闭眼
        for name, ch_id in EYE_LIDS.items():
            pca, l_ch = self.get_target(ch_id)
            if pca:
                pulse = angle_to_pulse_us(BLINK_POS[name], 0, 180, 600, 2400)
                with self.lock: pca.set_servo_pulse_us(l_ch, pulse)
        
        time.sleep(0.12)
        
        # 睁眼
        for name, ch_id in EYE_LIDS.items():
            pca, l_ch = self.get_target(ch_id)
            if pca:
                pulse = angle_to_pulse_us(OPEN_POS[name], 0, 180, 600, 2400)
                with self.lock: pca.set_servo_pulse_us(l_ch, pulse)
        
        # 结束后重新生成下次眨眼时间
        self.blink_interval = random.uniform(1.0, 5.0)
        self.is_blinking = False

    def track(self, error_x, error_y, is_real_data=True):
        self.is_sleeping = False
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        
        self.kf.predict(dt)
        if is_real_data:
            self.kf.update(np.array([error_x, error_y]))
            self.has_detected_first_time = True
            self.prediction_count = 0
        else:
            self.prediction_count += 1
            error_x, error_y = self.kf.state[0], self.kf.state[1]

        if self.prediction_count > self.MAX_PREDICTIONS:
            self.sleep()
            return

        # 核心追踪逻辑
        move_x, move_y = error_x * self.kp_x, error_y * self.kp_y
        if abs(error_x) < self.micro_zone_x:
            self.update_servo(23, self.curr_angles[23] - move_x * 0.9)
        else:
            self.update_servo(23, self.curr_angles[23] - move_x * 0.3)
            self.update_servo(32, self.curr_angles[32] - move_x * 0.7)

        if is_real_data and abs(error_x) < (self.micro_zone_x * 0.5):
            eye_x_offset = self.curr_angles[23] - self.eye_x_mid
            if abs(eye_x_offset) > self.align_deadzone:
                align_speed = eye_x_offset * self.align_p
                align_speed = max(-1.2, min(1.2, align_speed))
                self.update_servo(32, self.curr_angles[32] + align_speed)
                self.update_servo(23, self.curr_angles[23] - align_speed)

        if abs(error_y) < self.micro_zone_y:
            self.update_servo(6, self.curr_angles[6] - move_y * 1.0)
        else:
            self.update_servo(6, self.curr_angles[6] - move_y * 0.4)
            self.update_servo(30, self.curr_angles[30] - move_y * 0.8)
            self.update_servo(31, self.curr_angles[31] + move_y * 0.8)
            
        self.update_eyelids_follow()

    def reset_pose(self):
        """重置到初始姿态"""
        for ch in CONFIG.keys():
            self.update_servo(ch, CONFIG[ch][2])
        for name, ch_id in EYE_LIDS.items():
            pca, l_ch = self.get_target(ch_id)
            if pca:
                pulse = angle_to_pulse_us(OPEN_POS[name], 0, 180, 600, 2400)
                with self.lock: pca.set_servo_pulse_us(l_ch, pulse)

    def sleep(self):
        """丢失目标时的待机：关闭脖子动力，保留眼睑电源用于眨眼"""
        if not self.is_sleeping:
            print(">>> 目标丢失，进入待机模式（保持随机眨眼）")
            # 只关闭脖子和眼球相关的舵机电源
            for ch in CONFIG.keys():
                pca, local_ch = self.get_target(ch)
                if pca: pca.set_channel_full_off(local_ch, True)
            self.is_sleeping = True
            self.has_detected_first_time = False 

    def cleanup(self):
        """彻底关闭所有舵机"""
        for ch in list(CONFIG.keys()) + list(EYE_LIDS.values()):
            pca, local_ch = self.get_target(ch)
            if pca: pca.set_channel_full_off(local_ch, True)

def main():
    morpheus = MorpheusTracker()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 5005))
    sock.setblocking(0)
    
    print("\n" + "="*40)
    print(" Morpheus V3.4 [全时段随机眨眼版]")
    print(" 眨眼间隔: 1.0s - 5.0s 随机")
    print("="*40 + "\n")

    try:
        while True:
            # 1. 优先级最高：眨眼计时逻辑 (全时运行)
            now = time.time()
            if now - morpheus.last_blink_time > morpheus.blink_interval:
                if not morpheus.is_blinking:
                    threading.Thread(target=morpheus.auto_blink, daemon=True).start()
                    morpheus.last_blink_time = now

            # 2. 网络数据监听与处理
            ready = select.select([sock], [], [], 0.02) 
            if ready[0]:
                data, _ = sock.recvfrom(1024)
                msg = data.decode('utf-8').strip()
                try:
                    ex, ey = map(float, msg.split(','))
                    # 只有超出死区才触发追踪，否则保持静止
                    if abs(ex) > morpheus.deadzone or abs(ey) > morpheus.deadzone:
                        morpheus.track(ex, ey, is_real_data=True)
                except ValueError:
                    continue
            else:
                # 3. 无数据时的行为逻辑
                if morpheus.has_detected_first_time:
                    # 目标刚消失：执行卡尔曼滤波惯性预测
                    morpheus.track(0, 0, is_real_data=False)
                else:
                    # 目标彻底丢失：维持 sleep 状态 (sleep 内部已处理不再关眼睑)
                    morpheus.sleep()
                    
    except KeyboardInterrupt:
        print("\n正在安全退出...")
    finally:
        morpheus.cleanup()
        sock.close()

if __name__ == "__main__":
    main()