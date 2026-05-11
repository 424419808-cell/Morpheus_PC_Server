import socket
import time
import sys
import select
import numpy as np

try:
    from mor_servo_dev import PCA9685, angle_to_pulse_us_with_mid
except ImportError:
    print("错误: 未找到 mor_servo_dev.py 驱动文件。")
    sys.exit(1)

# --- 硬件与范围配置 ---
CONFIG = {
    6:  (30.0,  80.0,  60.0),   
    23: (55.0,  95.0,  75.0),   
    30: (70.0,  130.0, 105.0),  
    31: (55.0,  115.0, 80.0),   
    32: (0, 180.0, 90)   
}

class SimpleKalman:
    """简单的二维卡尔曼滤波器"""
    def __init__(self):
        # 状态向量 [x, y, vx, vy]
        self.state = np.zeros(4)
        # 协方差矩阵
        self.P = np.eye(4) * 10
        # 状态转移矩阵
        self.F = np.eye(4)
        # 观测矩阵 (只观测位置 x, y)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        # 过程噪声和测量噪声
        self.Q = np.eye(4) * 0.1
        self.R = np.eye(2) * 5

    def predict(self, dt):
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.state = dot(self.F, self.state)
        self.P = dot(dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        y = z - dot(self.H, self.state)  # 观测残差
        S = dot(dot(self.H, self.P), self.H.T) + self.R
        K = dot(dot(self.P, self.H.T), np.linalg.inv(S))  # 卡尔曼增益
        self.state = self.state + dot(K, y)
        self.P = dot((np.eye(4) - dot(K, self.H)), self.P)

def dot(A, B): return np.dot(A, B)

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

        self.curr_angles = {ch: cfg[2] for ch, cfg in CONFIG.items()}
        
        # --- 核心参数 ---
        self.kp_x, self.kp_y = 0.021, 0.022
        self.smooth = 0.2
        self.max_step = 1.5
        self.deadzone = 8
        self.micro_zone_x, self.micro_zone_y = 115, 50
        
        # --- 卡尔曼与状态控制 ---
        self.kf = SimpleKalman()
        self.last_time = time.time()
        self.has_detected_first_time = False # 是否捕获过目标
        self.prediction_count = 0            # 丢失目标后的预测次数
        self.MAX_PREDICTIONS = 100         # 最多预测100帧，防止预测飞了
        
        self.is_sleeping = False
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
            pca.set_servo_pulse_us(local_ch, pulse)
            self.curr_angles[ch] = new_angle

    def track(self, error_x, error_y, is_real_data=True):
        """执行追踪，支持预测模式"""
        self.is_sleeping = False
        
        # 如果是真实数据，更新卡尔曼状态
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
            # 获取预测的位置
            error_x, error_y = self.kf.state[0], self.kf.state[1]

        # 超过最大预测次数还没找回目标，则真正休眠
        if self.prediction_count > self.MAX_PREDICTIONS:
            self.sleep()
            return

        # --- 追踪逻辑 (保持你之前的分段逻辑) ---
        move_x, move_y = error_x * self.kp_x, error_y * self.kp_y

        if abs(error_x) < self.micro_zone_x:
            self.update_servo(23, self.curr_angles[23] - move_x * 0.9)
        else:
            self.update_servo(23, self.curr_angles[23] - move_x * 0.3)
            self.update_servo(32, self.curr_angles[32] - move_x * 0.7)

        if abs(error_y) < self.micro_zone_y:
            self.update_servo(6, self.curr_angles[6] - move_y * 1.0)
        else:
            self.update_servo(6, self.curr_angles[6] - move_y * 0.4)
            self.update_servo(30, self.curr_angles[30] - move_y * 0.8)
            self.update_servo(31, self.curr_angles[31] + move_y * 0.8)

    def reset_pose(self):
        print(">>> 启动：归位...")
        for ch in CONFIG.keys():
            self.update_servo(ch, CONFIG[ch][2])

    def sleep(self):
        if not self.is_sleeping:
            print(">>> 目标丢失，休眠释放...")
            for ch in CONFIG.keys():
                pca, local_ch = self.get_target(ch)
                pca.set_channel_full_off(local_ch, True)
            self.is_sleeping = True
            self.has_detected_first_time = False # 重置状态，等待下一次“第一次”捕获

    def cleanup(self):
        print(">>> 清理退出...")
        for ch in CONFIG.keys():
            pca, local_ch = self.get_target(ch)
            if pca: pca.set_channel_full_off(local_ch, True)

def main():
    morpheus = MorpheusTracker()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 5005))
    sock.setblocking(0)
    
    print("\n" + "="*40)
    print(" Morpheus 卡尔曼智能追踪系统")
    print(" [特性: 目标丢失预测 + 初次捕获前休眠]")
    print("="*40 + "\n")

    try:
        while True:
            ready = select.select([sock], [], [], 0.02) # 提高采样频率
            
            if ready[0]:
                data, _ = sock.recvfrom(1024)
                msg = data.decode('utf-8').strip()
                ex, ey = map(float, msg.split(','))
                
                if abs(ex) > morpheus.deadzone or abs(ey) > morpheus.deadzone:
                    morpheus.track(ex, ey, is_real_data=True)
            else:
                # 关键修改：只有在“曾经捕获过目标”的情况下，才进行预测追踪
                if morpheus.has_detected_first_time:
                    morpheus.track(0, 0, is_real_data=False)
                else:
                    morpheus.sleep()
                    
    except KeyboardInterrupt:
        print("\n[!] 退出...")
    finally:
        morpheus.cleanup()
        sock.close()

if __name__ == "__main__":
    main()