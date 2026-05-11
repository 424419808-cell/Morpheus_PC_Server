import sys
import time
import random
import argparse
import requests

# 复用驱动和配置模块
from mor_servo_dev import PCA9685, angle_to_pulse_us
from set_start_bound import TABLE_V_CONFIG, get_hardware_target, ServoStateManager

class MorpheusCollector:
    # 删除了 num_expressions 参数，因为现在由电脑端控制何时停止
    def __init__(self, symmetry_prob=0.7, move_speed=50.0, pc_ip="172.16.3.246"):
        self.symmetry_prob = symmetry_prob
        self.move_speed = move_speed
        self.capture_step_degrees = 10.0
        self.capture_pause_ms = 300
        self.pc_server_url = f"http://{pc_ip}:5000/capture"

        self.state_manager = ServoStateManager()
        self.current_angles = {}
        for ch in range(33):
            if ch in TABLE_V_CONFIG:
                self.current_angles[ch] = TABLE_V_CONFIG[ch][2]
            else:
                self.current_angles[ch] = 90.0

        self.strict_symmetric_pairs = [(19, 22)]
        # 移除了眉毛 (2,27) (3,26) 和眼睑 (4,25) (5,24) 对称对
        self.symmetric_pairs = [
            (0, 29), (1, 28), (8, 21), (13, 18), (20, 9), (15, 16)
        ]
        self.center_channels = [7, 14]
        # 眉毛和眼睑舵机固定为初始值，不参与随机运动
        self.fixed_channels = {2, 3, 4, 5, 24, 25, 26, 27}

        self.pcas = {
            0x40: PCA9685(bus_id=1, address=0x40, freq_hz=50),
            0x41: PCA9685(bus_id=1, address=0x41, freq_hz=50),
            0x42: PCA9685(bus_id=1, address=0x42, freq_hz=50)
        }
        self.global_sample_id = 0

    def get_angle_from_step(self, ch, step):
        min_a, max_a, _ = TABLE_V_CONFIG[ch]
        return min_a + (max_a - min_a) * step

    def _send_angles_concurrently(self, target_dict):
        for ch, target in target_dict.items():
            pca, local_ch = get_hardware_target(ch, self.pcas)
            if pca:
                pulse = angle_to_pulse_us(target, 0.0, 180.0, 600.0, 2400.0)
                pca.set_servo_pulse_us(local_ch, pulse)

    def _move_and_capture_interpolated(self, target_commands):
        """返回 True 代表收到停止指令，返回 False 代表继续"""
        max_delta = 0.0
        for ch, target in target_commands.items():
            delta = abs(target - self.current_angles[ch])
            if delta > max_delta:
                max_delta = delta

        if max_delta < 1.0:
            return False

        num_steps = max(1, int(max_delta / self.capture_step_degrees))

        for step in range(1, num_steps + 1):
            fraction = step / num_steps
            step_angles = {}
            for ch, target in target_commands.items():
                start = self.current_angles[ch]
                step_angles[ch] = start + (target - start) * fraction

            self._send_angles_concurrently(step_angles)
            time.sleep(self.capture_pause_ms / 1000.0)

            payload = {
                "sample_id": self.global_sample_id,
                "motor_commands": step_angles.copy()
            }

            try:
                response = requests.post(self.pc_server_url, json=payload, timeout=10)
                if response.status_code == 200:
                    res_data = response.json()
                    # 检查电脑是否发出了停机指令
                    if res_data.get("stop") is True:
                        return True
                else:
                    print(f"\n[警告] PC 端状态码: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"\n[错误] 网络断开: {e}")
                time.sleep(2)

            self.global_sample_id += 1

        for ch, target in target_commands.items():
            self.current_angles[ch] = target

        return False

    def move_and_collect(self):
        print(f"当前目标 PC 服务器: {self.pc_server_url}")
        print("准备开始无限随机探索（眉毛眼睑固定），直到 PC 端发送停止指令...")

        try:
            self._send_angles_concurrently(self.current_angles)
            time.sleep(1)

            route_count = 1
            # 核心修改：变成无限循环
            while True:
                current_commands = {}
                for left_ch, right_ch in self.strict_symmetric_pairs:
                    step_l = random.choice([0, 0.25, 0.5, 0.75, 1.0])
                    current_commands[left_ch] = self.get_angle_from_step(left_ch, step_l)
                    current_commands[right_ch] = self.get_angle_from_step(right_ch, 1.0 - step_l)

                for left_ch, right_ch in self.symmetric_pairs:
                    if random.random() < self.symmetry_prob:
                        step_l = random.choice([0, 0.25, 0.5, 0.75, 1.0])
                        step_r = 1.0 - step_l
                    else:
                        step_l = random.choice([0, 0.25, 0.5, 0.75, 1.0])
                        step_r = random.choice([0, 0.25, 0.5, 0.75, 1.0])
                    current_commands[left_ch] = self.get_angle_from_step(left_ch, step_l)
                    current_commands[right_ch] = self.get_angle_from_step(right_ch, step_r)

                for ch in self.center_channels:
                    step = random.choice([0, 0.25, 0.5, 0.75, 1.0])
                    current_commands[ch] = self.get_angle_from_step(ch, step)

                # 眉毛和眼睑舵机固定为初始值，不参与随机运动
                for ch in self.fixed_channels:
                    current_commands[ch] = TABLE_V_CONFIG[ch][2]

                print(f"\r[随机动作序列 {route_count}] 正在执行并与 PC 同步...", end="", flush=True)

                # 执行动作并检查是否收到停止指令
                should_stop = self._move_and_capture_interpolated(current_commands)

                if should_stop:
                    print("\n\n[指令下达] 收到 PC 端停止信号，有效数据采集目标已达成！")
                    break # 跳出 while 循环，结束程序

                route_count += 1

        except KeyboardInterrupt:
            print("\n[!] 采集被手动强制中断。")
        finally:
            for p in self.pcas.values():
                p.close()
            print("[✓] 舵机板卡资源已释放。")

if __name__ == "__main__":
    # 请确保 pc_ip 为你电脑真实的局域网 IP
    collector = MorpheusCollector(symmetry_prob=0.8, move_speed=50.0, pc_ip="172.16.3.245")
    collector.move_and_collect()
