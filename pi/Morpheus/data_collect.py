import sys
import time
import json
import random
import argparse

# 复用驱动和配置模块
from mor_servo_dev import PCA9685, angle_to_pulse_us
from set_start_bound import TABLE_V_CONFIG, get_hardware_target, ServoStateManager

class MorpheusCollector:
    def __init__(self, num_expressions=10, symmetry_prob=0.7, move_speed=50.0):
        self.num_expressions = num_expressions # 计划生成多少个随机目标表情
        self.symmetry_prob = symmetry_prob 
        
        # 核心控制参数
        self.move_speed = move_speed  # 整体过渡速度 (度/秒)
        self.capture_step_degrees = 10.0 # 每移动多少度停下来采集一次 (决定了过渡切片的密度，越小切片越多)
        self.capture_pause_ms = 300   # 停下来等待物理稳定并拍照的时间 (毫秒)
        
        self.state_manager = ServoStateManager()
        # 【优化】删除了无用的 self.i2c_bus_lock
        
        # 记录当前每个通道的角度
        self.current_angles = {}
        for ch in range(33):
            if ch in TABLE_V_CONFIG:
                self.current_angles[ch] = TABLE_V_CONFIG[ch][2]
            else:
                self.current_angles[ch] = 90.0

        # ==========================================================
        # 【修改点】：去掉了 (12, 17) 下巴前后运动，只保留嘴巴张合 (19, 22)
        self.strict_symmetric_pairs = [(19, 22)]
        # ==========================================================
        
        self.symmetric_pairs = [
            (0, 29), (1, 28), (2, 27), (3, 26), (4, 25), 
            (5, 24), (8, 21), (13, 18), (20, 9), (15, 16)
        ]
        self.center_channels = [7, 14]
        
        self.pcas = {
            0x40: PCA9685(bus_id=1, address=0x40, freq_hz=50),
            0x41: PCA9685(bus_id=1, address=0x41, freq_hz=50),
            0x42: PCA9685(bus_id=1, address=0x42, freq_hz=50)
        }
        
        self.dataset = []
        self.global_sample_id = 0 # 因为现在一个表情会产生多张切片数据，使用全局ID

    def get_angle_from_step(self, ch, step):
        min_a, max_a, _ = TABLE_V_CONFIG[ch]
        return min_a + (max_a - min_a) * step

    def _send_angles_concurrently(self, target_dict):
        """内部方法：瞬间将各个舵机驱动到指定的插值点 (已优化为最高效的单线程)"""
        # 【优化】去掉了多线程和锁，直接使用最快的单线程串行写入 I2C
        for ch, target in target_dict.items():
            pca, local_ch = get_hardware_target(ch, self.pcas)
            if pca:
                pulse = angle_to_pulse_us(target, 0.0, 180.0, 600.0, 2400.0)
                pca.set_servo_pulse_us(local_ch, pulse)

    def _move_and_capture_interpolated(self, target_commands):
        """
        核心切片采集逻辑：计算目标与当前的差距，按照速度和步长进行插值采集
        """
        # 1. 找出这次表情切换中，移动幅度最大的那个通道的角度差
        max_delta = 0.0
        for ch, target in target_commands.items():
            delta = abs(target - self.current_angles[ch])
            if delta > max_delta:
                max_delta = delta

        # 如果几乎不需要移动，跳过
        if max_delta < 1.0:
            return

        # 2. 计算需要切分成多少个中间帧
        num_steps = max(1, int(max_delta / self.capture_step_degrees))
        
        # 3. 按照插值步数，进行 步进->暂停->拍照 的循环
        for step in range(1, num_steps + 1):
            fraction = step / num_steps
            step_angles = {}
            
            # 计算这一步每个通道应该在的插值角度
            for ch, target in target_commands.items():
                start = self.current_angles[ch]
                step_angles[ch] = start + (target - start) * fraction
                
            # 发送插值角度，让舵机动到中间态
            self._send_angles_concurrently(step_angles)
            
            # 等待舵机到位，并在物理上停稳 (消抖)，准备拍照
            time.sleep(self.capture_pause_ms / 1000.0)
            
            # =================== 接入视觉采集 ===================
            # status, frame = cap.read()
            # landmarks = get_blendshapes(frame)
            landmarks = {"status": "success", "info": f"transition_step_{step}/{num_steps}"} 
            # ==================================================

            # 记录当前这个中间态的精确角度和表情
            sample = {
                "sample_id": self.global_sample_id,
                "motor_commands": step_angles.copy(),
                "landmarks": landmarks
            }
            self.dataset.append(sample)
            self.global_sample_id += 1
            
        # 4. 这个表情过渡走完了，更新当前角度字典
        for ch, target in target_commands.items():
            self.current_angles[ch] = target

    def move_and_collect(self):
        print(f"开始生成 {self.num_expressions} 个随机表情过渡路线...")
        print(f"每移动 {self.capture_step_degrees} 度停顿 {self.capture_pause_ms}ms 采集一帧中间态数据。")
        
        try:
            # 初始归位
            print("正在使所有舵机归位至初始状态...")
            self._send_angles_concurrently(self.current_angles)
            time.sleep(1)
            
            for i in range(self.num_expressions):
                current_commands = {}
                
                # --- 生成随机目标表情 ---
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
                
                # --- 核心：将终点指令扔给切片插值函数执行并采集 ---
                print(f"\r[路线 {i+1}/{self.num_expressions}] 正在执行切片过渡...", end="", flush=True)
                self._move_and_capture_interpolated(current_commands)

                # 每走完一个表情路线，落盘保存一次，防止意外崩溃
                with open("motor_babbling_data.json", "w") as f:
                    json.dump(self.dataset, f, indent=4)

            print(f"\n[✓] 采集任务完成！共计获取 {len(self.dataset)} 帧微表情与角度的映射数据。")

        except KeyboardInterrupt:
            print("\n[!] 采集被手动中断，正在保存已采集数据...")
            with open("motor_babbling_data.json", "w") as f:
                json.dump(self.dataset, f, indent=4)
        finally:
            for p in self.pcas.values(): 
                p.close()

if __name__ == "__main__":
    # speed 50 是宏观速度概念，这里被转化为 capture_step_degrees 来控制步长
    collector = MorpheusCollector(num_expressions=20, symmetry_prob=0.7, move_speed=50.0)
    collector.move_and_collect()