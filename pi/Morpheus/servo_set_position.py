#!/usr/bin/env python3
"""
舵机位置设置程序 - 32通道增强版
支持两块 PCA9685 (0x40, 0x41) 同时控制
通道 0-15 映射到第一块板，16-31 映射到第二块板
"""

import sys
import time
import argparse
import threading
# 确保 mor_servo_dev.py 在同一目录下
from mor_servo_dev import PCA9685, pulse_us_to_angle, angle_to_pulse_us


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='控制两块 PCA9685 舵机平缓移动 (支持通道 0-31)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 控制全局通道 0 和 16 (分别对应两块板子的 0 号位) 到 90 度
  python3 servo_set_position.py --channel 0 16 --angle 90
  
  # 为多个通道设置不同角度
  python3 servo_set_position.py --channel 0 5 16 20 --angle 30 60 90 120 --speed 20
        '''
    )
    
    # 保留原有的所有参数
    parser.add_argument('--addr1', type=lambda x: int(x, 0), default=0x40, help='第一块板地址 (默认: 0x40)')
    parser.add_argument('--addr2', type=lambda x: int(x, 0), default=0x41, help='第二块板地址 (默认: 0x41)')
    parser.add_argument('--channel', type=int, nargs='+', required=True, help='要控制的全局通道号 (0-31)')
    parser.add_argument('--angle', type=float, nargs='+', required=True, help='目标角度')
    parser.add_argument('--speed', type=float, default=10.0, help='移动速度 (度/秒), 0表示最快速度')
    parser.add_argument('--initial-angle', type=float, default=None, help='初始角度')
    parser.add_argument('--step-ms', type=float, default=20.0, help='更新间隔 (ms)')
    parser.add_argument('--tolerance', type=float, default=1.0, help='角度容差')
    parser.add_argument('--release', action='store_true', help='完成后释放舵机')
    parser.add_argument('--min-us', type=float, default=600, help='最小脉冲 (us)')
    parser.add_argument('--max-us', type=float, default=2400, help='最大脉冲 (us)')
    parser.add_argument('--min-angle', type=float, default=0, help='最小角度')
    parser.add_argument('--max-angle', type=float, default=180, help='最大角度')
    parser.add_argument('--show-progress', action='store_true', help='显示移动进度')
    
    return parser.parse_args()


def get_current_angle(pca, channel, min_us, max_us, min_angle, max_angle, initial_angle):
    """获取舵机当前角度 (继承原逻辑)"""
    try:
        pulse_us = pca.get_servo_pulse_us(channel)
        if pulse_us <= 1.0:
            return initial_angle if initial_angle is not None else (min_angle + max_angle) / 2.0
        else:
            return pulse_us_to_angle(pulse_us, min_angle, max_angle, min_us, max_us)
    except:
        return initial_angle if initial_angle is not None else (min_angle + max_angle) / 2.0


def set_servo_angle_smooth(pca, channel, target_angle, current_angle, speed, 
                          min_us, max_us, min_angle, max_angle, 
                          step_ms=20.0, tolerance=1.0, show_progress=False, lock=None):
    """平滑移动逻辑 (保留原所有进度显示逻辑)"""
    target_angle = max(min_angle, min(max_angle, target_angle))
    
    if speed <= 0:
        pulse = angle_to_pulse_us(target_angle, min_angle, max_angle, min_us, max_us)
        if lock:
            with lock: pca.set_servo_pulse_us(channel, pulse)
        else:
            pca.set_servo_pulse_us(channel, pulse)
        if show_progress: print(f"通道 {channel}: 快速跳转到 {target_angle:.1f}°")
        return
    
    step_dt = max(0.01, step_ms / 1000.0)
    max_step = speed * step_dt
    angle = current_angle
    direction = 1.0 if target_angle >= angle else -1.0
    
    start_time = time.time()
    while True:
        delta = target_angle - angle
        if abs(delta) <= max(tolerance, max_step):
            angle = target_angle
        else:
            angle += direction * max_step
        
        pulse = angle_to_pulse_us(angle, min_angle, max_angle, min_us, max_us)
        if lock:
            with lock: pca.set_servo_pulse_us(channel, pulse)
        else:
            pca.set_servo_pulse_us(channel, pulse)
        
        if show_progress:
            # 这里的进度显示增加了物理板卡的区分提示
            progress = abs(angle - current_angle) / max(0.1, abs(target_angle - current_angle)) * 100
            print(f"\r 通道 {channel:2d}: {progress:5.1f}% | 当前: {angle:6.1f}°", end='', flush=True)
        
        if angle == target_angle: break
        time.sleep(step_dt)
    
    if show_progress:
        elapsed = time.time() - start_time
        print(f"\n ✓ 通道 {channel:2d} 完成 | 用时: {elapsed:.2f}s")


def main():
    args = parse_args()
    
    # 验证参数范围扩大到 0-31
    for ch in args.channel:
        if not 0 <= ch <= 31:
            print(f"错误: 全局通道号 {ch} 超出范围 (0-31)")
            return 1

    # 处理角度匹配逻辑 (保留原逻辑)
    if len(args.angle) == 1:
        angles = [args.angle[0]] * len(args.channel)
    elif len(args.angle) == len(args.channel):
        angles = args.angle
    else:
        print(f"错误: 角度数量必须为1或与通道数量 ({len(args.channel)}) 相同")
        return 1

    try:
        # 初始化两块板子
        print(f"初始化 PCA9685 板卡: 0x{args.addr1:02X} (0-15) 和 0x{args.addr2:02X} (16-31)...")
        pca1 = PCA9685(bus_id=1, address=args.addr1, freq_hz=50)
        pca2 = PCA9685(bus_id=1, address=args.addr2, freq_hz=50)
        
        # 共享同一个锁，保护整个 I2C 总线
        pca_lock = threading.Lock()
        
        def move_servo_worker(global_channel, target_angle):
            """内部线程函数：处理通道映射"""
            try:
                # 核心路由逻辑：分配物理板卡和本地通道
                if global_channel < 16:
                    pca_target = pca1
                    local_channel = global_channel
                else:
                    pca_target = pca2
                    local_channel = global_channel - 16
                
                # 获取当前角度
                with pca_lock:
                    current_angle = get_current_angle(
                        pca_target, local_channel, 
                        args.min_us, args.max_us,
                        args.min_angle, args.max_angle,
                        args.initial_angle
                    )
                
                # 平滑移动
                set_servo_angle_smooth(
                    pca_target, local_channel, target_angle, current_angle,
                    args.speed, args.min_us, args.max_us,
                    args.min_angle, args.max_angle,
                    args.step_ms, args.tolerance,
                    args.show_progress, pca_lock
                )
                
                if args.release:
                    with pca_lock: pca_target.set_channel_full_off(local_channel, True)
            except Exception as e:
                print(f"\n全局通道 {global_channel} 错误: {e}")

        # 创建并启动线程
        threads = []
        for channel, target_angle in zip(args.channel, angles):
            thread = threading.Thread(target=move_servo_worker, args=(channel, target_angle))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        print("-" * 50)
        print(f"✓ 成功完成 {len(args.channel)} 个通道的设置")
        
        pca1.close()
        pca2.close()
        
    except IOError as e:
        print(f"\nI2C 访问失败: {e}。请检查地址 0x40 和 0x41 是否都已在线。")
        return 1
    except Exception as e:
        print(f"\n运行时错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())