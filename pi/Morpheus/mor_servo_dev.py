#!/usr/bin/env python3
import argparse
import sys
import time
from typing import Tuple

# Prefer smbus2; fallback to smbus
try:
    from smbus2 import SMBus  # type: ignore
except Exception:
    try:
        from smbus import SMBus  # type: ignore
    except Exception:
        SMBus = None  # type: ignore


class PCA9685:
    MODE1 = 0x00
    MODE2 = 0x01
    PRESCALE = 0xFE

    LED0_ON_L = 0x06
    LED0_ON_H = 0x07
    LED0_OFF_L = 0x08
    LED0_OFF_H = 0x09

    RESTART = 0x80
    SLEEP = 0x10
    ALLCALL = 0x01
    OUTDRV = 0x04
    AI = 0x20

    FULL_ON_BIT = 0x10
    FULL_OFF_BIT = 0x10

    # 使用说明：
    # pca = PCA9685(bus_id=1, address=0x40, freq_hz=50)
    # 初始化后，芯片即进入工作状态。
    def __init__(self, bus_id: int = 1, address: int = 0x40, freq_hz: float = 50.0, osc_hz: float = 25_000_000.0):
        if SMBus is None:
            sys.stderr.write(
                "ERROR: Missing I2C library. Install one of:\n"
                " - sudo apt-get install -y python3-smbus\n"
                " - pip install smbus2\n"
            )
            sys.exit(1)
        self.address = address
        self.bus = SMBus(bus_id)
        self.osc_hz = float(osc_hz)
        self.period_us = 0.0
        # MODE2: Totem pole; MODE1: ALLCALL
        self._write_byte(self.MODE2, self.OUTDRV)
        self._write_byte(self.MODE1, self.ALLCALL)
        time.sleep(0.005)
        # Clear sleep and enable auto-increment
        mode1 = self._read_byte(self.MODE1)
        mode1 = (mode1 & ~self.SLEEP) | self.AI
        self._write_byte(self.MODE1, mode1)
        time.sleep(0.005)
        self.set_pwm_freq(freq_hz)

    def close(self):
        try:
            self.bus.close()
        except Exception:
            pass

    def _read_byte(self, reg: int) -> int:
        return self.bus.read_byte_data(self.address, reg)

    def _write_byte(self, reg: int, value: int) -> None:
        self.bus.write_byte_data(self.address, reg, value & 0xFF)

    def _write_word(self, reg: int, low: int, high: int) -> None:
        self._write_byte(reg, low)
        self._write_byte(reg + 1, high)
        
    # 使用说明：
    # pca.set_pwm_freq(60)
    # 修改 PWM 频率。通常舵机使用 50Hz，高性能数字舵机可用 200Hz+。
    def set_pwm_freq(self, freq_hz: float) -> None:
        freq_hz = float(freq_hz)
        if freq_hz <= 0:
            raise ValueError("freq_hz must be positive")
        prescale_val = self.osc_hz / (4096.0 * freq_hz) - 1.0
        prescale = int(round(prescale_val))
        prescale = max(3, min(255, prescale))
        oldmode = self._read_byte(self.MODE1)
        newmode = (oldmode & ~self.RESTART) | self.SLEEP
        self._write_byte(self.MODE1, newmode)
        self._write_byte(self.PRESCALE, prescale)
        self._write_byte(self.MODE1, oldmode)
        time.sleep(0.005)
        self._write_byte(self.MODE1, oldmode | self.RESTART | self.AI)
        actual_freq = self.osc_hz / (4096.0 * (prescale + 1))
        self.period_us = 1_000_000.0 / actual_freq

    def set_pwm(self, channel: int, on_count: int, off_count: int) -> None:
        if not (0 <= channel <= 15):
            raise ValueError("channel must be 0..15")
        base = self.LED0_ON_L + 4 * channel
        self._write_word(base + 0, on_count & 0xFF, (on_count >> 8) & 0x0F)
        self._write_word(base + 2, off_count & 0xFF, (off_count >> 8) & 0x0F)

    def read_pwm_counts_and_flags(self, channel: int) -> Tuple[int, int, bool, bool]:
        if not (0 <= channel <= 15):
            raise ValueError("channel must be 0..15")
        base = self.LED0_ON_L + 4 * channel
        on_l = self.bus.read_byte_data(self.address, base + 0)
        on_h = self.bus.read_byte_data(self.address, base + 1)
        off_l = self.bus.read_byte_data(self.address, base + 2)
        off_h = self.bus.read_byte_data(self.address, base + 3)
        full_on = bool(on_h & self.FULL_ON_BIT)
        full_off = bool(off_h & self.FULL_OFF_BIT)
        on = ((on_h & 0x0F) << 8) | on_l
        off = ((off_h & 0x0F) << 8) | off_l
        return on, off, full_on, full_off

    def set_channel_full_off(self, channel: int, enable: bool = True) -> None:
        if not (0 <= channel <= 15):
            raise ValueError("channel must be 0..15")
        base = self.LED0_ON_L + 4 * channel
        on_h = self.bus.read_byte_data(self.address, base + 1)
        off_h = self.bus.read_byte_data(self.address, base + 3)
        if enable:
            self._write_byte(base + 1, on_h & 0x0F)
            self._write_byte(base + 3, (off_h & 0x0F) | self.FULL_OFF_BIT)
        else:
            self._write_byte(base + 3, (off_h & 0x0F))

    def set_servo_pulse_us(self, channel: int, pulse_us: float) -> None:
        if self.period_us <= 0:
            raise RuntimeError("PWM frequency not initialized")
        counts = int(round(4096.0 * (pulse_us / self.period_us)))
        counts = max(0, min(4095, counts))
        self.set_pwm(channel, 0, counts)

    def get_servo_pulse_us(self, channel: int) -> float:
        if self.period_us <= 0:
            raise RuntimeError("PWM frequency not initialized")
        on, off, _full_on, full_off = self.read_pwm_counts_and_flags(channel)
        if full_off:
            return 0.0
        width = (off - on) & 0x0FFF
        return (width / 4096.0) * self.period_us


def angle_to_pulse_us(angle_deg: float, min_angle: float, max_angle: float, min_us: float, max_us: float) -> float:
    if max_angle == min_angle:
        raise ValueError("min_angle and max_angle must differ")
    angle_deg = max(min_angle, min(max_angle, angle_deg))
    ratio = (angle_deg - min_angle) / (max_angle - min_angle)
    return min_us + ratio * (max_us - min_us)


def pulse_us_to_angle(pulse_us: float, min_angle: float, max_angle: float, min_us: float, max_us: float) -> float:
    pulse_us = max(min_us, min(max_us, pulse_us))
    ratio = (pulse_us - min_us) / (max_us - min_us)
    return min_angle + ratio * (max_angle - min_angle)


def angle_to_pulse_us_with_mid(angle_deg: float, min_angle: float, mid_angle: float, max_angle: float, min_us: float, mid_us: float, max_us: float) -> float:
    if max_angle < min_angle:
        min_angle, max_angle = max_angle, min_angle
    angle_deg = max(min_angle, min(max_angle, angle_deg))
    if not (min_angle < mid_angle < max_angle) or not (min_us < mid_us < max_us):
        return angle_to_pulse_us(angle_deg, min_angle, max_angle, min_us, max_us)
    if angle_deg <= mid_angle:
        ratio = (angle_deg - min_angle) / (mid_angle - min_angle)
        return min_us + ratio * (mid_us - min_us)
    else:
        ratio = (angle_deg - mid_angle) / (max_angle - mid_angle)
        return mid_us + ratio * (max_us - mid_us)


def pulse_us_to_angle_with_mid(pulse_us: float, min_angle: float, mid_angle: float, max_angle: float, min_us: float, mid_us: float, max_us: float) -> float:
    if max_us < min_us:
        min_us, max_us = max_us, min_us
    pulse_us = max(min_us, min(max_us, pulse_us))
    if not (min_angle < mid_angle < max_angle) or not (min_us < mid_us < max_us):
        return pulse_us_to_angle(pulse_us, min_angle, max_angle, min_us, max_us)
    if pulse_us <= mid_us:
        ratio = (pulse_us - min_us) / (mid_us - min_us)
        return min_angle + ratio * (mid_angle - min_angle)
    else:
        ratio = (pulse_us - mid_us) / (max_us - mid_us)
        return mid_angle + ratio * (max_angle - mid_angle)


def cmd_get(args) -> int:
    pca = PCA9685(bus_id=args.bus, address=args.address, freq_hz=args.freq, osc_hz=args.osc)
    try:
        pulse = pca.get_servo_pulse_us(args.channel)
        # Remove bias for reporting logical angle
        pulse_for_angle = pulse - (args.bias_us or 0.0)
        mid_us = (args.min_us + args.max_us) / 2.0 if args.mid_us is None else args.mid_us
        mid_angle = (args.min_angle + args.max_angle) / 2.0 if args.mid_angle is None else args.mid_angle
        angle = pulse_us_to_angle_with_mid(pulse_for_angle, args.min_angle, mid_angle, args.max_angle, args.min_us, mid_us, args.max_us) if pulse > 1.0 else mid_angle
        print(f"channel={args.channel} pulse_us={pulse:.1f} angle_deg={angle:.1f}")
        return 0
    finally:
        pca.close()


def cmd_set(args) -> int:
    pca = PCA9685(bus_id=args.bus, address=args.address, freq_hz=args.freq, osc_hz=args.osc)
    try:
        # Determine current angle from current pulse; if disabled or near 0, use initial angle
        try:
            current_pulse = pca.get_servo_pulse_us(args.channel)
        except Exception:
            current_pulse = 0.0
        initial_angle_default = (args.min_angle + args.max_angle) / 2.0
        initial_angle = args.initial_angle if args.initial_angle is not None else initial_angle_default
        mid_us = (args.min_us + args.max_us) / 2.0 if args.mid_us is None else args.mid_us
        mid_angle = (args.min_angle + args.max_angle) / 2.0 if args.mid_angle is None else args.mid_angle
        if current_pulse <= 1.0:
            current_angle = max(args.min_angle, min(args.max_angle, initial_angle))
        else:
            current_angle = pulse_us_to_angle_with_mid(current_pulse, args.min_angle, mid_angle, args.max_angle, args.min_us, mid_us, args.max_us)

        target_angle = max(args.min_angle, min(args.max_angle, args.angle))

        def write_angle(a: float) -> None:
            pulse = angle_to_pulse_us_with_mid(a, args.min_angle, mid_angle, args.max_angle, args.min_us, mid_us, args.max_us)
            # Apply constant bias in microseconds
            pulse += (args.bias_us or 0.0)
            # Avoid mid-area window that may cause hunting by nudging by one step
            if (args.avoid_center_window_us or 0.0) > 0.0:
                avoid_c = mid_us
                win = args.avoid_center_window_us
                if abs(pulse - avoid_c) <= win:
                    # Nudge away by at least one count step
                    step_us = pca.period_us / 4096.0
                    if pulse >= avoid_c:
                        pulse = avoid_c + win + step_us
                    else:
                        pulse = avoid_c - win - step_us
            # Clamp to min/max
            pulse = max(args.min_us, min(args.max_us, pulse))
            pca.set_servo_pulse_us(args.channel, pulse)

        if args.speed <= 0:
            write_angle(target_angle)
            if args.release:
                pca.set_channel_full_off(args.channel, True)
            msg = f"set angle_deg={target_angle:.1f} (no speed limit)" + (" and released" if args.release else "")
            print(msg)
            return 0

        step_dt = max(0.01, args.step_ms / 1000.0)
        max_step = args.speed * step_dt
        angle = current_angle
        direction = 1.0 if target_angle >= angle else -1.0
        while True:
            delta = target_angle - angle
            if abs(delta) <= max(args.tolerance_deg, max_step):
                angle = target_angle
            else:
                angle += direction * max_step
            write_angle(angle)
            if angle == target_angle:
                break
            time.sleep(step_dt)
        if args.release:
            pca.set_channel_full_off(args.channel, True)
        msg = (
            f"moved channel={args.channel} to angle_deg={target_angle:.1f} "
            f"with speed={args.speed} deg/s (start={current_angle:.1f})"
        ) + (" and released" if args.release else "")
        print(msg)
        return 0
    finally:
        pca.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PCA9685 servo tester: get current command angle and set target angle with limited speed.")
    parser.add_argument("--bus", type=int, default=1, help="I2C bus number (default: 1)")
    parser.add_argument("--address", type=lambda x: int(x, 0), default=0x40, help="I2C address (default: 0x40)")
    parser.add_argument("--freq", type=float, default=50.0, help="PWM frequency in Hz (default: 50.0)")
    parser.add_argument("--osc", type=float, default=25_000_000.0, help="Oscillator frequency in Hz (default: 25e6)")
    parser.add_argument("--channel", type=int, default=0, help="Servo channel 0-15 (default: 0)")
    parser.add_argument("--min-us", dest="min_us", type=float, default=600.0, help="Min pulse width in us for min_angle (default: 600)")
    parser.add_argument("--max-us", dest="max_us", type=float, default=2400.0, help="Max pulse width in us for max_angle (default: 2400)")
    parser.add_argument("--min-angle", dest="min_angle", type=float, default=0.0, help="Minimum angle in degrees (default: 0)")
    parser.add_argument("--max-angle", dest="max_angle", type=float, default=180.0, help="Maximum angle in degrees (default: 180)")
    parser.add_argument("--mid-us", dest="mid_us", type=float, default=None, help="Pulse width (us) for mid-angle calibration. Default: midpoint of [min_us, max_us]")
    parser.add_argument("--mid-angle", dest="mid_angle", type=float, default=None, help="Angle (deg) corresponding to mid-us. Default: midpoint of [min_angle, max_angle]")

    sub = parser.add_subparsers(dest="command", required=True)

    p_get = sub.add_parser("get", help="Read current command angle and pulse width")
    p_get.set_defaults(func=cmd_get)

    p_set = sub.add_parser("set", help="Move servo to target angle with limited speed")
    p_set.add_argument("--angle", type=float, required=True, help="Target angle in degrees")
    p_set.add_argument("--speed", type=float, default=60.0, help="Speed limit in deg/s (<=0 means jump). Default: 60")
    p_set.add_argument("--step-ms", dest="step_ms", type=float, default=20.0, help="Update interval in ms. Default: 20")
    p_set.add_argument("--initial-angle", dest="initial_angle", type=float, default=None, help="Initial angle when channel is disabled/uninitialized. Default: midpoint")
    p_set.add_argument("--tolerance-deg", dest="tolerance_deg", type=float, default=1.0, help="Stop when within this angular tolerance of target (deg). Default: 1.0")
    p_set.add_argument("--release", dest="release", action="store_true", help="After reaching target, disable channel (FULL_OFF) to remove holding torque/noise")
    p_set.add_argument("--bias-us", dest="bias_us", type=float, default=0.0, help="Constant pulse-width bias in microseconds to shift mapping (default: 0)")
    p_set.add_argument("--avoid-center-window-us", dest="avoid_center_window_us", type=float, default=0.0, help="Avoid pulse widths within +/- this window around mid-us by nudging away (default: 0=disabled)")
    p_set.set_defaults(func=cmd_set)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
