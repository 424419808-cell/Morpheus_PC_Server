#!/usr/bin/env python3
"""
PC 端：文字 → 中文拼音/英文G2P → edge-tts 语音 → 韵母/音素→BS唇形映射 → MLP → UDP
完整唇形同步：支持中英双语自动识别与混合处理。
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import asyncio
import io
import json
import re
import socket
import sys
import threading
import time
import torch
import torch.nn as nn
import numpy as np
import sounddevice as sd
from g2p_en import G2p  # 新增：英文音素解析库

# ================= 配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RPI_IP = "172.16.0.166"
RPI_PORT = 8888
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "lower_face_bs2angle.pth")
FORWARD_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "angle2bs_full.pth")

# 全局缩放
JAW_GAIN = 1.0         # 张嘴幅度 (按你原代码保持为0，可按需调大)
LIP_GAIN = 1.8       # 唇形幅度

# 插值参数 (针对中英双语优化)
# 将 SMOOTH 从 0.22 调高至 0.35，增强对英文短促爆破音(B,P)和摩擦音(F,V)的机械响应速度
SMOOTH = 0.35         
FRAME_MS = 15         # 推理帧间隔

# TTS 配置
TTS_VOICE = "zh-CN-XiaoxiaoNeural"  # Xiaoxiao 自带优秀的中英混读能力
TTS_SPEED = "-80%"

# 初始化英文 G2P
g2p = G2p()

# ================= 拼音声母/韵母分割 =================
_INITIALS_PAT = re.compile(r'^(zh|ch|sh|[bpmfdtnlgkhjqxrzcsyw])?(.+)$')

def split_pinyin(py_str: str):
    """拆分拼音为 (声母, 韵母)，如 'xiao' → ('x', 'iao')"""
    m = _INITIALS_PAT.match(py_str)
    if m:
        return m.group(1) or '', m.group(2)
    return '', py_str

# ================= BS 字典构造工具 =================
def _mk_bs(jaw=0.0, funnel=0.0, pucker=0.0, stretch=0.0, smile=0.0,
           close=0.0, lower_down=0.0, press=0.0, roll_upper=0.0, roll_lower=0.0,
           dimple=0.0, left=0.0, right=0.0,
           jaw_fwd=0.0, jaw_left=0.0, jaw_right=0.0,
           frown=0.0, shrug_lower=0.0, shrug_upper=0.0, upper_up=0.0):
    """构造字母索引的 BS 值字典（只包含非零键）"""
    d = {}
    if jaw:        d['jawOpen'] = jaw
    if jaw_fwd:    d['jawForward'] = jaw_fwd
    if jaw_left:   d['jawLeft'] = jaw_left
    if jaw_right:  d['jawRight'] = jaw_right
    if funnel:     d['mouthFunnel'] = funnel
    if pucker:     d['mouthPucker'] = pucker
    if stretch:    d['mouthStretchLeft'] = stretch; d['mouthStretchRight'] = stretch
    if smile:      d['mouthSmileLeft'] = smile; d['mouthSmileRight'] = smile
    if close:      d['mouthClose'] = close
    if lower_down: d['mouthLowerDownLeft'] = lower_down; d['mouthLowerDownRight'] = lower_down
    if press:      d['mouthPressLeft'] = press; d['mouthPressRight'] = press
    if roll_upper: d['mouthRollUpper'] = roll_upper
    if roll_lower: d['mouthRollLower'] = roll_lower
    if dimple:     d['mouthDimpleLeft'] = dimple; d['mouthDimpleRight'] = dimple
    if left:       d['mouthLeft'] = left
    if right:      d['mouthRight'] = right
    if frown:      d['mouthFrownLeft'] = frown; d['mouthFrownRight'] = frown
    if shrug_lower: d['mouthShrugLower'] = shrug_lower
    if shrug_upper: d['mouthShrugUpper'] = shrug_upper
    if upper_up:   d['mouthUpperUpLeft'] = upper_up; d['mouthUpperUpRight'] = upper_up
    return d

# ================= 中文映射表 =================
VOWEL_BS = {
    'a':  _mk_bs(jaw=0.55, stretch=0.10, lower_down=0.12),
    'o':  _mk_bs(jaw=0.35, funnel=0.45, pucker=0.35),
    'e':  _mk_bs(jaw=0.30, stretch=0.20, smile=0.08),
    'i':  _mk_bs(jaw=0.10, stretch=0.50, smile=0.20),
    'u':  _mk_bs(jaw=0.08, funnel=0.60, pucker=0.50),
    'v':  _mk_bs(jaw=0.08, funnel=0.45, pucker=0.55, stretch=0.05),  # ü
    'ü':  _mk_bs(jaw=0.08, funnel=0.45, pucker=0.55, stretch=0.05),
}

DIPHTHONG_PHASES = {
    'ai':  [(0.35, 'a'), (0.65, 'i')],
    'ei':  [(0.30, 'e'), (0.70, 'i')],
    'ao':  [(0.35, 'a'), (0.65, 'u')],
    'ou':  [(0.30, 'o'), (0.70, 'u')],
    'ia':  [(0.30, 'i'), (0.70, 'a')],
    'ie':  [(0.30, 'i'), (0.70, 'e')],
    'ua':  [(0.30, 'u'), (0.70, 'a')],
    'uo':  [(0.30, 'u'), (0.70, 'o')],
    've':  [(0.30, 'v'), (0.70, 'e')],  # üe
    'üe':  [(0.30, 'v'), (0.70, 'e')],
    'ui':  [(0.30, 'u'), (0.70, 'i')],
    'iu':  [(0.30, 'i'), (0.70, 'u')],
    'er':  [(1.0,  'e')],
}

NASAL_FINALS = {
    'an':  ('a', 'n'),  'en':  ('e', 'n'),  'in':  ('i', 'n'),
    'un':  ('u', 'n'),  'vn':  ('v', 'n'),  'ün':  ('v', 'n'),
    'ian': ('a', 'n'),  'uan': ('a', 'n'),  'van': ('a', 'n'),
    'ang': ('a', 'ng'), 'eng': ('e', 'ng'), 'ing': ('i', 'ng'),
    'ong': ('u', 'ng'), 'iong':('u', 'ng'),
    'iang':('a', 'ng'), 'uang':('a', 'ng'),
}

CONSONANT_MOD = {
    'b':  _mk_bs(close=0.45, press=0.35),
    'p':  _mk_bs(close=0.45, press=0.35),
    'm':  _mk_bs(close=0.40, press=0.30),
    'f':  _mk_bs(lower_down=0.10, stretch=0.10),
    'zh': _mk_bs(funnel=0.15, pucker=0.12),
    'ch': _mk_bs(funnel=0.15, pucker=0.12),
    'sh': _mk_bs(funnel=0.15, pucker=0.12),
    'r':  _mk_bs(funnel=0.15, pucker=0.10),
    'j':  _mk_bs(stretch=0.40, smile=0.20),
    'q':  _mk_bs(stretch=0.40, smile=0.20),
    'x':  _mk_bs(stretch=0.40, smile=0.20),
}

# ================= 英文映射表 =================
ENGLISH_PHONEME_BS = {
    # === 元音 ===
    'AA': _mk_bs(jaw=0.50, stretch=0.10),
    'AE': _mk_bs(jaw=0.40, stretch=0.20),
    'AH': _mk_bs(jaw=0.30, stretch=0.10),
    'AO': _mk_bs(jaw=0.40, funnel=0.30),
    'AW': _mk_bs(jaw=0.50, funnel=0.40, pucker=0.20),
    'AY': _mk_bs(jaw=0.40, stretch=0.30),
    'EH': _mk_bs(jaw=0.30, stretch=0.20),
    'ER': _mk_bs(jaw=0.20, pucker=0.20),
    'EY': _mk_bs(jaw=0.20, stretch=0.40),
    'IH': _mk_bs(jaw=0.10, stretch=0.30),
    'IY': _mk_bs(jaw=0.05, stretch=0.50, smile=0.10),
    'OW': _mk_bs(jaw=0.20, funnel=0.50, pucker=0.40),
    'OY': _mk_bs(jaw=0.30, funnel=0.40, stretch=0.20),
    'UH': _mk_bs(jaw=0.10, funnel=0.30),
    'UW': _mk_bs(jaw=0.05, funnel=0.60, pucker=0.50),
    # === 爆破音 ===
    'B':  _mk_bs(close=0.50, press=0.30),
    'P':  _mk_bs(close=0.50, press=0.30),
    'T':  _mk_bs(close=0.50, press=0.30),
    'D':  _mk_bs(close=0.50, press=0.30),
    'K':  _mk_bs(close=0.50, press=0.30),
    'G':  _mk_bs(close=0.50, press=0.30),
    # === 鼻音 ===
    'M':  _mk_bs(close=0.40, press=0.20),
    'N':  _mk_bs(jaw=0.08, close=0.15),
    'NG': _mk_bs(jaw=0.10, close=0.10),
    # === 摩擦音 ===
    'F':  _mk_bs(lower_down=0.20, roll_lower=0.20),
    'V':  _mk_bs(lower_down=0.20, roll_lower=0.20),
    'S':  _mk_bs(jaw=0.05, stretch=0.15),
    'Z':  _mk_bs(jaw=0.05, stretch=0.15),
    'TH': _mk_bs(jaw=0.08, lower_down=0.10),
    'DH': _mk_bs(jaw=0.08, lower_down=0.10),
    'HH': _mk_bs(jaw=0.15),
    # === 塞擦音 / 颚音 ===
    'CH': _mk_bs(jaw=0.10, funnel=0.30, pucker=0.20),
    'SH': _mk_bs(jaw=0.10, funnel=0.30, pucker=0.20),
    'ZH': _mk_bs(jaw=0.10, funnel=0.30, pucker=0.20),
    'JH': _mk_bs(jaw=0.10, funnel=0.30, pucker=0.20),
    # === 半元音 / 流音 ===
    'W':  _mk_bs(funnel=0.50, pucker=0.40),
    'R':  _mk_bs(funnel=0.20, pucker=0.20),
    'Y':  _mk_bs(stretch=0.30),
    'L':  _mk_bs(jaw=0.10),
}

# ================= 模型结构 =================
class LowerFaceBS2Angle(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128, 64]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_models(device):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    lower_motor_ids = checkpoint['lower_motor_ids']
    motor_ranges = checkpoint['motor_ranges']
    used_motors_full = checkpoint['used_motors_full']

    forward_ckpt = torch.load(FORWARD_MODEL_PATH, map_location=device)
    bs_keys_full = forward_ckpt['bs_keys']
    lower_bs_idx = checkpoint['lower_bs_idx']
    lower_bs_keys = [bs_keys_full[i] for i in lower_bs_idx]

    model = LowerFaceBS2Angle(len(lower_bs_keys), len(lower_motor_ids)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, lower_bs_keys, lower_motor_ids, motor_ranges, used_motors_full

# ================= 解析与合成 =================
def get_pinyin_list(text: str) -> list:
    from pypinyin import pinyin, Style
    py_list = pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)
    results = []
    for i, py_item in enumerate(py_list):
        py_str = py_item[0]
        ch = text[i]
        if not py_str or py_str == ch or py_str.startswith('_'):
            continue
        py_no_tone = re.sub(r'[1-5]', '', py_str).lower()
        py_no_tone = py_no_tone.replace('ü', 'v')
        init, final = split_pinyin(py_no_tone)
        if init in ('j', 'q', 'x', 'y') and final.startswith('u'):
            final = 'v' + final[1:]
        results.append((ch, init, final))
    return results

async def synthesize_speech(text: str):
    import edge_tts
    communicate = edge_tts.Communicate(text, TTS_VOICE, rate=TTS_SPEED, boundary="WordBoundary")
    mp3_data = b""
    word_events = []

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_data += chunk["data"]
        elif chunk["type"] == "WordBoundary":
            offset_s = chunk["offset"] / 10_000_000
            dur_s = chunk["duration"] / 10_000_000
            word_events.append((chunk["text"].strip(), offset_s, dur_s))

    if not mp3_data:
        raise RuntimeError("TTS 合成失败")
    return mp3_data, word_events

def load_audio_from_mp3(mp3_data: bytes):
    import torchaudio
    waveform, sr = torchaudio.load(io.BytesIO(mp3_data), format="mp3")
    audio = waveform.mean(dim=0).numpy().astype(np.float32)
    return audio, sr

# ================= 关键帧构建 =================
def bs_dict_to_vector(bs_dict: dict, bs_key_to_idx: dict) -> np.ndarray:
    vec = np.zeros(len(bs_key_to_idx), dtype=np.float32)
    for name, val in bs_dict.items():
        if name in bs_key_to_idx:
            vec[bs_key_to_idx[name]] = val
    return vec

def get_final_bs_sequence(final_str: str):
    if final_str in NASAL_FINALS:
        base_vowel, nasal_type = NASAL_FINALS[final_str]
        base_bs = VOWEL_BS.get(base_vowel, VOWEL_BS.get('a', {}))
        nasal_close = 0.20 if nasal_type == 'n' else 0.10
        nasal_bs = dict(base_bs)
        nasal_bs['mouthClose'] = nasal_bs.get('mouthClose', 0.0) + nasal_close
        return [(0.70, base_bs), (0.30, nasal_bs)]
    if final_str in DIPHTHONG_PHASES:
        phases = DIPHTHONG_PHASES[final_str]
        return [(frac, VOWEL_BS.get(v, VOWEL_BS.get('a', {}))) for frac, v in phases]
    if final_str in VOWEL_BS:
        return [(1.0, VOWEL_BS[final_str])]
    return [(1.0, VOWEL_BS['a'])]

def _add_kf(keyframes, offset_s, dur_s, init, final, bs_key_to_idx):
    cons_frac = 0.25 if init in CONSONANT_MOD else 0.0
    if cons_frac > 0:
        cons_mod = CONSONANT_MOD[init]
        cons_bs = dict(VOWEL_BS.get('a', {}))
        cons_bs.update(cons_mod)
        keyframes.append((offset_s, bs_dict_to_vector(cons_bs, bs_key_to_idx)))
    vowel_start = offset_s + dur_s * cons_frac
    vowel_dur = dur_s * (1 - cons_frac)
    final_phases = get_final_bs_sequence(final)
    cum_frac = 0.0
    for frac, bs_dict in final_phases:
        phase_start = vowel_start + vowel_dur * cum_frac
        cum_frac += frac
        keyframes.append((phase_start, bs_dict_to_vector(bs_dict, bs_key_to_idx)))
    end_time = offset_s + dur_s
    keyframes.append((end_time + 0.02, np.zeros(len(bs_key_to_idx), dtype=np.float32)))

def _add_english_kf(keyframes, offset_s, dur_s, word_text, bs_key_to_idx):
    """为一个英文单词添加关键帧"""
    raw_phonemes = g2p(word_text)
    phonemes = [re.sub(r'\d+', '', p) for p in raw_phonemes if p.isalpha()]
    
    if not phonemes:
        return

    p_dur = dur_s / len(phonemes)
    for i, p in enumerate(phonemes):
        p_start = offset_s + i * p_dur
        bs_dict = ENGLISH_PHONEME_BS.get(p, {}) 
        keyframes.append((p_start, bs_dict_to_vector(bs_dict, bs_key_to_idx)))

def build_viseme_timeline(pinyin_list: list, word_events: list, bs_key_to_idx: dict):
    """构建唇形关键帧时间线 (双语自动流转)"""
    keyframes = []
    neutral = np.zeros(len(bs_key_to_idx), dtype=np.float32)
    keyframes.append((0.0, neutral.copy()))

    py_idx = 0
    n_pinyin = len(pinyin_list)

    for we_text, we_offset, we_dur in word_events:
        has_chinese = bool(re.search(r'[\u4e00-\u9fa5]', we_text))
        has_english = bool(re.search(r'[A-Za-z]', we_text))

        if has_chinese:
            han_chars = [c for c in we_text if '一' <= c <= '鿿']
            n_han = len(han_chars)
            if n_han > 0:
                sub_dur = we_dur / n_han
                for chi, sub_ch in enumerate(han_chars):
                    if py_idx >= n_pinyin: break
                    while py_idx < n_pinyin and pinyin_list[py_idx][0] != sub_ch:
                        py_idx += 1
                    if py_idx >= n_pinyin: break
                    _, s_init, s_final = pinyin_list[py_idx]
                    sub_offset = we_offset + chi * sub_dur
                    _add_kf(keyframes, sub_offset, sub_dur, s_init, s_final, bs_key_to_idx)
                    py_idx += 1

        if has_english:
            clean_word = re.sub(r'[^A-Za-z\']', '', we_text)
            if clean_word:
                _add_english_kf(keyframes, we_offset, we_dur, clean_word, bs_key_to_idx)
                # 英文单词结束加一个瞬间回归中性的帧，收口干脆
                keyframes.append((we_offset + we_dur, neutral.copy()))

        if not has_chinese and not has_english:
            # 纯标点/静音 → 长停顿加中性帧
            if we_dur > 0.25:
                keyframes.append((we_offset + we_dur * 0.3, neutral.copy()))

    if keyframes:
        last_time = keyframes[-1][0]
        keyframes.append((last_time + 0.15, neutral.copy()))

    return keyframes

def denormalize_angle(norm, motor_id, motor_ranges):
    minv, maxv = motor_ranges[motor_id]
    return float(norm * (maxv - minv) + minv)

# ================= 主函数 =================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print("加载模型...")
    model, bs_keys, motor_ids, motor_ranges, used_motors = load_models(device)
    bs_key_to_idx = {name: i for i, name in enumerate(bs_keys)}
    print(f"  BS 键数: {len(bs_keys)}, 舵机数: {len(motor_ids)}")

    default_angles = {}
    for mid in used_motors:
        minv, maxv = motor_ranges[mid]
        default_angles[str(mid)] = float((minv + maxv) / 2.0)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"UDP → {RPI_IP}:{RPI_PORT}")

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("请输入文字(支持中英混合): ").strip()
    if not text:
        return

    pinyin_list = get_pinyin_list(text)
    
    print(f"TTS 合成 ({TTS_VOICE})...")
    mp3_data, word_events = asyncio.run(synthesize_speech(text))
    print(f"  音频: {len(mp3_data)/1024:.1f} KB, WordBoundary 事件: {len(word_events)}")

    keyframes = build_viseme_timeline(pinyin_list, word_events, bs_key_to_idx)
    print(f"WordBoundary 事件: {len(word_events)} 条")
    for we in word_events[:10]:
        print(f"  [{we[1]:.2f}s +{we[2]:.2f}s] \"{we[0]}\"")
    print(f"提取关键帧数: {len(keyframes)}")

    audio, sr = load_audio_from_mp3(mp3_data)
    duration = len(audio) / sr
    print(f"音频时长: {duration:.1f}s, sr={sr}Hz")

    print("\n开始播放，双语唇形实时驱动中... (Ctrl+C 停止)\n")

    def play_audio():
        sd.play(audio, sr)

    play_thread = threading.Thread(target=play_audio, daemon=True)
    play_thread.start()

    start_time = time.time()
    smoothed_bs = np.zeros(len(bs_keys), dtype=np.float32)
    last_frame_idx = -1

    try:
        while True:
            elapsed = time.time() - start_time
            frame_idx = int(elapsed / (FRAME_MS / 1000.0))

            if frame_idx == last_frame_idx:
                time.sleep(0.002)
                continue
            last_frame_idx = frame_idx

            if elapsed > duration + 0.3:
                if frame_idx > int(duration / (FRAME_MS / 1000.0)) + 3:
                    time.sleep(duration + 0.3 - elapsed + 0.05)
                    break

            target_bs = np.zeros(len(bs_keys), dtype=np.float32)
            for i in range(len(keyframes) - 1):
                t0, bs0 = keyframes[i]
                t1, bs1 = keyframes[i + 1]
                if t0 <= elapsed < t1:
                    alpha = (elapsed - t0) / (t1 - t0 + 1e-8)
                    target_bs = bs0 * (1 - alpha) + bs1 * alpha
                    break
            else:
                if elapsed >= keyframes[-1][0] if keyframes else 0:
                    target_bs = np.zeros(len(bs_keys), dtype=np.float32)

            # 指数平滑 - 此时 SMOOTH=0.35 让运动更敏捷
            smoothed_bs = SMOOTH * target_bs + (1 - SMOOTH) * smoothed_bs

            scaled_bs = smoothed_bs.copy()
            for name, idx in bs_key_to_idx.items():
                if name == 'jawOpen':
                    scaled_bs[idx] = min(scaled_bs[idx] * JAW_GAIN, 1.0)
                else:
                    scaled_bs[idx] = min(scaled_bs[idx] * LIP_GAIN, 1.0)

            input_tensor = torch.tensor([scaled_bs], dtype=torch.float32).to(device)
            with torch.no_grad():
                pred_norm = model(input_tensor).cpu().numpy()[0]

            angles_dict = default_angles.copy()
            for i, mid in enumerate(motor_ids):
                angle = denormalize_angle(pred_norm[i], mid, motor_ranges)
                angles_dict[str(mid)] = round(angle, 2)

            data = json.dumps(angles_dict) + '\n'
            sock.sendto(data.encode('utf-8'), (RPI_IP, RPI_PORT))

            if frame_idx % 30 == 0:
                jaw = scaled_bs[bs_key_to_idx['jawOpen']]
                funnel = scaled_bs[bs_key_to_idx['mouthFunnel']]
                stretch = scaled_bs[bs_key_to_idx['mouthStretchLeft']]
                close = scaled_bs[bs_key_to_idx['mouthClose']]
                pucker = scaled_bs[bs_key_to_idx['mouthPucker']]
                bar = '█' * int(jaw * 20)
                shape = ''
                if funnel > 0.1: shape += '⭕圆唇 '
                if stretch > 0.1: shape += '↔展唇 '
                if close > 0.1: shape += '🔒闭唇 '
                if pucker > 0.1: shape += '💋撅唇 '
                if not shape: shape = '😐中性'
                print(f"  t={elapsed:.2f}s jaw={bar:{20 if jaw>0 else 0}} {jaw:.2f} | {shape}")

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        sd.stop()
        sock.close()
        print("程序结束。")

if __name__ == "__main__":
    main()