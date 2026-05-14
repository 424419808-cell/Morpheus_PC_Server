# DashScope 在线语音识别服务
# 替代 xunfei_ear_realtime.py，使用 DashScope Paraformer + CAM++ 声纹
#
# ========== 运行示例 ==========
# 1) 列出音频设备:
#    python emotion_MLP/audio/sensevoice_ear_realtime.py --list-devices
#
# 2) 使用指定麦克风启动(推荐):
#    python emotion_MLP/audio/sensevoice_ear_realtime.py --input-device 2
#
# 3) 使用默认设备启动:
#    python emotion_MLP/audio/sensevoice_ear_realtime.py
#
# 注意: 如果 conda activate 有问题, 用以下方式替代:
#    conda run -p H:\APP\conda\envs\emotion_mlp python emotion_MLP/audio/sensevoice_ear_realtime.py --input-device 2
# =============================
import argparse
import os
import pickle
import queue
import socket
import sys
import threading
import time
from collections import deque

import numpy as np
import pyaudio
import tempfile
import torch
from dashscope.audio.asr import Recognition
from dashscope.audio.asr.recognition import RecognitionCallback
from funasr import AutoModel
from scipy.signal import butter, correlate, istft, lfilter, resample, stft

# ===== AEC 条件导入 =====
try:
    from aec_audio_processing import AudioProcessor
    HAS_AEC = True
except ImportError:
    HAS_AEC = False
    print("[WARN] aec_audio_processing 未找到，AEC 已禁用。继续运行但无回声消除。")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ================= 1. 网络与音频配置 =================
WSL_IP = "172.20.195.170"
UDP_PORT = 5006
GEMMA_VOICE_PORT = 5008
EMOTION_COPROCESSOR_PORT = 5012
LOCK_PORT = 5005               # 单例锁定端口（防止重复启动）

FORMAT = pyaudio.paInt16
CHANNELS = 4
RATE = 16000
CHUNK = 1280          # 80ms
SUBFRAME = 160        # 10ms for AEC

# --- DashScope 配置 ---
DASHSCOPE_API_KEY = "sk-0cf953e4c6c34f7089ffd6f7335235c1"

# --- 声源定位参数 ---
UPSAMPLE_FACTOR = 10
DISTANCE = 0.035
SOUND_SPEED = 343.0
WINDOW_SIZE = 7

# --- VAD 参数 ---
SILENCE_TIMEOUT = 0.5
SILENCE_CHUNKS = int(SILENCE_TIMEOUT / (CHUNK / RATE))  # ~6 chunks
VAD_THRESHOLD = 0.5
MAX_UTTERANCE_CHUNKS = 375  # ~30 seconds

# --- AEC 参数 ---
USE_RESIDUAL_SUPPRESSION = True
RESIDUAL_ALPHA = 2.5
RESIDUAL_BETA = 0.01

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# --- 声纹参数 ---
VOICEPRINT_THRESHOLD = 0.6
NAMES_PKL = os.path.join(ROOT_DIR, "names.pkl")


# ================= 2. 信号处理函数（与原始一致） =================
def voice_filter(data, lowcut=300, highcut=3400, fs=16000, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return lfilter(b, a, data)


def residual_suppression(sig, fs, alpha=2.0, beta=0.01):
    sig_float = sig.astype(np.float32)
    f, t, Zxx = stft(sig_float, fs, nperseg=512, noverlap=256)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    noise_mag = np.mean(mag[:, :5], axis=1, keepdims=True)
    mag_clean = np.maximum(mag - alpha * noise_mag, beta * mag)
    Zxx_clean = mag_clean * np.exp(1j * phase)
    _, sig_clean = istft(Zxx_clean, fs)
    sig_clean = sig_clean[:len(sig)]
    return np.clip(sig_clean, -32768, 32767).astype(np.int16)


# ================= 3. Morpheus_Ear_System =================

class _DummyCB(RecognitionCallback):
    """DashScope Recognition 需要的占位回调"""
    pass


class Morpheus_Ear_System:
    def __init__(self, device_str="auto", input_device_idx=None):
        self.exit_flag = False
        self.is_visual_running = False
        self.current_vip_id = None
        self.input_device_idx = input_device_idx

        # --- 设备 ---
        if device_str == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_str
        print(f"[Morpheus Ear] 推理设备: {self.device}")

        # --- 加载模型 ---
        print("[Morpheus Ear] 初始化 DashScope ASR (Paraformer)...")
        import dashscope
        dashscope.api_key = DASHSCOPE_API_KEY
        self._asr_recognizer = Recognition(
            model='paraformer-realtime-v2',
            callback=_DummyCB(),
            format='pcm',
            sample_rate=RATE,
        )
        print("[Morpheus Ear] DashScope ASR 初始化完成。")

        print("[Morpheus Ear] 加载 CAM++ 声纹模型...")
        self.sv_model = AutoModel(
            model="iic/speech_campplus_sv_zh-cn_16k-common",
            disable_pbar=True,
            device=self.device,
        )
        print("[Morpheus Ear] CAM++ 声纹模型加载完成。")

        print("[Morpheus Ear] 加载 Silero VAD...")
        from silero_vad import load_silero_vad
        self.vad_model = load_silero_vad()
        print("[Morpheus Ear] Silero VAD 加载完成。")

        # --- 声纹嵌入缓存 ---
        self.speaker_embeddings = {}

        # --- 加载名称映射 ---
        if os.path.exists(NAMES_PKL):
            with open(NAMES_PKL, "rb") as f:
                self.names = pickle.load(f)
        else:
            self.names = {}

        # --- 状态变量 ---
        self.temp_text = ""
        self.last_h_val = 0.0
        self.last_v_val = 0.0

        # --- 声源定位历史 ---
        self.voice_history_h = deque(maxlen=WINDOW_SIZE)
        self.voice_history_v = deque(maxlen=WINDOW_SIZE)

        # --- 性能优化: 帧计数降频 ---
        self._frame_counter = 0
        self._vad_interval = 4       # 每 4 chunks 跑一次 VAD (~320ms)
        self._localize_interval = 3  # 每 3 chunks 定位一次 (~240ms)
        # 动态计算 VAD 降频后的静音超时阈值（保持约 500ms）
        self._silence_timeout_chunks = max(1, int(SILENCE_TIMEOUT / (self._vad_interval * CHUNK / RATE)))

        # --- VAD 状态机 ---
        self.vad_state = "IDLE"      # IDLE | SPEAKING
        self.speech_buffer = []      # list of np.int16 1D arrays
        self.silence_counter = 0

        # --- AEC ---
        if HAS_AEC:
            self.aec_ref_queue = queue.Queue(maxsize=500)
            self.aec_processor = AudioProcessor(enable_aec=True, enable_ns=False, enable_agc=False)
            self.aec_processor.set_stream_format(16000, 1)
            self.aec_processor.set_reverse_stream_format(16000, 1)
            self.aec_ref_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.aec_ref_sock.bind(("0.0.0.0", 5010))
            self.aec_ref_sock.setblocking(False)
            threading.Thread(target=self._aec_ref_receiver, daemon=True).start()
        else:
            self.aec_processor = None
            self.aec_ref_queue = None

        # --- UDP 输出 ---
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # ================= AEC =================
    def _aec_ref_receiver(self):
        while not self.exit_flag:
            try:
                data, _ = self.aec_ref_sock.recvfrom(4096)
                self.aec_ref_queue.put(data)
            except BlockingIOError:
                time.sleep(0.001)
            except Exception:
                pass

    def _process_aec(self, best_ch):
        if self.aec_processor is None:
            return best_ch
        subframes = [best_ch[i:i + SUBFRAME] for i in range(0, len(best_ch), SUBFRAME)]
        processed_subframes = []
        for sub in subframes:
            try:
                far_frame = self.aec_ref_queue.get_nowait()
            except queue.Empty:
                far_frame = None
            if far_frame is not None:
                near_frame = sub.tobytes()
                self.aec_processor.process_reverse_stream(far_frame)
                out_frame = self.aec_processor.process_stream(near_frame)
                processed_sub = np.frombuffer(out_frame, dtype=np.int16)
            else:
                processed_sub = sub
            processed_subframes.append(processed_sub)
        processed = np.concatenate(processed_subframes)
        if USE_RESIDUAL_SUPPRESSION:
            processed = residual_suppression(processed, RATE,
                                             alpha=RESIDUAL_ALPHA,
                                             beta=RESIDUAL_BETA)
        return processed

    # ================= 声源定位（与原始一致） =================
    def _localize(self, ch_up, ch_down, ch_left, ch_right):
        # 水平定位（左-右）
        l_f = voice_filter(ch_left)
        if np.max(np.abs(l_f)) > 600:
            l_up = resample(l_f, len(l_f) * UPSAMPLE_FACTOR)
            r_f = voice_filter(ch_right)
            r_up = resample(r_f, len(r_f) * UPSAMPLE_FACTOR)
            corr = correlate(l_up, r_up, mode='full')
            delay_h = np.argmax(corr) - (len(l_up) - 1)
            self.voice_history_h.append(delay_h)
            if len(self.voice_history_h) == WINDOW_SIZE:
                self.last_h_val = np.median(self.voice_history_h)

        # 垂直定位（上-下）
        u_f = voice_filter(ch_up)
        if np.max(np.abs(u_f)) > 600:
            u_up = resample(u_f, len(u_f) * UPSAMPLE_FACTOR)
            d_f = voice_filter(ch_down)
            d_up = resample(d_f, len(d_f) * UPSAMPLE_FACTOR)
            corr = correlate(u_up, d_up, mode='full')
            delay_v = np.argmax(corr) - (len(u_up) - 1)
            self.voice_history_v.append(delay_v)
            if len(self.voice_history_v) == WINDOW_SIZE:
                self.last_v_val = np.median(self.voice_history_v)

    def _localize_horizontal(self, ch_left, ch_right):
        """仅水平方向声源定位（双通道麦克风）"""
        l_f = voice_filter(ch_left)
        if np.max(np.abs(l_f)) > 600:
            l_up = resample(l_f, len(l_f) * UPSAMPLE_FACTOR)
            r_f = voice_filter(ch_right)
            r_up = resample(r_f, len(r_f) * UPSAMPLE_FACTOR)
            corr = correlate(l_up, r_up, mode='full')
            delay_h = np.argmax(corr) - (len(l_up) - 1)
            self.voice_history_h.append(delay_h)
            if len(self.voice_history_h) == WINDOW_SIZE:
                self.last_h_val = np.median(self.voice_history_h)

    # ================= VAD =================
    def _vad_speech_prob(self, audio_np):
        audio_float = audio_np.astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_float).unsqueeze(0)
        with torch.no_grad():
            prob = self.vad_model(audio_tensor, RATE).item()
        return prob

    def _vad_process_chunk(self, chunk_np):
        # 能量预检测：低能量且未在说话时跳过昂贵的 Silero VAD 推理
        rms = np.sqrt(np.mean((chunk_np.astype(np.float32) / 32768.0) ** 2))
        if rms < 0.005 and self.vad_state == "IDLE":
            return None, False, None

        # Silero VAD 要求严格 512 样本 (16kHz)，将 CHUNK 切分为子帧取最大值
        VAD_FRAME = 512
        num_sub = len(chunk_np) // VAD_FRAME
        if num_sub == 0:
            return None, False, None
        sub_probs = []
        for i in range(num_sub):
            sub = chunk_np[i * VAD_FRAME:(i + 1) * VAD_FRAME]
            sub_probs.append(self._vad_speech_prob(sub))
        speech_prob = max(sub_probs)

        if self.vad_state == "IDLE":
            if speech_prob > VAD_THRESHOLD:
                self.vad_state = "SPEAKING"
                self.speech_buffer = [chunk_np.copy()]
                self.silence_counter = 0
            return None, False, None

        elif self.vad_state == "SPEAKING":
            self.speech_buffer.append(chunk_np.copy())
            if speech_prob < VAD_THRESHOLD:
                self.silence_counter += 1
                if self.silence_counter >= self._silence_timeout_chunks:
                    vpr_audio = b"".join(chunk.tobytes() for chunk in self.speech_buffer)
                    asr_text = self._run_asr()
                    need_vpr = self._check_keywords(asr_text) if asr_text else False
                    self.speech_buffer = []
                    self.silence_counter = 0
                    self.vad_state = "IDLE"
                    return asr_text, need_vpr, vpr_audio
            else:
                self.silence_counter = 0
            # 超过 30 秒强制截断
            if len(self.speech_buffer) > MAX_UTTERANCE_CHUNKS:
                vpr_audio = b"".join(chunk.tobytes() for chunk in self.speech_buffer)
                asr_text = self._run_asr()
                need_vpr = self._check_keywords(asr_text) if asr_text else False
                self.speech_buffer = []
                self.silence_counter = 0
                self.vad_state = "IDLE"
                return asr_text, need_vpr, vpr_audio
            return None, False, None

        return None, False, None

    # ================= ASR =================
    def _run_asr(self):
        if len(self.speech_buffer) == 0:
            return None
        t0 = time.perf_counter()
        audio_data = b"".join(chunk.tobytes() for chunk in self.speech_buffer)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        duration = len(audio_np) / RATE
        if duration < 0.3:
            return None
        try:
            t1 = time.perf_counter()
            with tempfile.NamedTemporaryFile(suffix='.pcm', delete=False) as f:
                f.write(audio_np.tobytes())
                tmp = f.name
            try:
                result = self._asr_recognizer.call(file=tmp)
            finally:
                os.unlink(tmp)
            t2 = time.perf_counter()
            sentences = result.get_sentence()
            if isinstance(sentences, list):
                text = ' '.join(s.get('text', '') for s in sentences if isinstance(s, dict)).strip()
            elif isinstance(sentences, dict):
                text = sentences.get('text', '').strip()
            else:
                text = ''
            if text:
                print(f"\n[ASR 耗时] 音频={duration:.1f}s  网络+API={t2-t1:.2f}s  总计={t2-t0:.2f}s")
                return text
        except Exception as e:
            print(f"\n[ASR Error]: {e}")
        return None

    # ================= 声纹 =================
    def _extract_embedding(self, audio_np):
        try:
            res = self.sv_model.generate(input=audio_np)
            if isinstance(res, list) and len(res) > 0:
                item = res[0]
                if isinstance(item, np.ndarray):
                    return item
                if isinstance(item, torch.Tensor):
                    return item.cpu().numpy()
                if isinstance(item, dict):
                    for v in item.values():
                        if isinstance(v, (np.ndarray, torch.Tensor)):
                            return v.cpu().numpy() if isinstance(v, torch.Tensor) else v
            if isinstance(res, np.ndarray):
                return res
            if isinstance(res, torch.Tensor):
                return res.cpu().numpy()
        except Exception as e:
            print(f"\n[Voiceprint Extract Error]: {e}")
        return None

    def _verify_voiceprint(self, audio_data):
        if not self.speaker_embeddings:
            print("[Voiceprint] 未注册任何声纹参考。")
            return None
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            emb = self._extract_embedding(audio_np)
            if emb is None:
                return None
            best_match = None
            best_score = 0.0
            for speaker_id, ref_emb in self.speaker_embeddings.items():
                score = self._cosine_similarity(emb, ref_emb)
                if score > best_score:
                    best_score = score
                    best_match = speaker_id
            if best_score > VOICEPRINT_THRESHOLD:
                return {"score": float(best_score), "featureId": best_match}
        except Exception as e:
            print(f"\n[Voiceprint Error]: {e}")
        return None

    @staticmethod
    def _cosine_similarity(a, b):
        a = a.flatten()
        b = b.flatten()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def register_speaker(self, speaker_id, audio_paths):
        embeddings = []
        for path in audio_paths:
            if not os.path.exists(path):
                print(f"[Voiceprint] 文件不存在: {path}")
                continue
            try:
                # 读取 WAV/PCM 文件
                import wave
                with wave.open(path, 'rb') as wf:
                    sr = wf.getframerate()
                    raw = wf.readframes(wf.getnframes())
                    audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                # 重采样到 16kHz 如果需要
                if sr != RATE:
                    from scipy.signal import resample as scipy_resample
                    new_len = int(len(audio_np) * RATE / sr)
                    audio_np = scipy_resample(audio_np, new_len).astype(np.float32)
                emb = self._extract_embedding(audio_np)
                if emb is not None:
                    embeddings.append(emb)
            except Exception as e:
                print(f"[Voiceprint] 处理 {path} 失败: {e}")
        if embeddings:
            self.speaker_embeddings[speaker_id] = np.mean(embeddings, axis=0)
            name = self.names.get(speaker_id, f"ID_{speaker_id}")
            print(f"[Voiceprint] 已注册 {name} (ID:{speaker_id}), 样本数: {len(embeddings)}")
        else:
            print(f"[Voiceprint] 注册 ID:{speaker_id} 失败：无有效嵌入")

    def _register_all_speakers(self):
        speaker_dir = os.path.join(ROOT_DIR, "speaker_samples")
        if not os.path.isdir(speaker_dir):
            print("[Voiceprint] 未找到 speaker_samples/ 目录，跳过声纹注册。")
            return
        for person_name in os.listdir(speaker_dir):
            p_path = os.path.join(speaker_dir, person_name)
            if not os.path.isdir(p_path):
                continue
            target_id = None
            for tid, name in self.names.items():
                if name == person_name:
                    target_id = int(tid)
                    break
            if target_id is None:
                print(f"[Voiceprint] 未找到 {person_name} 的 ID 映射，跳过。")
                continue
            audio_files = [
                os.path.join(p_path, f)
                for f in os.listdir(p_path)
                if f.endswith(('.wav', '.pcm'))
            ]
            if audio_files:
                self.register_speaker(target_id, audio_files)

    # ================= 关键词与 UDP 转发 =================
    @staticmethod
    def _check_keywords(text):
        if not text:
            return False
        return any(word in text for word in ["启动", "关机"]) or "这里" in text

    def _dispatch_result(self, text, match_result):
        if not text:
            return

        start_triggered = any(word in text for word in ["启动"])
        stop_triggered = any(word in text for word in ["关机"])
        relocate_triggered = "这里" in text

        if start_triggered or stop_triggered or relocate_triggered:
            if match_result:
                confidence = match_result['score']
                target_id = int(match_result['featureId'])
                user_name = self.names.get(target_id, f"VIP_{target_id}")
                if confidence > 0.1:
                    if start_triggered and not self.is_visual_running:
                        print(f"[验证通过] VIP: {user_name} (H:{self.last_h_val:.1f}, V:{self.last_v_val:.1f}) -> 启动")
                        self.sock.sendto(
                            f"START:{target_id}:{self.last_h_val}:{self.last_v_val}".encode(),
                            (WSL_IP, UDP_PORT))
                        self.is_visual_running = True
                        self.current_vip_id = target_id
                    elif relocate_triggered and self.is_visual_running:
                        if target_id == self.current_vip_id:
                            print(f"[定位请求] VIP: {user_name} (H:{self.last_h_val:.1f}, V:{self.last_v_val:.1f})")
                            self.sock.sendto(
                                f"START:{target_id}:{self.last_h_val}:{self.last_v_val}".encode(),
                                (WSL_IP, UDP_PORT))
                        else:
                            print(f"[权限拒绝] 非当前操作者 {user_name}")
                    elif stop_triggered and self.is_visual_running:
                        if target_id == self.current_vip_id:
                            print(f"[验证通过] VIP: {user_name} -> 关机")
                            self.sock.sendto("STOP".encode(), (WSL_IP, UDP_PORT))
                            self.is_visual_running = False
                            self.current_vip_id = None
                        else:
                            owner_name = self.names.get(self.current_vip_id, "原开启者")
                            print(f"[权限拒绝] 当前由 {owner_name} 运行，{user_name} 无权干预。")
                else:
                    print(f"[验证失败] 声纹不匹配 (置信度: {confidence:.2f})")
            else:
                print(">>> 声纹识别未返回结果")
        if len(text) > 1:
            print(f"\n[对话转发]: {text}")
            self.sock.sendto(text.encode(), ("127.0.0.1", GEMMA_VOICE_PORT))
            self.sock.sendto(text.encode(), ("127.0.0.1", EMOTION_COPROCESSOR_PORT))
            if self.is_visual_running:
                self.sock.sendto(f"DIR:{self.last_h_val}:{self.last_v_val}".encode(), (WSL_IP, UDP_PORT))

    # ================= 主采集循环 =================
    def _capture_loop(self):
        p = pyaudio.PyAudio()
        # 自动检测默认输入设备支持的最大通道数
        if self.input_device_idx is not None:
            try:
                device_info = p.get_device_info_by_index(self.input_device_idx)
                max_input_ch = int(device_info['maxInputChannels'])
                device_name = device_info['name']
                print(f"[Morpheus Ear] 使用指定输入设备 [{self.input_device_idx}]: {device_name}")
            except Exception:
                print(f"[WARN] 设备索引 {self.input_device_idx} 无效，回退到默认设备")
                self.input_device_idx = None

        if self.input_device_idx is None:
            try:
                device_info = p.get_default_input_device_info()
                max_input_ch = int(device_info['maxInputChannels'])
            except Exception:
                max_input_ch = 1
        actual_channels = min(CHANNELS, max_input_ch)
        if actual_channels < 4:
            print(f"[WARN] 默认设备仅支持 {actual_channels} 通道 "
                  f"(配置需要 {CHANNELS} 通道用于声源定位)，声源定位将受限。")
        self.actual_channels = actual_channels

        stream_kwargs = dict(
            format=FORMAT,
            channels=actual_channels,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        if self.input_device_idx is not None:
            stream_kwargs['input_device_index'] = self.input_device_idx
        stream = p.open(**stream_kwargs)
        try:
            while not self.exit_flag:
                self._frame_counter += 1
                raw_data = stream.read(CHUNK, exception_on_overflow=False)
                data_np = np.frombuffer(raw_data, dtype=np.int16).reshape(-1, actual_channels)

                # 通道分离（根据实际通道数自适应）
                if actual_channels == 4:
                    ch_up = data_np[:, 0]
                    ch_down = data_np[:, 1]
                    ch_left = data_np[:, 2]
                    ch_right = data_np[:, 3]
                    energies = [
                        np.max(np.abs(ch_up)),
                        np.max(np.abs(ch_down)),
                        np.max(np.abs(ch_left)),
                        np.max(np.abs(ch_right)),
                    ]
                    best_idx = np.argmax(energies)
                    best_ch = [ch_up, ch_down, ch_left, ch_right][best_idx]
                    if self._frame_counter % self._localize_interval == 0:
                        self._localize(ch_up, ch_down, ch_left, ch_right)
                elif actual_channels == 2:
                    ch_up = data_np[:, 0]
                    ch_down = data_np[:, 0]  # 复用 ch0 作为占位
                    ch_left = data_np[:, 0]
                    ch_right = data_np[:, 1]
                    energies = [
                        np.max(np.abs(ch_left)),
                        np.max(np.abs(ch_right)),
                    ]
                    best_idx = np.argmax(energies)
                    best_ch = [ch_left, ch_right][best_idx]
                    if self._frame_counter % self._localize_interval == 0:
                        self._localize_horizontal(ch_left, ch_right)
                else:
                    best_ch = data_np[:, 0]
                    # 单通道无法定位

                # AEC 处理
                processed = self._process_aec(best_ch)

                # VAD + ASR（降频：每 N 帧跑一次完整 VAD）
                if self._frame_counter % self._vad_interval == 0:
                    asr_text, need_vpr, vpr_audio = self._vad_process_chunk(processed)
                else:
                    asr_text = None
                    need_vpr = False
                    vpr_audio = None
                    # 非 VAD 帧：如果在说话中，累积音频但不跑推理
                    if self.vad_state == "SPEAKING":
                        self.speech_buffer.append(processed.copy())
                        if len(self.speech_buffer) > MAX_UTTERANCE_CHUNKS:
                            vpr_audio = b"".join(chunk.tobytes() for chunk in self.speech_buffer)
                            asr_text = self._run_asr()
                            need_vpr = self._check_keywords(asr_text) if asr_text else False
                            self.speech_buffer = []
                            self.silence_counter = 0
                            self.vad_state = "IDLE"

                if asr_text:
                    print(f"\n[ASR]: {asr_text}")
                    match_result = None
                    if need_vpr and vpr_audio:
                        match_result = self._verify_voiceprint(vpr_audio)
                    self._dispatch_result(asr_text, match_result)

                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"\n[Capture Error]: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def start(self):
        print("\n" + "=" * 40)
        print("   Morpheus 语音指挥中心 [DashScope 在线版]")
        print("   模型: Paraformer + CAM++ 声纹 + Silero VAD")
        print(f"   设备: {self.device}")
        print("   唤醒: '启动' | 休眠: '关机'")
        print("=" * 40)

        self._register_all_speakers()
        self._capture_loop()


# ================= 4. 单例锁定 =================
def try_acquire_lock(port):
    """尝试绑定锁定端口，防止同一脚本启动多个实例。返回 (lock_sock, True) 成功或 (None, False) 失败。"""
    lock_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        lock_sock.bind(("127.0.0.1", port))
        lock_sock.setblocking(False)
        return lock_sock, True
    except OSError:
        lock_sock.close()
        return None, False


# ================= 5. 入口 =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Morpheus Ear - DashScope ASR + CAM++ 声纹")
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cuda:0", "cpu"],
        help="推理设备 (default: auto — 自动检测 CUDA)",
    )
    parser.add_argument(
        "--input-device", type=int, default=None,
        help="PyAudio 输入设备索引 (default: None = 系统默认设备)。用 --list-devices 查看可用设备",
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="列出所有音频输入设备并退出",
    )
    args = parser.parse_args()

    # ---- 单例锁定（防止重复启动） ----
    lock_sock, acquired = try_acquire_lock(LOCK_PORT)
    if not acquired:
        print(f"[错误] 端口 {LOCK_PORT} 已被占用，说明已有实例在运行。请先终止已有进程。")
        sys.exit(1)

    if args.list_devices:
        import pyaudio
        p = pyaudio.PyAudio()
        print("可用音频输入设备:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  [{i}] {info['name']}  (in={info['maxInputChannels']}, rate={int(info['defaultSampleRate'])})")
        p.terminate()
        sys.exit(0)

    try:
        morpheus = Morpheus_Ear_System(device_str=args.device, input_device_idx=args.input_device)
        morpheus.start()
    except KeyboardInterrupt:
        print("\n>>> 听觉大脑已安全退出。")
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
