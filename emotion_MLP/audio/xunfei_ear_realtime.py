#转发要转发有意义的信息
import websocket
import datetime
import hashlib
import hmac
import base64
import json
import time
import ssl
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
from time import mktime
import _thread as thread
import pyaudio
import pickle
import os
import socket
import requests
import numpy as np
from scipy.signal import correlate, butter, lfilter, resample, stft, istft
from collections import deque

# ===== 新增：AEC 相关库 =====
import queue
import threading
from aec_audio_processing import AudioProcessor
# ============================

# 语音识别服务。在windows端里运行，在WSL等待语音指令时这个服务必须一直打开
# ================= 1. 网络与权限配置 =================
WSL_IP = "172.20.195.170"  # 127.0.0.1用于本地测试，如果 WSL2 无法接收，请修改为 hostname -I 查到的 IP
UDP_PORT = 5006  # 对应 brain_v2.py 的监听端口
GEMMA_VOICE_PORT = 5008  # 新增：对接 Gemma 语音对话的端口

APPID = "34d6c5db"
APIKey = "2ea750fed0dc74d3abcbba4e5f0c7759"
APISecret = "M2QwNDEwYmQ5OWMyOGU2MTcxMWE0MmFm"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
GROUP_ID = "morpheus_vip_group"
NAMES_PKL = os.path.join(ROOT_DIR, "names.pkl")

# 音频流参数 (16k/16bit/四通道采集，识别时只取左声道)
FORMAT = pyaudio.paInt16
CHANNELS = 4          # 四通道
RATE = 16000
CHUNK = 1280          # 80ms，与 AEC 子帧切分兼容（1280/160=8）

# --- 声源定位专用参数 ---
UPSAMPLE_FACTOR = 10
DISTANCE = 0.035      # 3.5cm（仅用于参考）
SOUND_SPEED = 343.0
WINDOW_SIZE = 7
voice_history_h = deque(maxlen=WINDOW_SIZE)   # 水平方向历史
voice_history_v = deque(maxlen=WINDOW_SIZE)   # 垂直方向历史

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ===== 后处理谱减法参数（启用，增强AEC效果） =====
USE_RESIDUAL_SUPPRESSION = True   # 修改：改为 True，启用谱减法抑制残留回声
RESIDUAL_ALPHA = 2.5
RESIDUAL_BETA = 0.01
# =================================

# --- 1. 人声带通滤波器 ---
def voice_filter(data, lowcut=300, highcut=3400, fs=16000, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return lfilter(b, a, data)

# ===== 谱减法后处理函数 =====
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
# ===============================

# ================= 2. 初始化加载 =================
try:
    print(">>> 正在加载 Morpheus 听觉大脑 [四通道定位版]...")
    with open(NAMES_PKL, "rb") as f:
        names = pickle.load(f)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(">>> [Morpheus Ear] 加载成功。")
except Exception as e:
    print(f">>> 初始化失败: {e}")
    exit()


class Morpheus_Ear_System:
    def __init__(self):
        self.exit_flag = False
        self.is_visual_running = False
        self.current_vip_id = None

        # 核心逻辑变量
        self.temp_text = ""
        self.audio_buffer = []          # 用于声纹识别
        self.max_buffer_len = 50
        self.last_h_val = 0.0           # 水平方向偏移量
        self.last_v_val = 0.0           # 垂直方向偏移量

        # ===== AEC 初始化（优化） =====
        self.aec_ref_queue = queue.Queue(maxsize=500)   # 修改：增大队列，避免参考帧丢失
        # 关闭噪声抑制，避免与AEC冲突
        self.aec_processor = AudioProcessor(enable_aec=True, enable_ns=False, enable_agc=False)
        self.aec_processor.set_stream_format(16000, 1)
        self.aec_processor.set_reverse_stream_format(16000, 1)

        self.aec_ref_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.aec_ref_sock.bind(("0.0.0.0", 5010))
        self.aec_ref_sock.setblocking(False)
        threading.Thread(target=self._aec_ref_receiver, daemon=True).start()
        # ===========================

    def _aec_ref_receiver(self):
        """接收参考帧并放入队列"""
        while not self.exit_flag:
            try:
                data, _ = self.aec_ref_sock.recvfrom(4096)
                self.aec_ref_queue.put(data)
            except BlockingIOError:
                time.sleep(0.001)
            except Exception:
                pass

    def get_asr_url(self):
        url = 'wss://iat.xf-yun.com/v1'
        now = datetime.datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        signature_origin = f"host: iat.xf-yun.com\ndate: {date}\nGET /v1 HTTP/1.1"
        signature_sha = hmac.new(APISecret.encode('utf-8'), signature_origin.encode('utf-8'), hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')
        auth_origin = f'api_key="{APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha}"'
        authorization = base64.b64encode(auth_origin.encode('utf-8')).decode(encoding='utf-8')
        return url + '?' + urlencode({"authorization": authorization, "date": date, "host": "iat.xf-yun.com"})

    def get_vpr_url(self):
        host = "api.xf-yun.com"
        path = "/v1/private/s782b4996"
        now = datetime.datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        signature_origin = f"host: {host}\ndate: {date}\nPOST {path} HTTP/1.1"
        signature_sha = hmac.new(APISecret.encode('utf-8'), signature_origin.encode('utf-8'), hashlib.sha256).digest()
        signature = base64.b64encode(signature_sha).decode(encoding='utf-8')
        auth_origin = f'api_key="{APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature}"'
        authorization = base64.b64encode(auth_origin.encode('utf-8')).decode(encoding='utf-8')
        return f"https://{host}{path}?" + urlencode({"host": host, "date": date, "authorization": authorization})

    def verify_voice_cloud(self):
        url = self.get_vpr_url()
        audio_raw = b"".join(list(self.audio_buffer))
        body = {
            "header": {"app_id": APPID, "status": 3},
            "parameter": {
                "s782b4996": {
                    "func": "searchFea", "groupId": GROUP_ID, "topK": 1,
                    "searchFeaRes": {"encoding": "utf8", "compress": "raw", "format": "json"}
                }
            },
            "payload": {
                "resource": {
                    "encoding": "raw", "sample_rate": 16000, "channels": 1, "bit_depth": 16, "status": 3,
                    "audio": base64.b64encode(audio_raw).decode('utf-8')
                }
            }
        }
        try:
            res = requests.post(url, data=json.dumps(body), headers={'content-type': "application/json"},
                                timeout=5).json()
            if res['header']['code'] == 0:
                text_b64 = res['payload']['searchFeaRes']['text']
                vpr_res = json.loads(base64.b64decode(text_b64).decode())
                if vpr_res.get('scoreList'):
                    return vpr_res['scoreList'][0]
        except Exception:
            pass
        return None

    def on_message(self, ws, message):
        msg = json.loads(message)
        if msg["header"]["code"] != 0:
            print(f"\n[错误]: {msg['header']['message']}")
            ws.close()
            return

        if "payload" in msg:
            result = msg["payload"]["result"]
            text_data = json.loads(base64.b64decode(result["text"]).decode('utf-8'))

            for i in text_data['ws']:
                for j in i["cw"]:
                    self.temp_text += j["w"]

            if self.temp_text:
                print(f"\r--- 正在倾听: '{self.temp_text}' ---", end="", flush=True)

            if msg["header"]["status"] == 2:
                final_text = self.temp_text.replace("。", "").replace("，", "").replace("？", "").strip()
                start_triggered = any(word in final_text for word in ["启动"])
                stop_triggered = any(word in final_text for word in ["关机"])
                relocate_triggered = "这里" in final_text

                if start_triggered or stop_triggered or relocate_triggered:
                    print(f"\n[系统]: 检测到关键词，正在调取声纹波形与方位数据...")
                    match = self.verify_voice_cloud()
                    if match:
                        confidence = match['score']
                        target_id = int(match['featureId'])
                        user_name = names.get(target_id, f"VIP_{target_id}")
                        if confidence > 0.1:
                            # 发送格式：START:ID:水平:垂直
                            if start_triggered and not self.is_visual_running:
                                print(f"【验证通过】VIP: {user_name} (H:{self.last_h_val:.1f}, V:{self.last_v_val:.1f}) -> 执行开启")
                                sock.sendto(f"START:{target_id}:{self.last_h_val}:{self.last_v_val}".encode(), (WSL_IP, UDP_PORT))
                                self.is_visual_running = True
                                self.current_vip_id = target_id
                            elif relocate_triggered and self.is_visual_running:
                                if target_id == self.current_vip_id:
                                    print(f"【定位请求】VIP: {user_name} (H:{self.last_h_val:.1f}, V:{self.last_v_val:.1f}) -> 重新盲寻")
                                    sock.sendto(f"START:{target_id}:{self.last_h_val}:{self.last_v_val}".encode(), (WSL_IP, UDP_PORT))
                                else:
                                    print(f"【权限拒绝】非当前操作者 {user_name}，不予执行重定位")
                            elif stop_triggered and self.is_visual_running:
                                if target_id == self.current_vip_id:
                                    print(f"【验证通过】VIP: {user_name} -> 执行关闭")
                                    sock.sendto("STOP".encode(), (WSL_IP, UDP_PORT))
                                    self.is_visual_running = False
                                    self.current_vip_id = None
                                else:
                                    owner_name = names.get(self.current_vip_id, "原开启者")
                                    print(f"【权限拒绝】当前由 {owner_name} 运行，{user_name} 无权干预。")
                        else:
                            print(f"【验证失败】声纹特征不匹配 (置信度: {confidence:.2f})")
                    else:
                        print(">>> 声纹识别未返回结果")

                elif self.is_visual_running and len(final_text) > 1:
                    print(f"\n[对话转发]: {final_text}")
                    sock.sendto(final_text.encode(), ("127.0.0.1", GEMMA_VOICE_PORT))
                    # 发送格式：DIR:水平:垂直
                    sock.sendto(f"DIR:{self.last_h_val}:{self.last_v_val}".encode(), (WSL_IP, UDP_PORT))

                self.temp_text = ""
                ws.close()

    def on_open(self, ws):
        def run(*args):
            p = pyaudio.PyAudio()
            stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                            input=True, frames_per_buffer=CHUNK)
            seq = 0
            SUBFRAME = 160
            try:
                while ws.sock and ws.sock.connected:
                    raw_data = stream.read(CHUNK, exception_on_overflow=False)
                    data_np = np.frombuffer(raw_data, dtype=np.int16).reshape(-1, CHANNELS)

                    # 分离四通道（假设硬件顺序：0上,1下,2左,3右）
                    ch_up = data_np[:, 0]
                    ch_down = data_np[:, 1]
                    ch_left = data_np[:, 2]
                    ch_right = data_np[:, 3]

                    # --- 动态选择能量最强的通道用于识别 ---
                    energies = [
                        np.max(np.abs(ch_up)),
                        np.max(np.abs(ch_down)),
                        np.max(np.abs(ch_left)),
                        np.max(np.abs(ch_right))
                    ]
                    best_idx = np.argmax(energies)
                    if best_idx == 0:
                        best_ch = ch_up
                    elif best_idx == 1:
                        best_ch = ch_down
                    elif best_idx == 2:
                        best_ch = ch_left
                    else:
                        best_ch = ch_right

                    # --- 水平定位（左-右）---
                    l_f = voice_filter(ch_left)
                    if np.max(np.abs(l_f)) > 600:
                        l_up = resample(l_f, len(l_f) * UPSAMPLE_FACTOR)
                        r_f = voice_filter(ch_right)
                        r_up = resample(r_f, len(r_f) * UPSAMPLE_FACTOR)
                        corr = correlate(l_up, r_up, mode='full')
                        delay_h = np.argmax(corr) - (len(l_up) - 1)
                        voice_history_h.append(delay_h)
                        if len(voice_history_h) == WINDOW_SIZE:
                            self.last_h_val = np.median(voice_history_h)

                    # --- 垂直定位（上-下）---
                    u_f = voice_filter(ch_up)
                    if np.max(np.abs(u_f)) > 600:
                        u_up = resample(u_f, len(u_f) * UPSAMPLE_FACTOR)
                        d_f = voice_filter(ch_down)
                        d_up = resample(d_f, len(d_f) * UPSAMPLE_FACTOR)
                        corr = correlate(u_up, d_up, mode='full')
                        delay_v = np.argmax(corr) - (len(u_up) - 1)
                        voice_history_v.append(delay_v)
                        if len(voice_history_v) == WINDOW_SIZE:
                            self.last_v_val = np.median(voice_history_v)

                    # ===== AEC 处理（使用最佳通道 best_ch）=====
                    subframes = [best_ch[i:i+SUBFRAME] for i in range(0, len(best_ch), SUBFRAME)]
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

                    # ===== 后处理谱减法（启用，增强AEC效果） =====
                    if USE_RESIDUAL_SUPPRESSION:
                        processed = residual_suppression(processed, RATE,
                                                          alpha=RESIDUAL_ALPHA,
                                                          beta=RESIDUAL_BETA)

                    # 维护滑动窗口（用于声纹识别）
                    self.audio_buffer.append(processed.tobytes())
                    if len(self.audio_buffer) > self.max_buffer_len:
                        self.audio_buffer.pop(0)

                    # 发送处理后的音频给讯飞识别
                    audio_b64 = base64.b64encode(processed.tobytes()).decode('utf-8')
                    if seq == 0:
                        frame = {
                            "header": {"app_id": APPID, "status": 0},
                            "parameter": {
                                "iat": {"domain": "slm", "language": "zh_cn", "accent": "mandarin", "vinfo": 1,
                                        "dwa": "wpgs",
                                        "result": {"encoding": "utf8", "compress": "raw", "format": "json"}}
                            },
                            "payload": {
                                "audio": {"encoding": "raw", "sample_rate": 16000, "channels": 1, "bit_depth": 16,
                                          "seq": seq, "status": 0, "audio": audio_b64}
                            }
                        }
                    else:
                        frame = {
                            "header": {"app_id": APPID, "status": 1},
                            "payload": {
                                "audio": {"encoding": "raw", "sample_rate": 16000, "channels": 1, "bit_depth": 16,
                                          "seq": seq, "status": 1, "audio": audio_b64}
                            }
                        }
                    ws.send(json.dumps(frame))
                    seq += 1
                    time.sleep(0.04)
            except websocket.WebSocketConnectionClosedException:
                # 连接已关闭，正常退出
                pass
            except Exception as e:
                print(f"音频发送线程异常: {e}")
            finally:
                stream.stop_stream()
                stream.close()
                p.terminate()

        thread.start_new_thread(run, ())

    def start(self):
        print("\n" + "=" * 40)
        print("   Morpheus 语音指挥中心 [四通道定位版]")
        print("   唤醒词: '启动' | 休眠词: '关机'")
        print("=" * 40)
        while not self.exit_flag:
            try:
                ws_url = self.get_asr_url()
                ws = websocket.WebSocketApp(ws_url,
                                            on_message=self.on_message,
                                            on_error=lambda w, e: None,
                                            on_close=lambda w, a, b: None)
                ws.on_open = self.on_open
                ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
                time.sleep(0.3)
            except KeyboardInterrupt:
                self.exit_flag = True
                print("\n>>> 听觉大脑已安全退出。")


if __name__ == "__main__":
    morpheus = Morpheus_Ear_System()
    morpheus.start()