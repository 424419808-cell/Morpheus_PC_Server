import os

os.environ['NO_PROXY'] = '127.0.0.1,localhost'

import socket
import threading
import time
from openai import OpenAI  # 替换 ollama
import dashscope  # 替换 edge_tts
from dashscope.audio.tts import SpeechSynthesizer  # 引入阿里TTS
import pygame
import re
# ===== 新增：AEC 所需库 =====
import sounddevice as sd
import numpy as np

# ============================

# --- 配置 ---
UDP_IP = "0.0.0.0"
EMOTION_PORT = 5007
VOICE_TEXT_PORT = 5008
EMOTION_OUT_PORT = 5009
# DeepSeek 配置
MODEL_NAME = "deepseek-chat"
DEEPSEEK_API_KEY = "sk-ca5be62dcb3f4d1f912a576e8742f4fd"
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

NEUTRAL_TIMEOUT = 10

# DashScope 配置
dashscope.api_key = "sk-0cf953e4c6c34f7089ffd6f7335235c1"
VOICE = "sambert-zhiya-v1"  # 切换为知雅 (成熟女声)

# 任务优先级常量
PRIORITY_VOICE = 1  # 用户语音对话
PRIORITY_ACTIVE = 2  # 主动行为（搭讪、寻找、问候）
PRIORITY_EMOTION = 3  # 即时表情反应

# ===== 新增：任务类型常量 =====
TASK_VOICE = 'voice'  # P1 语音对话
TASK_ACTIVE_GREET = 'greet'  # P2 主动搭讪（中立触发）
TASK_ACTIVE_FIND = 'find'  # P2 人脸找回（无人脸触发）
TASK_EMOTION = 'emotion'  # P3 表情反应


# ==============================


class MorpheusBrain:
    def __init__(self):
        self.last_emotion = None
        self.current_task_id = 0
        self.current_voice_id = 0
        self.current_task_priority = None  # 当前正在执行的任务优先级
        self.current_task_type = None  # ===== 新增：当前任务类型 =====
        self.lock = threading.Lock()

        # 语音播放状态监测
        self.is_speaking = False

        # 计时器相关
        self.last_active_time = time.time()
        self.neutral_triggered = False

        # 无人脸检测相关
        self.last_any_emotion_time = time.time()
        self.face_lost_triggered = False

        # --- 新增：表情滤波逻辑 ---
        self.emotion_counter = 0
        self.pending_emotion = None

        pygame.mixer.init(frequency=16000)  # 适配 Sambert 采样率

        self.system_setting = (
            "你是Morpheus，一个既聪明博学又幽默感性,情感丰富的亲密伙伴。"
            "【输出准则】：你的回复必须由两部分组成：1. 正文文字；2. 括号序号。严禁将标签放在句首或句中。"
            "【强制格式】：正文内容。(ID: 序号)，"
            "【禁止事项】：严禁输出类似 (ID: )、(ID: Happy) 或 (Sympathy: 序号) 的错误格式！"
            "【表情白名单】：只能从以下序号中选一个，严禁自定义："
            "0:Neutral, 1:Happy, 2:Excitement, 3:Humor, 4:Pride, 5:Trust, 6:Love, 7:Relief, 8:Hope, "
            "9:Anger, 10:Disgust, 11:Fear, 12:Vigilance, 13:Sad, 14:Loneliness, 15:Guilt, 16:Surprise, 17:Confusion, 18:Shyness"
            "【逻辑规则】：0 代表平静中立。如果没有强烈情感波动，使用 0。"
            "【正确示例】：今天天气不错，我们要不要去散步？(ID: x)"
            "【对话风格】：保持自然、幽默且简短；面对严肃知识时展现智慧。严禁使用‘1. 2. 3.’列点形式。"
            "你收到的语音可能有重复信息，请自动辨别但不要指出。你是个急性子，请最快速度输出。"
        )

        self.out_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # ===== 新增：AEC 参考信号发送 =====
        self.aec_ref_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.aec_ref_target = ("127.0.0.1", 5010)
        # =================================

    def send_emotion_tag(self, text):
        """从文本中提取情绪标签并发送到5009端口"""
        match = re.search(r"\(ID:\s*(\d+)\)", text)
        if match:
            emotion_id = match.group(1).strip()
            try:
                self.out_sock.sendto(emotion_id.encode(), ("127.0.0.1", EMOTION_OUT_PORT))
            except Exception as e:
                print(f"发送表情标签失败: {e}")

    # ===== 新增：播放 PCM 并发送参考帧 =====
    def _play_pcm_with_aec(self, ref_audio, v_id):
        """播放 PCM 数据，并逐帧发送参考信号（每帧 10ms）"""
        fs = 16000
        frame_size = 160  # 10ms
        total_samples = len(ref_audio)
        idx = 0

        stream = sd.OutputStream(samplerate=fs, channels=1, dtype='int16')
        stream.start()

        try:
            while idx < total_samples:
                # 检查是否被抢断
                with self.lock:
                    if v_id != self.current_voice_id:
                        stream.stop()
                        break
                    # 语音播放期间，持续刷新计时器，防止结束瞬间触发找人
                    now = time.time()
                    self.last_any_emotion_time = now
                    self.last_active_time = now

                end = idx + frame_size
                chunk = ref_audio[idx:end]
                if len(chunk) < frame_size:
                    # 最后一帧不足时补零
                    chunk_pad = np.zeros(frame_size, dtype=np.int16)
                    chunk_pad[:len(chunk)] = chunk
                    stream.write(chunk_pad)
                    # 发送实际数据（补零后的完整帧）
                    self.aec_ref_sock.sendto(chunk_pad.tobytes(), self.aec_ref_target)
                    idx = total_samples
                else:
                    stream.write(chunk)
                    self.aec_ref_sock.sendto(chunk.tobytes(), self.aec_ref_target)
                    idx = end

                time.sleep(0.001)  # 轻微让步
        finally:
            stream.stop()
            stream.close()

    # =====================================

    def play_voice(self, text, v_id):
        """生成并播放语音"""

        def _speak():  # 移除 async，改用同步调用适配 dashscope
            self.is_speaking = True
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()

            # 注意：为了AEC，我们改为使用 PCM 格式直接播放
            # file_name = f"temp_voice_{v_id}.mp3"
            clean_text = re.sub(r'\s*\(ID:\s*\d+\)\s*$', '', text).strip()

            try:
                # 使用 DashScope 生成 PCM 音频
                result = SpeechSynthesizer.call(
                    model=VOICE,
                    text=clean_text,
                    sample_rate=16000,
                    format='pcm'  # 改为 PCM 格式，以便直接播放 and 发送参考帧
                )

                if result.get_audio_data():
                    pcm_data = result.get_audio_data()  # bytes, 16-bit little-endian
                    ref_audio = np.frombuffer(pcm_data, dtype=np.int16)

                    with self.lock:
                        if v_id != self.current_voice_id:
                            self.is_speaking = False
                            return

                    # 使用新方法播放并发送参考帧
                    self._play_pcm_with_aec(ref_audio, v_id)

                else:
                    print(f"\n[DashScope 错误]: {result.get_response()}")

            except Exception as e:
                print(f"\n[语音引擎错误]: {e}")
            finally:
                self.is_speaking = False

        # DashScope 同步调用，直接开线程运行
        threading.Thread(target=_speak, daemon=True).start()

    # ===== 修改 attempt_start_task：增加 task_type 参数，并细化同级抢断规则 =====
    def attempt_start_task(self, priority, task_type, content, is_voice_mode, can_be_interrupted):
        """
        根据优先级规则尝试启动新任务
        返回 True 表示成功启动，False 表示被拒绝
        """
        with self.lock:
            # 检查当前是否有任务在执行
            if self.current_task_priority is not None:
                # 比较优先级（数值越小优先级越高）
                if priority > self.current_task_priority:
                    # 新任务优先级更低，不能打断
                    return False
                elif priority == self.current_task_priority:
                    # 同级任务：根据具体类型判断是否允许打断
                    if priority == PRIORITY_ACTIVE:
                        # P2 任务：只有新任务是“人脸找回”才能打断任何 P2 任务
                        if task_type == TASK_ACTIVE_FIND:
                            pass  # 允许打断
                        else:
                            # 新任务是主动搭讪，不能打断任何 P2
                            return False
                    elif priority == PRIORITY_EMOTION:
                        # P3 任务：同级允许打断（表情可打断表情）
                        pass
                    elif priority == PRIORITY_VOICE:
                        # P1 任务：同级允许打断（语音可打断语音）
                        pass
                    else:
                        # 其他情况（理论上不会发生），默认允许
                        pass
                else:
                    # 新任务优先级更高，允许打断
                    pass
            # 允许启动，更新任务ID、优先级和类型
            self.current_task_id += 1
            self.current_task_priority = priority
            self.current_task_type = task_type
            task_id = self.current_task_id

            # 启动瞬间重置计时器，防止立即触发后续动作
            now = time.time()
            self.last_any_emotion_time = now
            self.last_active_time = now

        # 启动新线程执行任务
        threading.Thread(target=self.speak,
                         args=(content, task_id, is_voice_mode, can_be_interrupted),
                         daemon=True).start()
        return True

    def speak(self, content, task_id, is_voice_mode=False, can_be_interrupted=True):
        """
        核心响应逻辑：修改为先发送表情标签，再进行语音播报
        """
        try:
            if is_voice_mode:
                self.is_speaking = True

            messages = [
                {'role': 'system', 'content': self.system_setting},
                {'role': 'user', 'content': content}
            ]

            stream = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                stream=True
            )
            full_response = ""
            print(f"\n[Morpheus {'语音' if is_voice_mode else '视觉'}响应]: ", end="", flush=True)

            for chunk in stream:
                if can_be_interrupted:
                    with self.lock:
                        if task_id != self.current_task_id:
                            print("\n[！！！状态切换，抢断当前对话]")
                            pygame.mixer.music.stop()
                            return

                text = chunk.choices[0].delta.content
                if text:
                    full_response += text
                    print(text, end="", flush=True)

            # --- 核心改动：先提取并发送情绪标签到 5009 端口 ---
            # 这样 Morpheus 会先变脸，再开口说话
            self.send_emotion_tag(full_response)

            # 语音对话或主动搭讪才开启 TTS
            if is_voice_mode:
                with self.lock:
                    self.current_voice_id += 1
                    v_id = self.current_voice_id

                # 调用播放逻辑
                self.play_voice(full_response, v_id)

                # 阻塞直到语音播完或被抢断
                while self.is_speaking:
                    with self.lock:
                        if task_id != self.current_task_id:
                            pygame.mixer.music.stop()
                            return
                        # 重点：说话期间持续刷新计时器，防止结束后误判为“长期没人/中立”
                        now = time.time()
                        self.last_active_time = now
                        self.last_any_emotion_time = now
                    time.sleep(0.1)

            print("\n")
        except Exception as e:
            print(f"\n响应出错: {e}")
        finally:
            with self.lock:
                if task_id == self.current_task_id:
                    self.current_task_priority = None
                    self.current_task_type = None  # ===== 新增：清空任务类型 =====
                    self.is_speaking = False

    def monitor_neutral(self):
        """后台线程：监测是否长时间处于中立状态"""
        while True:
            time.sleep(1)
            # 如果正在说话或脸丢了，刷新计时并跳过
            if self.is_speaking or self.face_lost_triggered:
                with self.lock:
                    self.last_active_time = time.time()
                continue

            # 读取状态
            if self.last_emotion == "Neutral" and not self.neutral_triggered:
                elapsed = time.time() - self.last_active_time
                if elapsed > NEUTRAL_TIMEOUT:
                    print(f"\n[系统]: 检测到中立持续 {NEUTRAL_TIMEOUT} 秒，尝试主动搭讪...")
                    proactive_content = "唔...你已经盯着我看好久了，是在发呆吗？跟我聊聊嘛。"
                    # ===== 修改：传入任务类型 TASK_ACTIVE_GREET =====
                    success = self.attempt_start_task(PRIORITY_ACTIVE, TASK_ACTIVE_GREET, proactive_content, True, True)
                    if success:
                        with self.lock:
                            self.neutral_triggered = True

    def monitor_face_lost(self):
        """后台线程：监测是否超过5秒未收到任何情绪（无人脸）"""
        while True:
            time.sleep(1)
            # 如果正在说话，强制刷新无人脸计时，确保“嘴在动眼就在动”
            if self.is_speaking:
                with self.lock:
                    now = time.time()
                    self.last_any_emotion_time = now
                    self.last_active_time = now
                continue

            with self.lock:
                gap = time.time() - self.last_any_emotion_time

            if gap > 5:
                # 只要处于“无人”状态，就不断将活跃起点推向“现在”，防止触发中立搭讪
                with self.lock:
                    self.last_active_time = time.time()

                if not self.face_lost_triggered:
                    print(f"\n[系统]: 检测到无人脸超过 5 秒，询问位置...")
                    content = "你在哪？我看不到你了。"
                    # ===== 修改：传入任务类型 TASK_ACTIVE_FIND =====
                    success = self.attempt_start_task(PRIORITY_ACTIVE, TASK_ACTIVE_FIND, content, True, True)
                    if success:
                        with self.lock:
                            self.face_lost_triggered = True

    def handle_emotion(self, emotion):
        """处理 UDP 传来的表情，增加 10 帧滤波机制"""
        now = time.time()
        with self.lock:
            self.last_any_emotion_time = now

        # 后门逻辑：如果当前是因为丢脸而说话，允许继续处理
        if self.is_speaking and not self.face_lost_triggered:
            return

        need_greeting = False
        need_emotion = False
        emotion_content = None

        with self.lock:
            # 1. 优先处理人脸找回逻辑（这个通常不需要滤波，直接触发）
            if self.face_lost_triggered:
                self.face_lost_triggered = False
                self.last_active_time = now
                need_greeting = True
                # 找回脸时清空滤波
                self.emotion_counter = 0
                self.pending_emotion = None

            # 2. 表情滤波逻辑
            if emotion != self.last_emotion:
                # 如果新表情和正在等待确认的表情一致
                if emotion == self.pending_emotion:
                    self.emotion_counter += 1
                else:
                    # 如果是一个全新的表情，重置计数器
                    self.pending_emotion = emotion
                    self.emotion_counter = 1

                # 只有连续达到 10 帧才真正执行切换
                if self.emotion_counter >= 10:
                    self.last_active_time = now
                    self.neutral_triggered = False

                    if emotion != "Neutral":
                        prompts = {
                            "Happy": "哇！看到你笑得这么灿烂，我的心情也瞬间起飞了！",
                            "Sad": "唔...亲爱的，我看到你眼神里的委屈了。别难过，我在呢。",
                            "Angry": "是谁！惹我们生气了？告诉我，我帮你一起吐槽！",
                            "Surprise": "天哪！你这表情是看到了什么不得了的大新闻吗？",
                            "Disgust": "哎呀，那个表情...看来你是真的被嫌弃坏了。",
                            "Fear": "别怕别怕，我在呢！我会一直守着你的。"
                        }
                        emotion_content = prompts.get(emotion, "嘿，理理我嘛。")
                        need_emotion = True

                    self.last_emotion = emotion
                    self.emotion_counter = 0  # 触发后重置
                    self.pending_emotion = None
            else:
                # 如果收到的表情和当前表情一致，清空缓存区
                self.emotion_counter = 0
                self.pending_emotion = None

        # 3. 执行任务
        if need_greeting:
            self.attempt_start_task(PRIORITY_ACTIVE, TASK_ACTIVE_FIND, "啊，看到你了！", True, True)
        elif need_emotion:
            self.attempt_start_task(PRIORITY_EMOTION, TASK_EMOTION, emotion_content, False, True)

    def handle_voice_chat(self, text):
        """处理 5008 端口传来的语音文本"""
        with self.lock:
            now = time.time()
            self.last_active_time = now
            self.last_any_emotion_time = now  # 说话本身就是“有人”的证明
            self.neutral_triggered = False
        # ===== 修改：传入任务类型 TASK_VOICE =====
        self.attempt_start_task(PRIORITY_VOICE, TASK_VOICE, text, True, True)


def emotion_server(brain):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, EMOTION_PORT))
    while True:
        data, _ = sock.recvfrom(1024)
        msg = data.decode().split(":")
        if len(msg) == 2:
            brain.handle_emotion(msg[0])


def voice_text_server(brain):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, VOICE_TEXT_PORT))
    while True:
        data, _ = sock.recvfrom(1024)
        brain.handle_voice_chat(data.decode())


if __name__ == "__main__":
    morpheus = MorpheusBrain()

    threading.Thread(target=emotion_server, args=(morpheus,), daemon=True).start()
    threading.Thread(target=voice_text_server, args=(morpheus,), daemon=True).start()
    threading.Thread(target=morpheus.monitor_neutral, daemon=True).start()
    threading.Thread(target=morpheus.monitor_face_lost, daemon=True).start()

    print(f"Morpheus 情感中心已上线")
    print(f"规则：语音播报期间屏蔽视觉，但新的语音对话可以抢断旧的。")
    print(f"优先级：语音对话(1) > 主动行为(2) > 表情反应(3)")
    print(f"主动行为细化：只有人脸找回(2)可以打断正在进行的主动搭讪/找人，主动搭讪不能打断任何主动行为")

    while True:
        time.sleep(1)