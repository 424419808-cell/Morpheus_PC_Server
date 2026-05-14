"""
实时共情推理 — MediaPipe BS → BS2BS_Empathy → 双窗口展示

用法:
  python exp_bs/infer/run_empathy_live.py                            # 实时共情 + 预览窗口（默认）
  python exp_bs/infer/run_empathy_live.py --no-viewer                # 纯控制台模式（无窗口）
  python exp_bs/infer/run_empathy_live.py --blender-render           # 实时共情 + Blender 窗口化渲染
  python exp_bs/infer/run_empathy_live.py --blender-render --blender-background  # 旧模式：后台 PNG
  python exp_bs/infer/run_empathy_live.py --blender-render --blender-res 600     # 较小窗口

窗口说明（--blender-render 模式）：
  - 窗口 1 (OpenCV): 摄像头画面 + 用户情绪标注 + 共情情绪百分比 + BS 条形图
  - 窗口 2 (Blender): EEVEE RENDERED 视口实时显示共情数字人 + 3D 情绪文字叠加
"""
import argparse
import os
import sys
import time

import cv2
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.join(BASE_DIR, "..", "..")
sys.path.insert(0, PROJ_ROOT)

try:
    from utils.blender_live_renderer import BlenderLiveRenderer
    HAS_BLENDER = True
except ImportError:
    HAS_BLENDER = False

MODEL_PATH = os.path.join(PROJ_ROOT, "exp_bs", "models", "bs2bs_empathy.pth")
LANDMARKER_PATH = os.path.join(PROJ_ROOT, "face_landmarker.task")
EMOTION_MODEL_PATH = os.path.join(PROJ_ROOT, "emo_clf", "models", "emotion_model.pkl")

# 24 种情绪名称（与 emotion_model.pkl 的 classes_ 顺序一致）
EMOTIONS = [
    "Neutral", "Happy", "Excitement", "Humor", "Pride",
    "Trust", "Love", "Relief", "Hope",
    "Anger", "Disgust", "Fear", "Vigilance",
    "Sad", "Loneliness", "Guilt",
    "Surprise", "Confusion", "Shyness",
    "Comfort", "Playful", "Impressed", "Concerned", "Awkward",
]

# 中文情绪映射（用于显示）
EMOTION_CN = {
    "Neutral": "中性", "Happy": "开心", "Excitement": "兴奋", "Humor": "幽默", "Pride": "自豪",
    "Trust": "信任", "Love": "爱", "Relief": "宽慰", "Hope": "希望",
    "Anger": "愤怒", "Disgust": "厌恶", "Fear": "恐惧", "Vigilance": "警惕",
    "Sad": "悲伤", "Loneliness": "孤独", "Guilt": "内疚",
    "Surprise": "惊讶", "Confusion": "困惑", "Shyness": "害羞",
    "Comfort": "舒适", "Playful": "调皮", "Impressed": "佩服", "Concerned": "关切", "Awkward": "尴尬",
}

# ============================================================
# 中文文字渲染（PIL 替代 cv2.putText）
# ============================================================
FONT_PATH = "C:/Windows/Fonts/msyh.ttc"
_cn_texts = []  # 每帧收集: (x, y, text, font_size, color_bgr)

def enqueue_cn_text(x, y, text, font_size, color_bgr):
    """收集中文文字，帧末统一渲染"""
    _cn_texts.append((x, y, text, font_size, color_bgr))

def flush_cn_texts(canvas):
    """在 canvas（BGR ndarray）上统一渲染所有收集的中文文字"""
    global _cn_texts
    if not _cn_texts:
        return

    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # 按 font_size 分组缓存字体
    needed_sizes = {s for _, _, _, s, _ in _cn_texts}
    font_cache = {}
    for sz in needed_sizes:
        try:
            font_cache[sz] = ImageFont.truetype(FONT_PATH, sz)
        except Exception:
            font_cache[sz] = ImageFont.load_default()

    for x, y, text, size, color_bgr in _cn_texts:
        color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        font = font_cache[size]
        # 'ls' anchor = left-baseline（与 cv2.putText 坐标一致）
        draw.text((int(x), int(y)), text, font=font, fill=color_rgb, anchor='ls')

    canvas[:, :] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    _cn_texts = []


class BS2BS_Empathy(torch.nn.Module):
    """BS→BS 共情 MLP — 52维 blendshape → 52维共情 blendshape"""
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(52, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 52), torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def load_models(device):
    model = BS2BS_Empathy().to(device)
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=device)
        if not any(k.startswith("net.") for k in state_dict.keys()):
            state_dict = {"net." + k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f"[OK] 加载共情模型: {MODEL_PATH}")
    else:
        print(f"[警告] 未找到共情模型: {MODEL_PATH}，将使用恒等映射")

    model.eval()
    return model


def format_emotion_text(probs, emotions, top_n=3, threshold=0.05):
    """
    格式化共情情绪文字，用于 Blender 3D 文字叠加。
    返回多行字符串，如：
        共情
        Happy 62%
        Excitement 28%
    """
    indices = np.argsort(probs)[::-1]
    items = []
    for i in indices:
        if len(items) >= top_n:
            break
        if probs[i] >= threshold:
            items.append(f"{emotions[i]} {probs[i]*100:.0f}%")

    if not items:
        return "等待..."

    # 首行加 "共情" 标题
    return "共情\n" + "\n".join(items)


def format_emotion_line(probs, emotions, top_n=3, threshold=0.05):
    """
    格式化一行情绪文字，用于 OpenCV 状态栏。
    如 "Happy 62% | Excitement 28% | Neutral 10%"
    """
    indices = np.argsort(probs)[::-1]
    parts = []
    for i in indices:
        if len(parts) >= top_n:
            break
        if probs[i] >= threshold:
            cn_name = EMOTION_CN.get(emotions[i], emotions[i])
            parts.append(f"{cn_name} {probs[i]*100:.0f}%")
    return "  |  ".join(parts) if parts else "等待中..."


def main():
    parser = argparse.ArgumentParser(description="实时共情推理")
    parser.add_argument("--no-viewer", action="store_true", help="不显示预览窗口（纯控制台模式）")
    parser.add_argument("--blender-render", action="store_true", help="Blender 窗口化实时渲染预览")
    parser.add_argument("--blender-background", action="store_true",
                        help="Blender 后台 PNG 模式（旧模式，不推荐）")
    parser.add_argument("--blender-engine", default="BLENDER_EEVEE",
                        choices=["BLENDER_EEVEE", "CYCLES"], help="Blender 渲染引擎（仅后台模式有效）")
    parser.add_argument("--blender-res", type=int, default=800,
                        help="Blender 窗口分辨率 (默认 800)")
    parser.add_argument("--no-udp", action="store_true", help="不发送 UDP")
    parser.add_argument("--cam", type=int, default=1, help="摄像头 ID (默认 1=电脑摄像头)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载模型
    model = load_models(device)

    # MediaPipe
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=LANDMARKER_PATH),
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    # 情绪分类器
    rf_classifier = None
    try:
        import joblib
        import pandas as pd
        rf_classifier = joblib.load(EMOTION_MODEL_PATH)
        print(f"[OK] 加载情绪分类器: {EMOTION_MODEL_PATH}")
        print(f"  情绪类别: {len(rf_classifier.classes_)} 种")
    except Exception as e:
        print(f"[信息] 未加载情绪分类器: {e}")

    # 摄像头
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[错误] 无法打开摄像头 {args.cam}")
        sys.exit(1)

    # ---- Blender 渲染器 ----
    blender_renderer = None
    if args.blender_render:
        if not HAS_BLENDER:
            print("[错误] 找不到 utils/blender_live_renderer.py")
            sys.exit(1)

        windowed = not args.blender_background
        try:
            blender_renderer = BlenderLiveRenderer(
                width=args.blender_res, height=args.blender_res,
                engine=args.blender_engine,
                windowed=windowed,
            )
            blender_renderer.start()
            mode_str = "窗口化" if windowed else "后台PNG"
            print(f"[OK] Blender {mode_str}渲染已启动 ({args.blender_res}x{args.blender_res})")
            time.sleep(3)  # 等待 Blender 加载模型
        except Exception as e:
            print(f"[错误] 启动 Blender 失败: {e}")
            sys.exit(1)

    print("\n=== 实时共情推理运行中 ===")
    print("  摄像头 → MediaPipe 52 BS → BS2BS_Empathy → 共情 BS")
    if blender_renderer:
        print("  窗口 1: 摄像头 + 情绪标注 (OpenCV)")
        print("  窗口 2: 共情数字人 (Blender EEVEE)")
    print("  按 Q 退出\n")

    # 平滑滤波
    smooth_bs = np.zeros(52, dtype=np.float32)

    # 面部区域索引
    BROWS = [1, 2, 3, 4, 5]
    EYES = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    JAW = [23, 24, 25, 26]
    MOUTH = [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    NOSE = [51, 52]
    REGION_COLORS = {
        "brows": (78, 196, 196), "eyes": (109, 230, 255),
        "jaw": (107, 107, 255), "mouth": (179, 225, 149), "nose": (129, 129, 243),
    }

    def _region_color(i):
        if i in BROWS:
            return REGION_COLORS["brows"]
        if i in EYES:
            return REGION_COLORS["eyes"]
        if i in JAW:
            return REGION_COLORS["jaw"]
        if i in MOUTH:
            return REGION_COLORS["mouth"]
        return REGION_COLORS["nose"]

    def draw_bs_bars(canvas, bs_values, title, offset_x, offset_y, width, height):
        cv2.putText(canvas, title, (offset_x + 10, offset_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        bar_w = max(4, (width - 40) // 52)
        for i in range(52):
            bar_h = int(bs_values[i] * (height - 40))
            x = offset_x + 20 + i * bar_w
            y = offset_y + height - 20 - bar_h
            cv2.rectangle(canvas, (x, y), (x + bar_w - 1, offset_y + height - 20), _region_color(i), -1)
        cv2.line(canvas, (offset_x + 20, offset_y + height - 20),
                 (offset_x + 20 + 52 * bar_w, offset_y + height - 20), (100, 100, 100), 1)

    fps_counter = 0
    fps_time = time.time()
    fps = 0

    # 共情情绪缓存（用于无检测时的持续显示）
    last_empathy_probs = None
    last_empathy_line = "等待中..."

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # FPS
            fps_counter += 1
            now = time.time()
            if now - fps_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_time = now

            # MediaPipe
            timestamp = int(time.time() * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, timestamp)

            # 默认值
            user_emotion_cn = "?"
            user_bs = np.zeros(52, dtype=np.float32)
            empathy_probs = None
            empathy_line = ""
            diff = 0.0
            active_in = 0.0
            active_out = 0.0
            face_detected = False

            if result.face_blendshapes:
                face_detected = True
                user_bs = np.array([b.score for b in result.face_blendshapes[0]], dtype=np.float32)

                # BS2BS_Empathy 推理
                with torch.no_grad():
                    tensor = torch.tensor([user_bs], dtype=torch.float32, device=device)
                    empathy_bs = model(tensor).cpu().numpy()[0]

                smooth_bs = smooth_bs * 0.7 + empathy_bs * 0.3

                # 用户情绪分类（基于 user_bs）
                if rf_classifier is not None:
                    features = pd.DataFrame([user_bs], columns=[f"bs_{i}" for i in range(52)])
                    user_emotion = rf_classifier.predict(features)[0]
                    user_emotion_cn = EMOTION_CN.get(user_emotion, user_emotion)

                    # 共情情绪分类（基于 empathy_bs / smooth_bs）
                    empathy_features = pd.DataFrame([smooth_bs], columns=[f"bs_{i}" for i in range(52)])
                    empathy_probs = rf_classifier.predict_proba(empathy_features)[0]
                    last_empathy_probs = empathy_probs
                    empathy_line = format_emotion_line(empathy_probs, rf_classifier.classes_)
                    last_empathy_line = empathy_line

                    # 共情情绪文字（多行，供 Blender 叠加）
                    empathy_multiline = format_emotion_text(empathy_probs, rf_classifier.classes_)
                else:
                    empathy_multiline = ""

                # 统计数据
                diff = float(np.mean(np.abs(empathy_bs - user_bs)))
                active_in = float(np.mean(user_bs[user_bs > 0.05])) if np.any(user_bs > 0.05) else 0.0
                active_out = float(np.mean(empathy_bs[empathy_bs > 0.05])) if np.any(empathy_bs > 0.05) else 0.0

                # ---- 发送到 Blender ----
                if blender_renderer and blender_renderer.is_alive():
                    blender_renderer.send_bs(smooth_bs, emotion_text=empathy_multiline)

                # 控制台输出
                print(f"\r用户[{user_emotion_cn:8}] | {empathy_line[:40]:40s} "
                      f"| 强度={active_out:.3f}" + " " * 10, end="", flush=True)

            else:
                print(f"\r[未检测到人脸]" + " " * 60, end="", flush=True)
                smooth_bs *= 0.9  # 衰减
                # 保持上次共情情绪显示
                if last_empathy_probs is not None:
                    empathy_line = last_empathy_line

            # --- 预览窗口 ---
            if not args.no_viewer:
                h, w = frame.shape[:2]
                disp_w, disp_h = 1280, 720
                canvas = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)

                # ======== 左面板：摄像头画面 ========
                cam_aspect = w / h
                cam_disp_w = disp_w // 2
                cam_disp_h = int(cam_disp_w / cam_aspect)
                if cam_disp_h > disp_h - 120:
                    cam_disp_h = disp_h - 120
                    cam_disp_w = int(cam_disp_h * cam_aspect)

                frame_resized = cv2.resize(frame, (cam_disp_w, cam_disp_h))
                canvas[10:10 + cam_disp_h, 10:10 + cam_disp_w] = frame_resized

                # 用户情绪标注（大号绿色文字叠加在画面上方）
                if face_detected:
                    enqueue_cn_text(20, 40, f"用户: {user_emotion_cn}", 26, (0, 230, 0))
                else:
                    enqueue_cn_text(20, 40, "未检测到人脸", 24, (80, 80, 80))

                # 摄像头标注
                cv2.putText(canvas, "Camera Feed", (20, cam_disp_h + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

                # ======== 右面板：共情情绪信息 + BS 条形图 ========
                right_x = disp_w // 2 + 10
                right_w = disp_w // 2 - 20

                # 共情情绪文字面板
                empathy_panel_h = 100
                # 背景框
                cv2.rectangle(canvas, (right_x, 10), (right_x + right_w, 10 + empathy_panel_h),
                              (30, 30, 30), -1)
                enqueue_cn_text(right_x + 10, 22, "共情情绪 (Empathy)", 14, (200, 200, 200))

                # 显示共情情绪百分比（大号青色文字）
                if face_detected and rf_classifier is not None:
                    enqueue_cn_text(right_x + 10, 55, empathy_line, 20, (0, 210, 255))
                else:
                    enqueue_cn_text(right_x + 10, 55, "等待人脸检测...", 18, (100, 100, 100))

                # BS 条形图
                bar_top = 10 + empathy_panel_h + 5
                bar_h = (disp_h - bar_top - 60) // 2

                if face_detected:
                    draw_bs_bars(canvas, user_bs, "User BS", right_x, bar_top, right_w, bar_h)
                    draw_bs_bars(canvas, smooth_bs, "Empathy BS", right_x,
                                 bar_top + bar_h + 5, right_w, bar_h)
                else:
                    cv2.putText(canvas, "No face detected", (right_x + 20, disp_h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)

                # ======== 底部状态栏 ========
                status_y = disp_h - 50
                cv2.rectangle(canvas, (0, status_y), (disp_w, disp_h), (40, 40, 40), -1)
                metrics = [
                    f"FPS: {fps}", f"Input: {active_in:.3f}",
                    f"Output: {active_out:.3f}", f"BS Diff: {diff:.4f}",
                ]
                for i, m in enumerate(metrics):
                    cv2.putText(canvas, m, (20 + i * 320, status_y + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                # 右侧状态栏提示
                if blender_renderer:
                    status_text = "Blender: 窗口化运行中" if blender_renderer.windowed else "Blender: 后台渲染中"
                    enqueue_cn_text(disp_w - 250, status_y + 25, status_text, 14, (100, 200, 100))

                # 统一渲染中文文字（PIL 单次转换）
                flush_cn_texts(canvas)

                cv2.imshow("Morpheus — 实时共情 (摄像头 + 共情分析)", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        if blender_renderer:
            blender_renderer.stop()
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
        print("程序结束。")


if __name__ == "__main__":
    main()
