"""
Blender 渲染验证 — 共情模型输出 → 数字人表情渲染

用法:
  python exp_bs/val/render_blender.py                    # 渲染所有情绪
  python exp_bs/val/render_blender.py --emotions Happy Sad  # 指定情绪
  python exp_bs/val/render_blender.py --engine EEVEE     # 快速渲染模式
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.join(BASE_DIR, "..", "..")
EMPATHY_MODEL_PATH = os.path.join(PROJ_ROOT, "exp_bs", "models", "bs2bs_empathy.pth")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "..", "models")

# Blender 配置 — 引用 blender_test/config.py
BLENDER_TEST_DIR = os.path.join(PROJ_ROOT, "..", "blender_test")
BLENDER_CONFIG = os.path.join(BLENDER_TEST_DIR, "config.py")

EMOTION_NAMES = [
    "Neutral", "Happy", "Excitement", "Humor", "Pride",
    "Trust", "Love", "Relief", "Hope",
    "Anger", "Disgust", "Fear", "Vigilance",
    "Sad", "Loneliness", "Guilt",
    "Surprise", "Confusion", "Shyness",
]

TEMP_DIR = os.path.join(BASE_DIR, "temp")
OUTPUT_DIR = os.path.join(BASE_DIR, "render_output")


def import_blender_config():
    """动态导入 blender_test 配置"""
    sys.path.insert(0, BLENDER_TEST_DIR)
    import config as blender_cfg
    return blender_cfg


class BS2BS_Empathy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(52, 128), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 256), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 52), torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def load_empathy_model(device):
    """加载共情模型，失败时回退到恒等映射"""
    model = BS2BS_Empathy().to(device)
    if os.path.exists(EMPATHY_MODEL_PATH):
        model.load_state_dict(torch.load(EMPATHY_MODEL_PATH, map_location=device))
        print(f"[OK] 已加载共情模型: {EMPATHY_MODEL_PATH}")
        return model
    else:
        print(f"[警告] 共情模型不存在: {EMPATHY_MODEL_PATH}")
        print("[警告] 将使用恒等映射（输入=输出），仅渲染原始 BS")
        return None


def get_base_bs(emotion_name):
    """从 gen_batch_data 获取模板 BS"""
    try:
        sys.path.insert(0, os.path.join(PROJ_ROOT, "exp_bs", "scripts"))
        from gen_batch_data import get_base_bs as _get_bs
        return _get_bs(emotion_name)
    except ImportError:
        print(f"[错误] 无法导入 gen_batch_data.get_base_bs")
        return np.zeros(52, dtype=np.float32)


def make_blendshape_dict(bs_52, arkit_names):
    """52 维 BS 数组 → ARKit 命名字典"""
    bd = {"_neutral": 0.0}
    for i in range(51):
        bd[arkit_names[i + 1]] = float(bs_52[i])
    return bd


def write_temp_blendshape(bs_dict, emotion_name, intensity=1.0):
    """写入临时 JSON 文件供 Blender 读取"""
    os.makedirs(TEMP_DIR, exist_ok=True)
    path = os.path.join(TEMP_DIR, f"bs_{emotion_name}.json")
    payload = {
        "emotion": emotion_name,
        "intensity": intensity,
        "blendshapes": bs_dict,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def render_with_blender(blender_exe, model_path, bs_json_path, output_path, engine="BLENDER_EEVEE"):
    """调用 Blender 渲染单帧"""
    render_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_blender_render_frame.py")
    if not os.path.exists(render_script):
        print(f"[错误] 渲染脚本不存在: {render_script}")
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = [
        blender_exe,
        "--background",
        model_path,
        "--python", render_script,
        "--",
        bs_json_path,
        output_path,
        engine,
    ]

    print(f"  [Blender] 渲染中...")
    start = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=BLENDER_TEST_DIR,
    )

    if result.stdout:
        for line in result.stdout.strip().splitlines()[-5:]:
            if "Error" in line or "Warning" in line:
                print(f"           {line}")

    if result.returncode != 0:
        print(f"  [错误] Blender 返回码 {result.returncode}")
        if result.stderr:
            for line in result.stderr.strip().splitlines()[-3:]:
                print(f"         {line}")
        return False

    elapsed = time.time() - start
    print(f"  [OK] {elapsed:.1f}s → {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="共情模型 Blender 渲染验证")
    parser.add_argument("--engine", default="BLENDER_EEVEE", choices=["CYCLES", "BLENDER_EEVEE"])
    parser.add_argument("--emotions", nargs="+", default=None, help="要渲染的情绪列表")
    parser.add_argument("--intensity", type=float, default=0.8, help="BS 强度缩放")
    parser.add_argument("--blender", type=str, default=None, help="Blender 可执行路径")
    args = parser.parse_args()

    # 加载 Blender 配置
    cfg = import_blender_config()
    blender_exe = args.blender or cfg.BLENDER_EXE
    model_path = cfg.MODEL_PATH
    arkit_names = cfg.ARKIT_BLENDSHAPE_NAMES

    if not os.path.exists(blender_exe):
        print(f"[错误] Blender 未找到: {blender_exe}")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"[错误] 3D 模型未找到: {model_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_empathy_model(device)

    emotions = args.emotions if args.emotions else EMOTION_NAMES
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n渲染引擎: {args.engine}")
    print(f"情绪数量: {len(emotions)}")
    print(f"BS 强度: {args.intensity}")
    print()

    passed = 0
    failed = 0

    for emotion in emotions:
        print(f"[{emotion}]")

        # 获取原始 BS
        raw_bs = get_base_bs(emotion)

        # 共情推理
        if model is not None:
            with torch.no_grad():
                tensor = torch.tensor([raw_bs], dtype=torch.float32, device=device)
                empathy_bs = model(tensor).cpu().numpy()[0]
        else:
            empathy_bs = raw_bs

        # 强度缩放
        empathy_bs = np.clip(empathy_bs * args.intensity, 0.0, 1.0)

        # 构建命名字典
        bs_dict = make_blendshape_dict(empathy_bs, arkit_names)

        # 写入临时文件
        bs_json = write_temp_blendshape(bs_dict, emotion, args.intensity)

        # 渲染
        output = os.path.join(OUTPUT_DIR, f"{emotion}.png")
        ok = render_with_blender(blender_exe, model_path, bs_json, output, args.engine)
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n完成: {passed}/{passed + failed} 渲染成功")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
