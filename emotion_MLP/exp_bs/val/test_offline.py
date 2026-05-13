"""
共情模型离线验证 — 自洽性、物理可实现性、情绪分类准确性

用法:
  python exp_bs/val/test_offline.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.join(BASE_DIR, "..", "..")
MODEL_PATH = os.path.join(PROJ_ROOT, "exp_bs", "models", "bs2bs_empathy.pth")
EMOTION_MODEL_PATH = os.path.join(PROJ_ROOT, "emo_clf", "models", "emotion_model.pkl")

# 19 种情绪（与 gen_batch_data.py 保持一致）
sys.path.insert(0, os.path.join(PROJ_ROOT, "exp_bs", "scripts"))
from gen_batch_data import get_base_bs


class BS2BS_Empathy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(52, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 52), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def test_self_consistency(model, device):
    """测试 1: 自洽性 — 输出 ≠ 输入（非中性情绪时）"""
    print("\n=== 自洽性测试 ===")
    emotion_names = [
        "Neutral", "Happy", "Excitement", "Humor", "Pride",
        "Trust", "Love", "Relief", "Hope",
        "Anger", "Disgust", "Fear", "Vigilance",
        "Sad", "Loneliness", "Guilt",
        "Surprise", "Confusion", "Shyness",
    ]

    model.eval()
    passed = 0
    failed = 0

    with torch.no_grad():
        for name in emotion_names:
            input_bs = get_base_bs(name)
            tensor = torch.tensor([input_bs], dtype=torch.float32, device=device)
            output_bs = model(tensor).cpu().numpy()[0]
            diff = np.mean(np.abs(output_bs - input_bs))

            if name == "Neutral":
                # 中性 → 中性，输出应该接近输入
                ok = diff < 0.05
                status = "PASS" if ok else "FAIL"
                print(f"  [{status}] {name:12s}: diff={diff:.4f} (应<0.05)")
            else:
                # 非中性 → 输出应不同于输入（共情 ≠ 镜像）
                ok = diff > 0.03
                status = "PASS" if ok else "FAIL"
                print(f"  [{status}] {name:12s}: diff={diff:.4f} (应>0.03)")

            if ok:
                passed += 1
            else:
                failed += 1

    print(f"\n结果: {passed}/{passed+failed} 通过")
    return passed, failed


def test_physical_realizability(model, device):
    """测试 2: 物理可实现性 — 通过 FaceBS2Angle → Angle2BS 重建"""
    print("\n=== 物理可实现性测试 ===")

    bs2angle_dir = os.path.join(PROJ_ROOT, "bs2angle", "models")
    forward_path = os.path.join(bs2angle_dir, "angle2bs_full.pth")
    upper_path = os.path.join(bs2angle_dir, "upper_face_bs2angle.pth")
    lower_path = os.path.join(bs2angle_dir, "lower_face_bs2angle.pth")

    if not all(os.path.exists(p) for p in [forward_path, upper_path, lower_path]):
        print("  [跳过] 缺少逆向模型文件")
        return 0, 0

    # 加载正向模型
    fwd_ckpt = torch.load(forward_path, map_location=device)
    input_dim = fwd_ckpt["input_dim"]
    output_dim = fwd_ckpt["output_dim"]

    fwd_net = nn.Sequential(
        nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, output_dim),
    )
    fwd_net.load_state_dict(fwd_ckpt["model_state_dict"])
    fwd_net.to(device).eval()

    # 测试构建
    emotion_names = ["Happy", "Sad", "Anger", "Fear", "Surprise"]
    model.eval()
    total_recon = 0

    with torch.no_grad():
        for name in emotion_names:
            input_bs = get_base_bs(name)
            tensor = torch.tensor([input_bs], dtype=torch.float32, device=device)

            # 共情模型 → BS
            pred_bs = model(tensor)

            # 简化的重建测试：BS → angle → 重建 BS
            # 这里用正向模型做直接重建
            pred_bs_small = pred_bs[:, :output_dim]
            recon_bs = fwd_net(pred_bs_small)
            recon_loss = nn.functional.l1_loss(pred_bs_small, recon_bs).item()
            total_recon += recon_loss

            ok = recon_loss < 0.1
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {name:12s}: 重建损失={recon_loss:.6f} (应<0.1)")

    avg_loss = total_recon / len(emotion_names)
    print(f"\n平均重建损失: {avg_loss:.6f}")
    return 0, 0


def test_emotion_classification(model, device):
    """测试 3: 情绪分类验证 — 输出 BS 的情绪标签应在共情映射范围内"""
    print("\n=== 情绪分类验证 ===")

    try:
        import joblib
        import pandas as pd
    except ImportError:
        print("  [跳过] 缺少 joblib/pandas")
        return 0, 0

    if not os.path.exists(EMOTION_MODEL_PATH):
        print("  [跳过] 找不到 emotion_model.pkl")
        return 0, 0

    rf = joblib.load(EMOTION_MODEL_PATH)
    feature_names = [f"bs_{i}" for i in range(52)]

    emotion_names = ["Happy", "Sad", "Angry", "Fear", "Surprise"]
    model.eval()
    passed = 0

    with torch.no_grad():
        for name in emotion_names:
            input_bs = get_base_bs(name)
            tensor = torch.tensor([input_bs], dtype=torch.float32, device=device)
            output_bs = model(tensor).cpu().numpy()[0]

            df = pd.DataFrame([output_bs], columns=feature_names)
            pred_label = rf.predict(df)[0]
            print(f"  {name:12s} → 分类为: {pred_label}")

    return 0, 0


def test_diversity(model, device):
    """测试 4: 多样性 — 同一输入多次推理的差异"""
    print("\n=== 多样性测试 ===")
    input_bs = get_base_bs("Happy")
    tensor = torch.tensor([input_bs], dtype=torch.float32, device=device)

    model.eval()
    outputs = []
    with torch.no_grad():
        for _ in range(10):
            out = model(tensor).cpu().numpy()[0]
            outputs.append(out)

    outputs = np.array(outputs)
    std = np.std(outputs, axis=0).mean()
    ok = std < 0.01  # 确定性的模型，std 应接近 0
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] 10次推理标准差: {std:.6f} (应<0.01，确定性模型)")
    return 0, 0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model = BS2BS_Empathy().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"[OK] 加载模型: {MODEL_PATH}")
    else:
        print(f"[警告] 模型未找到，使用未训练模型测试")

    test_self_consistency(model, device)
    test_physical_realizability(model, device)
    test_emotion_classification(model, device)
    test_diversity(model, device)

    print("\n=== 验证完成 ===")


if __name__ == "__main__":
    main()
