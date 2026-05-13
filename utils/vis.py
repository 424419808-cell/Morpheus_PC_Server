import numpy as np
import torch

def bs_to_bar(bs_vector, title="Blendshape", width=80, max_bar=50):
    """将 BS 系数打印为横向条形图（终端可视化）。"""
    names = [
        "eyeBlinkL","eyeLookDL","eyeLookIL","eyeLookOL",
        "eyeLookUL","eyeSquintL","eyeWideL","eyeBlinkR",
        "eyeLookDR","eyeLookIR","eyeLookOR","eyeLookUR",
        "eyeSquintR","eyeWideR","jawForward","jawLeft",
        "jawRight","jawOpen","mouthClose","mouthFunnel",
        "mouthPucker","mouthLeft","mouthRight","mouthSmileL",
        "mouthSmileR","mouthFrownL","mouthFrownR","mouthDimpleL",
        "mouthDimpleR","mouthStretchL","mouthStretchR","mouthRollLo",
        "mouthRollUp","mouthShrugLo","mouthShrugUp","mouthPressL",
        "mouthPressR","mouthLowDownL","mouthLowDownR","mouthUpUpL",
        "mouthUpUpR","browDownL","browDownR","browInnerUp",
        "browOuterUL","browOuterUR","cheekPuff","cheekSquintL",
        "cheekSquintR","noseSneerL","noseSneerR","tongueOut",
    ]
    if isinstance(bs_vector, torch.Tensor):
        bs_vector = bs_vector.cpu().numpy()
    print(f"\n── {title} ──")
    for i, v in enumerate(bs_vector):
        bar_len = int(v * max_bar)
        bar = "█" * bar_len + "░" * (max_bar - bar_len)
        print(f"{names[i]:>14s} |{bar}| {v:.3f}")


def plot_bs_comparison(user_bs, response_bs, title="User → Empathy Response"):
    """对比用户 BS 和共情回应 BS（ASCII）。"""
    if isinstance(user_bs, torch.Tensor):
        user_bs = user_bs.cpu().numpy()
    if isinstance(response_bs, torch.Tensor):
        response_bs = response_bs.cpu().numpy()

    max_diff_idx = np.argmax(np.abs(user_bs - response_bs))

    print(f"\n═══ {title} ═══")
    print(f"{'Index':>5s} {'User':>6s} {'Resp':>6s} {'Diff':>6s}")
    print("-" * 30)
    for i in range(len(user_bs)):
        d = response_bs[i] - user_bs[i]
        if abs(d) > 0.05:  # 只打印变化明显的
            marker = " ◀" if abs(d) == np.abs(user_bs - response_bs).max() else ""
            print(f"{i:5d} {user_bs[i]:6.3f} {response_bs[i]:6.3f} {d:6.3f}{marker}")
    print(f"\n最大变化索引: {max_diff_idx}, 幅度: {np.abs(user_bs[max_diff_idx] - response_bs[max_diff_idx]):.3f}")
