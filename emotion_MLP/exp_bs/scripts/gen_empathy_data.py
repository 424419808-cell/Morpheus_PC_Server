"""
生成共情训练配对数据 — 基于 gen_batch_data.py 的 BS 模板自举

从 19 个手工 BS 模板出发，结合共情映射规则，生成 (user_BS → empathy_BS) 配对。

输出: ../data/empathy_training_data.npz
格式: {'inputs': (N, 52), 'targets': (N, 52), 'user_emotions': (N,), 'target_emotions': (N,), 'alphas': (N,)}
"""
import numpy as np
import os
import sys

# 复用现有的 BS 模板
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_batch_data import get_base_bs
from empathy_map import EMPATHY_MAP, OBSERVED_EMOTIONS, EMOTION_IDS


def generate_empathy_data(
    intensities=None,
    noise_levels=None,
    variants_per_noise=100,
    mix_ratio=0.2,
    neutral_ratio=0.1,
):
    """
    生成共情训练配对数据

    参数:
        intensities: 强度缩放因子列表，默认 [0.2, 0.4, 0.6, 0.8, 1.0]
        noise_levels: 输入噪声标准差列表，默认 [0.01, 0.02, 0.05]
        variants_per_noise: 每个强度+噪声组合的变体数
        mix_ratio: 混合情绪插值的比例
        neutral_ratio: 中性样本占总数据的比例
    """
    if intensities is None:
        intensities = [0.7, 0.85, 1.0]  # 去掉弱强度，让模型看到完整表情
    if noise_levels is None:
        noise_levels = [0.01, 0.02, 0.05]

    inputs, targets = [], []
    user_emos, tgt_emos, alphas = [], [], []

    # 遍历每种用户情绪
    for user_emo in OBSERVED_EMOTIONS:
        base_user_bs = get_base_bs(user_emo)
        candidates = EMPATHY_MAP.get(user_emo, [("Neutral", 1.0)])

        for tgt_emo, weight in candidates:
            base_tgt_bs = get_base_bs(tgt_emo)

            for alpha in intensities:
                # 随机缩放 + 微小随机扰动，保持信号强度
                scale = alpha * np.random.uniform(0.8, 1.0)
                target_bs = np.clip(base_tgt_bs * scale, 0, 1)

                for noise_std in noise_levels:
                    for _ in range(variants_per_noise):
                        noise = np.random.normal(0, noise_std, 52)
                        input_bs = np.clip(base_user_bs + noise, 0.0, 1.0)

                        inputs.append(input_bs)
                        targets.append(target_bs)
                        user_emos.append(EMOTION_IDS.get(user_emo, 0))
                        tgt_emos.append(EMOTION_IDS.get(tgt_emo, 0))
                        alphas.append(alpha)

    # 添加混合情绪样本（两个目标之间线性插值）
    for user_emo in OBSERVED_EMOTIONS:
        base_user_bs = get_base_bs(user_emo)
        candidates = EMPATHY_MAP.get(user_emo, [("Neutral", 1.0)])
        if len(candidates) < 2:
            continue

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                emo_a, _ = candidates[i]
                emo_b, _ = candidates[j]
                bs_a = get_base_bs(emo_a)
                bs_b = get_base_bs(emo_b)

                for mix in [0.3, 0.5, 0.7]:
                    target_bs = bs_a * (1 - mix) + bs_b * mix
                    target_bs = target_bs * np.random.choice(intensities)

                    for noise_std in noise_levels:
                        for _ in range(variants_per_noise // 5):
                            noise = np.random.normal(0, noise_std, 52)
                            input_bs = np.clip(base_user_bs + noise, 0.0, 1.0)

                            inputs.append(input_bs)
                            targets.append(target_bs)
                            user_emos.append(EMOTION_IDS.get(user_emo, 0))
                            tgt_emos.append(EMOTION_IDS.get(emo_a, 0))
                            alphas.append(mix)

    # 添加大量中性样本（输入中性→输出中性）
    neutral_bs = get_base_bs("Neutral")
    n_neutral = int(len(inputs) * neutral_ratio / (1 - neutral_ratio))
    for _ in range(n_neutral):
        noise = np.random.normal(0, 0.01, 52)
        input_bs = np.clip(neutral_bs + noise, 0.0, 1.0)
        inputs.append(input_bs)
        targets.append(neutral_bs.copy())
        user_emos.append(EMOTION_IDS["Neutral"])
        tgt_emos.append(EMOTION_IDS["Neutral"])
        alphas.append(0.0)

    # 转为 numpy 数组
    data = {
        "inputs": np.array(inputs, dtype=np.float32),
        "targets": np.array(targets, dtype=np.float32),
        "user_emotions": np.array(user_emos, dtype=np.int32),
        "target_emotions": np.array(tgt_emos, dtype=np.int32),
        "alphas": np.array(alphas, dtype=np.float32),
    }

    print(f"生成共情数据: {len(inputs)} 样本")
    print(f"  输入形状: {data['inputs'].shape}")
    print(f"  情绪范围: {data['user_emotions'].min()}~{data['user_emotions'].max()}")
    print(f"  中性比例: {n_neutral / len(inputs) * 100:.1f}%")

    return data


def save_empathy_data(output_path=None):
    """生成并保存共情训练数据"""
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "..", "..", "data", "empathy_training_data.npz")

    data = generate_empathy_data()
    np.savez_compressed(output_path, **data)
    print(f"已保存: {output_path}")
    return output_path


if __name__ == "__main__":
    save_empathy_data()
