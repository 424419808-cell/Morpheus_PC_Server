"""
模型快速验证脚本。不依赖摄像头和数据，纯代码测试。
验证：模型能跑通前向、反向、采样。

用法：
    conda run -n morpheus python test_model.py
"""
import sys
sys.path.insert(0, ".")

import torch
from config import Config
from model.empathy_model import EmpathyModel


def test_forward():
    print("=" * 50)
    print("测试1: 模型前向传播 + 训练 loss")
    print("=" * 50)

    cfg = Config()
    cfg.batch_size = 4
    model = EmpathyModel(cfg)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 模拟输入
    img = torch.randn(4, 3, 224, 224)
    target_bs = torch.rand(4, 52)  # [0,1] BS 系数

    # 前向 + loss
    pred_noise, target_noise = model(img, target_bs)
    loss = torch.nn.functional.mse_loss(pred_noise, target_noise)
    loss.backward()

    print(f"  img shape: {img.shape}")
    print(f"  target_bs shape: {target_bs.shape}")
    print(f"  pred_noise shape: {pred_noise.shape}")
    print(f"  loss: {loss.item():.6f}")
    print(f"  ✅ 前向 + 反向传播通过\n")


def test_stage1():
    print("=" * 50)
    print("测试2: 第一阶段 decoder_only_mode")
    print("=" * 50)

    cfg = Config()
    model = EmpathyModel(cfg)

    user_bs = torch.rand(4, 52)
    target_bs = torch.rand(4, 52)

    pred_noise, target_noise = model.decoder_only_mode(user_bs, target_bs)
    loss = torch.nn.functional.mse_loss(pred_noise, target_noise)

    print(f"  user_bs shape: {user_bs.shape}")
    print(f"  pred_noise shape: {pred_noise.shape}")
    print(f"  loss: {loss.item():.6f}")
    print(f"  ✅ Stage 1 训练通过\n")


def test_sample():
    print("=" * 50)
    print("测试3: DDIM 采样（共情 BS 生成）")
    print("=" * 50)

    cfg = Config()
    model = EmpathyModel(cfg)

    img = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        cond = model.encode_face(img)
        sampled_bs = model.sample(cond, steps=20, ddim=True)

    print(f"  cond shape: {cond.shape}")
    print(f"  sampled_bs shape: {sampled_bs.shape}")
    print(f"  BS range: [{sampled_bs.min():.4f}, {sampled_bs.max():.4f}]")
    print(f"  BS mean: {sampled_bs.mean():.4f}")
    # BS 应该在合理范围内（扩散采样加 sigmoid/约束后）
    print(f"  ✅ DDIM 采样通过（20步）\n")


def test_speed():
    print("=" * 50)
    print("测试4: 推理速度（端到端）")
    print("=" * 50)

    cfg = Config()
    model = EmpathyModel(cfg).eval()
    img = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        import time
        cond = model.encode_face(img)

        # warmup
        for _ in range(5):
            model.sample(cond, steps=20, ddim=True)

        # benchmark
        n_runs = 20
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model.sample(cond, steps=20, ddim=True)
        dt = (time.perf_counter() - t0) / n_runs

    print(f"  平均推理时间: {dt*1000:.1f} ms")
    print(f"  FPS: {1/dt:.1f}")
    print(f"  ✅ 速度测试通过\n")


if __name__ == "__main__":
    # 检查 CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    test_forward()
    test_stage1()
    test_sample()
    test_speed()
    print("=" * 50)
    print("全部测试通过 ✅")
    print("=" * 50)
