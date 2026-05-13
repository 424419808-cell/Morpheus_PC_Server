"""
实时摄像头共情推理。
1. 读取摄像头画面
2. MediaPipe 检测人脸并对齐
3. ViT 编码人脸 → DiT 采样 BS → 显示/发送 BS

用法：
    conda run -n morpheus python inference/webcam_demo.py
    conda run -n morpheus python inference/webcam_demo.py --no_display --udp_port 5005
"""
import sys
sys.path.insert(0, ".")

import cv2
import numpy as np
import torch
import time
import argparse

from config import Config
from model.empathy_model import EmpathyModel


def face_preprocess(frame, target_size=224):
    """
    用 MediaPipe 检测人脸，对齐并裁剪到 224x224。
    返回归一化的 tensor [1, 3, 224, 224] 或 None。
    """
    try:
        import mediapipe as mp
        mp_face = mp.solutions.face_detection
        with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as det:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = det.process(rgb)
            if not results.detections:
                return None

            h, w = frame.shape[:2]
            d = results.detections[0]
            bbox = d.location_data.relative_bounding_box
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)

            # 扩大裁剪区域
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            size = max(x2 - x1, y2 - y1)
            size = int(size * 1.5)
            x1 = max(0, cx - size // 2)
            y1 = max(0, cy - size // 2)
            x2 = min(w, cx + size // 2)
            y2 = min(h, cy + size // 2)

            face = frame[y1:y2, x1:x2]
            face = cv2.resize(face, (target_size, target_size))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # Normalize (ImageNet)
            face = face.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            face = (face - mean) / std
            face = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).float()
            return face, (x1, y1, x2, y2)
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/empathy_best.pth")
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--no_display", action="store_true")
    parser.add_argument("--udp_port", type=int, default=0)
    parser.add_argument("--ddim_steps", type=int, default=20)
    args = parser.parse_args()

    cfg = Config()
    device = cfg.device
    print(f"使用设备: {device}")

    # 加载模型
    model = EmpathyModel(cfg).to(device)
    try:
        state = torch.load(args.ckpt, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)  # 允许只加载部分权重
        print(f"已加载模型: {args.ckpt}")
    except FileNotFoundError:
        print(f"[警告] 未找到模型 {args.ckpt}，使用未训练的模型进行演示")
    model.eval()

    # UDP 发送（用于驱动外部角色）
    udp_sock = None
    if args.udp_port:
        import socket
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 摄像头
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头")
        return

    print("\n共情引擎已启动。按 ESC 退出。\n")

    last_bs = np.zeros(cfg.bs_dim, dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = face_preprocess(frame, cfg.img_size)
        display = frame.copy()

        if result is not None:
            face_tensor, bbox = result
            face_tensor = face_tensor.to(device)

            # 推理
            with torch.no_grad():
                cond = model.encode_face(face_tensor)
                bs = model.sample(cond, steps=args.ddim_steps, ddim=True)
                bs = bs.cpu().numpy().flatten()

            # 指数平滑去抖
            alpha = 0.7
            last_bs = alpha * last_bs + (1 - alpha) * bs
            bs = last_bs

            # 绘制
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 显示主要 BS 值
            info = [
                f"smile: {bs[23]:.2f}",
                f"brow: {bs[41]:.2f}",
                f"jaw: {bs[17]:.2f}",
            ]
            for i, text in enumerate(info):
                cv2.putText(display, text, (10, 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            # 通过 UDP 发送 BS
            if udp_sock:
                msg = ",".join(f"{v:.4f}" for v in bs)
                udp_sock.sendto(msg.encode(), ("127.0.0.1", args.udp_port))
        else:
            cv2.putText(display, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

        if not args.no_display:
            cv2.imshow("Morpheus Empathy Engine", display)
            if cv2.waitKey(1) == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()
    if udp_sock:
        udp_sock.close()


if __name__ == "__main__":
    main()
