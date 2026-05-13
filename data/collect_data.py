"""
数据采集工具：用 iPhone ARKit / MediaPipe 录制 (人脸, BS) 配对数据。

采集流程：
  1. 打开摄像头
  2. 每帧检测人脸
  3. 按 S 键保存当前帧和人工标注的 BS/情绪
  4. 按 E 键切换"共情回应"模式——录制你对屏幕上表情的回应

适用于第一轮 MVP 验证：
  - 自己录制各种表情（作为用户表情）
  - 对每个表情录制你的共情回应
  - 得到最真实的配对数据

用法：
    conda run -n morpheus python data/collect_data.py
"""
import sys
sys.path.insert(0, ".")

import cv2
import numpy as np
import torch
from pathlib import Path
import json
import time
from datetime import datetime


def main():
    save_dir = Path(__file__).parent / "real"
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头")
        return

    mode = "capture_user"  # or "capture_response"
    frame_count = 0
    pair_count = 0
    pending_user_bs = None
    subject_name = input("输入受试者名称（如: cyz）: ").strip() or f"subject_{datetime.now():%Y%m%d_%H%M%S}"
    subject_dir = save_dir / subject_name
    subject_dir.mkdir(exist_ok=True)

    print(f"\n采集模式说明：")
    print(f"  S → 保存当前帧（作为用户表情）")
    print(f"  R → 切换至「共情回应」模式：对上一张保存的表情做出回应")
    print(f"  Q / ESC → 退出\n")
    print(f"数据保存至: {subject_dir}")
    print("-" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display = frame.copy()
        h, w = frame.shape[:2]

        # 状态显示
        cv2.putText(display, f"Mode: {mode}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   (0, 255, 0) if mode == "capture_user" else (0, 165, 255), 2)
        cv2.putText(display, f"Saved: {pair_count} pairs", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if mode == "capture_response" and pending_user_bs is not None:
            cv2.putText(display, "MAKE YOUR EMPATHETIC RESPONSE!", (w//2-200, h-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            # 在左上角显示用户表情的小缩略图
            thumb = cv2.resize(pending_user_bs["img"], (160, 120))
            display[10:130, 10:170] = thumb
            cv2.rectangle(display, (10, 10), (169, 129), (255, 255, 255), 1)

        cv2.imshow("Data Collection - Morpheus Empathy", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and mode == "capture_user":
            # 保存用户表情
            img_path = str(subject_dir / f"user_{frame_count:06d}.jpg")
            cv2.imwrite(img_path, frame)
            pending_user_bs = {"img": frame.copy(), "path": img_path, "ts": time.time()}
            print(f"[保存] 用户表情: {img_path}")
            # 自动切换共情回应模式
            mode = "capture_response"

        elif key == ord('r') or (key == ord('s') and mode == "capture_response"):
            # 保存共情回应
            if pending_user_bs is not None:
                resp_path = str(subject_dir / f"resp_{frame_count:06d}.jpg")
                cv2.imwrite(resp_path, frame)
                pair_count += 1

                # 保存配对元数据
                meta = {
                    "pair_id": pair_count,
                    "user_img": pending_user_bs["path"],
                    "response_img": resp_path,
                    "timestamp": time.time(),
                }
                with open(subject_dir / "pairs.jsonl", "a") as f:
                    f.write(json.dumps(meta) + "\n")

                print(f"[配对 #{pair_count}] 用户: {pending_user_bs['path']} | 回应: {resp_path}")
                pending_user_bs = None
                mode = "capture_user"
            else:
                print("[提示] 没有待回应的表情，请先在用户模式下保存一张表情")

        elif key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n采集完成！共 {pair_count} 对数据")
    print(f"数据目录: {subject_dir}")

    # 提示后续处理
    if pair_count > 0:
        print(f"\n后续步骤：")
        print(f"  1. 查看数据: ls {subject_dir}")
        print(f"  2. 对每张图用 MediaPipe 提取 BS（使用 data/extract_bs.py）")
        print(f"  3. 在 empathy/data/real/ 下整理成 FaceToBSPairDataset 格式")


if __name__ == "__main__":
    main()
