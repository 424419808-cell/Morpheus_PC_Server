"""
从采集的人脸图像中提取近似 BS 系数（用 MediaPipe Face Landmarker）。

MediaPipe 的 blendshapes 输出有 52 维，与 ARKit BS 对齐。
用于将采集的 JPG 图像转换为 BS 向量，作为训练标签。

用法：
    conda run -n morpheus python data/extract_bs.py --input_dir data/real/subject_name
"""
import sys
sys.path.insert(0, ".")

import cv2
import numpy as np
import torch
from pathlib import Path
import argparse
from tqdm import tqdm

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    print("[错误] 需要 mediapipe: pip install mediapipe")
    sys.exit(1)


def extract_blendshapes(image_path, detector):
    """从单张图片提取 52 维 blendshape 系数。"""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)

    if not result.face_landmarks:
        return None

    bs = np.zeros(52, dtype=np.float32)
    if result.face_blendshapes:
        for bs_item in result.face_blendshapes[0]:
            idx = bs_item.index
            if idx < 52:
                bs[idx] = bs_item.score
    return bs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                       help="采集的图像目录（含 user_*.jpg 和 resp_*.jpg）")
    parser.add_argument("--model", type=str,
                       default="face_landmarker_v2_with_blendshapes.task",
                       help="MediaPipe Face Landmarker 模型路径")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"[错误] 目录不存在: {input_dir}")
        return

    # 加载 MediaPipe 模型
    model_path = Path(args.model)
    if not model_path.exists():
        # 尝试用项目目录中的模型
        alt_path = Path(__file__).parent.parent / ".." / "emotion_MLP" / args.model
        if alt_path.exists():
            model_path = alt_path
        else:
            print(f"[错误] 找不到模型文件。请下载:")
            print(f"  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task")
            print(f"  然后放到 {args.model}")
            return

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        output_face_blendshapes=True,
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # 提取所有图片的 BS
    img_files = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png"))
    if not img_files:
        print(f"[错误] {input_dir} 中没有图片")
        return

    results = {}
    for img_path in tqdm(img_files, desc="Extracting BS"):
        bs = extract_blendshapes(img_path, detector)
        if bs is not None:
            results[img_path.name] = bs
        else:
            print(f"[警告] 未能检测到人脸: {img_path.name}")

    # 保存
    save_path = input_dir / "bs_results.pt"
    torch.save(results, save_path)
    print(f"\n已保存 {len(results)} 帧的 BS 数据到 {save_path}")

    # 如果同时存在 user_* 和 resp_*，尝试配对并生成训练数据
    user_bs_list = []
    resp_bs_list = []

    # 按文件名序号配对
    for name, bs in results.items():
        if name.startswith("resp_"):
            # 找到对应的 user_
            user_name = name.replace("resp_", "user_")
            if user_name in results:
                resp_bs_list.append(bs)
                user_bs_list.append(results[user_name])

    if user_bs_list and resp_bs_list:
        paired = {
            "user_bs": torch.from_numpy(np.stack(user_bs_list)).float(),
            "response_bs": torch.from_numpy(np.stack(resp_bs_list)).float(),
        }
        pair_path = input_dir / "paired_bs.pt"
        torch.save(paired, pair_path)
        print(f"配对数据: {len(user_bs_list)} 对 -> {pair_path}")
    else:
        print("未找到配对的 user/resp 图像对")


if __name__ == "__main__":
    main()
