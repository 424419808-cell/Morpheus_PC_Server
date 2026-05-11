"""
入口脚本：读取 motor_babbling_data_PC.json → 提取指定 ID 的 blendshape
→ 调用 Blender 渲染新图像 → 生成原图 vs 渲染图的并排对比

用法:
    python render_blendshape.py --sample_id 42
    python render_blendshape.py --sample_id 42 --no-compare
    python render_blendshape.py --sample_id 42 --blender "C:\\...\\blender.exe"
    python render_blendshape.py --sample_id 42 --ref-dir "D:\\images"
"""
import argparse
import json
import os
import subprocess
import sys

from config import (
    DATA_JSON, MODEL_PATH, REFERENCE_IMAGE_DIR,
    OUTPUT_DIR, TEMP_DIR, TEMP_BLENDSHAPE_DATA,
    BLENDER_EXE, ARKIT_BLENDSHAPE_NAMES,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="从 motor_babbling_data_PC.json 提取 blendshape 并用 Blender 渲染对比图"
    )
    parser.add_argument("--sample_id", type=int, required=True,
                        help="要渲染的样本 ID（0-10999）")
    parser.add_argument("--no-compare", action="store_true",
                        help="只渲染，不生成对比图")
    parser.add_argument("--blender", type=str, default=BLENDER_EXE,
                        help=f"Blender 可执行文件路径（默认: {BLENDER_EXE}）")
    parser.add_argument("--ref-dir", type=str, default=REFERENCE_IMAGE_DIR,
                        help=f"参考图像目录（默认: {REFERENCE_IMAGE_DIR}）")
    return parser.parse_args()


def load_data():
    """加载 JSON 数据文件，返回整个列表"""
    if not os.path.exists(DATA_JSON):
        print(f"[ERROR] 数据文件不存在: {DATA_JSON}")
        sys.exit(1)
    print(f"[INFO] 正在加载数据文件...")
    with open(DATA_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[OK] 已加载 {len(data)} 条样本记录")
    return data


def find_sample(data, sample_id):
    """在数据中查找指定 sample_id 的条目"""
    for item in data:
        if item["sample_id"] == sample_id:
            return item
    return None


def write_temp_data(sample_id, blendshapes):
    """将 blendshape 数据写入临时 JSON 文件供 Blender 读取"""
    os.makedirs(TEMP_DIR, exist_ok=True)
    payload = {"sample_id": sample_id, "blendshapes": blendshapes}
    with open(TEMP_BLENDSHAPE_DATA, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"[OK] 已写入临时数据: {TEMP_BLENDSHAPE_DATA}")


def check_prerequisites(blender_exe):
    """检查 Blender 和模型文件是否存在"""
    if not os.path.exists(blender_exe):
        print(f"[ERROR] Blender 未找到: {blender_exe}")
        print(f"        请使用 --blender 参数指定正确路径")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] 3D 模型文件未找到: {MODEL_PATH}")
        print(f"        请将 .blend 或 .fbx 模型文件放置到此路径")
        print(f"        或在 config.py 中修改 MODEL_PATH 指向你的模型文件")
        sys.exit(1)


def run_blender(blender_exe, sample_id):
    """调用 Blender 子进程执行渲染"""
    blender_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blender_render.py")
    cmd = [
        blender_exe,
        "--background",
        "--python", blender_script,
        "--", str(sample_id),
    ]
    print(f"[INFO] 正在调用 Blender 渲染...")
    print(f"       命令: {' '.join(cmd)}")

    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    # 打印 Blender 的输出（包含脚本的日志信息）
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            print(f"       [blender] {line}")

    if result.returncode != 0:
        print(f"[ERROR] Blender 渲染失败 (返回码 {result.returncode})")
        if result.stderr:
            print(f"[ERROR] 错误输出:")
            for line in result.stderr.strip().splitlines():
                print(f"       {line}")
        sys.exit(result.returncode)

    print(f"[OK] Blender 渲染完成")


def create_comparison(sample_id, image_file, ref_dir):
    """使用 Pillow 创建原图与渲染图的并排对比图"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("[WARN] Pillow 未安装，跳过对比图生成")
        print("       安装命令: pip install Pillow")
        return

    render_path = os.path.join(OUTPUT_DIR, f"render_{sample_id}.png")
    ref_path = os.path.join(ref_dir, image_file)

    # 检查渲染图
    if not os.path.exists(render_path):
        print(f"[WARN] 渲染图不存在: {render_path}")
        return

    # 加载渲染图
    img_render = Image.open(render_path).convert("RGB")

    # 尝试加载参考图
    if os.path.exists(ref_path):
        img_ref = Image.open(ref_path).convert("RGB")
    else:
        print(f"[WARN] 参考图不存在: {ref_path}")
        print(f"       只保存渲染图，不生成对比")
        return

    # 统一高度（以较高的为准）
    h = max(img_ref.height, img_render.height)
    img_ref_resized = img_ref.resize(
        (int(img_ref.width * h / img_ref.height), h), Image.LANCZOS
    )
    img_render_resized = img_render.resize(
        (int(img_render.width * h / img_render.height), h), Image.LANCZOS
    )

    # 标题栏高度
    title_h = 40
    sep_w = 4

    total_w = img_ref_resized.width + sep_w + img_render_resized.width
    total_h = h + title_h

    # 创建画布
    canvas = Image.new("RGB", (total_w, total_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    # 尝试使用系统字体，失败则用默认
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 24)
        except (OSError, IOError):
            font = ImageFont.load_default()

    # 绘制标题
    draw.text((img_ref_resized.width // 2 - 40, 8), "Original", fill=(255, 255, 255), font=font)
    draw.text((img_ref_resized.width + sep_w + img_render_resized.width // 2 - 40, 8),
              "Rendered", fill=(255, 255, 255), font=font)

    # 拼接图像
    canvas.paste(img_ref_resized, (0, title_h))
    canvas.paste(img_render_resized, (img_ref_resized.width + sep_w, title_h))

    # 分隔线
    for y in range(title_h, total_h):
        for x in range(img_ref_resized.width, img_ref_resized.width + sep_w):
            canvas.putpixel((x, y), (255, 255, 255))

    # 保存
    compare_path = os.path.join(OUTPUT_DIR, f"compare_{sample_id}.png")
    canvas.save(compare_path)
    print(f"[OK] 对比图已保存: {compare_path}")


def main():
    args = parse_args()

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载数据
    data = load_data()

    # 2. 查找样本
    sample = find_sample(data, args.sample_id)
    if sample is None:
        sample_ids = [item["sample_id"] for item in data]
        print(f"[ERROR] 未找到 sample_id={args.sample_id}")
        print(f"        有效范围: {min(sample_ids)} ~ {max(sample_ids)}")
        sys.exit(1)

    print(f"[INFO] sample_id={sample['sample_id']}")
    print(f"[INFO] image_file={sample['image_file']}")
    print(f"[INFO] blendshapes 数量: {len(sample['blendshapes'])}")

    # 3. 写入临时 blendshape 数据
    write_temp_data(args.sample_id, sample["blendshapes"])

    # 4. 前置检查
    check_prerequisites(args.blender)

    # 5. 调用 Blender 渲染
    run_blender(args.blender, args.sample_id)

    # 6. 生成对比图
    if not args.no_compare:
        create_comparison(args.sample_id, sample["image_file"], args.ref_dir)

    # 7. 输出摘要
    print()
    print("=" * 60)
    print("完成！输出文件:")
    print(f"  渲染图: {os.path.join(OUTPUT_DIR, f'render_{args.sample_id}.png')}")
    if not args.no_compare:
        print(f"  对比图: {os.path.join(OUTPUT_DIR, f'compare_{args.sample_id}.png')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
