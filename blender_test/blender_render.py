"""
Blender 内部渲染脚本
由 render_blendshape.py 通过 subprocess 调用：
    blender --background --python blender_render.py -- <sample_id>

功能：加载 3D 模型 → 应用 blendshape → 渲染出图
"""
import sys
import os
import json

# 确保能 import 同目录下的 config.py
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import bpy
from config import (
    TEMP_BLENDSHAPE_DATA, OUTPUT_DIR, MODEL_PATH,
    ARKIT_BLENDSHAPE_NAMES, SHAPE_KEY_NAME_MAP,
    RENDER_WIDTH, RENDER_HEIGHT, RENDER_ENGINE,
    CYCLES_SAMPLES,
)


def parse_sample_id():
    """从命令行参数中提取 sample_id（Blender 会把 -- 后面的参数传入 sys.argv）"""
    try:
        idx = sys.argv.index("--")
        return sys.argv[idx + 1]
    except (ValueError, IndexError):
        print("[ERROR] 用法: blender --background --python blender_render.py -- <sample_id>")
        sys.exit(1)


def load_blendshape_data(sample_id):
    """读取临时 JSON 文件中的 blendshape 数据"""
    if not os.path.exists(TEMP_BLENDSHAPE_DATA):
        print(f"[ERROR] 找不到 blendshape 临时文件: {TEMP_BLENDSHAPE_DATA}")
        sys.exit(1)

    with open(TEMP_BLENDSHAPE_DATA, "r", encoding="utf-8") as f:
        data = json.load(f)

    if str(data.get("sample_id")) != str(sample_id):
        print(f"[WARN] 临时文件中 sample_id={data.get('sample_id')}，与参数 {sample_id} 不一致")
    return data["blendshapes"]


def open_model():
    """加载 3D 模型：根据扩展名自动选择 .blend 或 .fbx 导入方式"""
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] 模型文件不存在: {MODEL_PATH}")
        sys.exit(1)

    ext = os.path.splitext(MODEL_PATH)[1].lower()
    if ext == ".blend":
        bpy.ops.wm.open_mainfile(filepath=MODEL_PATH)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=MODEL_PATH)
    else:
        print(f"[ERROR] 不支持的模型格式: {ext}，请使用 .blend 或 .fbx")
        sys.exit(1)

    print(f"[OK] 已加载模型: {MODEL_PATH}")


def find_face_mesh():
    """在所有 mesh 对象中找出带 ARKit shape keys 的人脸 mesh"""
    best_obj = None
    best_count = 0

    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if not obj.data.shape_keys or not obj.data.shape_keys.key_blocks:
            continue

        key_names = {kb.name for kb in obj.data.shape_keys.key_blocks}
        # 计算该 mesh 的 shape key 与 ARKit 标准名匹配的数量
        matches = sum(
            1 for name in ARKIT_BLENDSHAPE_NAMES
            if name in key_names or SHAPE_KEY_NAME_MAP.get(name, name) in key_names
        )
        if matches > best_count:
            best_count = matches
            best_obj = obj

    if best_obj is None:
        print("[ERROR] 场景中找不到带 shape keys 的 mesh 对象")
        print("[INFO] 场景中的 mesh 对象:")
        for obj in bpy.data.objects:
            if obj.type == "MESH":
                keys = list(obj.data.shape_keys.key_blocks.keys()) if obj.data.shape_keys else []
                print(f"  - {obj.name}: shape_keys={keys[:5]}...")
        sys.exit(1)

    # 打印诊断信息：模型的 shape key 名称 vs ARKit 标准名
    key_names = sorted(best_obj.data.shape_keys.key_blocks.keys())
    print(f"[INFO] 模型实际 shape key 名称 (共 {len(key_names)} 个):")
    print(f"        {key_names[:10]}{'...' if len(key_names) > 10 else ''}")
    if best_count == 0:
        print(f"[WARN] 匹配数为 0！模型 shape key 名称与 ARKit 标准名完全不同")
        print(f"[WARN] 模型前5个: {key_names[:5]}")
        print(f"[WARN] ARKit 前5个: {ARKIT_BLENDSHAPE_NAMES[:5]}")
        print(f"[WARN] 请在 config.py 中配置 SHAPE_KEY_NAME_MAP 映射")
    else:
        print(f"[OK] 找到人脸 mesh: '{best_obj.name}' (匹配 {best_count}/{len(ARKIT_BLENDSHAPE_NAMES)} 个 shape keys)")
    return best_obj


def apply_blendshapes(face_obj, blendshape_values):
    # ++++ 新增：暴力清除自带的动画数据和驱动，防止数值在渲染时被引擎强制重置 ++++
    if face_obj.data.shape_keys.animation_data:
        face_obj.data.shape_keys.animation_data_clear()
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    """将 blendshape 值应用到 mesh 的 shape keys"""
    key_blocks = face_obj.data.shape_keys.key_blocks
    applied = 0
    skipped = 0

    for bs_name in ARKIT_BLENDSHAPE_NAMES:
        value = blendshape_values.get(bs_name, 0.0)
        # 应用名称映射
        key_name = SHAPE_KEY_NAME_MAP.get(bs_name, bs_name)

        if key_name in key_blocks:
            key_blocks[key_name].value = value
            applied += 1
        else:
            skipped += 1

    print(f"[OK] 已应用 {applied} 个 blendshape, 跳过 {skipped} 个（模型上不存在）")
    if applied == 0:
        print(f"[ERROR] 所有 blendshape 都跳过了！不同 sample_id 渲染结果将完全相同")
        print(f"[ERROR] 原因: 模型 shape key 名称与 ARKit 标准名不匹配")
        print(f"[ERROR] 请在 config.py 中配置 SHAPE_KEY_NAME_MAP，将 ARKit 名称映射到模型实际名称")


def setup_scene(face_obj):
    """根据人脸 mesh 的包围盒自动定位相机和灯光"""
    import mathutils

    scene = bpy.context.scene

    # 渲染引擎 & 分辨率
    scene.render.engine = RENDER_ENGINE
    scene.render.resolution_x = RENDER_WIDTH
    scene.render.resolution_y = RENDER_HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True

    # CYCLES 采样设置
    if RENDER_ENGINE == "CYCLES":
        scene.cycles.samples = CYCLES_SAMPLES
        scene.cycles.device = "CPU"  # 无头模式强制 CPU

    # 计算人脸 mesh 的世界空间包围盒
    bbox_corners = [face_obj.matrix_world @ mathutils.Vector(c) for c in face_obj.bound_box]
    cx = sum(v.x for v in bbox_corners) / 8
    cy = sum(v.y for v in bbox_corners) / 8
    cz = sum(v.z for v in bbox_corners) / 8
    center = mathutils.Vector((cx, cy, cz))

    xs = [v.x for v in bbox_corners]
    ys = [v.y for v in bbox_corners]
    zs = [v.z for v in bbox_corners]
    bbox_size = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))

    print(f"[INFO] 人脸包围盒中心: ({center.x:.3f}, {center.y:.3f}, {center.z:.3f})")
    print(f"[INFO] 人脸包围盒大小: {bbox_size:.3f}")

    # 删除场景中所有已有相机，重新创建
    for obj in list(bpy.data.objects):
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)

    # 相机放在脸前方（-Y 方向），距离约为包围盒的 2.5 倍，略高于中心
    cam_dist = bbox_size * 2
    cam_pos = center + mathutils.Vector((0, -cam_dist, bbox_size * 0.25))
    bpy.ops.object.camera_add(location=cam_pos)
    camera = bpy.context.object
    scene.camera = camera

    # 让相机看向人脸中心
    look_dir = (center - cam_pos).normalized()
    track_quat = look_dir.to_track_quat('-Z', 'Y')
    camera.rotation_euler = track_quat.to_euler()

    print(f"[OK] 相机位置: ({cam_pos.x:.3f}, {cam_pos.y:.3f}, {cam_pos.z:.3f})")

    # 删除已有灯光，重新创建三点照明（位置相对于人脸）
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

    # === 修改灯光设置：大幅降低亮度，增加发光面积（柔和阴影） ===
    
    # 1. 主光 (Key Light) - 放在右前上方
    bpy.ops.object.light_add(type="AREA", location=center + mathutils.Vector((bbox_size*0.8, -bbox_size * 1.5, bbox_size)))
    bpy.context.object.data.energy = 40 * bbox_size  # 从 300 降到 40
    bpy.context.object.data.size = bbox_size * 2.5   # 发光板变大，皮肤阴影更柔和
    
    # 2. 补光 (Fill Light) - 放在左前方，极低亮度，用来消除死黑的阴影
    bpy.ops.object.light_add(type="AREA", location=center + mathutils.Vector((-bbox_size, -bbox_size, bbox_size*0.5)))
    bpy.context.object.data.energy = 10 * bbox_size  # 从 150 降到 10
    bpy.context.object.data.size = bbox_size * 3.0
    
    # 3. 轮廓光/发丝光 (Rim Light) - 放在正后上方，打亮边缘增加立体感
    bpy.ops.object.light_add(type="AREA", location=center + mathutils.Vector((0, bbox_size, bbox_size * 1.5)))
    bpy.context.object.data.energy = 60 * bbox_size  # 降到 60
    bpy.context.object.data.size = bbox_size * 1.5
    
    print("[OK] 已应用柔光箱三点照明")

    # 世界背景设为中性灰
    world = bpy.data.worlds.get("World")
    if world and world.use_nodes:
        bg = world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs[0].default_value = (0.18, 0.18, 0.18, 1.0)
            bg.inputs[1].default_value = 1.0


def render(sample_id):
    """渲染并保存图像"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"render_{sample_id}.png")
    bpy.context.scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    print(f"[OK] 渲染完成: {out_path}")
    return out_path


def main():
    sample_id = parse_sample_id()
    blendshape_values = load_blendshape_data(sample_id)

    open_model()
    face_obj = find_face_mesh()
    apply_blendshapes(face_obj, blendshape_values)
    setup_scene(face_obj)
    render(sample_id)


if __name__ == "__main__":
    main()
