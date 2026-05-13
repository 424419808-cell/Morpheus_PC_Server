"""
Blender 内部脚本 — 被 render_blender.py 调用
在 Blender --background 模式下：
  1. 加载 blendshape JSON 数据
  2. 设置 shape key 值
  3. 渲染单帧并保存

用法 (由 render_blender.py 自动调用，不单独使用):
  blender --background model.fbx --python _blender_render_frame.py -- bs_data.json output.png BLENDER_EEVEE
"""
import json
import os
import sys

import bpy


def set_shape_keys(obj, blendshapes):
    """设置对象的 shape key 值"""
    if not obj.data.shape_keys:
        print(f"  [警告] 对象 '{obj.name}' 没有 shape keys")
        return

    key_blocks = obj.data.shape_keys.key_blocks
    set_count = 0

    for key_name, value in blendshapes.items():
        if key_name in key_blocks:
            key_blocks[key_name].value = value
            set_count += 1

    print(f"  [BS] 已设置 {set_count}/{len(blendshapes)} 个 shape keys")


def main():
    args = sys.argv[sys.argv.index("--") + 1:]
    if len(args) < 2:
        print("[错误] 参数不足: bs_json_path output_path [engine]")
        sys.exit(1)

    bs_json_path = args[0]
    output_path = args[1]
    engine = args[2] if len(args) > 2 else "BLENDER_EEVEE"

    # 加载 BS 数据
    if not os.path.exists(bs_json_path):
        print(f"[错误] BS 数据文件不存在: {bs_json_path}")
        sys.exit(1)

    with open(bs_json_path, "r") as f:
        data = json.load(f)

    emotion = data.get("emotion", "unknown")
    blendshapes = data.get("blendshapes", {})
    print(f"[INFO] 渲染: {emotion}, {len(blendshapes)} blendshapes")

    # 找到网格对象
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH' and obj.data.shape_keys]
    if not mesh_objects:
        print("[错误] 场景中没有带 shape keys 的网格对象")
        sys.exit(1)

    obj = mesh_objects[0]
    set_shape_keys(obj, blendshapes)

    # 配置渲染引擎
    scene = bpy.context.scene
    scene.render.engine = engine

    if engine == "CYCLES":
        scene.cycles.samples = 64

    # 设置输出路径
    scene.render.filepath = output_path
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024

    # 渲染
    print(f"[渲染] {output_path}")
    bpy.ops.render.render(write_still=True)
    print(f"[OK] 已保存: {output_path}")


if __name__ == "__main__":
    main()
