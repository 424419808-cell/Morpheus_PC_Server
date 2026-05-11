# Blender 离线渲染验证工具

利用 Blender 的 Cycles 渲染引擎，从训练数据中提取 52 维 Apple ARKit blendshape，驱动 MetaHuman 3D 人脸模型渲染出图，与原始摄像头采集图片并排对比，用于**验证训练数据质量**。

## 环境依赖

- **Blender 5.0+**（[blender.org](https://www.blender.org/download/) 下载安装）
- **无需 pip 安装** — `render_blendshape.py` 仅使用 Python 标准库，`blender_render.py` 使用 Blender 内置的 `bpy` 模块
- **可选**：`pip install Pillow`（用于生成原图 vs 渲染图的并排对比图；不装则只出渲染图）

## 文件说明

| 文件 | 说明 |
|------|------|
| `render_blendshape.py` | **入口脚本**：读取 JSON 数据 → 提取 blendshape → 调用 Blender 渲染 → 生成对比图 |
| `blender_render.py` | Blender 内部脚本（由 `render_blendshape.py` 通过 subprocess 调用）：加载 3D 模型 → 应用 blendshape → 渲染出图 |
| `config.py` | 共享配置：模型路径、Blender 路径、渲染参数、52 个 ARKit blendshape 名称列表、shape key 名称映射 |
| `metaHumanHead_52shapekeys_01.fbx` | 带 52 个 ARKit blendshape shape key 的 3D 人脸模型（MetaHuman 导出） |
| `output/` | 渲染输出目录（渲染图 + 对比图） |
| `temp/` | 临时文件目录（blendshape 数据中转 JSON） |

## 使用方法

```bash
# 基本用法：渲染指定 sample_id，生成渲染图 + 原图对比
python render_blendshape.py --sample_id 42

# 只渲染，不生成对比图（跳过原图查找 + Pillow 依赖）
python render_blendshape.py --sample_id 42 --no-compare

# 指定自定义 Blender 路径
python render_blendshape.py --sample_id 42 --blender "D:\Blender\blender.exe"

# 指定自定义参考图片目录
python render_blendshape.py --sample_id 42 --ref-dir "D:\captured_faces"
```

## 配置修改

首次使用前，在 `config.py` 中修改以下路径：

```python
BLENDER_EXE = r"H:\APP\Blender Foundation\Blender 5.0\blender.exe"  # 你的 Blender 安装路径
REFERENCE_IMAGE_DIR = r"I:\captured_faces"                           # 原始摄像头图片目录
```

## 数据来源

读取 `../emotion_MLP/data/motor_babbling_data_PC.json`（PC 端电机随机摆动 + MediaPipe 采集的 blendshape 训练数据集）。
