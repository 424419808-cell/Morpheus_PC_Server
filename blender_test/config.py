"""
共享配置：被 render_blendshape.py 和 blender_render.py 共同引用。
修改路径时只需改这一个文件。
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 路径配置（根据你的实际环境修改）
# ============================================================

# motor_babbling_data_PC.json 数据文件
DATA_JSON = os.path.join(BASE_DIR, "..", "emotion_MLP", "data", "motor_babbling_data_PC.json")

# 带 ARKit blendshape 的 3D 人脸模型（支持 .blend 或 .fbx）
# MODEL_PATH = os.path.join(BASE_DIR, "assets", "face_model.blend")
# 也可以直接用 FBX，例如:
MODEL_PATH = os.path.join(BASE_DIR, "metaHumanHead_52shapekeys_01.fbx")

# 参考图像（原图）所在目录
REFERENCE_IMAGE_DIR = r"I:\captured_faces"

# Blender 可执行文件路径
BLENDER_EXE = r"H:\APP\Blender Foundation\Blender 5.0\blender.exe"

# 临时文件和输出目录
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
TEMP_BLENDSHAPE_DATA = os.path.join(TEMP_DIR, "blendshape_data.json")

# ============================================================
# 渲染参数
# ============================================================
RENDER_WIDTH = 1024
RENDER_HEIGHT = 1024
RENDER_FORMAT = "PNG"
RENDER_ENGINE = "CYCLES"  # CYCLES 纯 CPU 渲染，无头模式下贴图正常
# 如需速度可选 EEVEE，但 --background 模式会丢失 GPU 贴图:
# RENDER_ENGINE = "BLENDER_EEVEE"

# CYCLES 渲染采样数（越大越精细，越慢）
CYCLES_SAMPLES = 64

# ============================================================
# 52 个 Apple ARKit blendshape 名称（与 JSON 中的 key 完全一致）
# ============================================================
ARKIT_BLENDSHAPE_NAMES = [
    "_neutral",
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]

# ============================================================
# Shape Key 名称映射（可选）
# 如果模型的 shape key 名称与 ARKit 标准名不同，在这里配置映射。
# 留空 {} 表示模型的 shape key 名与 ARKit 标准名完全一致。
#
# 示例（MetaHuman 命名风格）：
# SHAPE_KEY_NAME_MAP = {
#     "browDownLeft": "CTRL_expressions_browDown_L",
#     "browDownRight": "CTRL_expressions_browDown_R",
#     ...
# }
# ============================================================
SHAPE_KEY_NAME_MAP = {}
