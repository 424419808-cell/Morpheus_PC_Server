"""
生成 EmotionBrain 训练数据 — 19 种情绪的 52 维 blendshape 目标值

索引约定（重要）：
  EmotionBrain 输出 52 维向量，其中 output[i] 在伺服管线 (bs_to_servo)
  中映射到 ARKIT_NAMES[i+1]。即 output[0] 控制 browDownLeft，
  output[1] 控制 browDownRight，以此类推。output[51] 未被使用。

  因此定义 SERVO_IDX[name] = i，表示 output[i] 控制该 ARKit shape key。

  所有情绪数据均与原始硬编码索引保持 1:1 精确对应。
  原始索引 N → ARKIT_NAMES[N+1] 的对应关系标注在每行注释中。
"""
import numpy as np
import os
import sys

# ==================== ARKit 52 blendshape 名称（字母序） ====================
ARKIT_NAMES = [
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

# Servo 管线索引映射：
#   output[SERVO_IDX[name]] 在 servo pipeline 中控制名为 name 的 ARKit shape key
#   即 SERVO_IDX[name] = ARKIT_NAMES.index(name) - 1，取值范围 0..50
SERVO_IDX = {name: i for i, name in enumerate(ARKIT_NAMES[1:])}


def get_base_bs(emotion):
    """返回指定情绪的目标 52 维 BS 向量（EmotionBrain 输出空间）"""
    bs = np.zeros(52)

    if emotion == "Neutral":
        pass

    elif emotion == "Happy":
        bs[SERVO_IDX["mouthSmileLeft"]] = 1.0       # bs[43]
        bs[SERVO_IDX["mouthSmileRight"]] = 1.0      # bs[44]
        bs[SERVO_IDX["eyeSquintLeft"]] = 0.6        # bs[18]
        bs[SERVO_IDX["eyeSquintRight"]] = 0.6       # bs[19]
        bs[SERVO_IDX["cheekSquintLeft"]] = 0.5      # bs[6]
        bs[SERVO_IDX["cheekSquintRight"]] = 0.5     # bs[7]
        bs[SERVO_IDX["jawOpen"]] = 0.1              # bs[24]

    elif emotion == "Excitement":
        bs[SERVO_IDX["mouthSmileLeft"]] = 1.0       # bs[43]
        bs[SERVO_IDX["mouthSmileRight"]] = 1.0      # bs[44]
        bs[SERVO_IDX["eyeWideLeft"]] = 0.7          # bs[20]
        bs[SERVO_IDX["eyeWideRight"]] = 0.7         # bs[21]
        bs[SERVO_IDX["browInnerUp"]] = 0.8          # bs[2]
        bs[SERVO_IDX["jawOpen"]] = 0.3              # bs[24]

    elif emotion == "Humor":
        bs[SERVO_IDX["mouthSmileRight"]] = 0.7      # bs[44] 减弱（原1.0太夸张）
        bs[SERVO_IDX["mouthSmileLeft"]] = 0.3       # bs[43] 增加少许左侧平衡
        bs[SERVO_IDX["mouthDimpleRight"]] = 0.6     # bs[28]
        bs[SERVO_IDX["eyeSquintRight"]] = 0.6       # bs[19]
        bs[SERVO_IDX["browOuterUpRight"]] = 0.5     # bs[4]
        bs[SERVO_IDX["mouthRight"]] = 0.3           # bs[38] 减弱

    elif emotion == "Pride":
        bs[SERVO_IDX["mouthSmileLeft"]] = 0.3       # bs[43]
        bs[SERVO_IDX["mouthSmileRight"]] = 0.3      # bs[44]
        bs[SERVO_IDX["browDownLeft"]] = 0.4         # bs[0]
        bs[SERVO_IDX["browDownRight"]] = 0.4        # bs[1]
        bs[SERVO_IDX["eyeLookDownLeft"]] = 0.7      # bs[10]
        bs[SERVO_IDX["eyeLookDownRight"]] = 0.7     # bs[11]
        bs[SERVO_IDX["mouthUpperUpLeft"]] = 0.8     # bs[47]

    elif emotion == "Trust":
        bs[SERVO_IDX["mouthSmileLeft"]] = 0.4       # bs[43]
        bs[SERVO_IDX["mouthSmileRight"]] = 0.4      # bs[44]
        bs[SERVO_IDX["eyeSquintLeft"]] = 0.2        # bs[18]
        bs[SERVO_IDX["eyeSquintRight"]] = 0.2       # bs[19]
        bs[SERVO_IDX["browInnerUp"]] = 0.3          # bs[2]

    elif emotion == "Love":
        bs[SERVO_IDX["mouthSmileLeft"]] = 0.7       # bs[43]
        bs[SERVO_IDX["mouthSmileRight"]] = 0.7      # bs[44]
        bs[SERVO_IDX["eyeBlinkLeft"]] = 0.3         # bs[8]
        bs[SERVO_IDX["eyeBlinkRight"]] = 0.3        # bs[9]
        bs[SERVO_IDX["eyeSquintLeft"]] = 0.8        # bs[18]
        bs[SERVO_IDX["eyeSquintRight"]] = 0.8       # bs[19]
        bs[SERVO_IDX["cheekSquintLeft"]] = 0.6      # bs[6]
        bs[SERVO_IDX["cheekSquintRight"]] = 0.6     # bs[7]

    elif emotion == "Relief":
        bs[SERVO_IDX["eyeBlinkLeft"]] = 0.9         # bs[8]
        bs[SERVO_IDX["eyeBlinkRight"]] = 0.9        # bs[9]
        bs[SERVO_IDX["mouthFunnel"]] = 0.4          # bs[31]
        bs[SERVO_IDX["jawOpen"]] = 0.1              # bs[24]
        bs[SERVO_IDX["mouthSmileLeft"]] = 0.2       # bs[43]
        bs[SERVO_IDX["mouthSmileRight"]] = 0.2      # bs[44]

    elif emotion == "Hope":
        bs[SERVO_IDX["eyeWideLeft"]] = 0.6          # bs[20]
        bs[SERVO_IDX["eyeWideRight"]] = 0.6         # bs[21]
        bs[SERVO_IDX["browInnerUp"]] = 0.9          # bs[2]
        bs[SERVO_IDX["browOuterUpLeft"]] = 0.7      # bs[3]
        bs[SERVO_IDX["browOuterUpRight"]] = 0.7     # bs[4]
        bs[SERVO_IDX["eyeLookUpLeft"]] = 0.8        # bs[16]
        bs[SERVO_IDX["eyeLookUpRight"]] = 0.8       # bs[17]

    elif emotion == "Anger":
        bs[SERVO_IDX["browDownLeft"]] = 1.0         # bs[0]
        bs[SERVO_IDX["browDownRight"]] = 1.0        # bs[1]
        bs[SERVO_IDX["noseSneerLeft"]] = 0.8        # bs[49]
        bs[SERVO_IDX["noseSneerRight"]] = 0.8       # bs[50]
        bs[SERVO_IDX["mouthPressLeft"]] = 0.8       # bs[35]
        bs[SERVO_IDX["mouthPressRight"]] = 0.8      # bs[36]
        bs[SERVO_IDX["jawForward"]] = 0.5           # bs[22]

    elif emotion == "Disgust":
        bs[SERVO_IDX["noseSneerLeft"]] = 1.0        # bs[49]
        bs[SERVO_IDX["noseSneerRight"]] = 1.0       # bs[50]
        bs[SERVO_IDX["mouthFrownLeft"]] = 0.9       # bs[29]
        bs[SERVO_IDX["mouthFrownRight"]] = 0.9      # bs[30]
        bs[SERVO_IDX["mouthLowerDownLeft"]] = 0.8   # bs[33]
        bs[SERVO_IDX["mouthShrugLower"]] = 0.7      # bs[41]

    elif emotion == "Fear":
        bs[SERVO_IDX["eyeWideLeft"]] = 1.0          # bs[20]
        bs[SERVO_IDX["eyeWideRight"]] = 1.0         # bs[21]
        bs[SERVO_IDX["jawOpen"]] = 0.6              # bs[24]
        bs[SERVO_IDX["browInnerUp"]] = 0.9          # bs[2]
        bs[SERVO_IDX["browDownLeft"]] = 0.2         # bs[0]
        bs[SERVO_IDX["browDownRight"]] = 0.2        # bs[1]
        bs[SERVO_IDX["mouthFunnel"]] = 0.5          # bs[31]

    elif emotion == "Vigilance":
        bs[SERVO_IDX["eyeSquintLeft"]] = 0.7        # bs[18]
        bs[SERVO_IDX["eyeSquintRight"]] = 0.7       # bs[19]
        bs[SERVO_IDX["browDownLeft"]] = 0.6         # bs[0]
        bs[SERVO_IDX["browDownRight"]] = 0.6        # bs[1]
        bs[SERVO_IDX["mouthLeft"]] = 0.3            # bs[32]
        bs[SERVO_IDX["eyeLookInLeft"]] = 0.6        # bs[12]

    elif emotion == "Sad":
        bs[SERVO_IDX["browDownLeft"]] = 0.8         # bs[0]
        bs[SERVO_IDX["browDownRight"]] = 0.8        # bs[1]
        bs[SERVO_IDX["browInnerUp"]] = 0.9          # bs[2]
        bs[SERVO_IDX["mouthFrownLeft"]] = 1.0       # bs[29]
        bs[SERVO_IDX["mouthFrownRight"]] = 1.0      # bs[30]
        bs[SERVO_IDX["mouthRollLower"]] = 0.8       # bs[39]
        bs[SERVO_IDX["mouthRollUpper"]] = 0.8       # bs[40]

    elif emotion == "Loneliness":
        bs[SERVO_IDX["mouthFrownLeft"]] = 0.6       # bs[29]
        bs[SERVO_IDX["mouthFrownRight"]] = 0.6      # bs[30]
        bs[SERVO_IDX["eyeLookDownLeft"]] = 0.8      # bs[10]
        bs[SERVO_IDX["eyeLookDownRight"]] = 0.8     # bs[11]
        bs[SERVO_IDX["eyeBlinkLeft"]] = 0.2         # bs[8]
        bs[SERVO_IDX["eyeBlinkRight"]] = 0.2        # bs[9]

    elif emotion == "Guilt":
        bs[SERVO_IDX["browDownLeft"]] = 0.9         # bs[0]
        bs[SERVO_IDX["browDownRight"]] = 0.9        # bs[1]
        bs[SERVO_IDX["eyeLookDownLeft"]] = 1.0      # bs[10]
        bs[SERVO_IDX["eyeLookDownRight"]] = 1.0     # bs[11]
        bs[SERVO_IDX["eyeBlinkLeft"]] = 0.4         # bs[8]
        bs[SERVO_IDX["eyeBlinkRight"]] = 0.4        # bs[9]

    elif emotion == "Surprise":
        bs[SERVO_IDX["browInnerUp"]] = 1.0          # bs[2]
        bs[SERVO_IDX["browOuterUpLeft"]] = 1.0      # bs[3]
        bs[SERVO_IDX["browOuterUpRight"]] = 1.0     # bs[4]
        bs[SERVO_IDX["jawOpen"]] = 0.9              # bs[24]
        bs[SERVO_IDX["eyeWideLeft"]] = 0.9          # bs[20]
        bs[SERVO_IDX["eyeWideRight"]] = 0.9         # bs[21]
        bs[SERVO_IDX["mouthFunnel"]] = 0.3          # bs[31]

    elif emotion == "Confusion":
        bs[SERVO_IDX["browInnerUp"]] = 1.0          # bs[2]
        bs[SERVO_IDX["browDownLeft"]] = 0.8         # bs[0]
        bs[SERVO_IDX["mouthFunnel"]] = 0.6          # bs[31]
        bs[SERVO_IDX["mouthPucker"]] = 0.5          # bs[37]
        bs[SERVO_IDX["jawLeft"]] = 0.4              # bs[23]

    elif emotion == "Shyness":
        bs[SERVO_IDX["mouthClose"]] = 0.4           # bs[26]
        bs[SERVO_IDX["mouthDimpleLeft"]] = 0.4      # bs[27]
        bs[SERVO_IDX["mouthPressLeft"]] = 0.4       # bs[35]
        bs[SERVO_IDX["mouthPressRight"]] = 0.4      # bs[36]
        bs[SERVO_IDX["eyeLookOutLeft"]] = 0.5       # bs[14]
        bs[SERVO_IDX["eyeLookOutRight"]] = 0.5      # bs[15]
        bs[SERVO_IDX["mouthSmileLeft"]] = 0.2       # bs[43]
        bs[SERVO_IDX["mouthSmileRight"]] = 0.2      # bs[44]

    elif emotion == "Comfort":
        bs[SERVO_IDX["mouthSmileLeft"]] = 0.6       # bs[43] 温和微笑
        bs[SERVO_IDX["mouthSmileRight"]] = 0.6      # bs[44]
        bs[SERVO_IDX["eyeSquintLeft"]] = 0.5        # bs[18] 温柔眼神
        bs[SERVO_IDX["eyeSquintRight"]] = 0.5       # bs[19]
        bs[SERVO_IDX["browInnerUp"]] = 0.3          # bs[2]  关切抬眉
        # 区别于 Relief：没有 eyeBlink(8,9)+mouthFunnel(31) 的叹气感

    elif emotion == "Playful":
        bs[SERVO_IDX["mouthSmileRight"]] = 1.0      # bs[44] 不对称坏笑
        bs[SERVO_IDX["mouthSmileLeft"]] = 0.3       # bs[43]
        bs[SERVO_IDX["browOuterUpRight"]] = 0.8     # bs[4]  右眉挑起
        bs[SERVO_IDX["eyeSquintRight"]] = 0.6       # bs[19] 右眼眯起
        bs[SERVO_IDX["mouthRight"]] = 0.4           # bs[38] 嘴角右偏

    elif emotion == "Impressed":
        bs[SERVO_IDX["browOuterUpLeft"]] = 0.8      # bs[3]  眉毛上扬
        bs[SERVO_IDX["browOuterUpRight"]] = 0.8     # bs[4]
        bs[SERVO_IDX["browInnerUp"]] = 0.6          # bs[2]  抬眉
        bs[SERVO_IDX["eyeWideLeft"]] = 0.5          # bs[20] 眼睛睁大
        bs[SERVO_IDX["eyeWideRight"]] = 0.5         # bs[21]
        bs[SERVO_IDX["mouthSmileLeft"]] = 0.5       # bs[43] 赞赏微笑
        bs[SERVO_IDX["mouthSmileRight"]] = 0.5      # bs[44]
        bs[SERVO_IDX["jawOpen"]] = 0.2              # bs[24] 微微张嘴

    elif emotion == "Concerned":
        bs[SERVO_IDX["browInnerUp"]] = 0.7          # bs[2]  关切抬眉
        bs[SERVO_IDX["browDownLeft"]] = 0.4         # bs[0]  轻微皱眉
        bs[SERVO_IDX["browDownRight"]] = 0.4        # bs[1]
        bs[SERVO_IDX["eyeSquintLeft"]] = 0.5        # bs[18] 专注凝视
        bs[SERVO_IDX["eyeSquintRight"]] = 0.5       # bs[19]
        bs[SERVO_IDX["mouthFrownLeft"]] = 0.3       # bs[29] 嘴角微垂
        bs[SERVO_IDX["mouthFrownRight"]] = 0.3      # bs[30]

    elif emotion == "Awkward":
        bs[SERVO_IDX["mouthSmileRight"]] = 0.7      # bs[44] 不对称尬笑
        bs[SERVO_IDX["mouthSmileLeft"]] = 0.2       # bs[43]
        bs[SERVO_IDX["eyeSquintRight"]] = 0.4       # bs[19]
        bs[SERVO_IDX["eyeLookOutLeft"]] = 0.6       # bs[14] 视线躲闪
        bs[SERVO_IDX["eyeLookOutRight"]] = 0.6      # bs[15]
        bs[SERVO_IDX["mouthDimpleRight"]] = 0.5     # bs[28] 酒窝尬笑

    return bs


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 gen_batch_data.py [emotion_name]")
        sys.exit(1)

    emo_name = sys.argv[1]
    target_bs = get_base_bs(emo_name)
    frames = 60
    sequence = np.zeros((frames, 52))
    for i in range(frames):
        sequence[i] = target_bs * (i / (frames - 1))

    os.makedirs("./result/batch_data", exist_ok=True)
    np.save(f"./result/batch_data/{emo_name}.npy", sequence.astype(np.float32))
    print(f"Saved: ./result/batch_data/{emo_name}.npy")
