"""
共情映射规则 — 用户观察到7种情绪 → 机器人应表现的共情表情

每个条目包含多个候选共情响应（按优先级排序），训练时按概率选取。
核心原则：不镜像，表达共情。如用户悲伤 → 机器人表达关爱/同情。
"""
import random

# 观察到的用户情绪标签（来自 RandomForest 7 分类）
OBSERVED_EMOTIONS = [
    "Happy", "Sad", "Angry", "Fear", "Surprise", "Disgust", "Neutral"
]

# 24 种可表现的机器人情绪标签（与 EmotionBrain 一致）
ROBOT_EMOTIONS = [
    "Neutral", "Happy", "Excitement", "Humor", "Pride",
    "Trust", "Love", "Relief", "Hope",
    "Anger", "Disgust", "Fear", "Vigilance",
    "Sad", "Loneliness", "Guilt",
    "Surprise", "Confusion", "Shyness",
    "Comfort", "Playful", "Impressed", "Concerned", "Awkward",
]

# 共情映射矩阵：user_emotion → [(robot_emotion, weight), ...]
# weight 表示该共情响应的相对概率
EMPATHY_MAP = {
    "Happy": [
        ("Happy", 0.30), ("Excitement", 0.20), ("Humor", 0.15),
        ("Playful", 0.15), ("Impressed", 0.10),
        ("Love", 0.05), ("Neutral", 0.05),
        # 共享喜悦：积极共情，热情回应，加入调皮和赞赏
    ],
    "Sad": [
        ("Comfort", 0.25), ("Love", 0.20), ("Trust", 0.15),
        ("Concerned", 0.15), ("Neutral", 0.10),
        ("Relief", 0.08), ("Hope", 0.07),
        # 同情安慰：用安慰/关切替代悲伤，温暖陪伴
    ],
    "Angry": [
        ("Neutral", 0.30), ("Confusion", 0.20), ("Trust", 0.20),
        ("Sad", 0.15), ("Vigilance", 0.15),
        # 降级缓和：用中立、困惑、信任来降低冲突
    ],
    "Fear": [
        ("Trust", 0.25), ("Comfort", 0.20), ("Love", 0.15),
        ("Concerned", 0.15), ("Relief", 0.10),
        ("Neutral", 0.10), ("Hope", 0.05),
        # 安抚信赖：表达可靠、安全、安抚，加入关切和安慰
    ],
    "Surprise": [
        ("Surprise", 0.20), ("Impressed", 0.20), ("Confusion", 0.15),
        ("Humor", 0.15), ("Excitement", 0.10),
        ("Awkward", 0.10), ("Neutral", 0.10),
        # 共情惊讶：共享惊讶，转向赞赏/好奇/幽默
    ],
    "Disgust": [
        ("Neutral", 0.25), ("Confusion", 0.20), ("Awkward", 0.15),
        ("Vigilance", 0.15), ("Sad", 0.10), ("Trust", 0.10), ("Concerned", 0.05),
        # 中立好奇：不表现厌恶，用困惑/尴尬和中立回应
    ],
    "Neutral": [
        ("Neutral", 0.40), ("Happy", 0.25), ("Trust", 0.20),
        ("Hope", 0.10), ("Love", 0.05),
        # 温和互动：保持中性或温和积极
    ],
}

# 情绪ID映射（与 gen_batch_data.py 保持一致）
EMOTION_IDS = {
    "Neutral": 0, "Happy": 1, "Excitement": 2, "Humor": 3, "Pride": 4,
    "Trust": 5, "Love": 6, "Relief": 7, "Hope": 8,
    "Anger": 9, "Disgust": 10, "Fear": 11, "Vigilance": 12,
    "Sad": 13, "Loneliness": 14, "Guilt": 15,
    "Surprise": 16, "Confusion": 17, "Shyness": 18,
    "Comfort": 19, "Playful": 20, "Impressed": 21, "Concerned": 22, "Awkward": 23,
}


def sample_empathy(user_emotion):
    """从共情映射中按权重采样一个机器人响应情绪"""
    candidates = EMPATHY_MAP.get(user_emotion, [("Neutral", 1.0)])
    emotions, weights = zip(*candidates)
    return random.choices(emotions, weights=weights, k=1)[0]


def get_empathy_targets(user_emotion):
    """获取用户情绪对应的所有可能共情目标及其权重"""
    return EMPATHY_MAP.get(user_emotion, [("Neutral", 1.0)])


def emotion_to_id(emotion_name):
    """情绪名称 → ID"""
    return EMOTION_IDS.get(emotion_name, 0)


def id_to_emotion(emotion_id):
    """ID → 情绪名称"""
    for name, eid in EMOTION_IDS.items():
        if eid == emotion_id:
            return name
    return "Neutral"
