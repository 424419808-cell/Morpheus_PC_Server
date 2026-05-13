import json
import os

# ================= 配置区 =================
TARGET_IDS = [5258,5259,5260,5261]
INPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_coll", "raw_data", "motor_babbling_data_PC.json")
# ==========================================

LIP_KEYS = [
    "jawOpen", "mouthClose",
    "mouthFunnel", "mouthPucker",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthPressLeft", "mouthPressRight",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthRollUpper", "mouthRollLower",
    "mouthShrugUpper", "mouthShrugLower",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthDimpleLeft", "mouthDimpleRight",
    "mouthLeft", "mouthRight"
]

def compare_lip_blendshapes(target_ids, file_path):
    if not os.path.exists(file_path):
        print(f"找不到文件: {file_path}")
        return

    print(f"正在读取 {file_path} ...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    extracted_data = {}
    for item in data:
        sample_id = item.get("sample_id")
        if sample_id in target_ids:
            bs_dict = item.get("blendshapes", {})
            extracted_data[sample_id] = {k: bs_dict.get(k, 0.0) for k in LIP_KEYS}

    if not extracted_data:
        print("未能在 JSON 中找到任何指定的 sample_id。")
        return

    found_ids = [i for i in target_ids if i in extracted_data]
    print(f"成功找到数据，正在对比 ID: {found_ids}\n")

    col_width_name = 22
    col_width_val = 12

    header = f"{'Blendshape Name'.ljust(col_width_name)} |"
    for sid in found_ids:
        header += f" ID: {str(sid).ljust(col_width_val - 5)} |"

    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for key in LIP_KEYS:
        max_val_for_this_key = max([extracted_data[sid][key] for sid in found_ids])
        if max_val_for_this_key < 0.001:
            continue

        row_str = f"{key.ljust(col_width_name)} |"
        for sid in found_ids:
            val = extracted_data[sid][key]
            row_str += f" {val:.4f}".ljust(col_width_val) + " |"
        print(row_str)

    print("-" * len(header))
    print("提示：为了表格整洁，所有选中 ID 中最大值低于 0.001 的无效键值已被自动隐藏。")

if __name__ == "__main__":
    compare_lip_blendshapes(TARGET_IDS, INPUT_FILE)
