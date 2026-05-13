import json
import os

# ================= 配置区 =================
INPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_coll", "raw_data", "motor_babbling_data_PC.json")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_coll", "raw_data", "motor_babbling_data_PC_base.json")
# ==========================================

def clean_and_reindex_data():
    if not os.path.exists(INPUT_FILE):
        print(f"找不到文件: {INPUT_FILE}")
        return

    print(f"正在读取 {INPUT_FILE} ...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    original_count = len(data)
    print(f"成功读取，共 {original_count} 组数据。")

    filtered_data = []

    for i, item in enumerate(data):
        if (4999 <= i <= 10999):
            continue
        filtered_data.append(item)

    for new_id, item in enumerate(filtered_data):
        item["sample_id"] = new_id

    new_count = len(filtered_data)
    print(f"已删除指定区间的数据。")
    print(f"清理后剩余 {new_count} 组数据 (删除了 {original_count - new_count} 组)。")
    print(f"sample_id 已重新从 0 编号到 {new_count - 1}。")

    print(f"正在保存到新文件 {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)

    print("处理完成！")

if __name__ == "__main__":
    clean_and_reindex_data()
