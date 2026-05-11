import json
import os
from datetime import datetime
import cv2

# ================= 配置区 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_JSON_PATH = os.path.join(BASE_DIR, "..", "data", "motor_babbling_data_PC.json")
CLEAN_JSON_PATH = os.path.join(BASE_DIR, "..", "data", "motor_babbling_data_clean.json")
PROGRESS_PATH = os.path.join(BASE_DIR, "..", "data", "data_cleaner_progress.json")

IMAGE_DIR = r"I:\captured_faces"
# ==========================================

def save_progress(last_index, clean_dataset):
    data = {
        "last_reviewed_index": last_index,
        "clean_dataset": clean_dataset,
        "timestamp": datetime.now().isoformat()
    }
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_progress():
    if not os.path.exists(PROGRESS_PATH):
        return 0, []
    with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["last_reviewed_index"], data["clean_dataset"]

def delete_progress():
    if os.path.exists(PROGRESS_PATH):
        os.remove(PROGRESS_PATH)

def calculate_uncanny_score(bs):
    score = 0
    warnings = []

    smile_diff = abs(bs.get('mouthSmileLeft', 0) - bs.get('mouthSmileRight', 0))
    if smile_diff > 0.4:
        score += 50
        warnings.append(f"嘴角极度不对称 (差值:{smile_diff:.2f})")

    smile_avg = (bs.get('mouthSmileLeft', 0) + bs.get('mouthSmileRight', 0)) / 2
    eye_wide_avg = (bs.get('eyeWideLeft', 0) + bs.get('eyeWideRight', 0)) / 2
    if smile_avg > 0.5 and eye_wide_avg > 0.5:
        score += 80
        warnings.append("瞪眼假笑 (恐怖谷高发)")

    if bs.get('jawOpen', 0) > 0.6 and bs.get('mouthClose', 0) > 0.8:
        score += 60
        warnings.append("下巴张开但嘴唇紧闭")

    return score, warnings

def main():
    if not os.path.exists(DATA_JSON_PATH):
        print("找不到原始数据 JSON 文件！")
        return

    with open(DATA_JSON_PATH, "r", encoding='utf-8') as f:
        dataset = json.load(f)

    total = len(dataset)
    clean_dataset = []
    discarded_count = 0
    start_index = 0

    print("=========================================")
    print("      Morpheus 仿生人数据清洗工具启动")
    print("=========================================")

    saved_index, saved_clean = load_progress()
    if saved_index > 0:
        print(f"\n发现上次审核进度：上次审核到第 {saved_index + 1} 张（索引 {saved_index}），已保留 {len(saved_clean)} 张。")
        choice = input(f"是否从第 {saved_index + 2} 张继续？(Y/n): ").strip().lower()
        if choice in ('', 'y', 'yes'):
            start_index = saved_index + 1
            clean_dataset = saved_clean
            print(f"从第 {start_index + 1} 张继续审核。")
        else:
            delete_progress()
            print("已清除旧进度，将重新开始。")

    if start_index == 0:
        while True:
            raw = input(f"\n总样本数: {total}。请输入起始索引 (0-{total-1})，按回车默认从 0 开始: ").strip()
            if raw == '':
                break
            try:
                idx = int(raw)
                if 0 <= idx < total:
                    start_index = idx
                    break
                else:
                    print(f"索引超出范围，请输入 0-{total-1} 之间的整数。")
            except ValueError:
                print("请输入有效的整数。")

    if start_index > 0 and len(clean_dataset) == 0:
        clean_dataset = dataset[:start_index]
        print(f"索引 0-{start_index - 1} 的 {start_index} 条数据已自动保留。")

    print(f"\n从第 {start_index + 1} 张（索引 {start_index}）开始审核。")
    print("操作指南：")
    print("  - 窗口弹出后，按 [K] 键：保留 (Keep) 此数据")
    print("  - 窗口弹出后，按 [D] 键：删除 (Delete) 此数据")
    print("  - 按 [Q] 键：随时退出并保存进度")
    print("=========================================\n")

    completed_all = False

    for index in range(start_index, total):
        sample = dataset[index]
        img_filename = sample.get("image_file")
        img_path = os.path.join(IMAGE_DIR, img_filename)

        if not os.path.exists(img_path):
            print(f"找不到图片 {img_filename}，已自动跳过该条数据。")
            save_progress(index, clean_dataset)
            continue

        img = cv2.imread(img_path)
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        bs = sample.get("blendshapes", {})

        uncanny_score, warnings = calculate_uncanny_score(bs)

        display_img = img.copy()

        cv2.putText(display_img, f"Kept: {len(clean_dataset)} | ID: {sample['sample_id']} ({index+1}/{total})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if uncanny_score > 0:
            cv2.putText(display_img, f"Uncanny Score: {uncanny_score}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            y_offset = 140
            for w in warnings:
                cv2.putText(display_img, f"WARN: {w}", (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                y_offset += 40
        else:
            cv2.putText(display_img, "Status: Normal", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Data Cleaner (Press K to Keep, D to Delete, Q to Quit)", display_img)

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord('k') or key == ord('K'):
                clean_dataset.append(sample)
                print(f"[保留] Sample {sample['sample_id']}")
                save_progress(index, clean_dataset)
                break

            elif key == ord('d') or key == ord('D'):
                print(f"[删除] Sample {sample['sample_id']} 已丢弃，正在删除源文件...")
                try:
                    os.remove(img_path)
                except Exception as e:
                    print(f"  -> 删除图片失败: {e}")
                discarded_count += 1
                save_progress(index, clean_dataset)
                break

            elif key == ord('q') or key == ord('Q'):
                cv2.destroyAllWindows()
                print(f"\n[中断] 已审核到第 {index + 1} 张（索引 {index}）。")
                while True:
                    print("  [1] 保存进度并退出（下次可继续）")
                    print("  [2] 保留剩余未审核数据并退出")
                    print("  [3] 放弃本次所有操作")
                    choice = input("请选择 (1/2/3): ").strip()
                    if choice == '1':
                        save_progress(index, clean_dataset)
                        print("进度已保存，下次启动可从中断处继续。")
                        break
                    elif choice == '2':
                        clean_dataset.extend(dataset[index:])
                        save_progress(index, clean_dataset)
                        print(f"已保留剩余 {total - index} 条未审核数据，进度已保存。")
                        break
                    elif choice == '3':
                        print("已放弃本次操作。")
                        break
                    else:
                        print("无效选择，请输入 1、2 或 3。")
                break

        if key == ord('q') or key == ord('Q'):
            break
    else:
        completed_all = True

    cv2.destroyAllWindows()

    with open(CLEAN_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(clean_dataset, f, indent=4, ensure_ascii=False)

    if completed_all:
        delete_progress()
        print("\n=========================================")
        print("全部审核完成！")
        print(f"原始数据总量: {total}")
        print(f"人工删除数量: {discarded_count}")
        print(f"剩余纯净数据: {len(clean_dataset)}")
        print(f"干净的 JSON 已保存至: {CLEAN_JSON_PATH}")
        print("=========================================")
    else:
        print("\n=========================================")
        print("审核中断退出")
        print(f"本次删除数量: {discarded_count}")
        print(f"当前已保留: {len(clean_dataset)}")
        print(f"干净的 JSON 已保存至: {CLEAN_JSON_PATH}")
        print("=========================================")

if __name__ == '__main__':
    main()
