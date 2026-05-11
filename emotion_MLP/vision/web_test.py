import cv2

# 替换为你刚才成功的那个 Windows IP
win_ip = "172.16.1.55" 
url = f"http://{win_ip}:5000/video_feed"

print(f"正在打开实时画面: {url}")
print("提示：点击画面窗口按 'ESC' 或 'q' 键退出预览")

cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("错误：无法获取画面帧")
        break

    # 在画面上显示“WSL 实时接收中”的字样
    cv2.putText(frame, "WSL Remote Stream", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示窗口
    cv2.imshow('WSL Camera Preview', frame)

    # 按 ESC 键 (27) 或 'q' 键退出
    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()
