# -*- coding: utf-8 -*-
"""
RTSP Viewer + Grid Toggle (press 'g')
"""

import cv2
import numpy as np

#RTSP_URL = "rtsp://admin:cctv%40321@182.79.56.146:554/Streaming/Channels/401"

flag = 0
show_grid = True  # 🔥 toggle state

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

if not cap.isOpened():
    print("❌ Cannot open RTSP stream")
    exit()

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

cv2.namedWindow("RTSP Feed", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RTSP Feed", SCREEN_WIDTH, SCREEN_HEIGHT)

# ================= GRID =================
def draw_grid_with_coordinates(frame, gap=20):
    h, w = frame.shape[:2]

    for x in range(0, w, gap):
        cv2.line(frame, (x, 0), (x, h), (0, 255, 0), 1)
        cv2.putText(frame, str(x), (x, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)

    for y in range(0, h, gap):
        cv2.line(frame, (0, y), (w, y), (0, 255, 0), 1)
        cv2.putText(frame, str(y), (0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)

    return frame

# ================= CLICK =================
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: ({x}, {y})")

cv2.setMouseCallback("RTSP Feed", click_event)

def resize_with_aspect_ratio(frame, width, height):
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    return cv2.resize(frame, (int(w * scale), int(h * scale)))

# ================= LOOP =================
while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Frame not received")
        continue

    if flag == 0:
        print("✅ got it..!!")
        flag = 1

    frame = cv2.resize(frame, (960, 540))

    # 🔥 APPLY GRID ONLY IF TOGGLED ON
    if show_grid:
        frame = draw_grid_with_coordinates(frame, gap=20)

    resized = resize_with_aspect_ratio(frame, SCREEN_WIDTH, SCREEN_HEIGHT)

    canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)

    y_offset = (SCREEN_HEIGHT - resized.shape[0]) // 2
    x_offset = (SCREEN_WIDTH - resized.shape[1]) // 2

    canvas[y_offset:y_offset+resized.shape[0],
           x_offset:x_offset+resized.shape[1]] = resized

    cv2.imshow("RTSP Feed", canvas)

    key = cv2.waitKey(1) & 0xFF

    # 🔥 TOGGLE GRID
    if key == ord('a'):
        show_grid = not show_grid
        print(f"Grid {'ON' if show_grid else 'OFF'}")

    # ESC exit
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()