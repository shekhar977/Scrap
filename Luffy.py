# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:24:05 2026

@author: GTIS-
"""

import cv2
import numpy as np
import time
import smtplib
import torch
from ultralytics import YOLO
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ================= CONFIG =================
MODEL_PATH = r"D:\Python Intern\Scrap Factory\UTU.pt"
RTSP_URL = "rtsp://admin:cctv%40321@182.79.56.146:554/Streaming/Channels/101"

#BELT_CLASS_ID = 0
PERSON_CLASS_ID = 0

PERSON_BOX_SHRINK = 0.15

MOTION_PERCENT_THRESHOLD = 3
STABLE_FRAMES = 1
WARMUP_FRAMES = 3

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "vinodmarkad04@gmail.com"
SENDER_PASSWORD = "obgpkzuamhonjtha"
RECEIVER_EMAIL = "autadeshekhar1@gmail.com"

ALERT_COOLDOWN_SECONDS = 60
last_alert_time = 0

# ================= ROI FUNCTIONS =================

def get_belt_roi_cam1(frame):
    return np.array([
        [500, 1],
        [560, 1],
        [605, 540],
        [500, 540]
    ])


def get_belt_roi_cam2(frame):
    return np.array([
        [460, 140],
        [615, 140],
        [660, 513],
        [450, 520]
    ])


def get_belt_roi_cam3(frame):
    return np.array([
        [520, 60],
        [620, 60],
        [710, 520],
        [450, 520]
    ])


# 👉 SELECT CAMERA HERE
CAMERA_ID = 1


def get_belt_polygon(frame):
    if CAMERA_ID == 1:
        return get_belt_roi_cam1(frame)
    elif CAMERA_ID == 2:
        return get_belt_roi_cam2(frame)
    else:
        return get_belt_roi_cam3(frame)


# ================= EMAIL =================

def send_alert_email():
    global last_alert_time
    if time.time() - last_alert_time < ALERT_COOLDOWN_SECONDS:
        return
    last_alert_time = time.time()

    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg["Subject"] = "🚨 DANGER ALERT"
    msg.attach(MIMEText("Person detected near moving belt", "plain"))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("📧 ALERT EMAIL SENT")
    except Exception as e:
        print("❌ Email error:", e)


# ================= COMMON ALERT FUNCTION =================

# ✅ CHANGED: now uses track_id
def handle_roi_crossing(camera_id, is_belt_moving, track_id):
    """
    Called whenever a person crosses into the ROI.
    Alerts only if the belt is currently moving.
    Always prints the crossing event to console.
    Sends email alert only when belt is moving.

    Args:
        camera_id     (int): Active camera ID (1, 2, or 3)
        is_belt_moving (bool): Whether the belt is currently in motion
        track_id  (int): Optional person index for logging clarity
    """
    person_label = f"Person #{track_id}" if track_id is not None else "Person"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Always log the ROI crossing to console
    print(f"[{timestamp}] ⚠️  ROI CROSSED — Camera {camera_id} | {person_label} detected inside zone")

    if is_belt_moving:
        print(f"[{timestamp}] 🚨 DANGER — Belt is MOVING! Triggering alert for Camera {camera_id}")
        send_alert_email()
    else:
        print(f"[{timestamp}] ✅ SAFE   — Belt is stationary. No alert triggered for Camera {camera_id}")


# ================= INIT =================

model = YOLO(MODEL_PATH)

if torch.cuda.is_available():
    model.to("cuda")
    print("🚀 GPU")
else:
    print("⚠ CPU")

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=200, varThreshold=25, detectShadows=False
)

frame_count = 0

# ================= MAIN LOOP =================

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (960, 540))
    h, w = frame.shape[:2]

    frame_count += 1

    belt_polygon = get_belt_polygon(frame)

    # ===== PERSON DETECTION =====
    person_boxes = []

    # ✅ CHANGED: using tracking instead of plain detection
    results = model.track(
        frame,
        conf=0.25,
        iou=0.5,
        tracker="bytetrack.yaml",  # ✅ NEW
        persist=True,  # ✅ NEW (keeps IDs across frames)
        device=0,
        verbose=False
    )[0]

    if results.boxes is not None:
        boxes = results.boxes
        # ✅ CHANGED: loop rewritten to extract track IDs
        for i in range(len(boxes)):
            cls = int(boxes.cls[i])

            if cls == PERSON_CLASS_ID:
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])

                # ✅ NEW: tracking ID
                track_id = int(boxes.id[i]) if boxes.id is not None else -1

                pw = x2 - x1
                ph = y2 - y1
                sx = int(pw * PERSON_BOX_SHRINK)
                sy = int(ph * PERSON_BOX_SHRINK)

                # ✅ CHANGED: now includes track_id
                person_boxes.append((x1 + sx, y1 + sy, x2 - sx, y2 - sy, track_id))

    # ===== BELT MOTION =====
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [belt_polygon], 255)

    # ✅ Step 1: Grayscale + blur to reduce camera shake noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)

    # ✅ Step 2: CLAHE contrast enhancement for similar colors
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # ✅ Step 3: Convert back to BGR for MOG2
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    # ✅ Step 4: Apply mask to enhanced frame
    belt_roi = cv2.bitwise_and(enhanced_bgr, enhanced_bgr, mask=mask)

    # ✅ Step 5: Background subtraction
    lr = 0.05 if frame_count < WARMUP_FRAMES else 0.005
    fg_mask = bg_subtractor.apply(belt_roi, learningRate=lr)

    # ✅ Step 6: Morphology with different kernel sizes
    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((9, 9), np.uint8)
    kernel_dilate = np.ones((7, 7), np.uint8)

    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)
    fg_mask = cv2.dilate(fg_mask, kernel_dilate, iterations=1)

    # ✅ Step 7: Clip dilation back within belt ROI
    fg_mask = cv2.bitwise_and(fg_mask, mask)

    # ✅ Step 8: Count pixels
    moving_pixels = cv2.countNonZero(fg_mask)
    total_pixels = np.count_nonzero(mask)

    motion_percent = (moving_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    is_moving = motion_percent > MOTION_PERCENT_THRESHOLD

    color = (0, 0, 255) if is_moving else (0, 255, 0)
    label = "BELT MOVING" if is_moving else "BELT STATIONARY"

    cv2.polylines(frame, [belt_polygon], True, color, 2)
    cv2.putText(frame, label, tuple(belt_polygon[0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # ===== PERSON SAFETY =====
    # ✅ CHANGED: unpacking track_id
    for (px1, py1, px2, py2, track_id) in person_boxes:
        cx = (px1 + px2) // 2
        cy = (py1 + py2) // 2

        danger = False
        if cv2.pointPolygonTest(belt_polygon, (cx, cy), False) >= 0:
            # ✅ CHANGED: using real tracking ID instead of index
            handle_roi_crossing(
                camera_id=CAMERA_ID,
                is_belt_moving=is_moving,
                track_id=track_id
            )
            danger = is_moving  # Red box only when belt is moving

        pcolor = (0, 0, 255) if danger else (0, 255, 255)
        cv2.rectangle(frame, (px1, py1), (px2, py2), pcolor, 2)

        # ✅ NEW: show tracking ID on bounding box
        cv2.putText(frame, f"ID {track_id}", (px1, py1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Show danger label on frame
        if danger:
            cv2.putText(frame, "DANGER!", (px1, py1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif cv2.pointPolygonTest(belt_polygon, (cx, cy), False) >= 0:
            cv2.putText(frame, "IN ZONE (Safe)", (px1, py1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.namedWindow("Belt System", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Belt System",
                          cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    #print("Detections:", len(results.boxes) if results.boxes else 0)
    cv2.imshow("Belt System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()