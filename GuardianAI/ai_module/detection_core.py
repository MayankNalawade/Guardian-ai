import cv2
import numpy as np
import time
import requests
from ultralytics import YOLO

# CONFIG
BACKEND_URL = "http://localhost:5000/api/alerts"
CAMERA_ID = "CAM_01"
CONF_THRESHOLD = 0.5
LOITER_SECONDS = 15
FALL_ASPECT_RATIO = 1.4
RUN_SPEED_THRESHOLD = 35
SEND_COOLDOWN = 10

# MODELS
print("Loading YOLOv8 model...")
yolo_model = YOLO("yolov8n.pt")
print("YOLOv8 model loaded")

# Track per-person history: {track_id: {"first_seen": t, "last_pos": (x,y), "last_time": t}}
track_history = {}
last_sent = {}

def send_alert(alert_type, confidence, frame):
    """POST detection event to backend"""
    global last_sent
    now = time.time()
    
    if alert_type in last_sent and now - last_sent[alert_type] < SEND_COOLDOWN:
        return # avoid spam
    
    last_sent[alert_type] = now
    
    # Convert frame to jpg
    _, buffer = cv2.imencode('.jpg', frame)
    files = {'snapshot': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}
    data = {
        'cameraId': CAMERA_ID,
        'type': alert_type,
        'confidence': confidence,
        'timestamp': now
    }
    
    try:
        requests.post(BACKEND_URL, data=data, files=files, timeout=2)
        print(f"ALERT SENT: {alert_type} - {confidence:.2f}")
    except:
        print(f"Could not reach backend: {alert_type}")

def process_frame(frame, track_history):
    """Process one frame: detect, track, and check for suspicious activity."""
    alerts = []
    current_time = time.time()
    results = yolo_model(frame, verbose=False)
    
    for r in results:
        for box in r.boxes:
            cls = yolo_model.names[int(box.cls)]
            conf = float(box.conf)
            
            if cls == 'person' and conf > CONF_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Draw box
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,165,255), 2)
                cv2.putText(frame, f'person {conf:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
                
                track_id = f"person_{center_x}_{center_y}"
                
                # 1. FALL DETECTION: width > height * 1.4
                if w > h * FALL_ASPECT_RATIO:
                    alerts.append(("fall_detected", conf))
                    send_alert("fall_detected", conf, frame)
                
                # 2. LOITERING + RUNNING TRACKING
                if track_id in track_history:
                    prev = track_history[track_id]
                    time_in_frame = current_time - prev["first_seen"]
                    dx = center_x - prev["last_pos"][0]
                    dy = center_y - prev["last_pos"][1]
                    dt = current_time - prev["last_time"]
                    speed = np.sqrt(dx*dx + dy*dy) / (dt + 0.001)
                    
                    # Loitering
                    if time_in_frame > LOITER_SECONDS:
                        alerts.append(("loitering", conf))
                        send_alert("loitering", conf, frame)
                    
                    # Running
                    if speed > RUN_SPEED_THRESHOLD:
                        alerts.append(("running", conf))
                        send_alert("running", conf, frame)
                    
                    track_history[track_id] = {"first_seen": prev["first_seen"], "last_pos": (center_x, center_y), "last_time": current_time}
                else:
                    track_history[track_id] = {"first_seen": current_time, "last_pos": (center_x, center_y), "last_time": current_time}
    
    return frame, alerts