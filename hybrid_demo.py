import cv2
import requests
import time
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore

# ========== 1. PUT YOUR KEYS HERE ==========
ROBOFLOW_API_KEY = " yTpay5EcTZnc9SzA8zVT"
ROBOFLOW_URL = "https://detect.roboflow.com/violence-detection-nbx24/1"
FIREBASE_KEY_FILE = "/Users/mayanknalawade/Desktop/guardianai-key.json" # path to your firebase json
# ===========================================

# ========== 2. INIT EVERYTHING ==========
print("[INIT] Loading Local Model...")
model = YOLO("/Users/mayanknalawade/Desktop/best.pt") # your trained sit/stand/fall model

print("[INIT] Connecting Firebase...")
FIREBASE_KEY_FILE = "/Users/mayanknalawade/Desktop/guardianai-key.json"
cred = credentials.Certificate(FIREBASE_KEY_FILE)
firebase_admin.initialize_app(cred)
db = firestore.client()

cap = cv2.VideoCapture(0)
last_api_call = 0
API_CALL_INTERVAL = 2 # call roboflow every 2 seconds to save credits
# ===========================================

print("[START] Guardian AI Running. Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. LOCAL MODEL: sit/stand/fall - Green boxes
    results = model(frame, conf=0.4, verbose=False)
    annotated_frame = results[0].plot() # draws green boxes

    # 2. ROBOFLOW API: violence/fight - Red boxes every 2 sec
    if time.time() - last_api_call > API_CALL_INTERVAL:
        last_api_call = time.time()
        cv2.imwrite("temp.jpg", frame) # save frame to send

        try:
            res = requests.post(ROBOFLOW_URL,
                params={"api_key": ROBOFLOW_API_KEY},
                files={"file": open("temp.jpg", "rb")},
                timeout=15)
            api_data = res.json()
        except Exception as e:
            print("[ERROR] Roboflow API request failed:", e)

            # Draw red boxes for violence
            for pred in api_data.get('predictions', []):
                x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
                x1, y1 = x - w//2, y - h//2
                x2, y2 = x + w//2, y + h//2
                label = f"violence {pred['confidence']:.2f}"
                cv2.rectangle(annotated_frame, (x1,y1), (x2,y2), (0,0,255), 2) # Red
                cv2.putText(annotated_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        except requests.exceptions.Timeout:
                print("[WARN] Roboflow API timed out. Skipping this frame.")
        except Exception as e:
                print("[ERROR] Roboflow API failed:", e)
                

                # 3. FIREBASE: Save alert if violence detected with >50% confidence
                if pred['confidence'] > 0.5:
                    db.collection("alerts").add({
                        "type": "violence",
                        "confidence": float(pred['confidence']),
                        "timestamp": firestore.SERVER_TIMESTAMP,
                        "location": "Classroom Cam 1" # change this
                    })
                    print(f"[FIREBASE] ALERT SAVED: violence - {pred['confidence']:.2f}")

        except Exception as e:
            print("[ERROR] Roboflow API failed:", e)

    # 4. SHOW WINDOW
    cv2.imshow("Guardian AI - Green=Local Red=API", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[STOP] Guardian AI Stopped")
