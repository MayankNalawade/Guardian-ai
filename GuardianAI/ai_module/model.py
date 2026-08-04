import cv2
from ultralytics import YOLO

# Load model
model = YOLO('best.pt') # change to 'best.pt' later

def process_frame(frame, track_history):
    alerts = []
    results = model(frame, verbose=False)
    
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]
            
            # FIX: Convert to int and list
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            
            # For now detect 'person'. Change to 'fall' after training
            if class_name == 'person' and conf > 0.5:
                alerts.append((class_name, conf))
    
    return frame, alerts