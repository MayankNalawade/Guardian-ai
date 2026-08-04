import cv2
import firebase_admin
from firebase_admin import credentials, firestore, storage
from datetime import datetime
from model import process_frame

# 1. Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json") # make sure this file is in ai_module folder
firebase_admin.initialize_app(cred, {
    'storageBucket': 'YOUR-PROJECT-ID.appspot.com' # <-- CHANGE THIS to your Firebase Storage bucket name
})
db = firestore.client()
bucket = storage.bucket()

print("Loading YOLOv8 model...")

# 2. Open webcam
cap = cv2.VideoCapture(0) # 0 = default camera. Use 1 if you have 2 cameras
track_history = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Run YOLO detection - this now returns frame AND alerts
    frame, alerts = process_frame(frame, track_history)

    # 4. Send alert to Firebase if fall detected
    for alert in alerts:
        label = alert[0]
        confidence = alert[1]
        
        if label == 'fall' and confidence > 0.8: # Only send if >80% confident
            
            # 4a. Save the fall frame as image
            filename = f"fall_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            
            # 4b. Upload to Firebase Storage
            blob = bucket.blob(f"falls/{filename}")
            blob.upload_from_filename(filename)
            blob.make_public()
            image_url = blob.public_url
            
            # 4c. Send data to Firestore
            alert_data = {
                "camera_id": "CAM001",
                "timestamp": datetime.now(),
                "image_url": image_url,
                "lat": 19.0760, # Mumbai coords for now
                "lng": 72.8777,
                "status": "New",
                "confidence": float(confidence),
                "message": "Person Fell Detected"
            }
            db.collection("alerts").add(alert_data)
            print(f"🚨 FALL ALERT SENT! Confidence: {confidence}")
            print(f"📸 Image: {image_url}")

    # 5. Show video
    cv2.imshow("GuardianAI - Fall Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()