import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# 1. Connect to Firebase
# The ../ means "go up 1 folder" because serviceAccountKey.json is in GuardianAI folder
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

print("Connected to Firestore ✅")

# 2. Create a test fall alert
alert_data = {
    "camera_id": "CAM001",
    "timestamp": datetime.now(),
    "image_url": "local/fall_test.jpg",  # no storage yet, just a placeholder
    "lat": 19.0760,  # Mumbai lat
    "lng": 72.8777,  # Mumbai lng
    "status": "New",
    "confidence": 0.95,
    "message": "Person Fell Detected"
}

# 3. Push it to Firestore 'alerts' collection
db.collection("alerts").add(alert_data)
print("Test Alert Sent! Check Firebase Console > Firestore > alerts")