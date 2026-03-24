import cv2
import csv
from datetime import datetime

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

def log_attendance(status):
    """Saves the detection event to a CSV file"""
    with open('attendance.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow(["Student_User", timestamp, status])

print("System Starting... Press 'q' to exit.")

# To prevent flooding the CSV, we only log every few seconds
last_log_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    status = "No Face"
    color = (0, 0, 255)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Simple Attention Logic
        center_x = x + (w / 2)
        if abs(center_x - (frame.shape[1] / 2)) < 100:
            status = "Attentive"
            color = (0, 255, 0)
        else:
            status = "Distracted"
            color = (0, 255, 255)

        # Log to CSV every 5 seconds if a face is present
        if time.time() - last_log_time > 5:
            log_attendance(status)
            last_log_time = time.time()

    cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Attendance Tracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
