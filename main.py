import cv2
import time

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

print("System Starting... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for natural mirror view
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.営業_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        status = "No Face Detected"
        color = (0, 0, 255) # Red
    else:
        for (x, y, w, h) in faces:
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Simple Attention Logic: 
            # If the face is roughly centered, they are "Attentive"
            # If the face is too far to the edges, they are "Distracted"
            center_x = x + (w / 2)
            frame_center = frame.shape[1] / 2
            
            if abs(center_x - frame_center) < 100:
                status = "Status: Attentive"
                color = (0, 255, 0) # Green
            else:
                status = "Status: Distracted"
                color = (0, 255, 255) # Yellow

    # Display status on screen
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Show the output
    cv2.imshow('Attention Tracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
