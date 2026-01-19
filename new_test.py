from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(text):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(text)

# Open camera
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error: Could not access the camera.")
    exit()

print("Camera is open. Initializing attendance system...")

# Load face detection model
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Load face data
try:
    with open('data/names.pkl', 'rb') as w:
        LABELS = pickle.load(w)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)
except FileNotFoundError:
    print("Error: Face data files not found. Train the model first.")
    exit()

# Ensure consistent length of data
if len(FACES) != len(LABELS):
    min_len = min(len(FACES), len(LABELS))
    FACES = FACES[:min_len]
    LABELS = LABELS[:min_len]

FACES = np.array(FACES)
LABELS = np.array(LABELS)

print('Shape of Faces matrix --> ', FACES.shape)
print('Shape of Labels array --> ', LABELS.shape)

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load background image (if exists)
if os.path.exists("authentech.jpg"):
    imgBackground = cv2.imread("authentech.jpg")
else:
    imgBackground = None

# Ensure Attendance directory exists
os.makedirs("Attendance", exist_ok=True)

# Column names for attendance CSV
COL_NAMES = ['NAME', 'TIME']

# Keep track of already marked attendance
marked_attendance = set()

start_time = time.time()
exit_time = start_time + 10  # Set exit time 10 seconds from now

while time.time() < exit_time:
    ret, frame = video.read()

    if not ret or frame is None:
        print("Warning: Could not read frame from camera.")
        continue

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        target_size = (50, 50)
        resized_img = cv2.resize(crop_img, target_size).flatten().reshape(1, -1)

        if resized_img.shape[1] != FACES.shape[1]:
            print(f"Feature mismatch: Expected {FACES.shape[1]}, but got {resized_img.shape[1]}")
            continue

        output = knn.predict(resized_img)[0]

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Draw background rectangle for text
        cv2.rectangle(frame, (x, y - 25), (x + w, y), (0, 0, 0), -1)
        # Display person's name
        cv2.putText(frame, str(output), (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if output not in marked_attendance:
            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            attendance_file = f"Attendance/Attendance_{date}.csv"
            file_exists = os.path.isfile(attendance_file)

            attendance = [output, timestamp]
            with open(attendance_file, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(COL_NAMES)
                writer.writerow(attendance)

            speak(f"Attendance marked for {output}")
            marked_attendance.add(output)

    if imgBackground is not None:
        imgBackground[162:162 + frame.shape[0], 55:55 + frame.shape[1]] = frame
        cv2.imshow("Face Attendance System", imgBackground)
    else:
        cv2.imshow("Face Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Exiting after 10 seconds...")
video.release()
cv2.destroyAllWindows()