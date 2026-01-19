# from sklearn.neighbors import KNeighborsClassifier
# import cv2
# import pickle
# import numpy as np
# import os
# import csv
# import time
# from datetime import datetime

# # ---------------- VOICE (SAFE) ----------------
# try:
#     from win32com.client import Dispatch
#     speaker = Dispatch("SAPI.SpVoice")
#     def speak(text):
#         speaker.Speak(text)
# except:
#     def speak(text):
#         pass

# # ---------------- CAMERA ----------------
# video = cv2.VideoCapture(0)
# if not video.isOpened():
#     print("❌ Camera not accessible")
#     exit()

# # ---------------- FACE DETECTOR ----------------
# cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# facedetect = cv2.CascadeClassifier(cascade_path)

# if facedetect.empty():
#     print("❌ Haarcascade not loaded")
#     exit()

# # ---------------- LOAD DATA ----------------
# try:
#     with open("data/names.pkl", "rb") as f:
#         LABELS = pickle.load(f)
#     with open("data/faces_data.pkl", "rb") as f:
#         FACES = pickle.load(f)
# except:
#     print("❌ Training data not found")
#     exit()

# FACES = np.array(FACES)
# LABELS = np.array(LABELS)

# min_len = min(len(FACES), len(LABELS))
# FACES = FACES[:min_len]
# LABELS = LABELS[:min_len]

# print("Faces:", FACES.shape)
# print("Labels:", LABELS.shape)

# # ---------------- TRAIN MODEL ----------------
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(FACES, LABELS)

# # ---------------- BACKGROUND ----------------
# bg = None
# if os.path.exists("authentech.jpg"):
#     bg = cv2.imread("authentech.jpg")
#     bg = cv2.resize(bg, (800, 600))

# # ---------------- ATTENDANCE ----------------
# os.makedirs("Attendance", exist_ok=True)
# COL_NAMES = ["NAME", "TIME"]
# marked_attendance = set()
# exit_timer = None

# print("📸 Attendance Started (Press Q to quit)")

# # ---------------- MAIN LOOP ----------------
# while True:
#     ret, frame = video.read()
#     if not ret:
#         continue

#     frame = cv2.resize(frame, (640, 480))
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     faces = facedetect.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:
#         face = frame[y:y+h, x:x+w]
#         face = cv2.resize(face, (50, 50)).flatten().reshape(1, -1)

#         if face.shape[1] != FACES.shape[1]:
#             continue

#         name = knn.predict(face)[0]

#         if name not in marked_attendance:
#             ts = time.time()
#             date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
#             time_now = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

#             file = f"Attendance/Attendance_{date}.csv"
#             exists = os.path.isfile(file)

#             with open(file, "a", newline="") as f:
#                 writer = csv.writer(f)
#                 if not exists:
#                     writer.writerow(COL_NAMES)
#                 writer.writerow([name, time_now])

#             speak(f"Attendance marked for {name}")
#             marked_attendance.add(name)
#             exit_timer = time.time() + 10  # auto-exit after 10 sec

#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(frame, name, (x, y-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

#     # ---------------- DISPLAY ----------------
#     if bg is not None:
#         display = bg.copy()
#         display[60:540, 80:720] = frame
#         cv2.imshow("AuthenTech Face Attendance", display)
#     else:
#         cv2.imshow("Face Attendance", frame)

#     # Auto exit
#     if exit_timer and time.time() > exit_timer:
#         break

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # ---------------- CLEANUP ----------------
# video.release()
# cv2.destroyAllWindows()
# print("✅ Attendance session ended")










from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime

# ---------------- VOICE SUPPORT ----------------
try:
    from win32com.client import Dispatch
    speaker = Dispatch("SAPI.SpVoice")
    def speak(text):
        speaker.Speak(text)
except:
    def speak(text):
        pass

# ---------------- CAMERA ----------------
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("❌ Camera not accessible")
    exit()

# ---------------- FACE DETECTOR ----------------
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
facedetect = cv2.CascadeClassifier(cascade_path)

# ---------------- LOAD DATA ----------------
with open("data/names.pkl", "rb") as f:
    LABELS = pickle.load(f)
with open("data/faces_data.pkl", "rb") as f:
    FACES = pickle.load(f)

FACES = np.array(FACES)
LABELS = np.array(LABELS)

min_len = min(len(FACES), len(LABELS))
FACES = FACES[:min_len]
LABELS = LABELS[:min_len]

# ---------------- TRAIN MODEL ----------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# ---------------- BACKGROUND ----------------
bg = None
if os.path.exists("authentech.jpg"):
    bg = cv2.imread("authentech.jpg")
    bg = cv2.resize(bg, (800, 600))

# ---------------- ATTENDANCE ----------------
os.makedirs("Attendance", exist_ok=True)
COL_NAMES = ["NAME", "TIME"]
marked_attendance = set()

exit_after = 10   # seconds
exit_time = None

print("📸 Face Attendance Started (Auto Exit Enabled)")

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = video.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (50, 50)).flatten().reshape(1, -1)

        if face.shape[1] != FACES.shape[1]:
            continue

        name = knn.predict(face)[0]

        if name not in marked_attendance:
            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            time_now = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

            file = f"Attendance/Attendance_{date}.csv"
            exists = os.path.isfile(file)

            with open(file, "a", newline="") as f:
                writer = csv.writer(f)
                if not exists:
                    writer.writerow(COL_NAMES)
                writer.writerow([name, time_now])

            speak(f"Attendance marked for {name}")
            marked_attendance.add(name)

            # Start auto-exit timer
            exit_time = time.time() + exit_after

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # ---------------- DISPLAY ----------------
    if bg is not None:
        display = bg.copy()
        display[60:540, 80:720] = frame
        cv2.imshow("AuthenTech Face Attendance", display)
    else:
        cv2.imshow("Face Attendance", frame)

    # ---------------- AUTO EXIT ----------------
    if exit_time and time.time() > exit_time:
        break

    cv2.waitKey(1)  # Required for window refresh

# ---------------- CLEANUP ----------------
video.release()
cv2.destroyAllWindows()
print("✅ Attendance session ended automatically")





from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime

# ---------------- VOICE SUPPORT ----------------
try:
    from win32com.client import Dispatch
    speaker = Dispatch("SAPI.SpVoice")
    def speak(text):
        speaker.Speak(text)
except:
    def speak(text):
        pass

# ---------------- CAMERA ----------------
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("❌ Camera not accessible")
    exit()

# ---------------- FACE DETECTOR ----------------
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
facedetect = cv2.CascadeClassifier(cascade_path)

if facedetect.empty():
    print("❌ Haarcascade not loaded")
    exit()

# ---------------- LOAD TRAINING DATA ----------------
try:
    with open("data/names.pkl", "rb") as f:
        LABELS = pickle.load(f)
    with open("data/faces_data.pkl", "rb") as f:
        FACES = pickle.load(f)
except:
    print("❌ Training data not found")
    exit()

FACES = np.array(FACES)
LABELS = np.array(LABELS)

min_len = min(len(FACES), len(LABELS))
FACES = FACES[:min_len]
LABELS = LABELS[:min_len]

# ---------------- TRAIN MODEL ----------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# ---------------- LOAD YOUR BACKGROUND IMAGE ----------------
BG_PATH = "authentech.jpg"

if not os.path.exists(BG_PATH):
    print("❌ Background image not found")
    exit()

bg = cv2.imread(BG_PATH)
bg = cv2.resize(bg, (1000, 700))   # Safe fixed size

# ---------------- ATTENDANCE ----------------
os.makedirs("Attendance", exist_ok=True)
COL_NAMES = ["NAME", "TIME"]
marked_attendance = set()

exit_after = 10
exit_time = None

print("📸 Face Attendance Started (Auto Exit Enabled)")

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = video.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (500, 380))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (50, 50)).flatten().reshape(1, -1)

        if face.shape[1] != FACES.shape[1]:
            continue

        name = knn.predict(face)[0]

        if name not in marked_attendance:
            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            time_now = datetime
