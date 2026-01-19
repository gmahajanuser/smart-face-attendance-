'''import cv2
import pickle
import numpy as np
import os
import getpass

# Initialize camera
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not access the camera.")
    exit()

facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces_data = []
i = 0

# Ensure 'data' folder exists
os.makedirs('data', exist_ok=True)

# Load existing PINs if available
pins = {}
if os.path.exists('data/pins.pkl'):
    with open('data/pins.pkl', 'rb') as f:
        pins = pickle.load(f)

name = input("Enter Your Name: ")

# If user already exists, verify PIN; otherwise, create a new one
if name in pins:
    input_pin = getpass.getpass("Enter PIN to Modify Data: ")
    if input_pin != pins[name]:
        print("Incorrect PIN! Access Denied.")
        exit()
else:
    new_pin = getpass.getpass("Set a New PIN: ")
    confirm_pin = getpass.getpass("Confirm New PIN: ")
    if new_pin != confirm_pin:
        print("PINs do not match! Try again.")
        exit()
    pins[name] = new_pin
    with open('data/pins.pkl', 'wb') as f:
        pickle.dump(pins, f)
    print("PIN successfully set!")

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Could not read from camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50))  # Ensure consistent size
        
        if len(faces_data) < 100 and i % 10 == 0:
            faces_data.append(resized_img.flatten())  # Flatten image to match feature size
        i += 1

        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == 100:
        break

video.release()
cv2.destroyAllWindows()

# Convert to numpy array and reshape correctly
faces_data = np.asarray(faces_data).reshape(len(faces_data), -1)  # Ensure each row has same feature size

# Load existing names and faces if available
names = []
faces = np.empty((0, faces_data.shape[1]))

if os.path.exists('data/names.pkl'):
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
if os.path.exists('data/faces_data.pkl'):
    with open('data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)

# Ensure feature size consistency before appending
if faces.shape[1] == faces_data.shape[1]:
    faces = np.append(faces, faces_data, axis=0)
    names.extend([name] * len(faces_data))
    
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)
    print(f"Data successfully saved for {name}.")
else:
    print(f"Feature mismatch! Existing data: {faces.shape}, New data: {faces_data.shape}")'''






# import cv2
# import pickle
# import numpy as np
# import os
# import getpass

# # Initialize camera
# video = cv2.VideoCapture(0)
# if not video.isOpened():
#     print("Error: Could not access the camera.")
#     exit()

# facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# faces_data = []
# i = 0

# # Ensure 'data' folder exists
# os.makedirs('data', exist_ok=True)

# # Load existing PINs if available
# pins = {}
# if os.path.exists('data/pins.pkl'):
#     with open('data/pins.pkl', 'rb') as f:
#         pins = pickle.load(f)

# name = input("Enter Your Name: ")

# # If user already exists, verify PIN; otherwise, create a new one
# if name in pins:
#     input_pin = getpass.getpass("Enter PIN to Modify Data: ")
#     if input_pin != pins[name]:
#         print("Incorrect PIN! Access Denied.")
#         exit()
# else:
#     new_pin = getpass.getpass("Set a New PIN: ")
#     confirm_pin = getpass.getpass("Confirm New PIN: ")
#     if new_pin != confirm_pin:
#         print("PINs do not match! Try again.")
#         exit()
#     pins[name] = new_pin
#     with open('data/pins.pkl', 'wb') as f:
#         pickle.dump(pins, f)
#     print("PIN successfully set!")

# def detect_spoof(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
#     return laplacian_var > 15  # Threshold for real vs printed images

# while True:
#     ret, frame = video.read()
#     if not ret:
#         print("Error: Could not read from camera.")
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = facedetect.detectMultiScale(gray, 1.3, 5)

#     if len(faces) == 0 or not detect_spoof(frame):
#         print("Spoofing detected! Ensure a real face is used.")
#         continue

#     for (x, y, w, h) in faces:
#         crop_img = frame[y:y+h, x:x+w]
#         resized_img = cv2.resize(crop_img, (50, 50))  # Ensure consistent size
        
#         if len(faces_data) < 100 and i % 10 == 0:
#             faces_data.append(resized_img.flatten())  # Flatten image to match feature size
#         i += 1

#         cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

#     cv2.imshow("Frame", frame)
#     k = cv2.waitKey(1)
#     if k == ord('q') or len(faces_data) == 100:
#         break

# video.release()
# cv2.destroyAllWindows()

# # Convert to numpy array and reshape correctly
# faces_data = np.asarray(faces_data).reshape(len(faces_data), -1)  # Ensure each row has same feature size

# # Load existing names and faces if available
# names = []
# faces = np.empty((0, faces_data.shape[1]))

# if os.path.exists('data/names.pkl'):
#     with open('data/names.pkl', 'rb') as f:
#         names = pickle.load(f)
# if os.path.exists('data/faces_data.pkl'):
#     with open('data/faces_data.pkl', 'rb') as f:
#         faces = pickle.load(f)

# # Ensure feature size consistency before appending
# if faces.shape[1] == faces_data.shape[1]:
#     faces = np.append(faces, faces_data, axis=0)
#     names.extend([name] * len(faces_data))
    
#     with open('data/names.pkl', 'wb') as f:
#         pickle.dump(names, f)
#     with open('data/faces_data.pkl', 'wb') as f:
#         pickle.dump(faces, f)
#     print(f"Data successfully saved for {name}.")
# else:
#     print(f"Feature mismatch! Existing data: {faces.shape}, New data: {faces_data.shape}")
























import cv2
import pickle
import numpy as np
import os
import getpass

# -------------------- SETUP --------------------

os.makedirs("data", exist_ok=True)

video = cv2.VideoCapture(0)
if not video.isOpened():
    print("❌ Camera not accessible")
    exit()

# Load Haarcascade safely
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
facedetect = cv2.CascadeClassifier(cascade_path)

if facedetect.empty():
    print("❌ Haarcascade not loaded")
    exit()

# Load PINs
pins = {}
if os.path.exists("data/pins.pkl"):
    with open("data/pins.pkl", "rb") as f:
        pins = pickle.load(f)

name = input("Enter Your Name: ")

# PIN authentication
if name in pins:
    pin = getpass.getpass("Enter PIN to modify data: ")
    if pin != pins[name]:
        print("❌ Incorrect PIN")
        exit()
else:
    pin1 = getpass.getpass("Set new PIN: ")
    pin2 = getpass.getpass("Confirm PIN: ")
    if pin1 != pin2:
        print("❌ PINs do not match")
        exit()
    pins[name] = pin1
    with open("data/pins.pkl", "wb") as f:
        pickle.dump(pins, f)
    print("✅ PIN saved")

# -------------------- SPOOF CHECK --------------------

def detect_spoof(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() > 15

# -------------------- FACE CAPTURE --------------------

faces_data = []
count = 0

print("📸 Capturing face data (press Q to quit)")

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        if y < 0 or x < 0:
            continue

        face = frame[y:y+h, x:x+w]

        if not detect_spoof(face):
            cv2.putText(frame, "Spoof Detected", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            continue

        face = cv2.resize(face, (50, 50))

        if len(faces_data) < 100 and count % 10 == 0:
            faces_data.append(face.flatten())

        count += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{len(faces_data)}/100", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or len(faces_data) == 100:
        break

video.release()
cv2.destroyAllWindows()

# -------------------- SAVE DATA --------------------

if len(faces_data) == 0:
    print("❌ No face data collected")
    exit()

faces_data = np.array(faces_data).reshape(len(faces_data), -1)

names = []
faces = np.empty((0, faces_data.shape[1]))

if os.path.exists("data/names.pkl"):
    with open("data/names.pkl", "rb") as f:
        names = pickle.load(f)

if os.path.exists("data/faces_data.pkl"):
    with open("data/faces_data.pkl", "rb") as f:
        faces = pickle.load(f)

if faces.size == 0 or faces.shape[1] == faces_data.shape[1]:
    faces = np.vstack((faces, faces_data))
    names.extend([name] * len(faces_data))

    with open("data/faces_data.pkl", "wb") as f:
        pickle.dump(faces, f)

    with open("data/names.pkl", "wb") as f:
        pickle.dump(names, f)

    print(f"✅ Face data saved for {name}")
else:
    print("❌ Feature size mismatch — data not saved")
