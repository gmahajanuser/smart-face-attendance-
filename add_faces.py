'''
import cv2
import pickle
import numpy as np
import os
video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces_data=[]

i=0

name=input("Enter Your Name: ")

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3 ,5)
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        resized_img=cv2.resize(crop_img, (50,50))
        if len(faces_data)<=100 and i%10==0:
            faces_data.append(resized_img)
        i=i+1
        cv2.putText(frame, str(len(faces_data)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==ord('q') or len(faces_data)==20:
        break
video.release()
cv2.destroyAllWindows()

faces_data=np.asarray(faces_data)
faces_data=faces_data.reshape(100, -1)


if 'names.pkl' not in os.listdir('data/'):
    names=[name]*100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names=pickle.load(f)
    names=names+[name]*100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('data/faces_data.pkl', 'rb') as f:
        faces=pickle.load(f)
    faces=np.append(faces, faces_data, axis=0)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)
        '''




"""


import cv2
import pickle
import numpy as np
import os

# Initialize camera
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces_data = []
i = 0

name = input("Enter Your Name: ")

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
faces_data = np.asarray(faces_data).reshape(100, -1)  # Ensure each row has same feature size

# Ensure 'data' folder exists
os.makedirs('data', exist_ok=True)

# Save names
if not os.path.exists('data/names.pkl'):
    names = [name] * 100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names.extend([name] * 100)
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

# Save face data
if not os.path.exists('data/faces_data.pkl'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)

    # Ensure same feature size before appending
    if faces.shape[1] == faces_data.shape[1]:
        faces = np.append(faces, faces_data, axis=0)
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces, f)
    else:
        print(f"Feature mismatch! Existing data: {faces.shape}, New data: {faces_data.shape}")
"""





# import cv2
# import pickle
# import numpy as np
# import os

# # Initialize camera
# video = cv2.VideoCapture(0)
# if not video.isOpened():
#     print("Error: Could not access the camera.")
#     exit()

# facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# faces_data = []
# i = 0

# name = input("Enter Your Name: ")
    
# while True:
#     ret, frame = video.read()
#     if not ret:
#         print("Error: Could not read from camera.")
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = facedetect.detectMultiScale(gray, 1.3, 5)

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

# # Ensure 'data' folder exists
# os.makedirs('data', exist_ok=True)

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


# new text






import cv2
import pickle
import numpy as np
import os

# -------------------- SETUP --------------------

# Create data directory
os.makedirs("data", exist_ok=True)

# Initialize camera
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("❌ Error: Camera not accessible")
    exit()

# Load Haar cascade safely
facedetect = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

faces_data = []
frame_count = 0

name = input("Enter Your Name: ")

print("📸 Capturing face data... Press 'q' to stop")

# -------------------- FACE CAPTURE LOOP --------------------

while True:
    ret, frame = video.read()
    if not ret:
        print("❌ Error: Failed to read frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_crop = frame[y:y+h, x:x+w]
        face_resize = cv2.resize(face_crop, (50, 50))

        # Capture every 10th frame until 100 samples
        if len(faces_data) < 100 and frame_count % 10 == 0:
            faces_data.append(face_resize.flatten())

        frame_count += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Samples: {len(faces_data)}/100",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or len(faces_data) == 100:
        break

# -------------------- CLEANUP --------------------

video.release()
cv2.destroyAllWindows()

if len(faces_data) == 0:
    print("❌ No faces captured")
    exit()

faces_data = np.array(faces_data).reshape(len(faces_data), -1)

# -------------------- LOAD OLD DATA --------------------

names = []
faces = np.empty((0, faces_data.shape[1]))

if os.path.exists("data/names.pkl"):
    with open("data/names.pkl", "rb") as f:
        names = pickle.load(f)

if os.path.exists("data/faces_data.pkl"):
    with open("data/faces_data.pkl", "rb") as f:
        faces = pickle.load(f)

# -------------------- SAVE DATA --------------------

# Check feature compatibility
if faces.size == 0 or faces.shape[1] == faces_data.shape[1]:
    faces = np.vstack((faces, faces_data))
    names.extend([name] * len(faces_data))

    with open("data/faces_data.pkl", "wb") as f:
        pickle.dump(faces, f)

    with open("data/names.pkl", "wb") as f:
        pickle.dump(names, f)

    print(f"✅ Data saved successfully for {name}")
    print(f"📊 Total samples stored: {len(names)}")

else:
    print("❌ Feature size mismatch — data not saved")
























































































































































'''import cv2
import pickle
import numpy as np
import os

# Initialize camera
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Load the Haar cascade classifier for face detection
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces_data = []
i = 0

name = input("Enter Your Name: ")

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Could not read from camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Improved face detection parameters
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        # Ensure cropped region is within bounds
        if y >= 0 and x >= 0 and y + h <= frame.shape[0] and x + w <= frame.shape[1]:
            crop_img = frame[y:y+h, x:x+w]
            resized_img = cv2.resize(crop_img, (50, 50))  # Standardized size
            
            if len(faces_data) < 100 and i % 5 == 0:  # Capture every 5th frame for variety
                faces_data.append(resized_img.flatten())  # Flatten image for storage
            
            i += 1

        cv2.putText(frame, f"Captured: {len(faces_data)}/100", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)

    cv2.imshow("Frame", frame)
    
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == 100:  # Stop when 100 samples are collected
        break

video.release()
cv2.destroyAllWindows()

# Convert to numpy array and reshape correctly
faces_data = np.asarray(faces_data).reshape(len(faces_data), -1)  # Each row represents a face

# Ensure 'data' folder exists
os.makedirs('data', exist_ok=True)

# Load existing names and faces if available
names = []
faces = np.empty((0, faces_data.shape[1]))

if os.path.exists('data/names.pkl'):
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)

if os.path.exists('data/faces_data.pkl'):
    with open('data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)

# Append new data ensuring feature size consistency
if faces.size == 0 or faces.shape[1] == faces_data.shape[1]:
    faces = np.append(faces, faces_data, axis=0)
    names.extend([name] * len(faces_data))
    
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)

    print(f"Data successfully saved for {name}. Total faces stored: {len(faces)}")
else:
    print(f"Feature mismatch! Existing data: {faces.shape}, New data: {faces_data.shape}")
'''
