import cv2
import os
import tensorflow as tf
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

# Initialize mixer for alarm
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load Haar cascades
face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt2.xml')  # better for head tilt
face_with_glasses = cv2.CascadeClassifier('haar cascade files/haarcascade_eye_tree_eyeglasses.xml')  # for glasses
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')  # detect left eye
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')  # detect right eye

lbl = ['Close', 'Open']

# Load the trained model
model = load_model('./content/model-after-augmv2.h5')

# Start video capture
cap = cv2.VideoCapture(0)

# Set desired FPS and resolution
desired_fps = 60
cap.set(cv2.CAP_PROP_FPS, desired_fps)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce width for higher FPS
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Reduce height for higher FPS

# Verify the actual FPS
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Actual FPS: {actual_fps}")

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# Initialize variables
count = 0
score = 0
thicc = 2
rpred = [1, 0]
lpred = [1, 0]
val1 = 1
val2 = 1
frame_count = 0
skip_interval = 2  # Skip every other frame to increase perceived FPS

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame_count += 1

    # Skip processing for some frames
    if frame_count % skip_interval != 0:
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces1 = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    faces2 = face_with_glasses.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    faces = list(faces1) + list(faces2)

    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        r_eye = gray[y:y + h, x:x + w]
        count += 1
        r_eye = cv2.resize(r_eye, (224, 224))
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_GRAY2RGB)
        r_eye = r_eye / 255.0
        r_eye = np.expand_dims(r_eye, axis=0)

        rpred = model.predict(r_eye)
        prob_open = rpred[0][0]
        if prob_open > 0.5:
            val1 = 1
            lbl = 'Open'
        else:
            val1 = 0
            lbl = 'Closed'
        break

    for (x, y, w, h) in left_eye:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        l_eye = gray[y:y + h, x:x + w]
        count += 1
        l_eye = cv2.resize(l_eye, (224, 224))
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_GRAY2RGB)
        l_eye = l_eye / 255.0
        l_eye = np.expand_dims(l_eye, axis=0)

        lpred = model.predict(l_eye)
        prob_open = lpred[0][0]
        if prob_open > 0.5:
            val2 = 1
            lbl = 'Open'
        else:
            val2 = 0
            lbl = 'Closed'
        break

    if val1 == 0 and val2 == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score -= 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score < 0:
        score = 0
    cv2.putText(frame, 'Alarm count:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score > 10:
        try:
            sound.play()
        except:
            pass
        if thicc < 16:
            thicc += 2
        else:
            thicc -= 2
            if thicc < 2:
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
