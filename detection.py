import cv2
import os
import tensorflow as tf
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

from head_tilt import get_head_tilt_angle

# Initialize mixer for alarm
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load Haar cascades
face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# Add head tilt detection
tilt_start_time = None
tilt_angle_treshold = 15
tilt_duration_limit = 3000 # 3 seconds max of head tilt that exceeds 15 degrees angle treshold

lbl = ['Close', 'Open']

# Load the trained model
model = load_model('./content/model-after-augm.h5')

# Start video capture
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# Initialize variables
count = 0
score = 0
thicc = 2
rpred = [1, 0]
lpred = [1, 0]
val1 = 1
val2 = 1

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get head tilt angle
    angle = get_head_tilt_angle(gray)
    if (angle is not None) and (angle > tilt_angle_treshold):
        if tilt_start_time is None:
            tilt_start_time = time.time()
        elif time.time() - tilt_start_time > tilt_duration_limit:
            sound.play()
    else:
        tilt_start_time = None

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
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
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_GRAY2RGB)  # Convert to 3 channels
        r_eye = r_eye / 255.0
        r_eye = np.expand_dims(r_eye, axis=0)  # Add batch dimension

        # Make prediction
        rpred = model.predict(r_eye)  # Output is (1, 1)
        prob_open = rpred[0][0]  # Extract the single probability
        prob_closed = 1 - prob_open  # Compute the complementary probability

        # Determine the label
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
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_GRAY2RGB)  # Convert to 3 channels
        l_eye = l_eye / 255.0
        l_eye = np.expand_dims(l_eye, axis=0)  # Add batch dimension

        # Make prediction
        lpred = model.predict(l_eye)  # Output is (1, 1)
        prob_open = lpred[0][0]  # Extract the single probability
        prob_closed = 1 - prob_open  # Compute the complementary probability

        # Determine the label
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

    if score > 15:
        # Person is feeling sleepy, trigger alarm
        cv2.imwrite(os.path.join(os.getcwd(), 'image.jpg'), frame)
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
