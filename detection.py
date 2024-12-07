import cv2
import os
import tensorflow as tf
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

from head_tilt import get_head_tilt

# Initialize mixer for alarm
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load Haar cascades
face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')
nose = cv2.CascadeClassifier('haar cascade files/haarcascade_mcs_nose.xml')

# Thresholds
previous_nose_y = None
nose_disappear_count = 0
nose_disappear_threshold = 5  # Frames to wait before considering "head down"
y_diff_threshold = 50  # Change in y-coordinate for "head down"

# Load the trained model
model = load_model('./content/model-after-augmv2.h5')

# Start video capture
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# Set desired FPS and resolution
desired_fps = 60
cap.set(cv2.CAP_PROP_FPS, desired_fps)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce width for higher FPS
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Reduce height for higher FPS

# Initialize variables
lbl = ['Close', 'Open']
count = 0
score = 0
thicc = 2
rpred = [1, 0]
lpred = [1, 0]
val1 = 1
val2 = 1
tilt_score = 0
tilt_increment_delay = 1  # Number of consecutive frames to wait before incrementing
tilt_increment_count = 0  # Counter for consecutive frames of head tilt

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Step 1: Instantiate variables and get grayscale frame
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define positions for text display
    threshold_y = 30  # Top-left corner for thresholds
    value_y = height - 100  # Bottom-left corner for actual values
    line_spacing = 30  # Spacing between lines

    # Step 2: Get face and eye bounding boxes
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    # Display thresholds in the top-left corner
    cv2.putText(frame, f"Head Tilt Threshold: {y_diff_threshold}", 
                (10, threshold_y), font, 1, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Eyes Open Threshold: 30 frames", 
                (10, threshold_y + line_spacing), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Bounding Box for Whole face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    # Step 3: Get eye open/close state
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
        # prob_closed = 1 - prob_open  # Compute the complementary probability

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
        # prob_closed = 1 - prob_open  # Compute the complementary probability

        # Determine the label
        if prob_open > 0.5:
            val2 = 1
            lbl = 'Open'
        else:
            val2 = 0
            lbl = 'Closed'
        break

    # Step 4: Get nose tilt y-displacement
    nose_y = get_head_tilt(gray, face, nose)

    # Step 5: Update scores
    if val1 == 0 and val2 == 0:
        score += 1  # Increment when eyes are closed
    else:
        score = max(0, score - 1)  # Decrement score but keep it non-negative

    if nose_y is None:
        nose_disappear_count += 1
        if nose_disappear_count > nose_disappear_threshold:
            tilt_increment_count += 1
            if tilt_increment_count > tilt_increment_delay:
                print("Head down: Sustained nose disappearance")
                tilt_score += 1
                tilt_increment_count = 0  # Reset the delay counter
    else:
        if previous_nose_y is not None:
            y_diff = nose_y - previous_nose_y
            if y_diff > y_diff_threshold:
                tilt_increment_count += 1
                if tilt_increment_count > tilt_increment_delay:
                    print(f"Head down: Sustained nose movement, y_diff = {y_diff}")
                    tilt_score += 1
                    tilt_increment_count = 0  # Reset the delay counter
            else:
                tilt_score = max(0, tilt_score - 1)
                tilt_increment_count = 0  # Reset delay if condition is not met
        nose_disappear_count = 0  # Reset disappear count
        previous_nose_y = nose_y

    # Step 6: Display scores
    cv2.putText(frame, f"Eyes: {'Closed' if val1 == 0 and val2 == 0 else 'Open'}", 
                (10, value_y), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Tilt Score: {tilt_score}", 
                (10, value_y + line_spacing), font, 1, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Eyes Closed Count: {score}", 
                (10, value_y + 2 * line_spacing), font, 1, (255, 255, 255), 1, cv2.LINE_AA)


    # Step 7: Score conditions to trigger alarms
    if score > 15 or tilt_score > 8:
        # Person is feeling sleepy, trigger alarm
        # cv2.imwrite(os.path.join(os.getcwd(), 'image.jpg'), frame)
        try:
            sound.play()
            print("Eye alarm playing")
        except:
            pass
        if thicc < 16:
            thicc += 2
        else:
            thicc -= 2
            if thicc < 2:
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    # if tilt_score >10:
    #     try:
    #         sound.play()
    #         print("Tilt alarm playing")
    #     except:
    #         pass

    # Display frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
