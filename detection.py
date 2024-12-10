import cv2
import os
import tensorflow as tf
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

from head_tilt import get_head_tilt
from yawn import get_yawn

# Initialize mixer for alarm
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load Haar cascades
face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')
nose = cv2.CascadeClassifier('haar cascade files/haarcascade_mcs_nose.xml')

# Load the trained model
model = load_model('./content/model-after-augm.h5')

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
increment_value = 3.0

# Time-based thresholds in seconds
eye_closed_threshold = 1.5  # 2 seconds of sustained eye closure
head_tilt_threshold = 1.0  # 1.5 seconds of sustained head tilt
yawn_threshold_time = 2.0  # 3 seconds of sustained yawning
face_reset_threshold = 3.0  # Reset scores if no face detected for 5 seconds

# Dynamically calculate the actual FPS
actual_fps = cap.get(cv2.CAP_PROP_FPS)
if actual_fps <= 0:  # Fallback for systems that don't return FPS
    actual_fps = 30  # Assume 30 FPS if actual FPS can't be determined
print(f"Actual FPS: {actual_fps}")

# Convert thresholds to frame-based values
eye_closed_threshold_frames = int(eye_closed_threshold * actual_fps)
head_tilt_threshold_frames = int(head_tilt_threshold * actual_fps)
yawn_threshold_frames = int(yawn_threshold_time * actual_fps)

# Initialise scoring variables
eye_closed_score = 0
head_tilt_score = 0
yawn_score = 0

# Initialise nose tracking variables
previous_nose_y = None
y_diff_threshold = 50  # Change in y-coordinate for "head down"

# Initialise last face detected time stamp
last_face_detected_time = time.time()

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

    # Reset the 3 frame values if no face detected for 5s
    if len(faces) > 0:
        # Face detected, update the timer
        last_face_detected_time = time.time()
        print(f"last face time : {last_face_detected_time}")
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)
    else:
        # No face detected, check elapsed time
        time_since_last_face = time.time() - last_face_detected_time
        if time_since_last_face >= face_reset_threshold:
            print(f"No face detected for {face_reset_threshold} seconds. Resetting scores.")
            eye_closed_score = 0
            head_tilt_score = 0
            yawn_score = 0

    # Bounding Box for Whole face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    # Step 3.1: Get eye open/close state
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

        # Determine the label
        if prob_open > 0.5:
            val2 = 1
            lbl = 'Open'
        else:
            val2 = 0
            lbl = 'Closed'
        break

    # Step 3.2: Get nose tilt y-displacement
    nose_detected, nose_y = get_head_tilt(frame, gray, face, nose)

    # Step 3.3 : Detect yawns
    yawning, mar_value = get_yawn(frame)

    # Step 4: Update scores
    # Update Eye Score
    if val1 == 0 and val2 == 0:  # Eyes are closed
        eye_closed_score += increment_value + 1.0
    else:  # Eyes are open
        eye_closed_score = max(0, eye_closed_score - increment_value)  # Decrease score but keep it non-negative

    # Update Head Tilt Score
    if not nose_detected or (previous_nose_y is not None and abs(nose_y - previous_nose_y) > y_diff_threshold):
        head_tilt_score += increment_value
    else:
        head_tilt_score = max(0, head_tilt_score - increment_value)
    previous_nose_y = nose_y
    
    # Update Yawn Score
    if yawning:
        yawn_score += increment_value + 1.5
    else:
        yawn_score = max(0, yawn_score - increment_value)

    # Step 4 : Display thresholds in the top-left corner
    cv2.putText(frame, f"Eyes Closed Threshold: {eye_closed_threshold:.1f}s",
                (10, threshold_y), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.putText(frame, f"Head Tilt Threshold: {head_tilt_threshold:.1f}s",
                (10, threshold_y + line_spacing), font, 1, (0, 255, 255), 1, cv2.LINE_AA)
    
    cv2.putText(frame, f"Yawning Threshold: {yawn_threshold_time:.1f}s",
                (10, threshold_y + 2 * line_spacing), font, 1, (144, 238, 144), 1, cv2.LINE_AA)

    # Step 5: Display scores
    cv2.putText(frame, f"Eyes: {'Closed' if val1 == 0 and val2 == 0 else 'Open'}", 
                (10, value_y), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Eyes Closed: {eye_closed_score / actual_fps:.1f}s", 
                (10, value_y + line_spacing), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.putText(frame, f"Head Tilt: {head_tilt_score / actual_fps:.1f}s", 
                (10, value_y + 2 * line_spacing), font, 1, (0, 255, 255), 1, cv2.LINE_AA)
    
    cv2.putText(frame, f"Yawning: {yawn_score / actual_fps:.1f}s", 
                (10, value_y + 3 * line_spacing), font, 1, (144, 238, 144), 1, cv2.LINE_AA)

    # Step 6: Score conditions to trigger alarms
    if (eye_closed_score >= eye_closed_threshold_frames or
        head_tilt_score >= head_tilt_threshold_frames or
        yawn_score >= yawn_threshold_frames):
        # Person is feeling sleepy, trigger alarm
        # cv2.imwrite(os.path.join(os.getcwd(), 'image.jpg'), frame)
        try:
            sound.play()
            print("Alarm playing")
        except:
            pass
        if thicc < 16:
            thicc += 2
        else:
            thicc -= 2
            if thicc < 2:
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    # Display frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
