import math
import cv2

def get_head_tilt(frame, gray_frame, face_cascade, nose_cascade):
    """
    Calculate head tilt based on the y-coordinate of the nose tip.
    If the nose tip is not detected, assume the head is tilted.

    Args:
        frame: Original video frame.
        gray_frame: Grayscale frame of the video feed.
        face_cascade: Haar cascade for face detection.
        nose_cascade: Haar cascade for nose detection.

    Returns:
        True if nose tip was detected
        y-coordinate of the nose tip if detected, or None if nose tip undetected
    """
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    if len(faces) == 0:
        return (False, None)  # No face detected

    for (x, y, w, h) in faces:
        face_roi = gray_frame[y:y + h, x:x + w]
        noses = nose_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(noses) == 0:
            return (False, None)  # Assume tilt if no nose is detected

        for (nx, ny, nw, nh) in noses:
            nose_tip_y = y + ny + nh // 2  # Calculate nose tip y-coordinate
            return (True, nose_tip_y)

    return (False, None)

