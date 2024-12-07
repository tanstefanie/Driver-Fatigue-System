import math
# import dlib

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('dlib files/shape_predictor_68_face_landmarks.dat')  # Pre-trained landmarks model

def get_head_tilt(gray_frame, face_cascade, nose_cascade):
    """
    Calculate head tilt based on the y-coordinate of the nose tip.

    Args:
        gray_frame: Grayscale frame of the video feed.
        face_cascade: Haar cascade for face detection.
        nose_cascade: Haar cascade for nose detection.

    Returns:
        y-coordinate of the nose tip, or None if not detected.
    """
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    if len(faces) == 0:
        return None  # No face detected

    for (x, y, w, h) in faces:
        # Focus on the face area
        face_roi = gray_frame[y:y + h, x:x + w]
        noses = nose_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(noses) == 0:
            return None  # No nose detected

        for (nx, ny, nw, nh) in noses:
            # Calculate the absolute y-coordinate of the nose tip
            nose_tip_y = y + ny + nh // 2
            return nose_tip_y

    return None
