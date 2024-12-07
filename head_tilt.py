import dlib
import numpy as np

# Load Dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("dlib files/shape_predictor_68_face_landmarks.dat")

def get_head_tilt_angle(gray_frame):
    faces = detector(gray_frame)
    for face in faces:
        landmarks = predictor(gray_frame, face)
        nose = (landmarks.part(33).x, landmarks.part(33).y)
        chin = (landmarks.part(8).x, landmarks.part(8).y)

        # Calculate the tilt angle
        delta_x = chin[0] - nose[0]
        delta_y = chin[1] - nose[1]
        angle = np.arctan2(delta_y, delta_x) * (180 / np.pi)
        return angle
    return None  # Return None if no face is detected
