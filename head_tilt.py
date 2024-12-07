import math
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('dlib files/shape_predictor_68_face_landmarks.dat')  # Pre-trained landmarks model

def get_head_tilt_angle_vertical(gray_frame):
    faces = detector(gray_frame)

    if (len(faces)) == 0:
        return None # no face detected
    
    for face in faces:
        landmarks = predictor(gray_frame, face)

        # get the coordinates for key points: nose tip and chin
        nose_tip = (landmarks.part(30).x, landmarks.part(30).y) 
        chin = (landmarks.part(8).x, landmarks.part(8).y)

        # Compute vertical tilt angle with key points
        delta_y = chin[1] - nose_tip[1] # vertical distance
        delta_x = chin[0] - nose_tip[0] # horizontal distance
        angle = math.degrees(math.atan2(delta_y, delta_x))

        # Return absolute angle 
        return abs(angle)