import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import cv2

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Define indices for mouth landmarks
MOUTH_LANDMARKS = [
    61, 291, 78, 308, 191, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
]

# Function to calculate Mouth Aspect Ratio (MAR)
def calculate_mar(mouth_landmarks):
    # Validate the number of landmarks
    if len(mouth_landmarks) < 15:
        print("Error: Insufficient landmarks for MAR calculation")
        return 0

    # Vertical distances (top-to-bottom)
    vertical1 = distance.euclidean(mouth_landmarks[13], mouth_landmarks[14])
    vertical2 = distance.euclidean(mouth_landmarks[10], mouth_landmarks[9])

    # Horizontal distance (corner-to-corner)
    horizontal = distance.euclidean(mouth_landmarks[0], mouth_landmarks[6])

    # MAR formula
    mar = (vertical1 + vertical2) / (2.0 * horizontal)
    return mar

# Function to detect yawns using Mediapipe
def get_yawn(frame, mar_threshold=0.7):
    yawning = False
    mar_value = 0

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for facial landmarks
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract mouth landmarks
            height, width, _ = frame.shape
            mouth_landmarks = np.array([
                [int(face_landmarks.landmark[i].x * width), int(face_landmarks.landmark[i].y * height)]
                for i in MOUTH_LANDMARKS
            ])

            # Calculate MAR
            mar_value = calculate_mar(mouth_landmarks)
            if mar_value > mar_threshold:
                yawning = True

    return yawning, mar_value
