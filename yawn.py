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
def calculate_mar(mouth_landmarks, vertical_scaling=1.2, horizontal_scaling=1.0):
    """
    Calculate MAR with adjusted scaling for better distinction.
    """
    if len(mouth_landmarks) < 15:
        print("Error: Insufficient landmarks for MAR calculation")
        return 0

    # Calculate raw distances
    vertical1 = distance.euclidean(mouth_landmarks[13], mouth_landmarks[14])
    vertical2 = distance.euclidean(mouth_landmarks[10], mouth_landmarks[9])
    horizontal = distance.euclidean(mouth_landmarks[0], mouth_landmarks[6])

    # Apply scaling
    vertical1 *= vertical_scaling
    vertical2 *= vertical_scaling
    horizontal *= horizontal_scaling

    # Calculate MAR
    mar = (vertical1 + vertical2) / (2.0 * horizontal)
    return mar


# Function to detect yawns using Mediapipe
def get_yawn(frame, mar_threshold=0.61, vertical_scaling=1.2, horizontal_scaling=1.0):
    """
    Detect yawning with updated MAR calculation and threshold adjustment.
    """
    yawning = False
    mar_value = 0

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for facial landmarks
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            height, width, _ = frame.shape
            mouth_landmarks = np.array([
                [int(face_landmarks.landmark[i].x * width), int(face_landmarks.landmark[i].y * height)]
                for i in MOUTH_LANDMARKS
            ])

            # Draw bounding box
            x_min = max(0, np.min(mouth_landmarks[:, 0]) - 10)
            y_min = max(0, np.min(mouth_landmarks[:, 1]) - 10)
            x_max = min(frame.shape[1], np.max(mouth_landmarks[:, 0]) + 10)
            y_max = min(frame.shape[0], np.max(mouth_landmarks[:, 1]) + 10)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Calculate MAR with adjusted scaling
            mar_value = calculate_mar(mouth_landmarks, vertical_scaling, horizontal_scaling)
            if mar_value > mar_threshold:
                yawning = True

    # print(f"MAR Value: {mar_value:.2f}")
    return yawning, mar_value


