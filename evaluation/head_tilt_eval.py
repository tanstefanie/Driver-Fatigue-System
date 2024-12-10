import os
import cv2
import sys

# Determine the base directory dynamically (where Driver-Fatigue-System is located)
script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
base_dir = os.path.abspath(os.path.join(script_dir, os.pardir))  # One level up from the script's directory

# Add the base directory to Python's module search path
sys.path.append(base_dir)

from head_tilt import get_head_tilt

# Get Dataset Paths
head_tilt_dir = os.path.join(base_dir, "Datasets", "Head Tilt")
tilted_dir = os.path.join(head_tilt_dir, "Tilt")
upright_dir = os.path.join(head_tilt_dir, "No Tilt")

# Load Haar cascades
haar_cascade_path = os.path.join(os.getcwd(), "haar cascade files")
face_cascade = cv2.CascadeClassifier(os.path.join(haar_cascade_path, "haarcascade_frontalface_alt.xml"))
nose_cascade = cv2.CascadeClassifier(os.path.join(haar_cascade_path, "haarcascade_mcs_nose.xml"))

# Initialize result dictionary
results = {"True_Label": [], "Predicted_Label": [], "Nose_Tip_Y": []}

def head_tilt_accuracy(d: dict, folder_path: str, h_label: str):
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        frame = cv2.imread(img_path)
        if frame is not None:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            nose_detected, nose_tip_y = get_head_tilt(frame, gray_frame, face_cascade, nose_cascade)

            # If nose is not detected, assume "Tilt"
            if not nose_detected:
                predicted_label = "Tilt"
            else:
                # If nose is detected, classify based on y-coordinate
                predicted_label = "Tilt" if nose_tip_y > frame.shape[0] // 2 else "No Tilt"

            d["True_Label"].append(h_label)
            d["Predicted_Label"].append(predicted_label)
            d["Nose_Tip_Y"].append(nose_tip_y if nose_detected else "Not Detected")
    return d

# Process Tilted and Upright directories
results = head_tilt_accuracy(results, tilted_dir, "Tilt")
results = head_tilt_accuracy(results, upright_dir, "No Tilt")

# Evaluate Results
correct_predictions = sum(
    1 for true, pred in zip(results["True_Label"], results["Predicted_Label"]) if true == pred
)
accuracy = correct_predictions / len(results["True_Label"])

print(f"Accuracy: {accuracy:.2%}")

# Print detailed results
# for i, (true, pred, nose_tip_y) in enumerate(zip(results["True_Label"], results["Predicted_Label"], results["Nose_Tip_Y"])):
#     print(f"Image {i + 1}: True Label = {true}, Predicted Label = {pred}, Nose Tip Y = {nose_tip_y}")
