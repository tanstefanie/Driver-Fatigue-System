import os
import sys
import cv2

# Determine the base directory dynamically (where Driver-Fatigue-System is located)
script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
base_dir = os.path.abspath(os.path.join(script_dir, os.pardir))  # One level up from the script's directory

# Add the base directory to Python's module search path
sys.path.append(base_dir)

from yawn import get_yawn

# Get Dataset Paths
yawning_dir = os.path.join(base_dir, "Datasets", "Yawning")
yawn_dir = os.path.join(yawning_dir, "Yawn")
no_yawn_dir = os.path.join(yawning_dir, "No_Yawn")

# Initialize result dictionary
results = {"True_Label": [], "Predicted_Label": [], "MAR": []}

def yawn_accuracy(d: dict, folder_path: str, y_label: str):
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        frame = cv2.imread(img_path)
        if frame is not None:
            yawning, mar_value = get_yawn(frame)
            d["True_Label"].append(y_label)
            d["Predicted_Label"].append("Yawn" if yawning else "No Yawn")
            d["MAR"].append(mar_value)
    return d


# Yawn and No Yawn Directories
results = yawn_accuracy(results, yawn_dir, "Yawn")
results = yawn_accuracy(results, no_yawn_dir, "No Yawn")

# Evaluate Results
correct_predictions = sum(
    1 for true, pred in zip(results["True_Label"], results["Predicted_Label"]) if true == pred
)
accuracy = correct_predictions / len(results["True_Label"])

print(f"Accuracy: {accuracy:.2%}")