import matplotlib.pyplot as plt # type: ignore
import json
import os

# Define path to training history JSON file
BASE_DIR = "C:\\Users\\dtirt\\Desktop\\Folder\\Deepfake Detection Project\\deepfake_detection"
HISTORY_FILE = os.path.join(BASE_DIR, "training_history.json")

# Check if the file exists
if not os.path.exists(HISTORY_FILE):
    print("Error: training_history.json not found!")
    exit()

# Load training history
with open(HISTORY_FILE, "r") as f:
    history = json.load(f)

# Plot accuracy
plt.figure(figsize=(8, 6))
plt.plot(history["accuracy"], label="Train Accuracy", marker="o")
plt.plot(history["val_accuracy"], label="Validation Accuracy", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Over Epochs")
plt.legend()
plt.grid(True)
plt.show()
