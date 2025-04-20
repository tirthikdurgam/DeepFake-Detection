import cv2
import os
import sys
import shutil
import numpy as np
from mtcnn import MTCNN

# Define base directory
BASE_DIR = "C:\\Users\\dtirt\\Desktop\\Folder\\Deepfake Detection Project\\deepfake_detection"

# Input: Extracted frames from `frames/`
FRAMES_DIR = os.path.join(BASE_DIR, "frames")

# Output: Cropped faces stored in `cropped_faces/`
CROPPED_FACES_DIR = os.path.join(BASE_DIR, "cropped_faces")

# Ensure output folder is empty before processing
def clear_cropped_faces():
    if os.path.exists(CROPPED_FACES_DIR):
        shutil.rmtree(CROPPED_FACES_DIR)
    os.makedirs(CROPPED_FACES_DIR, exist_ok=True)

clear_cropped_faces()

# Initialize MTCNN detector
detector = MTCNN()

# Function to detect faces using MTCNN
def detect_faces(image):
    faces = detector.detect_faces(image)
    
    if not faces:
        print(" No faces detected.")
        return None

    # Choose the largest face
    largest_face = max(faces, key=lambda f: f["box"][2] * f["box"][3])

    return largest_face

# Process each frame and extract faces
for frame_file in os.listdir(FRAMES_DIR):
    frame_path = os.path.join(FRAMES_DIR, frame_file)

    if not frame_file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    # Read image
    image = cv2.imread(frame_path)
    if image is None:
        print(f" Error: Unable to read {frame_path}")
        continue

    # Detect faces
    largest_face = detect_faces(image)

    if largest_face is None:
        continue

    bbox = largest_face["box"]
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

    # Add margin to avoid tight crops
    margin_x, margin_y = int(w * 0.3), int(h * 0.3)
    x1, y1 = max(0, x - margin_x), max(0, y - margin_y)
    x2, y2 = min(image.shape[1], x + w + margin_x), min(image.shape[0], y + h + margin_y)

    # Crop the face
    cropped_face = image[y1:y2, x1:x2]
    cropped_face_path = os.path.join(CROPPED_FACES_DIR, f"{frame_file[:-4]}_face.png")

    # Save cropped face
    cv2.imwrite(cropped_face_path, cropped_face)
    print(f" Saved cropped face: {cropped_face_path}")

print(" Face cropping completed!")