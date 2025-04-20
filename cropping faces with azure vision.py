import os
import sys
import torch
from facenet_pytorch import MTCNN
from PIL import Image

# Ensure we use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {DEVICE}")

# Initialize MTCNN face detector
mtcnn = MTCNN(keep_all=True, device=DEVICE)

# Read Input Arguments (Image or Folder)
if len(sys.argv) < 3:
    print("Error: Please provide input and output folder paths.")
    print("Usage: python 'cropping faces with azure vision.py' <input_folder_or_file> <output_folder>")
    sys.exit(1)

INPUT_PATH = sys.argv[1]
OUTPUT_FOLDER = sys.argv[2]

# Ensure Output Folder Exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Get List of Image Files
if os.path.isdir(INPUT_PATH):
    image_files = [os.path.join(INPUT_PATH, f) for f in os.listdir(INPUT_PATH) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
elif os.path.isfile(INPUT_PATH):
    image_files = [INPUT_PATH]  # Handle single image input
else:
    print(" Error: Invalid input path. Must be a folder or an image file.")
    sys.exit(1)

if not image_files:
    print(" No images found for processing.")
    sys.exit(1)

#  Process Each Image
for image_path in image_files:
    image_name = os.path.basename(image_path)
    print(f" Processing image: {image_name}")

    # 🔹 Load Image
    image = Image.open(image_path).convert("RGB")
    
    # 🔹 Detect Faces with MTCNN
    boxes, _ = mtcnn.detect(image)
    if boxes is None:
        print(f" No faces detected in {image_name}")
        continue

    print(f" Detected {len(boxes)} face(s) in {image_name}")

    # 🔹 Crop and Save Faces
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Expand bounding box slightly
        margin_x, margin_y = int((x2 - x1) * 0.2), int((y2 - y1) * 0.2)
        x1, y1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
        x2, y2 = min(image.width, x2 + margin_x), min(image.height, y2 + margin_y)

        # Crop and Save
        cropped_face = image.crop((x1, y1, x2, y2))
        face_filename = os.path.join(OUTPUT_FOLDER, f"{image_name[:-4]}_face_{i}.png")
        cropped_face.save(face_filename)
        print(f" Saved: {face_filename}")

print(" Finished processing all images.")