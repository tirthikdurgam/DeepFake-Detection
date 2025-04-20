import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from flask import Flask, request, jsonify, render_template
from torchvision import models
from PIL import Image
import subprocess

app = Flask(__name__)

# Paths
BASE_DIR = "C:\\Users\\dtirt\\Desktop\\Folder\\Deepfake Detection Project\\deepfake_detection"
MODEL_PATH = os.path.join(BASE_DIR, "tmp_checkpoint", "best_model.pt")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
FRAMES_FOLDER = os.path.join(BASE_DIR, "frames")
CROPPED_FACES_FOLDER = os.path.join(BASE_DIR, "cropped_faces")
TEMP_IMAGE_FOLDER = os.path.join(BASE_DIR, "temp_image")

# Ensure necessary folders exist
for folder in [UPLOAD_FOLDER, FRAMES_FOLDER, CROPPED_FACES_FOLDER, TEMP_IMAGE_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Load PyTorch Model
class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=1):
        super(DeepfakeDetector, self).__init__()
        self.model = models.efficientnet_b1(weights=None)  # Use EfficientNetB1
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )  # No Sigmoid since BCEWithLogitsLoss was used

    def forward(self, x):
        return self.model(x)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
# Load the trained model
model = DeepfakeDetector().to(DEVICE)
# Load the model weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded successfully!")
torch.cuda.empty_cache()

# Define Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']  
    file_ext = file.filename.split('.')[-1].lower()
    if file_ext not in ['mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png']:
        return jsonify({'error': 'Unsupported file type'}), 400

    print("Clearing previous data...")
    for folder in [CROPPED_FACES_FOLDER, FRAMES_FOLDER, TEMP_IMAGE_FOLDER]:
        for filename in os.listdir(folder):  
            os.remove(os.path.join(folder, filename))

    if file_ext in ['mp4', 'avi', 'mov']:  # Video Processing
        for video_file in os.listdir(UPLOAD_FOLDER):  
            os.remove(os.path.join(UPLOAD_FOLDER, video_file))  

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    try:
        if file_ext in ['mp4', 'avi', 'mov']:  
            print(f"🎥 Processing video: {file_path}")
            file.save(file_path)

            subprocess.run(["python", os.path.join(BASE_DIR, "converting into frames.py"), file_path, FRAMES_FOLDER], check=True)
            subprocess.run(["python", os.path.join(BASE_DIR, "cropping faces with azure vision.py"), FRAMES_FOLDER, CROPPED_FACES_FOLDER], check=True)

        elif file_ext in ['jpg', 'jpeg', 'png']:  
            print(f"Processing image: {file.filename}")

            # Clear any previous images in TEMP_IMAGE_FOLDER
            for img_file in os.listdir(TEMP_IMAGE_FOLDER):
                os.remove(os.path.join(TEMP_IMAGE_FOLDER, img_file))

            # Save the new image
            temp_image_path = os.path.join(TEMP_IMAGE_FOLDER, file.filename)
            file.save(temp_image_path)

            subprocess.run(["python", os.path.join(BASE_DIR, "cropping faces with azure vision.py"), TEMP_IMAGE_FOLDER, CROPPED_FACES_FOLDER], check=True)


            if not os.listdir(CROPPED_FACES_FOLDER):
                return jsonify({'error': 'No faces detected!'}), 400

        # Batch Inference for Deepfake Detection
        face_images = [transform(Image.open(os.path.join(CROPPED_FACES_FOLDER, face)).convert("RGB")) for face in os.listdir(CROPPED_FACES_FOLDER)]
        
        if not face_images:
            return jsonify({'error': 'No valid faces detected for prediction.'}), 400

        images_tensor = torch.stack(face_images).to(DEVICE).float()  # Convert to float

        with torch.no_grad():
            predictions = torch.sigmoid(model(images_tensor)).squeeze().tolist()  # Ensure sigmoid

        if isinstance(predictions, float):  
            predictions = [predictions]  # Convert single prediction to list

        confidence_percentage = np.mean(predictions) * 100  # Correct confidence scaling

        return jsonify({'confidence': round(confidence_percentage, 2)})  # Proper JSON response

    except subprocess.CalledProcessError as e:
        print("Subprocess Error:", e.stderr)
        return jsonify({'error': f'Script failed: {e.stderr}'}), 500
    except Exception as e:
        print("General Error:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)