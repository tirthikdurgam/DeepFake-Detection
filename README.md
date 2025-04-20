# DeepFake Detection Web App

A real-time deepfake detection system that identifies manipulated (fake) faces in images and videos. It leverages a **Convolutional Neural Network (CNN)** and integrates **Azure Vision API** for accurate face detection. A **Flask web interface** allows users to upload content and instantly receive classification results.

---

## Features

- Accepts both **images and videos** as input  
- Converts videos into frames automatically  
- Crops faces using **Azure Vision API**  
- Classifies each face as **Real** or **DeepFake**  
- Displays confidence score for predictions  
- User-friendly web interface built with Flask  

---

## Tech Stack

- **Language**: Python  
- **Framework**: Flask  
- **ML/DL Library**: TensorFlow / Keras  
- **Face Detection**: Azure Vision API  
- **Video Processing**: OpenCV  
- **Frontend**: HTML, CSS  

---
## Datasets Used

This project uses the following datasets for training and testing the deepfake detection model:


1. **Celeb-DF v2**

- **Source**: [https://github.com/yuezunli/Celeb-DF](https://github.com/yuezunli/Celeb-DF)
- **License**: Academic research use only
- **Description**:  
  - High-quality DeepFake and real videos featuring celebrities  
  - Also includes **cropped face frames** for deepfake training  
  - Labels are provided in JSON format

---

2. **DFDC (DeepFake Detection Challenge)**

- **Source**: [https://ai.facebook.com/datasets/dfdc/](https://ai.facebook.com/datasets/dfdc/)
- **License**: For non-commercial research use
- **Description**:  
  - Large-scale dataset with **100,000+** real and fake videos  
  - Includes metadata files that label each video as real or fake  
  - Requires extraction of face frames for model training

---

3. **Custom Dataset**
   - Manually collected real and fake videos
   - Converted to frames and cropped using Azure Vision API

> ⚠️ Due to size and licensing restrictions, datasets are **not included** in this repository.
---
## Project Structure

```
deepfake-detection/
├── app.py                     # Flask backend
├── requirements.txt
├── README.md
├── static/                    # Static frontend assets
├── templates/                 # index.html for UI
├── DeepFakeDetection/        # Core logic and scripts
│   ├── 00-convert_video_to_image.py
│   ├── 01b-crop_faces_with_azure-vision-api.py
│   ├── 03-train_cnn.py
│   └── ...
├── tmp_uploaded_files/       # Stores uploaded user files
├── tmp_faces/                # Cropped face images
├── tmp_checkpoint/           # Trained models (e.g., best_model.keras)
└── training-history.json     # Model training performance log
```

---

##  How to Run the Project Locally

1. **Clone the repository**  
   ```
   git clone https://github.com/your-username/deepfake-detection.git
   cd deepfake-detection
   ```

2. **Create and activate virtual environment**  
   ```
   python -m venv .venv
   source .venv/bin/activate      # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**  
   ```
   pip install -r requirements.txt
   ```

4. **Run the Flask app**  
   ```
   python app.py
   ```

5. **Visit in browser**  
   Open `http://127.0.0.1:5000` to use the web app.

---

## Model Training

- CNN model trained on cropped face images  
- Supports real-time predictions  
- Stored in `tmp_checkpoint/best_model.keras`

To retrain the model manually:
```
python DeepFakeDetection/03-train_cnn.py
```

---

## Future Improvements

- Replace Azure Vision API with open-source face detectors (e.g., MTCNN or Mediapipe)  
- Enhance model using transfer learning (e.g., EfficientNet, ViT)  
- Add LIME/SHAP for explainable predictions  
- Deploy on cloud (e.g., AWS, GCP, Azure)  

---

## Contact

For queries or collaboration, feel free to reach out via LinkedIn or open an issue.

---

## License

This project is open-source and available under the MIT License.
