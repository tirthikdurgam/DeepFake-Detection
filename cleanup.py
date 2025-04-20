import os
import shutil
import glob

# Define paths
MODEL_DIR = "C:\\Users\\dtirt\\Desktop\\Folder\\Deepfake Detection Project\\deepfake_detection\\tmp_checkpoint"
UPLOAD_DIR = "C:\\Users\\dtirt\\Desktop\\Folder\\Deepfake Detection Project\\deepfake_detection\\uploads"
MAX_MODELS = 5
MAX_VIDEOS = 50

def cleanup_old_models():
    models = sorted(glob.glob(os.path.join(MODEL_DIR, '*.keras')), key=os.path.getmtime, reverse=True)
    if len(models) > MAX_MODELS:
        for old_model in models[MAX_MODELS:]:
            os.remove(old_model)
            print(f"Deleted old model: {old_model}")

def cleanup_old_videos():
    videos = sorted(glob.glob(os.path.join(UPLOAD_DIR, '*')), key=os.path.getmtime, reverse=True)
    if len(videos) > MAX_VIDEOS:
        for old_video in videos[MAX_VIDEOS:]:
            os.remove(old_video)
            print(f"Deleted old video: {old_video}")

# Call cleanup functions before saving new data
cleanup_old_models()
cleanup_old_videos()
