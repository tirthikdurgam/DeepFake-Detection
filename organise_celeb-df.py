import os
import cv2
import random
import shutil
import glob
from tqdm import tqdm

# Define paths
BASE_DIR = "C:\\Users\\dtirt\\Desktop\\Folder\\Deepfake Detection Project\\deepfake_detection"
CELEB_DF_PATH = "C:\\Users\\dtirt\\Desktop\\Folder\\Deepfake Detection Project\\Celeb-DF"
REAL_VIDEOS = os.path.join(CELEB_DF_PATH, "Celeb-real")
FAKE_VIDEOS = os.path.join(CELEB_DF_PATH, "Celeb-synthesis")
OUTPUT_DATASET = os.path.join(BASE_DIR, "split_dataset")

# Create output directories
for split in ["train", "val", "test"]:
    for label in ["real", "fake"]:
        os.makedirs(os.path.join(OUTPUT_DATASET, split, label), exist_ok=True)

# Define split ratios
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.2, 0.1

# Function to extract frames from videos
def extract_frames(video_path, output_folder, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = sorted(random.sample(range(total_frames), min(max_frames, total_frames)))
    count = 0
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success:
            frame_filename = f"{os.path.basename(video_path).replace('.mp4', '')}_frame_{count}.jpg"
            cv2.imwrite(os.path.join(output_folder, frame_filename), frame)
            count += 1
    cap.release()

# Function to distribute videos into train, val, test
def process_videos(video_folder, label):
    videos = glob.glob(os.path.join(video_folder, "*.mp4"))
    random.shuffle(videos)
    train_split = int(len(videos) * TRAIN_RATIO)
    val_split = int(len(videos) * (TRAIN_RATIO + VAL_RATIO))
    
    sets = {"train": videos[:train_split], "val": videos[train_split:val_split], "test": videos[val_split:]}
    
    for split, split_videos in sets.items():
        output_folder = os.path.join(OUTPUT_DATASET, split, label)
        for video in tqdm(split_videos, desc=f"Processing {split}/{label}"):
            extract_frames(video, output_folder)

# Process real and fake videos
process_videos(REAL_VIDEOS, "real")
process_videos(FAKE_VIDEOS, "fake")

print("Celeb-DF processing complete! Frames saved in train, val, and test folders.")
