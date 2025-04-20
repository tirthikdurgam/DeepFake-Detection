import os
import cv2
import random
import shutil
import math

# Define Paths
BASE_DIR = "C:\\Users\\dtirt\\Desktop\\Folder\\Deepfake Detection Project\\deepfake_detection"
RAW_FAKE_VIDEOS = os.path.join(BASE_DIR, "prepared_dataset", "fake")
RAW_REAL_VIDEOS = os.path.join(BASE_DIR, "prepared_dataset", "real")
SPLIT_DATASET = os.path.join(BASE_DIR, "split_dataset")

# Define Train-Val-Test split ratio
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15  # (15% validation, 15% test)
TEST_RATIO = 0.15

# Ensure necessary folders exist
for category in ["train", "val", "test"]:
    for label in ["real", "fake"]:
        os.makedirs(os.path.join(SPLIT_DATASET, category, label), exist_ok=True)

# Function to extract frames from a video
def extract_frames(video_path, output_folder, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    
    # Ensure at least 1 frame is extracted per second
    frame_interval = max(1, int(frame_rate))

    count = 0
    extracted_frames = []
    while cap.isOpened():
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if not ret or count >= max_frames:  
            break  # Stop when video ends or max frames reached

        # Extract frames at interval
        if frame_id % frame_interval == 0:
            frame_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{count:03d}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            extracted_frames.append(frame_path)
            count += 1

    cap.release()
    return extracted_frames

# Function to process a folder of videos
def process_videos(video_folder, label):
    all_videos = [os.path.join(video_folder, v) for v in os.listdir(video_folder) if v.endswith(('.mp4', '.avi', '.mov'))]
    random.shuffle(all_videos)

    train_split = int(len(all_videos) * TRAIN_RATIO)
    val_split = int(len(all_videos) * (TRAIN_RATIO + VAL_RATIO))

    for idx, video in enumerate(all_videos):
        if idx < train_split:
            split = "train"
        elif idx < val_split:
            split = "val"
        else:
            split = "test"

        output_folder = os.path.join(SPLIT_DATASET, split, label)
        frames = extract_frames(video, output_folder)

        print(f"Extracted {len(frames)} frames from {video} -> {split}/{label}")

# Process both fake and real videos
process_videos(RAW_FAKE_VIDEOS, "fake")
process_videos(RAW_REAL_VIDEOS, "real")

print("✅ Dataset preparation complete! Videos have been converted into frames and sorted.")

















'''import json
import os
from shutil import copytree
import shutil
import numpy as np
import splitfolders
import time

base_path = 'C:\\Users\\dtirt\\Desktop\\Folder\\Deepfake Detection Project\\deepfake_detection\\train_sample_videos'
dataset_path = 'C:\\Users\\dtirt\\Desktop\\Folder\\Deepfake Detection Project\\deepfake_detection\\prepared_dataset'
print('Creating Directory: ' + dataset_path)
os.makedirs(dataset_path, exist_ok=True)

tmp_fake_path = '.\\tmp_fake_faces'
print('Creating Directory: ' + tmp_fake_path)

# Check if the directory exists and delete it if necessary
if os.path.exists(tmp_fake_path):
    if os.path.isdir(tmp_fake_path):
        shutil.rmtree(tmp_fake_path)  # Remove existing directory
        time.sleep(1)  # Add a short delay to ensure it's removed

os.makedirs(tmp_fake_path, exist_ok=True)

def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only

with open(os.path.join(base_path, 'metadata.json')) as metadata_json:
    metadata = json.load(metadata_json)
    print(len(metadata))

real_path = os.path.join(dataset_path, 'real')
print('Creating Directory: ' + real_path)
os.makedirs(real_path, exist_ok=True)

fake_path = os.path.join(dataset_path, 'fake')
print('Creating Directory: ' + fake_path)
os.makedirs(fake_path, exist_ok=True)

for filename in metadata.keys():
    print(filename)
    print(metadata[filename]['label'])
    tmp_path = os.path.join(base_path, get_filename_only(filename), 'faces')
    print(tmp_path)
    
    if os.path.exists(tmp_path):
        if metadata[filename]['label'] == 'REAL':    
            print('Copying to :' + real_path)
            # Copy individual files to avoid directory conflict
            for file in os.listdir(tmp_path):
                file_path = os.path.join(tmp_path, file)
                if os.path.isfile(file_path):
                    shutil.copy(file_path, real_path)
        elif metadata[filename]['label'] == 'FAKE':
            print('Copying to :' + tmp_fake_path)
            # Copy individual files to avoid directory conflict
            for file in os.listdir(tmp_path):
                file_path = os.path.join(tmp_path, file)
                if os.path.isfile(file_path):
                    shutil.copy(file_path, tmp_fake_path)
        else:
            print('Ignored..')


all_real_faces = [f for f in os.listdir(real_path) if os.path.isfile(os.path.join(real_path, f))]
print('Total Number of Real faces: ', len(all_real_faces))
all_fake_faces = [f for f in os.listdir(tmp_fake_path) if os.path.isfile(os.path.join(tmp_fake_path, f))]
print('Total Number of Fake faces: ', len(all_fake_faces))

random_faces = np.random.choice(all_fake_faces, len(all_real_faces), replace=False)
for fname in random_faces:
    src = os.path.join(tmp_fake_path, fname)
    dst = os.path.join(fake_path, fname)
    shutil.copyfile(src, dst)

print('Down-sampling Done!')

# Split into Train/ Val/ Test folders
splitfolders.ratio(dataset_path, output='split_dataset', seed=1377, ratio=(.8, .1, .1)) # default values
print('Train/ Val/ Test Split Done!')
'''