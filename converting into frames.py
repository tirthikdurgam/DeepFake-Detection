import os
import cv2
import math
import sys

# Ensure a video file is provided
if len(sys.argv) < 2:
    print("Error: No video file provided.")
    sys.exit(1)

video_file = sys.argv[1]  # Get the uploaded video path from Flask

# Ensure the video file exists
if not os.path.exists(video_file):
    print(f"Error: The file '{video_file}' does not exist.")
    sys.exit(1)

# Define base directory for frame storage
base_directory = "C:\\Users\\dtirt\\Desktop\\Folder\\Deepfake Detection Project\\deepfake_detection"

# Define output directory for extracted frames
output_dir = os.path.join(base_directory, "frames")
os.makedirs(output_dir, exist_ok=True)

print(f"Processing video: {video_file}")
cap = cv2.VideoCapture(video_file)

if not cap.isOpened():
    print("Error: Unable to open the video file.")
    sys.exit(1)

frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get video frame rate
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames
print(f"Video FPS: {frame_rate}, Total Frames: {frame_count}")

count = 0

while cap.isOpened():
    frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Current frame number
    ret, frame = cap.read()
    
    if not ret:
        break  # Stop when video ends
    
    # Extract 1 frame per second
    if frame_id % math.floor(frame_rate) == 0:
        width, height = frame.shape[1], frame.shape[0]

        # Resize frames if necessary
        if width > 1000:
            scale_ratio = 0.5  # Reduce size for large images
        elif width < 300:
            scale_ratio = 2  # Increase size for small images
        else:
            scale_ratio = 1  # Keep original size
        
        new_width = int(width * scale_ratio)
        new_height = int(height * scale_ratio)
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Save frame
        frame_filename = os.path.join(output_dir, f"frame-{count:03d}.png")
        cv2.imwrite(frame_filename, resized_frame)
        print(f"Saved: {frame_filename}")
        count += 1

cap.release()
print(f"Done! Extracted {count} frames in '{output_dir}'")