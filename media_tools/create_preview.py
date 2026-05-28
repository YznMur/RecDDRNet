import cv2
import os
from glob import glob
from tqdm import tqdm  # <-- added

# Paths to your frame folders
folder1 = "/home/trainer/DDRNet.pytorch/data/rsm/test/camera_10_29-05-2024/"
folder2 = "/home/trainer/DDRNet.pytorch/RSM_v2_rsm_640x480_5_super_classes_weighted_CombinedLoss/rsm/ddrnet23_slim_640x480_5classes/val_result_rsmv2/"

# Output video
output_path = "preview_camera_10_29-05-2024.mp4"
fps = 120

# Get sorted frame lists
frames1 = sorted(glob(os.path.join(folder1, "*.*")))
frames2 = sorted(glob(os.path.join(folder2, "*.*")))

print(f"Frames1: {len(frames1)}, Frames2: {len(frames2)}")

# Ensure equal length (use shortest)
num_frames = min(len(frames1), len(frames2))

# Read first frame to get size
img1 = cv2.imread(frames1[0])
img2 = cv2.imread(frames2[0])

# Resize to same height if needed
height = min(img1.shape[0], img2.shape[0])

def resize_to_height(img, h):
    scale = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * scale), h))

img1 = resize_to_height(img1, height)
img2 = resize_to_height(img2, height)

# Combined width
combined_width = img1.shape[1] + img2.shape[1]

# Video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, height))

# Progress bar here
for i in tqdm(range(num_frames), desc="Processing frames"):
    frame1 = cv2.imread(frames1[i])
    frame2 = cv2.imread(frames2[i])

    # Optional safety check
    if frame1 is None or frame2 is None:
        print(f"Skipping frame {i} (read error)")
        continue

    frame1 = resize_to_height(frame1, height)
    frame2 = resize_to_height(frame2, height)

    combined = cv2.hconcat([frame1, frame2])
    out.write(combined)

out.release()
print("Preview video saved to:", output_path)