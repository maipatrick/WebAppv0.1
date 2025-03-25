import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import os

# === CONFIG ===
input_path = r"C:\Users\adpatrick\OneDrive - nih.no\Desktop\WebApp1080sync\WebAppv0.1\linear\Video.mp4"
output_path = os.path.splitext(input_path)[0] + "_output.mp4"
output_csv = os.path.splitext(input_path)[0] + "_movenet_landmarks.csv"
input_size = 256  # MoveNet Thunder expects 256x256 input

# === MOVENET LOAD ===
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")

# === KEYPOINT SETUP ===
keypoint_names = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
skeleton = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

columns = [f"{name}_{axis}" for name in keypoint_names for axis in ["x", "y"]]
df = pd.DataFrame(columns=["frame"] + columns)

# === HELPER: Detect Pose ===
def detect_pose(frame):
    img = cv2.resize(frame, (input_size, input_size))
    img = tf.convert_to_tensor(img, dtype=tf.int32)
    input_tensor = tf.expand_dims(img, axis=0)
    outputs = movenet.signatures['serving_default'](input_tensor)
    keypoints = outputs['output_0'].numpy()[0][0]  # shape: (17, 3)
    return keypoints

# === HELPER: Draw Skeleton ===
def draw_skeleton(frame, keypoints, frame_width, frame_height):
    for idx, (y, x, confidence) in enumerate(keypoints):
        if confidence > 0.3:
            cx, cy = int(x * frame_width), int(y * frame_height)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    for joint1, joint2 in skeleton:
        y1, x1, c1 = keypoints[joint1]
        y2, x2, c2 = keypoints[joint2]
        if c1 > 0.3 and c2 > 0.3:
            p1 = int(x1 * frame_width), int(y1 * frame_height)
            p2 = int(x2 * frame_width), int(y2 * frame_height)
            cv2.line(frame, p1, p2, (0, 0, 255), 2)

# === PROCESS VIDEO ===
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print(f"❌ Failed to open video: {input_path}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    keypoints = detect_pose(frame_rgb)

    # Draw on the frame
    draw_skeleton(frame, keypoints, width, height)

    # Save the pose points
    row = {"frame": frame_idx}
    for i, name in enumerate(keypoint_names):
        row[f"{name}_x"] = keypoints[i][1]
        row[f"{name}_y"] = keypoints[i][0]
    df.loc[frame_idx] = row

    # Write frame to output video
    out_video.write(frame)
    frame_idx += 1
    print(f"\rProcessing frame {frame_idx}/{total_frames}...", end="")

cap.release()
out_video.release()

# Save DataFrame
df.to_csv(output_csv, index=False)

print(f"\n✅ Done! Video saved to: {output_path}")
print(f"📁 Landmark CSV saved to: {output_csv}")
