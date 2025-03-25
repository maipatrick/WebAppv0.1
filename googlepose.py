import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import tempfile
import os

# Load MoveNet Thunder
movenet_model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
movenet = movenet_model.signatures['serving_default']

# Skeleton connections based on COCO keypoint format
SKELETON = [
    (0, 1), (1, 3), (0, 2), (2, 4),  # eyes to ears
    (5, 7), (7, 9),                  # left arm
    (6, 8), (8, 10),                 # right arm
    (5, 6),                         # shoulders
    (5, 11), (6, 12),               # torso sides
    (11, 13), (13, 15),             # left leg
    (12, 14), (14, 16),             # right leg
    (11, 12)                        # hips
]

def detect_pose_movenet(frame):
    input_size = 256
    ih, iw, _ = frame.shape

    # Resize with padding to square
    image = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), input_size, input_size)
    input_tensor = tf.cast(image, dtype=tf.int32)

    # Run MoveNet
    outputs = movenet(input_tensor)
    keypoints = outputs["output_0"].numpy()[0, 0]  # (17, 3)

    # Scale and padding adjustment
    scale = min(input_size / ih, input_size / iw)
    pad_h = (input_size - scale * ih) / 2
    pad_w = (input_size - scale * iw) / 2

    keypoints_xy = []
    for kp in keypoints:
        y, x, conf = kp
        x_scaled = (x * input_size - pad_w) / scale
        y_scaled = (y * input_size - pad_h) / scale
        keypoints_xy.append((y_scaled, x_scaled, conf))

    return keypoints_xy  # (y, x, confidence) in original resolution

def process_video_blured(video_path, show_pose=True, blur_faces=False):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capturing_length = total_frames / fps

    # Output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.join(tempfile.gettempdir(), 'processed_video.mp4')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Output frame images
    frame_output_dir = os.path.join(tempfile.gettempdir(), "pose_frames")
    os.makedirs(frame_output_dir, exist_ok=True)

    landmarks_data = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints = detect_pose_movenet(frame_rgb)

        frame_data = {'frame': frame_count}
        for idx, (y, x, conf) in enumerate(keypoints):
            frame_data[f'keypoint_{idx}_x'] = x
            frame_data[f'keypoint_{idx}_y'] = y
            frame_data[f'keypoint_{idx}_confidence'] = conf

        landmarks_data.append(frame_data)

        if show_pose:
            # Draw keypoints
            for y, x, conf in keypoints:
                if conf > 0.3:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

            # Draw skeleton
            for start, end in SKELETON:
                y1, x1, c1 = keypoints[start]
                y2, x2, c2 = keypoints[end]
                if c1 > 0.3 and c2 > 0.3:
                    pt1 = (int(x1), int(y1))
                    pt2 = (int(x2), int(y2))
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

        # Optional face blurring
        if blur_faces:
            face_ids = [0, 1, 2, 3, 4]
            for i in face_ids:
                x, y, conf = keypoints[i][1], keypoints[i][0], keypoints[i][2]
                if conf > 0.3:
                    cv2.circle(frame, (int(x), int(y)), 30, (0, 0, 0), -1)

        # Save frame image
        frame_filename = f"frame_{frame_count:04d}.jpg"
        cv2.imwrite(os.path.join(frame_output_dir, frame_filename), frame)

        # Save frame to output video
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    df_landmarks = pd.DataFrame(landmarks_data)
    return df_landmarks, fps, capturing_length, total_frames, output_video_path, frame_output_dir



video_path = r'C:\Users\adpatrick\OneDrive - nih.no\Desktop\WebApp1080sync\WebAppv0.1\linear\3. SSL 8kg.MOV'


df_landmarks, fps, capturing_length, total_frames, output_video_path = process_video_blured(video_path, show_pose=1, blur_faces=False)