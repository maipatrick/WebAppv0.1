import cv2
import mediapipe as mp
import pandas as pd
import os

input_path = r"C:\Users\adpatrick\OneDrive - nih.no\Desktop\WebApp1080sync\WebAppv0.1\linear\3. SSL 8kg.MOV"

def test(input_path):
    # Initialize MediaPipe pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False)
    mp_drawing = mp.solutions.drawing_utils

    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {input_path}")
        exit()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # DataFrame with 'frame' as first column
    landmark_names = [l.name.lower() for l in mp_pose.PoseLandmark]
    columns = [f"{name}_{axis}" for name in landmark_names for axis in ["x", "y"]]
    df = pd.DataFrame(columns=["frame"] + columns)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        row_data = {}
        if results.pose_landmarks:
            for id, lm in enumerate(results.pose_landmarks.landmark):
                name = landmark_names[id]
                row_data[f"{name}_x"] = lm.x
                row_data[f"{name}_y"] = lm.y
        else:
            row_data = {col: None for col in columns}

        # Add frame number as first column
        df.loc[frame_idx] = {"frame": frame_idx, **row_data}
        frame_idx += 1
        print(f"\rProcessing frame {frame_idx}/{total_frames}...", end="")

    cap.release()
    pose.close()
    return df


df = test(input_path)
