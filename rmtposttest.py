import cv2
import pandas as pd
from rtmlib import PoseTracker, BodyWithFeet
import streamlit as st
from anytree import RenderTree

# Halpe-26 landmark names
landmark_names = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "head", "neck", "hip",
    "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe",
    "left_heel", "right_heel"
]

def pose_estimation_rmt_pose(video_path, max_people=10):
    # Streamlit UI
    progress_bar = st.progress(0)
    status_text = st.empty()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    pose_tracker = PoseTracker(
        BodyWithFeet,
        det_frequency=1,
        mode="balanced",
        backend="onnxruntime",
        device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu",
        tracking=True,
        to_openpose=False
    )

    keypoints_data = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Initialize row with NaNs for all people
        row = [frame_count]
        for _ in range(max_people):
            row.extend([None] * len(landmark_names) * 2)

        try:
            keypoints_all, scores_all = pose_tracker(frame)

            for pid, keypoints in enumerate(keypoints_all, start=1):
                if pid > max_people:
                    continue
                for i, name in enumerate(landmark_names):
                    if i < keypoints.shape[0]:
                        x, y = keypoints[i]
                    else:
                        x, y = None, None
                    base_col = 1 + (pid - 1) * len(landmark_names) * 2 + i * 2
                    row[base_col] = x
                    row[base_col + 1] = y

            # Optional visualization
            for keypoints in keypoints_all:
                for x, y in keypoints:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

        except IndexError as e:
            print(f"⚠️ Tracker error at frame {frame_count}: {e} — writing NaNs for this frame.")

        # Save the row (whether valid or all NaNs)
        keypoints_data.append(row)

        # Streamlit progress
        progress = frame_count / total_frames
        progress_bar.progress(min(progress, 1.0))
        status_text.text(f"Processing frame {frame_count}/{total_frames}")
        print(f"Processing frame {frame_count}/{total_frames}")

    # Create headers
    headers = ['Frame']
    for pid in range(1, max_people + 1):
        for name in landmark_names:
            headers.append(f'person_{pid}_{name}_x')
            headers.append(f'person_{pid}_{name}_y')

    keypoints_df = pd.DataFrame(keypoints_data, columns=headers)

    # Remove unused people (all-NaN columns)
    for pid in range(1, max_people + 1):
        person_cols = [f'person_{pid}_{name}_{axis}' for name in landmark_names for axis in ['x', 'y']]
        if keypoints_df[person_cols].isnull().all().all():
            keypoints_df.drop(columns=person_cols, inplace=True)

    cap.release()
    return fps, total_frames, duration, video_path, keypoints_df

# Example usage
pose_estimation_rmt_pose(r"C:\Users\adpatrick\OneDrive - nih.no\Desktop\Testdata\linear\converted.mp4")