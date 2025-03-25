import cv2
import numpy as np
import streamlit as st
import pandas as pd
import tempfile
import os
import mediapipe as mp

# Enhance contrast using CLAHE
def enhance_contrast(frame_bgr):
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def process_video_blured_new(video_path, show_pose=1, blur_faces=False):
    progress_bar = st.progress(0)
    status_text = st.empty()

    cap = cv2.VideoCapture(video_path)

    # Initial confidence thresholds
    min_detection_confidence = 0.6
    min_tracking_confidence = 0.6

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )

    mp_drawing = mp.solutions.drawing_utils
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    capturing_length = total_frames / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.join(tempfile.gettempdir(), 'processed_video.mp4')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    landmarks_data = []
    frame_count = 0
    status_text.text("Processing video frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: Apply CLAHE to improve contrast
        frame = enhance_contrast(frame)

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Step 2: Primary pose estimation
        results = pose.process(frame_rgb)

        # Fallback: Retry with lower confidence if pose wasn't detected
        if not results.pose_landmarks:
            fallback_pose = mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                smooth_landmarks=True,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
            results = fallback_pose.process(frame_rgb)

        # Extract and store landmarks
        frame_data = {'frame': frame_count}
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                name = mp_pose.PoseLandmark(idx).name.lower()
                frame_data[f'{name}_x'] = landmark.x
                frame_data[f'{name}_y'] = landmark.y

            # Optional: draw landmarks
            if show_pose:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
                )
        else:
            for idx in range(len(mp_pose.PoseLandmark)):
                name = mp_pose.PoseLandmark(idx).name.lower()
                frame_data[f'{name}_x'] = float('nan')
                frame_data[f'{name}_y'] = float('nan')

        landmarks_data.append(frame_data)

        # Optional face blur
        if blur_faces and results.pose_landmarks:
            head_landmarks = [
                mp_pose.PoseLandmark.NOSE,
                mp_pose.PoseLandmark.LEFT_EYE_INNER,
                mp_pose.PoseLandmark.LEFT_EYE,
                mp_pose.PoseLandmark.LEFT_EYE_OUTER,
                mp_pose.PoseLandmark.RIGHT_EYE_INNER,
                mp_pose.PoseLandmark.RIGHT_EYE,
                mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
                mp_pose.PoseLandmark.LEFT_EAR,
                mp_pose.PoseLandmark.RIGHT_EAR,
                mp_pose.PoseLandmark.MOUTH_LEFT,
                mp_pose.PoseLandmark.MOUTH_RIGHT
            ]
            for landmark in head_landmarks:
                x = int(results.pose_landmarks.landmark[landmark].x * width)
                y = int(results.pose_landmarks.landmark[landmark].y * height)
                cv2.circle(frame, (x, y), 30, (0, 0, 0), -1)

        # Save to video
        out.write(frame)

        # Update Streamlit UI
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count + 1}/{total_frames}")
        frame_count += 1

    # Cleanup
    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()

    df_landmarks = pd.DataFrame(landmarks_data)
    return df_landmarks, fps, capturing_length, total_frames, output_video_path



video_path = r'C:\Users\adpatrick\OneDrive - nih.no\Desktop\WebApp1080sync\WebAppv0.1\linear\3. SSL 8kg.MOV'

df_landmarks, fps, capturing_length, total_frames, output_video_path = process_video_blured_new(video_path, show_pose=1, blur_faces=False)