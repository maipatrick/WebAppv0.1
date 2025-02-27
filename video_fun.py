import cv2
import tempfile
import mediapipe as mp
import streamlit as st

def process_video(uploaded_video, show_pose=1):
    # Save the uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    # Load the video file
    cap = cv2.VideoCapture(tfile.name)

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # Create a placeholder for the video frames
    frame_placeholder = st.empty()
    progress_bar = st.progress(0)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Process the video frame by frame
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform pose estimation
        results = pose.process(frame_rgb)

        # Draw the pose annotation on the frame if show_pose is 1
        if show_pose == 1 and results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Update the placeholder with the new frame
        frame_placeholder.image(frame, channels="BGR")

        # Update the progress bar
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)
    cap.release()