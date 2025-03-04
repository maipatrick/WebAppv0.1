import cv2
import tempfile
import mediapipe as mp
import streamlit as st
import os
from Sports2D.Sports2D import process, DEFAULT_CONFIG

def run_pose_estimation(video_path):
    """
    Run pose estimation on the given video path.

    Parameters:
    video_path (str): The path to the video file.
    """
    # Modify the default configuration as needed
    config = DEFAULT_CONFIG.copy()
    config['project']['video_input'] = [video_path]
    config['process']['result_dir'] = ''
    config['process']['show_realtime_results'] = False  # Disable real-time display
    config['process']['save_vid'] = True
    config['process']['save_img'] = False
    config['process']['save_pose'] = True
    config['process']['calculate_angles'] = True
    config['process']['save_angles'] = True
    config['process']['multiperson'] = True
    # px_to_meters_conversion
    config['px_to_meters_conversion']['to_meters'] = True
    config['px_to_meters_conversion']['make_c3d'] = False
    # filter
    config['post-processing']['butterworth']['cut_off_frequency'] = 6
    config['post-processing']['show_graphs'] = False

    # Call the process function with the modified configuration
    process(config)
        
    processed_video_name = "fp22_m505_left 2_Sports2D.mp4"
    processed_video_path = os.path.join("fp22_m505_left 2_Sports2D", processed_video_name)

        # Allow the user to download the processed video
    st.markdown(f"Download the processed video [here]('./fp22_m505_left 2_Sports2D/fp22_m505_left 2_Sports2D.mp4')")
    st.video(processed_video_path)
    
    
    processed_video_name = r"fp22_m505_left 2_Sports2D.mp4"
    processed_video_path = ('fp22_m505_left 2_Sports2D/'+ processed_video_name)

    # Allow the user to download the processed video
    st.markdown(f"Download the processed video [here](./{processed_video_path})")
    st.video(processed_video_path)

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