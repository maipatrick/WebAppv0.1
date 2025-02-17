# processing.py

import cv2
import tempfile
import mediapipe as mp
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from designidea import plot_hand_drawn_style

def process_excel(uploaded_excel, progress_callback=None):
    """
    Process the uploaded Excel file.

    :param uploaded_excel: The uploaded Excel file object.
    :param progress_callback: A callback function accepting a float (0.0 to 1.0) to update progress.
    :return: A tuple (df, fig) where df is the DataFrame from the Excel file and fig is a matplotlib figure
             generated using a hand-drawn style plot if "Speed [m/s]" is found, otherwise None.
    """
    # Read the Excel file into a DataFrame
    df = pd.read_excel(uploaded_excel, header=0)
    
    # Simulate processing progress
    if progress_callback:
        for i in range(100):
            time.sleep(0.0005)  # Simulated delay for processing
            progress_callback((i + 1) / 100)
    
    fig = None
    # # If the "Speed [m/s]" column exists, generate a plot
    # if "Speed [m/s]" in df.columns:
    #     y = df["Speed [m/s]"]
    #     x = np.arange(len(y))
    #     fig = plot_hand_drawn_style(x, y)
    
    return df, fig

def process_video(uploaded_video, progress_callback=None, frame_callback=None):
    """
    Process the uploaded video file for pose estimation.

    :param uploaded_video: The uploaded video file object.
    :param progress_callback: A callback function accepting a float (0.0 to 1.0) to update progress.
    :param frame_callback: A callback function accepting an image frame for display.
    """
    # Save the uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    
    # Open the video file
    cap = cv2.VideoCapture(tfile.name)
    
    # Initialize MediaPipe Pose and drawing utilities
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils
    
    # Get total frame count for progress calculation
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame for pose estimation
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        # Draw pose landmarks if detected
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Use the frame callback to update the UI if provided
        if frame_callback:
            frame_callback(frame)
        
        frame_count += 1
        if progress_callback and total_frames > 0:
            progress_callback(frame_count / total_frames)
    
    cap.release()
