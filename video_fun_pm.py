import os
os.environ['OPENCV_VIDEOIO_PRIORITY_BACKEND'] = '0'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

import cv2
import mediapipe as mp
from scipy.signal import butter, filtfilt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive 'Agg'
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import tempfile
import streamlit as st


import cv2
import tempfile
import streamlit as st
import os

# Define the segments and their respective weights
segmentspairs = [
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'),
    ('right_shoulder', 'right_elbow'),
    ('left_elbow', 'left_wrist'),
    ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip', 'left_knee'),
    ('right_hip', 'right_knee'),
    ('left_knee', 'left_ankle'),
    ('right_knee', 'right_ankle'),
    ('left_heel', 'left_foot_index'),
    ('right_heel', 'right_foot_index'),
    ('left_hip', 'right_hip')
]

def process_and_overlay_videoStreamlit_None(video_path, df_landmarks_filtered):
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Error opening video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Use MPEG4 codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_output = os.path.join(temp_dir, 'temp_output.mp4')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            temp_output = os.path.join(temp_dir, 'temp_output.avi')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

            if not out.isOpened():
                raise Exception("Could not initialize video writer. No compatible codec found.")

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Update progress
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"{round((frame_count / total_frames) * 100)}%")

            # Get the landmarks for the current frame, skipping the first column (frame count)
            landmarks = df_landmarks_filtered.iloc[frame_count, 1:]

            # Plot landmarks on the frame
            for i in range(0, len(landmarks), 2):
                x = int(landmarks[i] * width)
                y = int(landmarks[i + 1] * height)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

            # Draw lines between the segments
            for segment in segmentspairs:
                landmark1, landmark2 = segment
                x1 = int(df_landmarks_filtered[f'{landmark1}_x'].iloc[frame_count] * width)
                y1 = int(df_landmarks_filtered[f'{landmark1}_y'].iloc[frame_count] * height)
                x2 = int(df_landmarks_filtered[f'{landmark2}_x'].iloc[frame_count] * width)
                y2 = int(df_landmarks_filtered[f'{landmark2}_y'].iloc[frame_count] * height)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # Write the frame
            out.write(frame)
            frame_count += 1

        # Clean up
        cap.release()
        out.release()

        # Read the final video file
        with open(temp_output, 'rb') as f:
            video_data = f.read()

        # Clear the progress bar and status text
        progress_bar.empty()
        status_text.empty()

        return video_data






def process_and_overlay_videoStreamlit_force(video_path, sync_a, total_time):
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Open the video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Ensure video length matches sync_a
        if total_frames != len(sync_a):
            raise ValueError("Mismatch: Video frames and force signal length must be equal.")

        # Use MPEG4 codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_output = temp_dir + '/temp_output.mp4'
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            temp_output = temp_dir + '/temp_output.avi'
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

            if not out.isOpened():
                raise Exception("Could not initialize video writer. No compatible codec found.")

        # Set style for better visualization
        plt.style.use('dark_background')

        # Calculate the time per frame
        time_per_frame = total_time / total_frames
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Update progress
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"{round((frame_count / total_frames)*100)}%")

            # Create and save plot
            temp_plot_path = temp_dir + '/temp_plot.png'
            create_force_plot(time_values=[i * time_per_frame for i in range(frame_count + 1)],
                              sync_a_slice=sync_a[:frame_count + 1],
                              total_time=total_time,
                              sync_a=sync_a,
                              output_path=temp_plot_path)

            # Overlay plot on frame
            plot_img = cv2.imread(temp_plot_path, cv2.IMREAD_UNCHANGED)
            if plot_img is not None:
                overlay_plot_on_frame(frame, plot_img, width, height)

            # Write the frame
            out.write(frame)
            frame_count += 1

        # Clean up
        cap.release()
        out.release()

        # Read the final video file
        with open(temp_output, 'rb') as f:
            video_data = f.read()

        # Clear the progress bar and status text
        progress_bar.empty()
        status_text.empty()

        return video_data

def create_force_plot(time_values, sync_a_slice, total_time, sync_a, output_path):
    plt.figure(figsize=(8, 4), facecolor='none')
    ax = plt.gca()
    ax.set_facecolor('none')

    # Plot Force signal
    ax.plot(time_values, sync_a_slice, 
             color='red',
             label='Force (N)',
             linewidth=3,
             alpha=0.8)

    # Add gradient fill
    ax.fill_between(time_values, sync_a_slice,
                    alpha=0.2, color='red')

    # Customize plot appearance
    ax.set_xlabel('Time [s]', color='white', fontsize=15, fontweight='bold', labelpad=10)
    ax.set_ylabel('Force [N]', color='white', fontsize=15, fontweight='bold', labelpad=10)
    ax.grid(True, alpha=0.2, linestyle='--', color='white')

    # Style axes
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.tick_params(colors='white', grid_color='white')

    # Add legend
    legend = ax.legend(facecolor='none', edgecolor='none', loc='upper right', fontsize=15)
    plt.setp(legend.get_texts(), color='white')

    # Set plot limits
    ax.set_xlim(0, total_time)
    ax.set_ylim(sync_a.min() - 0.1, sync_a.max() + 0.1)

    # Add background panel
    ax.add_patch(Rectangle((0, 0), 1, 1, 
                         transform=ax.transAxes,
                         facecolor='black',
                         alpha=0.7,
                         zorder=-1))

    # Save plot
    plt.savefig(output_path, 
                transparent=True, 
                bbox_inches='tight', 
                pad_inches=0.2,
                dpi=300)
    plt.close()

def overlay_plot_on_frame(frame, plot_img, width, height):
    # Resize plot
    plot_height = 250
    plot_width = 500
    plot_img = cv2.resize(plot_img, (plot_width, plot_height))

    # Create mask
    if plot_img.shape[2] == 4:
        mask = plot_img[:, :, 3] / 255.0
        mask = np.expand_dims(mask, axis=-1)
        plot_img = plot_img[:, :, :3]
    else:
        mask = np.ones((plot_height, plot_width, 1))

    # Position plot
    y_offset = 30
    x_offset = width - plot_width - 30

    # Overlay plot
    roi = frame[y_offset:y_offset+plot_height, x_offset:x_offset+plot_width]
    frame[y_offset:y_offset+plot_height, x_offset:x_offset+plot_width] = (
        roi * (1 - mask) + plot_img * mask
    ).astype(np.uint8)

def process_and_overlay_videoStreamlit(video_path, df_pos_com, sync_a, lag, cut_index, total_time, df_distance):
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Crop the data in df_pos_com to the lag and cut_index
        df_pos_com = df_pos_com.iloc[lag:cut_index, :].reset_index(drop=True)
        df_distance = df_distance.iloc[lag:cut_index, :].reset_index(drop=True)
        
        # Debugging: Print lengths before padding
        #st.write(f"Lengths before padding: sync_a={len(sync_a)}, df_pos_com={len(df_pos_com)}, df_distance={len(df_distance)}")
        
        # Ensure sync_a, df_pos_com, and df_distance have the same length
        max_length = max(len(sync_a), len(df_pos_com), len(df_distance))
        sync_a = np.pad(sync_a, (0, max_length - len(sync_a)), 'edge')
        df_pos_com = df_pos_com.reindex(range(max_length), method='ffill')
        df_distance = df_distance.reindex(range(max_length), method='ffill')
        
        # Debugging: Print lengths after padding
        #st.write(f"Lengths after padding: sync_a={len(sync_a)}, df_pos_com={len(df_pos_com)}, df_distance={len(df_distance)}")
        
        # Open the video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Use MPEG4 codec which is more widely supported
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_output = os.path.join(temp_dir, 'temp_output.mp4')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        
        if not out.isOpened():
            # Fallback to other codecs if mp4v fails
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            temp_output = os.path.join(temp_dir, 'temp_output.avi')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
            
            if not out.isOpened():
                # Last resort - try MJPG
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                temp_output = os.path.join(temp_dir, 'temp_output.avi')
                out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise Exception("Could not initialize video writer. No compatible codec found.")
        
        # Set style for better visualization
        plt.style.use('dark_background')
        
        # Calculate the time per frame
        time_per_frame = total_time / (cut_index - lag)
        
        # Process each frame
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Debugging: Print total frames in the video
        #st.write(f"Total frames in the video: {total_frames}")
        
        # Adjust lengths if necessary
        if total_frames > max_length:
            #st.write("Video has more frames than data. Padding data.")
            sync_a = np.pad(sync_a, (0, total_frames - max_length), 'edge')
            df_pos_com = df_pos_com.reindex(range(total_frames), method='ffill')
            df_distance = df_distance.reindex(range(total_frames), method='ffill')
        elif total_frames < max_length:
            #st.write("Video has fewer frames than data. Trimming data.")
            sync_a = sync_a[:total_frames]
            df_pos_com = df_pos_com.iloc[:total_frames]
            df_distance = df_distance.iloc[:total_frames]
        
        # Debugging: Print lengths after adjustment
        #st.write(f"Lengths after adjustment: sync_a={len(sync_a)}, df_pos_com={len(df_pos_com)}, df_distance={len(df_distance)}, total_frames={total_frames}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count < lag or frame_count > cut_index:
                frame_count += 1
                continue
            
            # Update progress
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}/{total_frames}")
            
            # Add com_x and com_y circles
            if frame_count - lag < len(df_pos_com):
                com_x = int(df_pos_com.loc[frame_count - lag, 'com_x'] * width)
                com_y = int(df_pos_com.loc[frame_count - lag, 'com_y'] * height)
                cv2.circle(frame, (com_x, com_y), 5, (0, 0, 255), -1)
            
            # Draw horizontal line from lag position to current position
            if frame_count - lag < len(df_pos_com):
                start_x = int(df_pos_com.loc[0, 'com_x'] * width)
                end_x = int(df_pos_com.loc[frame_count - lag, 'com_x'] * width)
                y_position = int(df_pos_com.loc[frame_count - lag, 'com_y'] * height)
                cv2.line(frame, (start_x, y_position), (end_x, y_position), (255, 0, 0), 2)
                
                # Add current value of df_distance at this frame
                distance_value = df_distance.loc[frame_count - lag].values[0]
                cv2.putText(frame, f"{distance_value:.2f} m", (end_x + 10, y_position), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Create and save plot
            temp_plot_path = os.path.join(temp_dir, 'temp_plot.png')
            create_plot(time_values=[i * time_per_frame for i in range(frame_count - lag + 1)],
                       sync_a_slice=sync_a[lag:frame_count + 1],
                       df_distance=df_distance.iloc[:frame_count - lag + 1],
                       total_time=total_time,
                       sync_a=sync_a,
                       lag=lag,
                       cut_index=cut_index,
                       output_path=temp_plot_path)
            
            # Debugging: Print lengths during frame processing
            #st.write(f"Frame {frame_count}: sync_a_slice={len(sync_a[lag:frame_count + 1])}, df_distance_slice={len(df_distance.iloc[:frame_count - lag + 1])}")
            
            # Overlay plot on frame
            if os.path.exists(temp_plot_path):
                plot_img = cv2.imread(temp_plot_path, cv2.IMREAD_UNCHANGED)
                if plot_img is not None:
                    overlay_plot_on_frame(frame, plot_img, width, height)
            
            # Add frame count text
            frame_text = f"Frame {frame_count}/{total_frames}"
            cv2.putText(frame, frame_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Write the frame
            out.write(frame)
            frame_count += 1
        
        # Clean up
        cap.release()
        out.release()
        
        # Convert the output to MP4 using FFmpeg if necessary
        if not temp_output.endswith('.mp4'):
            status_text.text("Converting video format...")
            final_output = os.path.join(temp_dir, 'output.mp4')
            os.system(f'ffmpeg -i {temp_output} -c:v libx264 -preset medium -crf 23 -c:a aac -b:a 128k {final_output} -y')
            temp_output = final_output
        
        # Read the final video file
        with open(temp_output, 'rb') as f:
            video_data = f.read()
        
        # Clear the progress bar and status text
        progress_bar.empty()
        status_text.empty()
        
        return video_data
    
def create_plot(time_values, sync_a_slice, df_distance, total_time, sync_a, lag, cut_index, output_path):
    plt.figure(figsize=(8, 4), facecolor='none')
    ax = plt.gca()
    ax.set_facecolor('none')
    
    # Plot the speed signal
    ax.plot(time_values, sync_a_slice, 
             color='#00ff00',
             label='Velocity Profile',
             linewidth=3,
             alpha=0.8)
    
    # Add gradient fill
    ax.fill_between(time_values, sync_a_slice,
                    alpha=0.2, color='#00ff00')
    
    # Customize plot appearance
    ax.set_xlabel('Time [s]', color='white', fontsize=15, fontweight='bold', labelpad=10)
    ax.set_ylabel('Speed [m/s]', color='white', fontsize=15, fontweight='bold', labelpad=10)
    ax.grid(True, alpha=0.2, linestyle='--', color='white')
    
    # Style axes
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.tick_params(colors='white', grid_color='white')
    
    # Add legend
    legend = ax.legend(facecolor='none', edgecolor='none', 
                       loc='upper right', fontsize=15)
    plt.setp(legend.get_texts(), color='white')
    
    # Set plot limits
    ax.set_xlim(0, total_time)
    ax.set_ylim(sync_a[lag:cut_index].min() - 0.1, sync_a[lag:cut_index].max() + 0.1)
    
    # Add background panel
    ax.add_patch(Rectangle((0, 0), 1, 1, 
                         transform=ax.transAxes,
                         facecolor='black',
                         alpha=0.7,
                         zorder=-1))
    
    # Add distance plot
    ax2 = ax.twinx()
    ax2.plot(time_values, df_distance, 
             color='yellow',
             label='Distance',
             linewidth=2,
             alpha=0.8)
    ax2.set_ylabel('Distance [m]', color='yellow', fontsize=15, fontweight='bold', labelpad=10)
    ax2.tick_params(axis='y', colors='yellow')
    
    # Add second legend
    legend2 = ax2.legend(facecolor='none', edgecolor='none', 
                         loc='upper left', fontsize=15)
    plt.setp(legend2.get_texts(), color='yellow')
    
    # Save plot
    plt.savefig(output_path, 
                transparent=True, 
                bbox_inches='tight', 
                pad_inches=0.2,
                dpi=300)
    plt.close()

def overlay_plot_on_frame(frame, plot_img, width, height):
    # Resize plot
    plot_height = 250
    plot_width = 500
    plot_img = cv2.resize(plot_img, (plot_width, plot_height))
    
    # Create mask
    if plot_img.shape[2] == 4:
        mask = plot_img[:, :, 3] / 255.0
        mask = np.expand_dims(mask, axis=-1)
        plot_img = plot_img[:, :, :3]
    else:
        mask = np.ones((plot_height, plot_width, 1))
    
    # Position plot
    y_offset = 30
    x_offset = width - plot_width - 30
    
    # Add shadow
    shadow_offset = 5
    shadow_color = np.array([0, 0, 0])
    shadow_alpha = 0.3
    shadow_mask = mask * shadow_alpha
    shadow_roi = frame[y_offset+shadow_offset:y_offset+plot_height+shadow_offset, 
                      x_offset+shadow_offset:x_offset+plot_width+shadow_offset]
    frame[y_offset+shadow_offset:y_offset+plot_height+shadow_offset, 
          x_offset+shadow_offset:x_offset+plot_width+shadow_offset] = (
        shadow_roi * (1 - shadow_mask) + shadow_color * shadow_mask
    ).astype(np.uint8)
    
    # Overlay plot
    roi = frame[y_offset:y_offset+plot_height, x_offset:x_offset+plot_width]
    frame[y_offset:y_offset+plot_height, x_offset:x_offset+plot_width] = (
        roi * (1 - mask) + plot_img * mask
    ).astype(np.uint8)

def filter_landmarks(df_landmarks_raw, fps_video, cutoff_frequency):
    # Define the Butterworth filter
    nyquist_freq = 0.5 * fps_video
    normal_cutoff = cutoff_frequency / nyquist_freq
    b, a = butter(4, normal_cutoff, btype='low', analog=False)

    # Initialize a DataFrame to store the filtered data
    df_filtered = pd.DataFrame()

    # Apply the filter to each landmark's x and y coordinates
    for column in df_landmarks_raw.columns:
        if column != 'frame':
            df_filtered[column] = filtfilt(b, a, df_landmarks_raw[column])
        else:
            df_filtered[column] = df_landmarks_raw[column]

    return df_filtered

def process_video(video_path, show_pose=1):
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Load the video file
    cap = cv2.VideoCapture(video_path)

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the capturing length in seconds
    capturing_length = total_frames / fps

    # Initialize a list to store the landmarks data
    landmarks_data = []

    # Process the video frame by frame
    frame_count = 0
    
    status_text.text("Processing video frames...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform pose estimation
        results = pose.process(frame_rgb)

        # Extract and store landmarks data if available
        if results.pose_landmarks:
            frame_data = {'frame': frame_count}
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmark_name = mp_pose.PoseLandmark(idx).name.lower()
                frame_data[f'{landmark_name}_x'] = landmark.x
                frame_data[f'{landmark_name}_y'] = landmark.y
            landmarks_data.append(frame_data)

        # Update the progress
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count + 1}/{total_frames}")
        
        # Update the frame count
        frame_count += 1

    # Clean up
    cap.release()

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    # Convert the landmarks data to a DataFrame
    df_landmarks = pd.DataFrame(landmarks_data)

    return df_landmarks, fps, capturing_length, total_frames

# Define the segments and their respective weights
segments = [
    ('left_shoulder', 'right_shoulder', 0.15),
    ('left_hip', 'right_hip', 0.15),
    ('left_shoulder', 'left_hip', 0.10),
    ('right_shoulder', 'right_hip', 0.10),
    ('left_hip', 'left_knee', 0.10),
    ('right_hip', 'right_knee', 0.10),
    ('left_knee', 'left_ankle', 0.10),
    ('right_knee', 'right_ankle', 0.10),
    ('left_elbow', 'left_wrist', 0.05),
    ('right_elbow', 'right_wrist', 0.05)
] 

# segments = [
#     ('head', 'neck', 0.08),  # Head and neck (~8%)
#     ('left_shoulder', 'right_shoulder', 0.15),  # Thorax (~15%)
#     ('left_hip', 'right_hip', 0.14),  # Pelvis (~14%)
#     ('left_shoulder', 'left_hip', 0.11),  # Left torso half (~11%)
#     ('right_shoulder', 'right_hip', 0.11),  # Right torso half (~11%)
#     ('left_hip', 'left_knee', 0.10),  # Left thigh (~10%)
#     ('right_hip', 'right_knee', 0.10),  # Right thigh (~10%)
#     ('left_knee', 'left_ankle', 0.05),  # Left lower leg (~5%)
#     ('right_knee', 'right_ankle', 0.05),  # Right lower leg (~5%)
#     ('left_elbow', 'left_wrist', 0.016),  # Left forearm (~1.6%)
#     ('right_elbow', 'right_wrist', 0.016),  # Right forearm (~1.6%)
#     ('left_shoulder', 'left_elbow', 0.028),  # Left upper arm (~2.8%)
#     ('right_shoulder', 'right_elbow', 0.028),  # Right upper arm (~2.8%)
#     ('left_ankle', 'left_foot', 0.014),  # Left foot (~1.4%)
#     ('right_ankle', 'right_foot', 0.014)  # Right foot (~1.4%)
# ]

def calculate_segment_midpoints(df_landmarks_filtered):
    # Initialize lists to store the segment midpoints
    midpoints_x = []
    midpoints_y = []

    # Iterate over each frame
    for index, row in df_landmarks_filtered.iterrows():
        frame_midpoints_x = []
        frame_midpoints_y = []

        # Calculate the midpoints for each segment
        for segment in segments:
            landmark1, landmark2, _ = segment
            midpoint_x = (row[f'{landmark1}_x'] + row[f'{landmark2}_x']) / 2
            midpoint_y = (row[f'{landmark1}_y'] + row[f'{landmark2}_y']) / 2
            frame_midpoints_x.append(midpoint_x)
            frame_midpoints_y.append(midpoint_y)

        midpoints_x.append(frame_midpoints_x)
        midpoints_y.append(frame_midpoints_y)

    return midpoints_x, midpoints_y

def calculate_com(df_landmarks_filtered):
    # Calculate segment midpoints
    midpoints_x, midpoints_y = calculate_segment_midpoints(df_landmarks_filtered)

    # Initialize lists to store the COM positions
    com_x = []
    com_y = []

    # Iterate over each frame
    for i in range(len(midpoints_x)):
        frame_midpoints_x = midpoints_x[i]
        frame_midpoints_y = midpoints_y[i]

        # Calculate the weighted sum of the midpoints
        weighted_sum_x = sum(frame_midpoints_x[j] * segments[j][2] for j in range(len(segments)))
        weighted_sum_y = sum(frame_midpoints_y[j] * segments[j][2] for j in range(len(segments)))

        # Calculate the COM for the current frame
        com_x.append(weighted_sum_x)
        com_y.append(weighted_sum_y)

    # Create a DataFrame to store the COM positions
    df_com = pd.DataFrame({
        'frame': df_landmarks_filtered['frame'],
        'com_x': com_x,
        'com_y': com_y
    })

    df_landmarks_filtered['com_x'] = df_com['com_x']
    df_landmarks_filtered['com_y'] = df_com['com_y']
    return df_landmarks_filtered


    # Crop the data in df_pos_com to the lag and cut_index
    df_pos_com = df_pos_com.iloc[lag:cut_index, :].reset_index(drop=True)
    df_distance = df_distance.iloc[lag:cut_index, :].reset_index(drop=True)
    
    # Open the video
    directory, filename = os.path.split(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # Generate output video path based on input video path
    output_video_path = os.path.join(directory, f"{os.path.splitext(filename)[0]}_with_overlay.avi")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Set style for better visualization
    plt.style.use('dark_background')
    
    # Calculate the time per frame
    time_per_frame = total_time / (cut_index - lag)
    
    # Process each frame
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count < lag or frame_count > cut_index:
            frame_count += 1
            continue
        
        # Add com_x and com_y circles
        if frame_count - lag < len(df_pos_com):
            com_x = int(df_pos_com.loc[frame_count - lag, 'com_x'] * width)
            com_y = int(df_pos_com.loc[frame_count - lag, 'com_y'] * height)
            cv2.circle(frame, (com_x, com_y), 5, (0, 0, 255), -1)
        
        # Draw horizontal line from lag position to current position
        if frame_count - lag < len(df_pos_com):
            start_x = int(df_pos_com.loc[0, 'com_x'] * width)
            end_x = int(df_pos_com.loc[frame_count - lag, 'com_x'] * width)
            y_position = int(df_pos_com.loc[frame_count - lag, 'com_y'] * height)
            cv2.line(frame, (start_x, y_position), (end_x, y_position), (255, 0, 0), 2)
            
            # Add current value of df_distance at this frame
            distance_value = df_distance.loc[frame_count - lag].values[0]
            cv2.putText(frame, f"{distance_value:.2f} m", (end_x + 10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Create figure with dark theme
        plt.figure(figsize=(8, 4), facecolor='none')
        ax = plt.gca()
        ax.set_facecolor('none')
        
        # Plot the speed signal with enhanced styling
        time_values = [i * time_per_frame for i in range(frame_count - lag + 1)]
        sync_a_slice = sync_a[lag:frame_count + 1]
        if len(time_values) != len(sync_a_slice):
            # Adjust time_values to match sync_a_slice length
            time_values = time_values[:len(sync_a_slice)]
        ax.plot(time_values, sync_a_slice, 
                 color='#00ff00',  # Bright green color
                 label='Velocity Profile',
                 linewidth=3,
                 alpha=0.8)
        
        # Add a gradient fill under the line
        ax.fill_between(time_values, sync_a_slice,
                        alpha=0.2, color='#00ff00')
        
        # Customize the plot appearance
        ax.set_xlabel('Time [s]', color='white', fontsize=15, fontweight='bold', labelpad=10)
        ax.set_ylabel('Speed [m/s]', color='white', fontsize=15, fontweight='bold', labelpad=10)
        ax.set_title('', color='white', fontsize=15, fontweight='bold', pad=15)
        
        # Style the grid
        ax.grid(True, alpha=0.2, linestyle='--', color='white')
        
        # Style the axes
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.tick_params(colors='white', grid_color='white')
        
        # Add legend with custom styling
        legend = ax.legend(facecolor='none', edgecolor='none', 
                           loc='upper right', fontsize=15)
        plt.setp(legend.get_texts(), color='white')
        
        # Set plot limits with some padding
        ax.set_xlim(0, total_time)
        ax.set_ylim(sync_a[lag:cut_index].min() - 0.1, sync_a[lag:cut_index].max() + 0.1)
        
        # Add a semi-transparent background panel
        ax.add_patch(Rectangle((0, 0), 1, 1, 
                                     transform=ax.transAxes,
                                     facecolor='black',
                                     alpha=0.7,
                                     zorder=-1))
        
        # Create a secondary y-axis for df_distance
        ax2 = ax.twinx()
        ax2.plot(time_values, df_distance.iloc[:frame_count - lag + 1], 
                 color='yellow',  # Yellow color
                 label='Distance',
                 linewidth=2,
                 alpha=0.8)
        ax2.set_ylabel('Distance [m]', color='yellow', fontsize=15, fontweight='bold', labelpad=10)
        ax2.tick_params(axis='y', colors='yellow')
        
        # Add legend for the secondary y-axis
        legend2 = ax2.legend(facecolor='none', edgecolor='none', 
                             loc='upper left', fontsize=15)
        plt.setp(legend2.get_texts(), color='yellow')
        
        # Convert plot to image with transparency
        plt.savefig('temp_plot.png', 
                    transparent=True, 
                    bbox_inches='tight', 
                    pad_inches=0.2,
                    dpi=300)
        plt.close()
        
        # Read the plot image
        plot_img = cv2.imread('temp_plot.png', cv2.IMREAD_UNCHANGED)
        
        # Resize plot to fit in the top-right corner of the video
        plot_height = 250  # Increased size
        plot_width = 500   # Increased size
        plot_img = cv2.resize(plot_img, (plot_width, plot_height))
        
        # Create a mask for the plot (using alpha channel)
        if plot_img.shape[2] == 4:  # If image has alpha channel
            mask = plot_img[:, :, 3] / 255.0
            mask = np.expand_dims(mask, axis=-1)
            plot_img = plot_img[:, :, :3]  # Remove alpha channel
        else:
            mask = np.ones((plot_height, plot_width, 1))
        
        # Position the plot in the top-right corner with some padding
        y_offset = 30
        x_offset = width - plot_width - 30
        
        # Add a subtle shadow effect
        shadow_offset = 5
        shadow_color = np.array([0, 0, 0])
        shadow_alpha = 0.3
        
        # Create shadow
        shadow_mask = mask * shadow_alpha
        shadow_roi = frame[y_offset+shadow_offset:y_offset+plot_height+shadow_offset, 
                          x_offset+shadow_offset:x_offset+plot_width+shadow_offset]
        frame[y_offset+shadow_offset:y_offset+plot_height+shadow_offset, 
              x_offset+shadow_offset:x_offset+plot_width+shadow_offset] = (
            shadow_roi * (1 - shadow_mask) + shadow_color * shadow_mask
        ).astype(np.uint8)
        
        # Overlay the plot on the video frame
        roi = frame[y_offset:y_offset+plot_height, x_offset:x_offset+plot_width]
        frame[y_offset:y_offset+plot_height, x_offset:x_offset+plot_width] = (
            roi * (1 - mask) + plot_img * mask
        ).astype(np.uint8)
        
        # Add frame count text
        frame_text = f"Frame {frame_count}/{total_frames}"
        cv2.putText(frame, frame_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Write the frame
        out.write(frame)
        frame_count += 1
    
    # Clean up
    cap.release()
    out.release()
    #cv2.destroyAllWindows()
    if os.path.exists('temp_plot.png'):
        os.remove('temp_plot.png')
    
    print(f'Video with speed signal overlay created successfully: {output_video_path}')


    # Load the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw the COM position on the frame
        if frame_count < len(df_com):
            com_x = int(df_com.loc[frame_count, 'com_x'] * width)
            com_y = int(df_com.loc[frame_count, 'com_y'] * height)
            cv2.circle(frame, (com_x, com_y), 5, (0, 0, 255), -1)
        
        # Write the frame to the output video
        out.write(frame)
        
        frame_count += 1
    
    cap.release()
    out.release()
    #cv2.destroyAllWindows()