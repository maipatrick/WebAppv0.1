import os
os.environ['OPENCV_VIDEOIO_PRIORITY_BACKEND'] = '0'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
import numpy as np
from scipy.spatial import distance as dist
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
from scipy.signal import savgol_filter
import os
import cv2
import numpy as np
import tempfile
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import cv2
import tempfile
import streamlit as st
import os

from rtmlib import PoseTracker, BodyWithFeet


import os
import cv2
import numpy as np
from rtmlib import PoseTracker, BodyWithFeet, draw_skeleton
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.signal import savgol_filter

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




# Halpe-26 landmark names in the order used by BodyWithFeet
landmark_names = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "head", "neck", "hip",
    "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe",
    "left_heel", "right_heel"
]

def pose_estimation_rmt_pose_new(video_path, max_people=10):
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
    headers = ['frame']
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
            # if frame_count == 706:
            #     a = 2
            # Get the landmarks for the current frame, skipping the first column (frame count)
            
            landmarks = df_landmarks_filtered.iloc[frame_count, 1:]
            #TODO if not in the image it fails
            # Plot landmarks on the frame
            for i in range(0, len(landmarks), 2):
                #x = int(landmarks[i] * width)
                #y = int(landmarks[i + 1] * height)
                if np.isnan(landmarks[i]) :
                    pass
                else:
                    x = int(landmarks[i] )
                    y = int(landmarks[i + 1] )
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                    # put the landmarks name as text to the coordinate
                    cv2.putText(frame, landmarks.index[i].replace('_x', ''), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw lines between the segments
            # for segment in segmentspairs:
            #     landmark1, landmark2 = segment
            #     x1 = int(df_landmarks_filtered[f'{landmark1}_x'].iloc[frame_count] * width)
            #     y1 = int(df_landmarks_filtered[f'{landmark1}_y'].iloc[frame_count] * height)
            #     x2 = int(df_landmarks_filtered[f'{landmark2}_x'].iloc[frame_count] * width)
            #     y2 = int(df_landmarks_filtered[f'{landmark2}_y'].iloc[frame_count] * height)
            #     cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            

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




def filter_landmarks_new(df: pd.DataFrame, window_length: int = 7, polyorder: int = 2) -> pd.DataFrame:
    """
    Applies smoothing to x and y columns in the landmark DataFrame.
    
    Parameters:
        df (pd.DataFrame): Raw landmark DataFrame (output of process_video_blured_new)
        window_length (int): Length of the filter window (must be odd and > polyorder)
        polyorder (int): Order of the polynomial to fit
        
    Returns:
        pd.DataFrame: Smoothed landmark DataFrame with same structure
    """
    df_filtered = df.copy()
    
    # Ensure window length is valid
    if window_length >= len(df):
        window_length = len(df) - 1 if len(df) % 2 == 0 else len(df)
    if window_length < 3:
        return df  # Not enough data to smooth

    for col in df.columns:
        if '_x' in col or '_y' in col:
            # Interpolate missing values before smoothing
            series = df_filtered[col].interpolate(method='linear', limit_direction='both')
            try:
                smoothed = savgol_filter(series, window_length=window_length, polyorder=polyorder)
                df_filtered[col] = smoothed
            except Exception:
                # In case of error, fallback to original/interpolated
                df_filtered[col] = series

    return df_filtered




def pose_estimation_rmt_pose(video_path):
     # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()

    #video_path = Path(video_path)
    #output_path = video_path.with_name(video_path.stem + '_output.mp4')

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    caputure_length = total_number_of_frames *(1/ fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Set up pose tracker  ('lightweight', 'balanced', 'performance') 
    pose_tracker = PoseTracker(
        BodyWithFeet,
        det_frequency=1,
        mode="balanced",
        backend="onnxruntime",
        device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu",
        tracking=True,
        to_openpose=False
    )

    # Get keypoints structure
    model = BodyWithFeet
    #keypoints_ids = [node.id for _, _, node in RenderTree(model) if node.id is not None]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        keypoints_all, scores_all = pose_tracker(frame)
        
        for keypoints in keypoints_all:
            X = keypoints[:, 0]
            Y = keypoints[:, 1]
            # draw circles with cv2.circle
            for i, (x, y) in enumerate(zip(X, Y)):
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
            #draw_skel(frame, [X], [Y], model)
            #draw_keypts(frame, [X], [Y], scores_all, cmap_str='RdYlGn')
        #print(f"\rProcessing frame {frame_number}/{total_number_of_frames}...", end="")
        # Update the progress
        progress = frame_number / total_number_of_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {int(frame_number) + 1}/{total_number_of_frames}")
        #out.write(frame)

    cap.release()
    #out.release()
    #print(f"✅ Output saved to: {output_path}")
    return fps, total_number_of_frames, caputure_length ,video_path 

def process_video_blured_new(video_path, show_pose=1, blur_faces=False):
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Load the video file
    cap = cv2.VideoCapture(video_path)
    min_detection_confidence = 0.6
    min_tracking_confidence = 0.6
    # Initialize MediaPipe Pose and Face Detection
    mp_pose = mp.solutions.pose
    # 🔧 2. Enable landmark smoothing for video stability
    # MediaPipe has an optional smoothing filter that reduces jitter between frames — especially useful when dealing with subtle pose changes or motion blur.

    # ✅ Make sure this is set:
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    #pose = mp_pose.Pose(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)

    mp_drawing = mp.solutions.drawing_utils
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection()

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the capturing length in seconds
    capturing_length = total_frames / fps

    # Initialize a list to store the landmarks data
    landmarks_data = []

    # Prepare video writer for output video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') #mp4v avc1
    output_video_path = os.path.join(tempfile.gettempdir(), 'processed_video.mp4')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Process the video frame by frame
    frame_count = 0
    
    status_text.text("Processing video frames...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame (e.g., histogram equalization)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame = cv2.equalizeHist(frame)
        #frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform pose estimation
        results = pose.process(frame_rgb)

        # Extract and store landmarks data if available
        frame_data = {'frame': frame_count}
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmark_name = mp_pose.PoseLandmark(idx).name.lower()
                frame_data[f'{landmark_name}_x'] = landmark.x
                frame_data[f'{landmark_name}_y'] = landmark.y
            # Draw the pose landmarks on the frame
            #mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        else:
            for idx in range(len(mp_pose.PoseLandmark)):
                landmark_name = mp_pose.PoseLandmark(idx).name.lower()
                frame_data[f'{landmark_name}_x'] = float('nan')
                frame_data[f'{landmark_name}_y'] = float('nan')
        landmarks_data.append(frame_data)

        # Blur faces if blur_faces is True
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

        # Write the frame to the output video
        out.write(frame)

        # Update the progress
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count + 1}/{total_frames}")
        
        # Update the frame count
        frame_count += 1

    # Clean up
    cap.release()
    out.release()

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    # Convert the landmarks data to a DataFrame
    df_landmarks = pd.DataFrame(landmarks_data)

    return df_landmarks, fps, capturing_length, total_frames, output_video_path


def process_video_blured(video_path, show_pose=1, blur_faces=False):
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Load the video file
    cap = cv2.VideoCapture(video_path)
    min_detection_confidence = 0.3
    min_tracking_confidence = 0.3
    # Initialize MediaPipe Pose and Face Detection
    mp_pose = mp.solutions.pose
    # 🔧 2. Enable landmark smoothing for video stability
    # MediaPipe has an optional smoothing filter that reduces jitter between frames — especially useful when dealing with subtle pose changes or motion blur.

    # ✅ Make sure this is set:
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    #pose = mp_pose.Pose(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)

    mp_drawing = mp.solutions.drawing_utils
    mp_face_detection = mp.solutions.face_detection
    mp_face_detection.FaceDetection()

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the capturing length in seconds
    capturing_length = total_frames / fps

    # Initialize a list to store the landmarks data
    landmarks_data = []

    # Prepare video writer for output video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') #mp4v avc1
    output_video_path = os.path.join(tempfile.gettempdir(), 'processed_video.mp4')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Process the video frame by frame
    frame_count = 0
    
    status_text.text("Processing video frames...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame (e.g., histogram equalization)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame = cv2.equalizeHist(frame)
        #frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform pose estimation
        results = pose.process(frame_rgb)

        # Extract and store landmarks data if available
        frame_data = {'frame': frame_count}
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmark_name = mp_pose.PoseLandmark(idx).name.lower()
                frame_data[f'{landmark_name}_x'] = landmark.x
                frame_data[f'{landmark_name}_y'] = landmark.y
        else:
            for idx in range(len(mp_pose.PoseLandmark)):
                landmark_name = mp_pose.PoseLandmark(idx).name.lower()
                frame_data[f'{landmark_name}_x'] = float('nan')
                frame_data[f'{landmark_name}_y'] = float('nan')
        landmarks_data.append(frame_data)

        # Blur faces if blur_faces is True
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

        # Write the frame to the output video
        out.write(frame)

        # Update the progress
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count + 1}/{total_frames}")
        
        # Update the frame count
        frame_count += 1

    # Clean up
    cap.release()
    out.release()

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    # Convert the landmarks data to a DataFrame
    df_landmarks = pd.DataFrame(landmarks_data)

    return df_landmarks, fps, capturing_length, total_frames, output_video_path

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
        frame_data = {'frame': frame_count}
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmark_name = mp_pose.PoseLandmark(idx).name.lower()
                frame_data[f'{landmark_name}_x'] = landmark.x
                frame_data[f'{landmark_name}_y'] = landmark.y
        else:
            for idx in range(len(mp_pose.PoseLandmark)):
                landmark_name = mp_pose.PoseLandmark(idx).name.lower()
                frame_data[f'{landmark_name}_x'] = float('nan')
                frame_data[f'{landmark_name}_y'] = float('nan')
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

def process_video_woNaNs(video_path, show_pose=1):
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
#for mediapipe
# segments = [
#     ('left_shoulder', 'right_shoulder', 0.15),
#     ('left_hip', 'right_hip', 0.15),
#     ('left_shoulder', 'left_hip', 0.10),
#     ('right_shoulder', 'right_hip', 0.10),
#     ('left_hip', 'left_knee', 0.10),
#     ('right_hip', 'right_knee', 0.10),
#     ('left_knee', 'left_ankle', 0.10),
#     ('right_knee', 'right_ankle', 0.10),
#     ('left_elbow', 'left_wrist', 0.05),
#     ('right_elbow', 'right_wrist', 0.05)
# ] 
segments = [
    ('LShoulder', 'RShoulder', 0.15),
    ('LHip', 'RHip', 0.15),
    ('LShoulder', 'LHip', 0.10),
    ('RShoulder', 'RHip', 0.10),
    ('LHip', 'LKnee', 0.10),
    ('RHip', 'RKnee', 0.10),
    ('LKnee', 'LAnkle', 0.10),
    ('RKnee', 'RAnkle', 0.10),
    ('LElbow', 'LWrist', 0.05),
    ('RElbow', 'RWrist', 0.05)
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

def convert_pose_data_to_meters_multiple(pose_data, reference_height_m):
    """
    Convert pose_data pixel coordinates to meters using reference height for each person.

    Parameters:
        pose_data (dict): Original pose data with pixel coordinates for multiple persons.
        reference_height_m (float): Real-world height of the reference person (in meters).

    Returns:
        pose_data_m (dict): Converted pose data in meters, same format as pose_data.
    """
    pose_data_m = {}

    for person_id, person_data in pose_data.items():
        coords_list = person_data['coords']

        # Estimate height in pixels per frame using top and bottom keypoints
        heights_px = []
        for frame_coords in coords_list:
            frame_coords = np.array(frame_coords)
            if frame_coords.shape[0] < 2:
                continue
            # Use min and max Y values to estimate person height (head to foot)
            y_values = frame_coords[:, 1]
            if np.isnan(y_values).all():
                continue
            height_px = np.nanmax(y_values) - np.nanmin(y_values)
            if height_px > 0:
                heights_px.append(height_px)

        if len(heights_px) == 0:
            raise ValueError(f"Could not compute valid reference height in pixels for person ID '{person_id}'.")

        # Calculate the average height in pixels for this person
        avg_height_px = np.mean(heights_px)
        pixels_per_meter = avg_height_px / reference_height_m

        # Convert all pose data for this person to meters
        coords_meters = [np.array(frame_coords) / pixels_per_meter for frame_coords in person_data['coords']]
        pose_data_m[person_id] = {
            'coords': coords_meters,
            'scores': person_data['scores'],
            'frames': person_data['frames']
        }

    return pose_data_m

def convert_pose_data_to_meters(pose_data, reference_height_m, reference_person_id):
    """
    Convert pose_data pixel coordinates to meters using reference height of one tracked person.

    Parameters:
    - pose_data (dict): Original pose data with pixel coordinates.
    - reference_height_m (float): Real-world height of the reference person (in meters).
    - reference_person_id (str or int): The ID of the reference person in the pose_data.

    Returns:
    - pose_data_m (dict): Converted pose data in meters, same format as pose_data.
    """

    if str(reference_person_id) not in pose_data:
        raise ValueError(f"Reference person ID '{reference_person_id}' not found in pose_data.")

    person_data = pose_data[str(reference_person_id)]
    coords_list = person_data['coords']

    # Estimate height in pixels per frame using top and bottom keypoints
    heights_px = []
    for frame_coords in coords_list:
        frame_coords = np.array(frame_coords)
        if frame_coords.shape[0] < 2:
            continue
        # Use min and max Y values to estimate person height (head to foot)
        y_values = frame_coords[:, 1]
        if np.isnan(y_values).all():
            continue
        height_px = np.nanmax(y_values) - np.nanmin(y_values)
        if height_px > 0:
            heights_px.append(height_px)

    if len(heights_px) == 0:
        raise ValueError("Could not compute valid reference height in pixels.")

    avg_height_px = np.mean(heights_px)
    pixels_per_meter = avg_height_px / reference_height_m

    # Convert all pose data to meters
    pose_data_m = {}
    for pid, pdata in pose_data.items():
        coords_meters = [np.array(frame_coords) / pixels_per_meter for frame_coords in pdata['coords']]
        pose_data_m[pid] = {
            'coords': coords_meters,
            'scores': pdata['scores'],
            'frames': pdata['frames']
        }

    return pose_data_m

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

def process_video_multi_person_rstLIB(input_path: str):
    assert os.path.exists(input_path), f"❌ File not found: {input_path}"
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    # === Load video ===
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    base, ext = os.path.splitext(input_path)
    #output_path = f"{base}_pose{ext}"
    #fourcc = cv2.VideoWriter_fourcc(*("mp4v" if ext.lower() != ".avi" else "XVID"))
    #writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    #print(f"📹 Input: {input_path}")
    #print(f"💾 Output: {output_path}")

    # === RTMLib Pose Estimation ===
    pose_tracker = PoseTracker(
        BodyWithFeet,
        det_frequency=5,
        mode='balanced',
        backend='onnxruntime',
        device='cpu',
        tracking=False
    ) #balanced performance
    #BodyWithFeet.keypoints = BodyWithFeet.keypoints[:25]  # 25 keypoints

    # === Deep SORT Tracker ===
    tracker = DeepSort(max_age=30, n_init=3)
    pose_data = {}  # person_id: {coords: [], scores: []}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints_list, scores_list = pose_tracker(frame)
        keypoints = np.array(keypoints_list)  # (N, K, 2)
        scores = np.array(scores_list)        # (N, K)

        detections = []
        for i, kps in enumerate(keypoints):
            if np.isnan(kps).any():
                continue
            x1, y1 = np.min(kps, axis=0)
            x2, y2 = np.max(kps, axis=0)
            w, h = x2 - x1, y2 - y1
            bbox = [x1, y1, w, h]
            conf = float(np.nanmean(scores[i]))
            detections.append((bbox, conf, {'keypoints': kps, 'score': scores[i]}))

        tracks = tracker.update_tracks(detections, frame=frame)

        for t in tracks:
            if not t.is_confirmed():
                continue

            track_id = t.track_id
            detection = t.det_class  # dict: {'keypoints', 'score'}
            if detection is None or 'keypoints' not in detection:
                continue

            kps = detection['keypoints']
            score = detection['score']

            # Save data for smoothing later
            # if track_id not in pose_data:
            #     pose_data[track_id] = {'coords': [], 'scores': []}
            # pose_data[track_id]['coords'].append(kps)
            # pose_data[track_id]['scores'].append(score)
            if track_id not in pose_data:
                pose_data[track_id] = {'coords': [], 'scores': [], 'frames': []}
            pose_data[track_id]['coords'].append(kps)
            pose_data[track_id]['scores'].append(score)
            pose_data[track_id]['frames'].append(frame_idx)

            # Draw skeleton + ID
            draw_skeleton(frame, np.array([kps]), np.array([score]), kpt_thr=0.1)
            l, t_, r, b = t.to_ltrb()
            cv2.putText(frame, f"ID {track_id}", (int(l), int(t_) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        progress = frame_idx / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_idx + 1}/{total_frames}")
        #writer.write(frame)
        frame_idx += 1
        #if frame_idx % 30 == 0:
        #    print(f"🧠 Frame {frame_idx}/{total_frames}")

    cap.release()
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    #writer.release()
    #print("✅ Saved video with skeletons:", output_path)

    # === Smooth keypoints ===
    #print("🔄 Smoothing landmarks with Savitzky-Golay filter...")

    # for track_id, person in pose_data.items():
    #     coords = np.array(person['coords'])  # (T, K, 2)
    #     scores = np.array(person['scores'])  # (T, K)
    #     T, K, _ = coords.shape

    #     if T < 5:
    #         continue  # not enough frames

    #     for k in range(K):
    #         for d in range(2):  # x and y
    #             seq = coords[:, k, d]
    #             conf = scores[:, k]

    #             # Replace low confidence with NaN
    #             seq[conf < 0.3] = np.nan
    #             if np.all(np.isnan(seq)):
    #                 continue

    #             # Interpolate missing
    #             nans = np.isnan(seq)
    #             seq[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), seq[~nans])

    #             # Smooth with Savitzky-Golay
    #             window = min(7, len(seq) if len(seq) % 2 == 1 else len(seq)-1)
    #             if window >= 5:
    #                 coords[:, k, d] = savgol_filter(seq, window_length=window, polyorder=2)

    #     pose_data[track_id]['coords'] = coords


    #print("🎉 Done smoothing. Everything worked!")
    capture_length = frame_idx / (1/fps)
    return pose_data, fps, total_frames, capture_length

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



######################### NEW

# Define the skeleton connections with color information
skeleton_connections = [
    ("LBigToe", "LHeel", (0, 255, 0)),  # Green for left side
    ("RBigToe", "RHeel", (255, 0, 0)),  # Blue for right side
    ("LAnkle", "LKnee", (0, 255, 0)),
    ("RAnkle", "RKnee", (255, 0, 0)),
    ("LKnee", "LHip", (0, 255, 0)),
    ("RKnee", "RHip", (255, 0, 0)),
    ("RHip", "Hip", (255, 0, 0)),
    ("Hip", "LHip", (0, 255, 0)),
    ("Hip", "Neck", (255, 255, 255)),  # White for central connections
    ("Neck", "RShoulder", (255, 0, 0)),
    ("Neck", "LShoulder", (0, 255, 0)),
    ("RShoulder", "RElbow", (255, 0, 0)),
    ("LShoulder", "LElbow", (0, 255, 0)),
    ("LElbow", "LWrist", (0, 255, 0)),
    ("RElbow", "RWrist", (255, 0, 0)),
]

def new_video_for_linear(video_path, sync_a, start_frame, end_frame, df_landmarks_filtered, landmarks_draw, distance_1080_in_video, pose_data, blur_faces, draw_skeleton_in_video):
    """
    Processes a video and overlays the sync_a signal, selected landmarks, skeleton, and optionally blurs faces on it for frames between start_frame and end_frame.

    Parameters:
        video_path (str): Path to the input video.
        sync_a (np.ndarray): Signal to overlay, with as many frames as the video.
        start_frame (int): The starting frame index for processing.
        end_frame (int): The ending frame index for processing.
        df_landmarks_filtered (pd.DataFrame): DataFrame containing landmark coordinates (_x and _y) for each frame.
        landmarks_draw (list): List of landmark names (without _x and _y) to be plotted.
        distance_1080_in_video (np.ndarray): Distance signal to overlay.
        pose_data (dict): Pose data containing landmarks for multiple persons.
        blur_faces (bool): Whether to blur faces in the video.

    Returns:
        video_data (bytes): Processed video data.
    """
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
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Validate start_frame and end_frame
        if start_frame < 0 or end_frame > total_frames or start_frame >= end_frame:
            raise ValueError("Invalid start_frame or end_frame values.")

        # Slice sync_a, distance_1080_in_video, and landmarks to match the range of start_frame to end_frame
        sync_a = sync_a[start_frame:end_frame]
        distance_1080_in_video = distance_1080_in_video[start_frame:end_frame]
        df_landmarks_filtered = df_landmarks_filtered[start_frame:end_frame]
        angles_df = pose_data['1']['Angles_filt'].iloc[start_frame:end_frame].reset_index(drop=True)
        scores_df = pose_data['1']['scores'].iloc[start_frame:end_frame].reset_index(drop=True)
        right_knee_angles = angles_df['RightKnee'].to_numpy()
        left_knee_angles = angles_df['LeftKnee'].to_numpy()


        # Use MPEG4 codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_output = os.path.join(temp_dir, 'temp_output.mp4')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

        if not out.isOpened():
            raise Exception("Could not initialize video writer. No compatible codec found.")

        # Set style for better visualization
        plt.style.use('dark_background')

        # Calculate the time per frame
        time_per_frame = 1 / fps
        frame_count = 0

        # Set the video to start at the start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count >= (end_frame - start_frame):
                break

            # Get the landmarks for the current frame
            landmarks = df_landmarks_filtered.iloc[frame_count]

            # Draw only the specified landmarks
            for landmark in landmarks_draw:
                x_col = f"{landmark}_x"
                y_col = f"{landmark}_y"
                if x_col in landmarks.index and y_col in landmarks.index:
                    # and scores_df[landmark].iloc[frame_count]>=0.6:
                    x = int(landmarks[x_col]) if not np.isnan(landmarks[x_col]) else None
                    y = int(landmarks[y_col]) if not np.isnan(landmarks[y_col]) else None
                    if x is not None and y is not None:
                        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Draw a red circle
            if draw_skeleton_in_video == True:
                # Draw the skeleton with colors
                for connection in skeleton_connections:
                    landmark1, landmark2, color = connection
                    x1_col = f"{landmark1}_x"
                    y1_col = f"{landmark1}_y"
                    x2_col = f"{landmark2}_x"
                    y2_col = f"{landmark2}_y"
                    if (
                        x1_col in landmarks.index and y1_col in landmarks.index and
                        x2_col in landmarks.index and y2_col in landmarks.index
                    ):
                        x1 = int(landmarks[x1_col]) if not np.isnan(landmarks[x1_col]) else None
                        y1 = int(landmarks[y1_col]) if not np.isnan(landmarks[y1_col]) else None
                        x2 = int(landmarks[x2_col]) if not np.isnan(landmarks[x2_col]) else None
                        y2 = int(landmarks[y2_col]) if not np.isnan(landmarks[y2_col]) else None
                        if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                            cv2.line(frame, (x1, y1), (x2, y2), color, 2)  # Use the color from the connection

            face_landmarks = ["REye", "LEye", "Nose", "REar", "LEar"]  # Example face landmarks
            # Blur faces if blur_faces is True
            if blur_faces:
                # Filter pose_data for the range [start_frame, end_frame]
                filtered_pose_data = {
                    person_key: {
                        'filled': person_data['filled'].iloc[start_frame:end_frame].reset_index(drop=True)
                    }
                    for person_key, person_data in pose_data.items()
                }

                # Process each frame
                for person_key, person_data in filtered_pose_data.items():
                    person_df = person_data['filled']

                    # Ensure the frame index is within the bounds of the person's data
                    if frame_count < len(person_df):
                        person_landmarks = person_df.iloc[frame_count]
                        face_coords = []

                        # Extract face landmarks for the current person
                        for landmark in face_landmarks:
                            x_col = f"{landmark}_x"
                            y_col = f"{landmark}_y"
                            if x_col in person_landmarks.index and y_col in person_landmarks.index:
                                x = person_landmarks[x_col]
                                y = person_landmarks[y_col]
                                if not np.isnan(x) and not np.isnan(y):
                                    # Use the raw coordinates (no scaling)
                                    x = int(x)
                                    y = int(y)
                                    face_coords.append((x, y))

                        if face_coords:
                            # Calculate bounding box around the face
                            x_coords = [coord[0] for coord in face_coords]
                            y_coords = [coord[1] for coord in face_coords]
                            x_min, x_max = max(0, min(x_coords)), min(width, max(x_coords))
                            y_min, y_max = max(0, min(y_coords)), min(height, max(y_coords))

                            # Calculate circle parameters
                            center_x = (x_min + x_max) // 2
                            center_y = (y_min + y_max) // 2
                            radius = int(max((x_max - x_min) // 2, (y_max - y_min) // 2) * 1.5)  # Increase size by 1.5x

                            # Validate circle parameters and apply pixelation
                            if 0 <= center_x < width and 0 <= center_y < height and radius > 0:
                                # Define the bounding box for the face
                                x_min = max(0, center_x - radius)
                                x_max = min(width, center_x + radius)
                                y_min = max(0, center_y - radius)
                                y_max = min(height, center_y + radius)

                                # Extract the region of interest (ROI) around the face
                                face_roi = frame[y_min:y_max, x_min:x_max]

                                # Pixelate the ROI by resizing it to a smaller size and then back to the original size
                                if face_roi.size > 0:
                                    small_size = (10, 10)  # Size for pixelation
                                    face_roi_small = cv2.resize(face_roi, small_size, interpolation=cv2.INTER_LINEAR)
                                    face_roi_pixelated = cv2.resize(face_roi_small, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_NEAREST)

                                    # Replace the original ROI with the pixelated version
                                    frame[y_min:y_max, x_min:x_max] = face_roi_pixelated
                            else:
                                print(f"Invalid circle parameters for person {person_key} at frame {frame_count}: center_x={center_x}, center_y={center_y}, radius={radius}")
                        else:
                            print(f"No valid face coordinates for person {person_key} at frame {frame_count}")

            # Create and save plot
            temp_plot_path = os.path.join(temp_dir, 'temp_plot.png')
            create_sync_a_plot_linear(
                time_values=[i * time_per_frame for i in range(frame_count + 1)], 
                velocity_1080_in_video=sync_a[:frame_count + 1], 
                distance_1080_in_video=distance_1080_in_video[:frame_count + 1], 
                total_time=(end_frame - start_frame) / fps, 
                output_path=temp_plot_path
            )

                # Create and save knee angle plot
            temp_knee_plot_path = os.path.join(temp_dir, 'temp_knee_plot.png')
            create_knee_angle_plot(
                time_values=[i * time_per_frame for i in range(frame_count + 1)],
                right_knee_angles=right_knee_angles[:frame_count + 1],
                left_knee_angles=left_knee_angles[:frame_count + 1],
                total_time=(end_frame - start_frame) / fps,
                output_path=temp_knee_plot_path
            )


            # Overlay plot on frame
            plot_img = cv2.imread(temp_plot_path, cv2.IMREAD_UNCHANGED)
            if plot_img is not None:
                overlay_plot_on_frame_linear(frame, plot_img, width, height)

            # Overlay knee angle plot on the upper-left corner
            knee_plot_img = cv2.imread(temp_knee_plot_path, cv2.IMREAD_UNCHANGED)
            if knee_plot_img is not None:
                overlay_plot_on_frame_upper_left(frame, knee_plot_img, width, height)

            # Write the frame
            out.write(frame)
            progress = (frame_count / len(sync_a)) * 100
            progress_bar.progress(int(progress))
            status_text.text(f" {int(progress)}%")
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

def create_sync_a_plot_linear(time_values, velocity_1080_in_video, distance_1080_in_video, total_time, output_path):
    """
    Creates a plot of velocity_1080_in_video and distance_1080_in_video signals.

    Parameters:
        time_values (list): Time values for the x-axis.
        velocity_1080_in_video (np.ndarray): Velocity signal to plot.
        distance_1080_in_video (np.ndarray): Distance signal to plot.
        total_time (float): Total duration of the video in seconds.
        output_path (str): Path to save the plot.
    """
    plt.figure(figsize=(8, 4), facecolor='black')
    ax1 = plt.gca()
    ax1.set_facecolor('black')

    # Plot velocity_1080_in_video on the left y-axis
    ax1.plot(time_values, velocity_1080_in_video, color='red', label='Velocity (m/s)', linewidth=2, alpha=0.8)
    ax1.fill_between(time_values, velocity_1080_in_video, alpha=0.2, color='red')
    ax1.set_xlabel('Time [s]', color='white', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Velocity [m/s]', color='red', fontsize=12, fontweight='bold', labelpad=10)
    ax1.tick_params(axis='y', colors='red')
    ax1.grid(True, alpha=0.2, linestyle='--', color='white')

    # Add a secondary y-axis for distance_1080_in_video
    ax2 = ax1.twinx()
    ax2.plot(time_values, distance_1080_in_video, color='yellow', label='Distance (m)', linewidth=2, alpha=0.8)
    ax2.set_ylabel('Distance [m]', color='yellow', fontsize=12, fontweight='bold', labelpad=10)
    ax2.tick_params(axis='y', colors='yellow')

    # Style axes
    for spine in ax1.spines.values():
        spine.set_color('white')
    for spine in ax2.spines.values():
        spine.set_color('white')
    ax1.tick_params(colors='white', grid_color='white')

    # Add legends
    ax1.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)

    # Set plot limits
    ax1.set_xlim(0, total_time)
    ax1.set_ylim(velocity_1080_in_video.min() - 0.1, velocity_1080_in_video.max() + 0.1)
    ax2.set_ylim(distance_1080_in_video.min() - 0.1, distance_1080_in_video.max() + 0.1)

    # Save plot
    plt.savefig(output_path, transparent=True, bbox_inches='tight', pad_inches=0.2, dpi=300)
    plt.close()


def create_knee_angle_plot(time_values, right_knee_angles, left_knee_angles, total_time, output_path):
    """
    Creates a plot of right and left knee angles.

    Parameters:
        time_values (list): Time values for the x-axis.
        right_knee_angles (np.ndarray): Right knee angles to plot.
        left_knee_angles (np.ndarray): Left knee angles to plot.
        total_time (float): Total duration of the video in seconds.
        output_path (str): Path to save the plot.
    """
    plt.figure(figsize=(8, 4), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')

    # Plot right knee angles
    ax.plot(time_values, right_knee_angles, color=(0, 0, 1), label='Right Knee Angle', linewidth=2, alpha=0.8)
    ax.fill_between(time_values, right_knee_angles, alpha=0.2, color=(0, 0, 1))

    # Plot left knee angles
    ax.plot(time_values, left_knee_angles, color=(0, 1, 0), label='Left Knee Angle', linewidth=2, alpha=0.8)
    ax.fill_between(time_values, left_knee_angles, alpha=0.2, color=(0, 1, 0))

    # Customize the plot
    ax.set_xlabel('Time [s]', color='white', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Angle [°]', color='white', fontsize=12, fontweight='bold', labelpad=10)
    ax.tick_params(axis='both', colors='white')
    ax.grid(True, alpha=0.2, linestyle='--', color='white')

    # Style axes
    for spine in ax.spines.values():
        spine.set_color('white')

    # Add legend
    ax.legend(loc='upper right', fontsize=10, facecolor='none', edgecolor='none')
    plt.setp(ax.get_legend().get_texts(), color='white')

    # Set plot limits
    ax.set_xlim(0, total_time)
    ax.set_ylim(min(right_knee_angles.min(), left_knee_angles.min()) - 5,
                max(right_knee_angles.max(), left_knee_angles.max()) + 5)

    # Save the plot
    plt.savefig(output_path, transparent=True, bbox_inches='tight', pad_inches=0.2, dpi=300)
    plt.close()


def overlay_plot_on_frame_upper_left(frame, plot_img, width, height):
    """
    Overlays a plot image on the upper-left corner of a video frame.

    Parameters:
        frame (np.ndarray): Video frame.
        plot_img (np.ndarray): Plot image to overlay.
        width (int): Width of the video frame.
        height (int): Height of the video frame.
    """
    # Resize plot
    plot_height = int(height * 0.25)  # 25% of the video height
    plot_width = int(width * 0.4)    # 40% of the video width
    plot_img = cv2.resize(plot_img, (plot_width, plot_height))

    # Create mask
    if plot_img.shape[2] == 4:  # If the plot image has an alpha channel
        mask = plot_img[:, :, 3] / 255.0
        mask = np.expand_dims(mask, axis=-1)
        plot_img = plot_img[:, :, :3]  # Remove the alpha channel
    else:
        mask = np.ones((plot_height, plot_width, 1))

    # Position plot in the upper-left corner
    y_offset = int(height * 0.05)  # 5% padding from the top
    x_offset = int(width * 0.05)   # 5% padding from the left

    # Overlay plot
    roi = frame[y_offset:y_offset + plot_height, x_offset:x_offset + plot_width]
    frame[y_offset:y_offset + plot_height, x_offset:x_offset + plot_width] = (
        roi * (1 - mask) + plot_img * mask
    ).astype(np.uint8)

def overlay_plot_on_frame_linear(frame, plot_img, width, height):
    """
    Overlays a plot image on a video frame.

    Parameters:
        frame (np.ndarray): Video frame.
        plot_img (np.ndarray): Plot image to overlay.
        width (int): Width of the video frame.
        height (int): Height of the video frame.
    """
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
    roi = frame[y_offset:y_offset + plot_height, x_offset:x_offset + plot_width]
    frame[y_offset:y_offset + plot_height, x_offset:x_offset + plot_width] = (
        roi * (1 - mask) + plot_img * mask
    ).astype(np.uint8)