import cv2
import mediapipe as mp
import pandas as pd
from scipy.signal import butter, filtfilt
import pandas as pd
from default_processing_pm import upsample_signal, sync_signals
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import pandas as pd
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tempfile

import tempfile

def process_and_overlay_videoStreamlit(video_path, df_pos_com, sync_a, lag, cut_index, total_time, df_distance):
    # Crop the data in df_pos_com to the lag and cut_index
    df_pos_com = df_pos_com.iloc[lag:cut_index, :].reset_index(drop=True)
    df_distance = df_distance.iloc[lag:cut_index, :].reset_index(drop=True)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # Create a temporary file for the output video
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
    output_video_path = temp_video.name
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
    cv2.destroyAllWindows()
    if os.path.exists('temp_plot.png'):
        os.remove('temp_plot.png')
    
    print(f'Video with speed signal overlay created successfully: {output_video_path}')
    return output_video_path

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

def cut_video(video_path, lag, cut_index):
    directory, filename = os.path.split(video_path)
    output_path = os.path.join(directory, 'cutted_video.avi')
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (1920, 1080))
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i >= lag and i <= cut_index:
            out.write(frame)
        if i > cut_index:
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return output_path

def process_video(video_path, show_pose=1):
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

            # Draw the pose annotation on the frame if show_pose is 1
            if show_pose == 1:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame
        cv2.imshow('Pose Estimation', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update the progress
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

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



def process_and_overlay_video(video_path, df_pos_com, sync_a, lag, cut_index, total_time, df_distance):
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
    cv2.destroyAllWindows()
    if os.path.exists('temp_plot.png'):
        os.remove('temp_plot.png')
    
    print(f'Video with speed signal overlay created successfully: {output_video_path}')

def draw_com_on_video(video_path, df_com, output_path):
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
    cv2.destroyAllWindows()