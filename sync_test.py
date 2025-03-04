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

# File paths
a1080 = r'C:\Users\adpatrick\OneDrive - nih.no\Desktop\WebApp1080sync\WebAppv0.1\speed_1080.xlsx'
comx = r'C:\Users\adpatrick\OneDrive - nih.no\Desktop\WebApp1080sync\WebAppv0.1\velocity_com.xlsx'
video = r'C:\Users\adpatrick\OneDrive - nih.no\Desktop\WebApp1080sync\WebAppv0.1\fp22_m505_left 2.MOV'
com_position = r'C:\Users\adpatrick\OneDrive - nih.no\Desktop\WebApp1080sync\WebAppv0.1\com_pos.xlsx'

time = r'C:\Users\adpatrick\OneDrive - nih.no\Desktop\WebApp1080sync\WebAppv0.1\time_1080.xlsx'
distance = r'C:\Users\adpatrick\OneDrive - nih.no\Desktop\WebApp1080sync\WebAppv0.1\distance_1080.xlsx'
# Read the Excel file into a DataFrame
df_comx = pd.read_excel(comx, header=0)
df_a1080 = pd.read_excel(a1080, header=0)
df_pos_com = pd.read_excel(com_position, header=0)
df_time = pd.read_excel(time, header=0)
df_distance = pd.read_excel(distance, header=0)

def replace_non_finite_values(data):
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        cleaned_data = data.replace([np.inf, -np.inf], np.nan)
        cleaned_data = cleaned_data.ffill().bfill()
        cleaned_data = cleaned_data.interpolate(method='nearest')
    elif isinstance(data, np.ndarray):
        data = pd.Series(data)
        cleaned_data = data.replace([np.inf, -np.inf], np.nan)
        cleaned_data = cleaned_data.ffill().bfill()
        cleaned_data = cleaned_data.interpolate(method='nearest')
        cleaned_data = cleaned_data.to_numpy()
    else:
        raise ValueError("Input data must be a pandas DataFrame, Series, or numpy array.")
    
    return cleaned_data

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

def downsample_df(df, original_freq, target_freq):
    downsample_factor = int(original_freq / target_freq)
    downsampled_df = df.iloc[::downsample_factor, :].reset_index(drop=True)
    return downsampled_df




# Clean the data from NaNs, -inf, and inf values
df_comx = replace_non_finite_values(df_comx)
df_a1080 = replace_non_finite_values(df_a1080)
signal_a = downsample_df(df_a1080, 333.33, 60)
# downsample also df_distance
df_distance = replace_non_finite_values(df_distance)
df_distance = downsample_df(df_distance, 333.33, 60)

signal_b = df_comx

# Calculate a factor based on the peak of signal_a and scale the signal_a peak according to the factor to match the peak of signal_b
factor = signal_a['Speed [m/s]'].max() / signal_b['com_x'].max()
signal_b = signal_b * factor

# Use cross-correlation to sync the two signals
sync_a, sync_b, lag, cut_index = sync_signals(signal_a['Speed [m/s]'], signal_b['com_x'])


def pad_df(df, signal_length, lag, cut_index):
    # Get the nearest values at lag and cut_index
    start_value = df.iloc[0]
    end_value = df.iloc[-1]
    
    # Create padding for the beginning and end
    start_padding = pd.DataFrame([start_value] * lag, columns=df.columns)
    end_padding = pd.DataFrame([end_value] * (signal_length - cut_index), columns=df.columns)
    
    # Concatenate the padding with the original df_distance
    padded_df_distance = pd.concat([start_padding, df_distance, end_padding], ignore_index=True)
    
    return padded_df_distance

padded_df_distance = pad_df(df_distance,  len(sync_a) , lag, cut_index)



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
        ax.plot(time_values, sync_a[lag:frame_count + 1], 
                 color='#00ff00',  # Bright green color
                 label='Velocity Profile',
                 linewidth=3,
                 alpha=0.8)
        
        # Add a gradient fill under the line
        ax.fill_between(time_values, sync_a[lag:frame_count + 1],
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


total_time = df_time.iloc[-1].values[0]
process_and_overlay_video(video, df_pos_com, sync_a, lag, cut_index, total_time, padded_df_distance)