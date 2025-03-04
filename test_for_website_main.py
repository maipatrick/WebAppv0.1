
from video_fun_pm import process_video, filter_landmarks, calculate_com, draw_com_on_video, process_and_overlay_video
from d1080_fun_pm import read_1080, filter_1080_data
from default_processing_pm import pad_df,replace_non_finite_values, sync_signals, upsample_signal, sync_signals2, sync_signals22, downsample_df, replace_non_finite_valuesDF
import pandas as pd


video_path = r'C:\Users\adpatrick\OneDrive - nih.no\Desktop\WebApp1080sync\WebAppv0.1\fp22_m505_left 2.MOV'
a1080_file = r'C:\Users\adpatrick\OneDrive - nih.no\Desktop\WebApp1080sync\WebAppv0.1\fp22_m505_left 2.xlsx'

## 1080 DATA PROCESSING
df_1080_unfiltered, capturing_length_1080, total_frames_1080, fps_10802, fps_1080  = read_1080(a1080_file)
df_1080_filtered = filter_1080_data(df_1080_unfiltered, fps_1080, 10)

## VIDEO PROCESSING
# process the video pose estimation and video metadata
df_landmarks_raw, fps_video, capturing_length_video, total_frames_video = process_video(video_path, show_pose=1)
# Filter the landmarks data
df_landmarks_filtered = filter_landmarks(df_landmarks_raw, fps_video, 5)
# Calculate the center of mass position
df_landmarks_filtered = calculate_com(df_landmarks_filtered)
# Calculate the velocity and acceleration
df_velocity = df_landmarks_filtered.diff() * fps_video
df_acceleration = df_velocity.diff() * fps_video

# not optimal TODO
# abs the df_velocity['com_x'] and f_1080_filtered['Speed [m/s]']
df_velocity['com_x'] = df_velocity['com_x'].abs()
df_1080_filtered['Speed [m/s]'] = df_1080_filtered['Speed [m/s]'].abs()


# upsampling the signal with the smaller samling frequency to the one with higher sampling frequency
if fps_video > fps_1080:
    signal_a = upsample_signal(df_1080_filtered['Speed [m/s]'], fps_1080, fps_video)
    signal_b = df_velocity['com_x']
else:
    signal_a = upsample_signal(df_velocity['com_x'], fps_video, fps_1080)
    signal_b = df_1080_filtered['Speed [m/s]']


# cross-correlation to sync the signals
synced_signal_a, synced_signal_b, lag = sync_signals22(signal_a, signal_b)


# Clean the data from NaNs, -inf, and inf values
#df_comx = replace_non_finite_values(df_velocity['com_x'])

df_velocity = replace_non_finite_valuesDF(df_velocity)
df_1080_filtered = replace_non_finite_valuesDF(df_1080_filtered)
#df_distance = replace_non_finite_values(df_1080_filtered['Distance since start [m]'])

signal_a = downsample_df(df_1080_filtered['Speed [m/s]'], 333.33, 60)

df_distance = downsample_df(df_1080_filtered['Distance since start [m]'], 333.33, 60)

signal_b = df_velocity['com_x']

# Calculate a factor based on the peak of signal_a and scale the signal_a peak according to the factor to match the peak of signal_b
factor = signal_a.max() / signal_b.max()
signal_b = signal_b * factor

# Use cross-correlation to sync the two signals
sync_a, sync_b, lag, cut_index = sync_signals(signal_a, signal_b)
# plot
# import matplotlib.pyplot as plt
# plt.plot(sync_a)
# plt.plot(sync_b)
# plt.show()

padded_df_distance = pad_df(df_distance, len(sync_a), lag, cut_index)

df_pos_com = pd.DataFrame({
     'com_x': df_landmarks_filtered['com_x'],
     'com_y': df_landmarks_filtered['com_y']
 })

total_time = df_1080_unfiltered['Time since start [s]'].iloc[-1]
process_and_overlay_video(video_path, df_pos_com, sync_a, lag, cut_index, total_time, padded_df_distance)