import streamlit as st
from video_fun_pm import process_video, filter_landmarks, calculate_com, process_and_overlay_videoStreamlit
from d1080_fun_pm import read_1080, filter_1080_data
from default_processing_pm import pad_df, sync_signals, upsample_signal, sync_signals22, downsample_df, replace_non_finite_valuesDF
import pandas as pd
import tempfile

st.title("Video and Excel File Processing App")

# Upload video file
video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

# Upload Excel file
excel_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

if video_file and excel_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    with tempfile.NamedTemporaryFile(delete=False) as temp_excel:
        temp_excel.write(excel_file.read())
        excel_path = temp_excel.name

    # 1080 DATA PROCESSING
    df_1080_unfiltered, capturing_length_1080, total_frames_1080, fps_10802, fps_1080 = read_1080(excel_path)
    df_1080_filtered = filter_1080_data(df_1080_unfiltered, fps_1080, 10)

    # VIDEO PROCESSING
    df_landmarks_raw, fps_video, capturing_length_video, total_frames_video = process_video(video_path, show_pose=1)
    df_landmarks_filtered = filter_landmarks(df_landmarks_raw, fps_video, 5)
    df_landmarks_filtered = calculate_com(df_landmarks_filtered)
    df_velocity = df_landmarks_filtered.diff() * fps_video
    df_acceleration = df_velocity.diff() * fps_video

    df_velocity['com_x'] = df_velocity['com_x'].abs()
    df_1080_filtered['Speed [m/s]'] = df_1080_filtered['Speed [m/s]'].abs()

    if fps_video > fps_1080:
        signal_a = upsample_signal(df_1080_filtered['Speed [m/s]'], fps_1080, fps_video)
        signal_b = df_velocity['com_x']
    else:
        signal_a = upsample_signal(df_velocity['com_x'], fps_video, fps_1080)
        signal_b = df_1080_filtered['Speed [m/s]']

    synced_signal_a, synced_signal_b, lag = sync_signals22(signal_a, signal_b)

    df_velocity = replace_non_finite_valuesDF(df_velocity)
    df_1080_filtered = replace_non_finite_valuesDF(df_1080_filtered)

    signal_a = downsample_df(df_1080_filtered['Speed [m/s]'], 333.33, 60)
    df_distance = downsample_df(df_1080_filtered['Distance since start [m]'], 333.33, 60)
    signal_b = df_velocity['com_x']

    factor = signal_a.max() / signal_b.max()
    signal_b = signal_b * factor

    sync_a, sync_b, lag, cut_index = sync_signals(signal_a, signal_b)
    padded_df_distance = pad_df(df_distance, len(sync_a), lag, cut_index)

    df_pos_com = pd.DataFrame({
        'com_x': df_landmarks_filtered['com_x'],
        'com_y': df_landmarks_filtered['com_y']
    })

    total_time = df_1080_unfiltered['Time since start [s]'].iloc[-1]
    output_video_path = process_and_overlay_videoStreamlit(video_path, df_pos_com, sync_a, lag, cut_index, total_time, padded_df_distance)

    with open(output_video_path, "rb") as file:
        btn = st.download_button(
            label="Download Processed Video",
            data=file,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )