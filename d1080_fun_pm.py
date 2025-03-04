import pandas as pd
from scipy.signal import butter, filtfilt

def filter_1080_data(df_1080_unfiltered, fps_1080, cutoff_frequency):
    # Define the Butterworth filter
    nyquist_freq = 0.5 * fps_1080
    normal_cutoff = cutoff_frequency / nyquist_freq
    b, a = butter(4, normal_cutoff, btype='low', analog=False)

    # Initialize a DataFrame to store the filtered data
    df_filtered = pd.DataFrame()

    # Apply the filter to each column
    for column in df_1080_unfiltered.columns:
        try:
            df_filtered[column] = filtfilt(b, a, df_1080_unfiltered[column])
        except Exception as e:
            print(f"Skipping column {column} due to error: {e}")
            df_filtered[column] = df_1080_unfiltered[column]

    return df_filtered

def read_1080(xlsx_path):
    df = pd.read_excel(xlsx_path, header=0)
    # get the last entry in Time since start
    capturing_length_1080 = df['Time since start [s]'].iloc[-1]
    # get the number of frames in the 1080 data
    total_frames_1080 = len(df)
    # get the frame rate of the 1080 data
    fps_1080 = total_frames_1080 / capturing_length_1080
    #fps_video, capturing_length_video, total_frames_video
    median_sampling_freq =1/( df['Time since start [s]'].diff().median())
    return df, capturing_length_1080, total_frames_1080, fps_1080, median_sampling_freq