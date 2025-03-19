import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate

def pad_sync_signal(sync_a, lag, cut_index, total_frames_video):
    """
    Pads the sync_a signal to make it as long as total_frames_video using the lag and cut_index.
    
    Parameters:
    sync_a (pd.Series): The original sync_a signal.
    lag (int): The lag value.
    cut_index (int): The cut index value.
    total_frames_video (int): The total number of frames in the video.
    
    Returns:
    np.ndarray: The padded sync_a signal.
    """
    # Initialize the final sync_a array with the length of total_frames_video
    final_sync_a = np.zeros(total_frames_video)
    
    # Fill the initial part of the array (from 0 to lag) with the first value of sync_a
    final_sync_a[:lag] = sync_a.iloc[0]
    
    # Copy the values from sync_a to the new array starting from lag to cut_index
    final_sync_a[lag:lag + len(sync_a)] = sync_a
    
    # Fill the remaining part of the array (from cut_index to total_frames_video) with the last value of sync_a
    final_sync_a[lag + len(sync_a):] = sync_a.iloc[-1]
    
    return final_sync_a

def calculate_joint_angles(df_landmarks_filtered):
    def calculate_angle(a, b, c):
        """
        Calculate the angle between three points.
        a, b, c are tuples representing the (x, y) coordinates of the points.
        """
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    # Define the joints to calculate angles for
    joint_pairs = [
        ('left_shoulder', 'left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow', 'right_wrist'),
        ('left_hip', 'left_knee', 'left_ankle'),
        ('right_hip', 'right_knee', 'right_ankle'),
        ('left_shoulder', 'left_hip', 'left_knee'),
        ('right_shoulder', 'right_hip', 'right_knee')
    ]

    # Initialize a list to store the joint angles
    joint_angles_list = []

    # Iterate over each frame
    for index, row in df_landmarks_filtered.iterrows():
        frame_angles = {'frame': row['frame']}
        
        # Calculate the angles for each joint pair
        for joint_pair in joint_pairs:
            joint1, joint2, joint3 = joint_pair
            point1 = (row[f'{joint1}_x'], row[f'{joint1}_y'])
            point2 = (row[f'{joint2}_x'], row[f'{joint2}_y'])
            point3 = (row[f'{joint3}_x'], row[f'{joint3}_y'])
            angle = calculate_angle(point1, point2, point3)
            frame_angles[f'{joint2}_angle'] = angle
        
        joint_angles_list.append(frame_angles)

    # Convert the list of joint angles to a DataFrame
    df_joint_angles = pd.DataFrame(joint_angles_list)

    return df_joint_angles

def sync_signals2(signal_a, signal_b):
    # Ensure both signals are numpy arrays
    signal_a = signal_a.values if isinstance(signal_a, pd.Series) else np.array(signal_a)
    signal_b = signal_b.values if isinstance(signal_b, pd.Series) else np.array(signal_b)
    
    # Clean the signals by replacing NaN and inf values with zero
    signal_a = np.nan_to_num(signal_a, nan=0.0, posinf=0.0, neginf=0.0)
    signal_b = np.nan_to_num(signal_b, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Determine which signal is longer
    if len(signal_a) > len(signal_b):
        longer_signal = signal_a
        shorter_signal = signal_b
    else:
        longer_signal = signal_b
        shorter_signal = signal_a
    
    # Compute cross-correlation
    correlation = correlate(longer_signal, shorter_signal, mode='full', method='direct')
    lag = np.argmax(correlation) - (len(shorter_signal) - 1)
    
    # Shift the shorter signal to best match the longer signal
    if lag > 0:
        synchronized_shorter_signal = np.pad(shorter_signal, (lag, 0), 'edge')[:len(longer_signal)]
    else:
        synchronized_shorter_signal = np.pad(shorter_signal[-lag:], (0, -lag), 'edge')

    # Ensure the synchronized shorter signal has the same length as the longer signal
    if len(synchronized_shorter_signal) < len(longer_signal):
        synchronized_shorter_signal = np.pad(synchronized_shorter_signal, (0, len(longer_signal) - len(synchronized_shorter_signal)), 'edge')

    return longer_signal, synchronized_shorter_signal, lag

def replace_non_finite_valuesDF(df):
    """
    Replace NaNs, -inf, and inf values in a DataFrame with column-wise mean or zero if the column is entirely non-finite.
    """
    for column in df.columns:
        if df[column].dtype.kind in 'biufc':  # Check if the column is of a numeric type
            finite_values = df[column][np.isfinite(df[column])]
            if not finite_values.empty:
                mean_value = finite_values.mean()
                df[column] = df[column].replace([np.inf, -np.inf, np.nan], mean_value)
            else:
                df[column] = df[column].replace([np.inf, -np.inf, np.nan], 0)
    return df

def sync_signals22(signal_a, signal_b):
    # Ensure both signals are numpy arrays
    signal_a = np.array(signal_a)
    signal_b = np.array(signal_b)
    
    # Determine which signal is longer
    if len(signal_a) > len(signal_b):
        longer_signal = signal_a
        shorter_signal = signal_b
    else:
        longer_signal = signal_b
        shorter_signal = signal_a
    
    # Compute cross-correlation
    correlation = np.correlate(longer_signal, shorter_signal, mode='full')
    lag = np.argmax(correlation) - (len(shorter_signal) - 1)
    
    # Sync the signals
    if lag > 0:
        synced_longer_signal = longer_signal[lag:]
        synced_shorter_signal = np.pad(shorter_signal, (0, len(synced_longer_signal) - len(shorter_signal)), 'edge')
    else:
        synced_longer_signal = longer_signal[:lag]
        synced_shorter_signal = np.pad(shorter_signal, (abs(lag), 0), 'edge')
    
    return synced_longer_signal, synced_shorter_signal, lag

def pad_df(df, signal_length, lag, cut_index):
    # Ensure df is a DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # Get the nearest values at lag and cut_index
    start_value = df.iloc[0]
    end_value = df.iloc[-1]
    
    # Create padding for the beginning and end
    start_padding = pd.DataFrame([start_value] * lag, columns=df.columns)
    end_padding = pd.DataFrame([end_value] * (signal_length - cut_index), columns=df.columns)
    
    # Concatenate the padding with the original df
    padded_df = pd.concat([start_padding, df, end_padding], ignore_index=True)
    
    return padded_df

def downsample_df(df, original_freq, target_freq):
    downsample_factor = int(original_freq / target_freq)
    downsampled_df = df.iloc[::downsample_factor].reset_index(drop=True)
    return downsampled_df

def sync_signals(signal_a, signal_b):
    # Ensure signal_a is the shorter signal
    if len(signal_a) > len(signal_b):
        signal_a, signal_b = signal_b, signal_a

    # Compute cross-correlation
    correlation = np.correlate(signal_b, signal_a, mode='full')
    lag = np.argmax(correlation) - (len(signal_a) - 1)

    # Shift the shorter signal to align with the longer signal
    if lag > 0:
        synced_signal_a = np.pad(signal_a, (lag, 0), 'constant', constant_values=(0, 0))[:len(signal_b)]
        synced_signal_b = signal_b
    else:
        synced_signal_a = signal_a[:len(signal_b) + lag]
        synced_signal_b = signal_b[-lag:]

    # Ensure the lengths of the synced signals are the same
    min_length = min(len(synced_signal_a), len(synced_signal_b))
    synced_signal_a = synced_signal_a[:min_length]
    synced_signal_b = synced_signal_b[:min_length]

    # Cut signal_b at the end of signal_a
    cut_index = len(synced_signal_a)
    synced_signal_b = synced_signal_b[:cut_index]

    return synced_signal_a, synced_signal_b, lag, cut_index

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

def upsample_signal(data, current_fps, target_fps):
    # Calculate the time intervals for the current and target sampling frequencies
    current_time = np.arange(0, len(data)) / current_fps
    target_time = np.arange(0, len(data) * target_fps / current_fps) / target_fps

    if isinstance(data, pd.Series):
        # Interpolate the Series to the target time intervals
        upsampled_data = np.interp(target_time, current_time, data)
        return pd.Series(upsampled_data, name=data.name)
    elif isinstance(data, pd.DataFrame):
        # Initialize a DataFrame to store the upsampled data
        df_upsampled = pd.DataFrame()

        # Interpolate each column to the target time intervals
        for column in data.columns:
            df_upsampled[column] = np.interp(target_time, current_time, data[column])

        return df_upsampled
    else:
        raise ValueError("Input data must be a pandas DataFrame or Series")