import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.signal import butter, filtfilt

import numpy as np



import pandas as pd
import numpy as np

import numpy as np

def filter_landmarks_by_confidence(pose_data_df, confidence_threshold=0.6):
    """
    Filters landmarks in 'raw' and 'scaled_raw' based on confidence scores in 'scores'.
    Sets coordinates (_x and _y) to NaN if the confidence score is below the threshold.

    Parameters:
        pose_data_df (dict): Dictionary containing pose data for multiple persons.
        confidence_threshold (float): Confidence threshold for filtering landmarks.

    Returns:
        dict: Updated pose_data_df with filtered 'raw' and 'scaled_raw' data.
    """
    # Iterate over each person in pose_data_df
    for person_id, data in pose_data_df.items():
        # Check if 'scores', 'raw', and 'scaled_raw' exist for the person
        if 'scores' in data and 'raw' in data and 'scaled_raw' in data:
            scores_df = data['scores']
            raw_df = data['raw']
            scaled_raw_df = data['scaled_raw']

            # Ensure the DataFrames are valid
            if not scores_df.empty and not raw_df.empty and not scaled_raw_df.empty:
                # Iterate over each landmark in the scores DataFrame
                for landmark in scores_df.columns:
                    # Skip the 'frames' column
                    if landmark == 'frames':
                        continue

                    # Get the confidence scores for the landmark
                    confidence_scores = scores_df[landmark]

                    # Identify frames where the confidence score is below the threshold
                    low_confidence_frames = confidence_scores < confidence_threshold

                    # Set the corresponding coordinates in 'raw' and 'scaled_raw' to NaN
                    raw_df.loc[low_confidence_frames, f"{landmark}_x"] = np.nan
                    raw_df.loc[low_confidence_frames, f"{landmark}_y"] = np.nan
                    scaled_raw_df.loc[low_confidence_frames, f"{landmark}_x"] = np.nan
                    scaled_raw_df.loc[low_confidence_frames, f"{landmark}_y"] = np.nan

                # Ensure the 'frames' column remains unchanged
                raw_df['frames'] = raw_df['frames']
                scaled_raw_df['frames'] = scaled_raw_df['frames']
            else:
                print(f"Warning: Missing or empty DataFrames for person {person_id}. Skipping.")
        else:
            print(f"Warning: Missing 'scores', 'raw', or 'scaled_raw' for person {person_id}. Skipping.")

    return pose_data_df

def process_scores(pose_data, pose_data_df, total_frames_video):
    """
    Processes scores for each person in pose_data and adds them to pose_data_df.

    Parameters:
        pose_data (dict): Dictionary containing pose data for multiple persons.
        pose_data_df (dict): Dictionary to store processed data for each person.
        total_frames_video (int): Total number of frames in the video.

    Returns:
        dict: Updated pose_data_df with scores added for each person.
    """
    # Define landmark names
    landmark_names = [
        "Nose", "LEye", "REye", "LEar", "REar", "LShoulder", "RShoulder",
        "LElbow", "RElbow", "LWrist", "RWrist", "LHip", "RHip", "LKnee",
        "RKnee", "LAnkle", "RAnkle", "Head", "Neck", "Hip", "LBigToe",
        "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel"
    ]

    # Iterate over each person in pose_data
    for person_id, person_data in pose_data.items():
        # Initialize a DataFrame with NaN values for all frames
        scores_df = pd.DataFrame(np.nan, index=range(total_frames_video), columns=landmark_names)

        # Get the frames where the person was tracked
        tracked_frames = person_data.get('frames', [])

        # Get the scores for the tracked frames
        scores = person_data.get('scores', [])

        # Fill the DataFrame with scores for the tracked frames
        for i, frame_idx in enumerate(tracked_frames):
            if frame_idx < total_frames_video:  # Ensure the frame index is within bounds
                scores_df.iloc[frame_idx] = scores[i]

        # Add the scores DataFrame to pose_data_df
        pose_data_df[person_id]['scores'] = scores_df

    return pose_data_df

def interpolate_pose_data(pose_data_df):
    """
    Interpolates NaN values in pose_data_df[Id]['raw'] and pose_data_df[Id]['scaled_raw']
    using cubic spline interpolation and adds new entries for 'filled' and 'scaled_filled'.

    Parameters:
        pose_data_df (dict): Dictionary containing pose data for multiple user IDs.

    Returns:
        dict: Updated pose_data_df with interpolated data added as 'filled' and 'scaled_filled'.
    """
    # Iterate over each user ID in pose_data_df
    for user_id, data in pose_data_df.items():
        # Interpolate 'raw' data
        raw_df = data['raw'].copy()
        raw_filled = raw_df.copy()
        for col in raw_df.columns:
            if col != 'frames':  # Skip the 'frames' column
                # Interpolate using cubic spline
                raw_filled[col] = raw_df[col].interpolate(method='spline', order=3)
                # Handle boundary NaNs with forward-fill and backward-fill
                raw_filled[col] = raw_filled[col].fillna(method='ffill').fillna(method='bfill')

        # Interpolate 'scaled_raw' data
        scaled_raw_df = data['scaled_raw'].copy()
        scaled_filled = scaled_raw_df.copy()
        for col in scaled_raw_df.columns:
            if col != 'frames':  # Skip the 'frames' column
                # Interpolate using cubic spline
                scaled_filled[col] = scaled_raw_df[col].interpolate(method='spline', order=3)
                # Handle boundary NaNs with forward-fill and backward-fill
                scaled_filled[col] = scaled_filled[col].fillna(method='ffill').fillna(method='bfill')

        # Add the interpolated DataFrames to pose_data_df
        pose_data_df[user_id]['filled'] = raw_filled
        pose_data_df[user_id]['scaled_filled'] = scaled_filled

    return pose_data_df



def filter_pose_data(pose_data_df, parametername, fps_video, cut_off_frequency):
    """
    Applies a low-pass Butterworth filter to the specified parameter for all subjects in pose_data_df.

    Parameters:
        pose_data_df (dict): Dictionary containing pose data for multiple subjects.
        parametername (str): The key in pose_data_df to filter (e.g., 'Angles').
        fps_video (float): Frames per second of the video.
        cut_off_frequency (float): Cut-off frequency for the low-pass filter.

    Returns:
        dict: Updated pose_data_df with filtered data added under parametername+'_filt' for each subject.
    """
    # Design a 4th-order Butterworth low-pass filter
    nyquist = 0.5 * fps_video
    normal_cutoff = cut_off_frequency / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)

    # Loop through all subjects in pose_data_df
    for person_id, data in pose_data_df.items():
        # Check if the parameter exists for the current person
        if parametername in data:
            # Get the DataFrame for the parameter
            parameter_df = data[parametername]

            # Ensure the DataFrame is valid
            if isinstance(parameter_df, pd.DataFrame) and not parameter_df.empty:
                # Create a new DataFrame to store filtered data
                filtered_df = pd.DataFrame()

                # Copy the 'frames' column directly without filtering
                if 'frame' in parameter_df.columns:
                    filtered_df['frame'] = parameter_df['frame']

                # Apply the filter to each column except 'frames'
                for col in parameter_df.columns:
                    if col != 'frame':  # Skip the 'frames' column
                        filtered_df[col] = filtfilt(b, a, parameter_df[col])

                # Store the filtered DataFrame in pose_data_df
                pose_data_df[person_id][f"{parametername}_filt"] = filtered_df
            else:
                print(f"Warning: {parametername} for person {person_id} is not a valid DataFrame or is empty.")
        else:
            print(f"Warning: {parametername} not found for person {person_id}.")

    return pose_data_df


def process_pose_data(total_frames_video, pose_data, pose_data_m):
    """
    Processes pose data for all user IDs and creates labeled DataFrames for each user.

    Parameters:
        total_frames_video (int): Total number of frames in the video.
        pose_data (dict): Pose data containing landmarks for multiple persons.
        pose_data_m (dict): Pose data in meters for multiple persons.

    Returns:
        dict: A dictionary where keys are user IDs, and values are labeled DataFrames for each user.
    """
    # Define landmark names
    landmark_names = [
        "Nose", "LEye", "REye", "LEar", "REar", "LShoulder", "RShoulder",
        "LElbow", "RElbow", "LWrist", "RWrist", "LHip", "RHip", "LKnee",
        "RKnee", "LAnkle", "RAnkle", "Head", "Neck", "Hip", "LBigToe",
        "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel"
    ]

    # Generate column headers for x and y coordinates
    headers = [f"{name}_{axis}" for name in landmark_names for axis in ("x", "y")]

    # Initialize the output dictionary
    pose_data_df = {}

    # Process each user ID in pose_data
    for user_id in pose_data.keys():
        # Extract raw pose data
        raw_coords = pose_data[user_id]['coords']
        raw_frames = pose_data[user_id]['frames']

        # Extract scaled pose data (in meters)
        scaled_coords = pose_data_m[user_id]['coords']
        scaled_frames = pose_data_m[user_id]['frames']

        # Initialize DataFrames with NaNs for all frames
        raw_df = pd.DataFrame(np.nan, index=range(total_frames_video), columns=headers)
        scaled_df = pd.DataFrame(np.nan, index=range(total_frames_video), columns=headers)

        # Fill the DataFrames with the tracked data
        for i, frame_idx in enumerate(raw_frames):
            raw_df.loc[frame_idx] = np.array(raw_coords[i]).flatten()

        for i, frame_idx in enumerate(scaled_frames):
            scaled_df.loc[frame_idx] = np.array(scaled_coords[i]).flatten()

        # Add the 'frames' column to both DataFrames
        raw_df.insert(0, "frames", range(total_frames_video))
        scaled_df.insert(0, "frames", range(total_frames_video))

        # Store the DataFrames in the output dictionary
        pose_data_df[user_id] = {
            'raw': raw_df,  # Raw pose data with NaNs for untracked frames
            'scaled_raw': scaled_df  # Scaled pose data with NaNs for untracked frames
        }

    return pose_data_df


def calculate_joint_angles(pose_data_df):
    """
    Calculates joint angles for each person in pose_data_df and adds them to pose_data_df['Id']['Angles'].

    Parameters:
        pose_data_df (dict): Dictionary containing pose data for multiple user IDs.

    Returns:
        dict: Updated pose_data_df with joint angles added under 'Angles' for each person.
    """
    def calculate_angle(a, b, c):
        """
        Calculate the angle between three points.
        a, b, c are tuples representing the (x, y) coordinates of the points.
        """
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip to handle numerical issues
        return np.degrees(angle)

    # Define the joints to calculate angles for
    joint_pairs = {
        'LeftElbow': ('LShoulder', 'LElbow', 'LWrist'),
        'RightElbow': ('RShoulder', 'RElbow', 'RWrist'),
        'LeftKnee': ('LHip', 'LKnee', 'LAnkle'),
        'RightKnee': ('RHip', 'RKnee', 'RAnkle'),
        'LeftShoulder': ('Neck', 'LShoulder', 'LElbow'),
        'RightShoulder': ('Neck', 'RShoulder', 'RElbow'),
        'LeftHip': ('LShoulder', 'LHip', 'LKnee'),
        'RightHip': ('RShoulder', 'RHip', 'RKnee')
    }

    # Iterate over each person in pose_data_df
    for person_id, data in pose_data_df.items():
        # Get the filled DataFrame for the person
        filled_df = data['filled']
        if not isinstance(filled_df, pd.DataFrame):
            print(f"Person {person_id}: 'filled' is not a DataFrame. Skipping.")
            pose_data_df[person_id]['Angles'] = pd.DataFrame()  # Add an empty DataFrame for 'Angles'
            continue

        if filled_df.empty:
            print(f"Person {person_id}: 'filled' DataFrame is empty. Skipping.")
            pose_data_df[person_id]['Angles'] = pd.DataFrame()  # Add an empty DataFrame for 'Angles'
            continue

        angles_list = []

        # Iterate over each frame
        for _, row in filled_df.iterrows():
            frame_angles = {'frame': row['frames']}  # Start with the frame number

            # Calculate angles for each joint
            for joint_name, (p1, p2, p3) in joint_pairs.items():
                try:
                    # Get the coordinates for the three points
                    a = (row[f"{p1}_x"], row[f"{p1}_y"])
                    b = (row[f"{p2}_x"], row[f"{p2}_y"])
                    c = (row[f"{p3}_x"], row[f"{p3}_y"])

                    # Ensure all points are valid (not NaN)
                    if not any(np.isnan([a[0], a[1], b[0], b[1], c[0], c[1]])):
                        angle = calculate_angle(a, b, c)
                        frame_angles[joint_name] = angle
                    else:
                        frame_angles[joint_name] = np.nan  # Set NaN if any point is invalid
                except KeyError:
                    frame_angles[joint_name] = np.nan  # Handle missing landmarks gracefully

            angles_list.append(frame_angles)

        # Convert the list of angles to a DataFrame
        angles_df = pd.DataFrame(angles_list)

        # Add the angles DataFrame to pose_data_df
        pose_data_df[person_id]['Angles'] = angles_df

    return pose_data_df

def find_backwards_turning_point(signal, threshold):
    """
    Find the index in a signal where:
    1. It first crosses the given threshold.
    2. Then go backwards to find:
        a. A zero crossing (signal changes sign).
        b. Or a point where the signal starts increasing again.
    
    Parameters:
        signal (pd.Series or np.ndarray): 1D signal
        threshold (float): Threshold to detect crossing
    
    Returns:
        int or None: Index of the detected turning point
    """
    x = np.asarray(signal)

    # Step 1: Find first crossing of the threshold
    cross_indices = np.where(x > threshold)[0]
    if len(cross_indices) == 0 or cross_indices[0] == 0:
        return None

    cross_idx = cross_indices[0]

    # Step 2: Check for zero crossing going backward
    for i in range(cross_idx - 1, 0, -1):
        if x[i] * x[i + 1] <= 0:
            return i

    # Step 3: Check for turning point (signal starts increasing again)
    dx = np.diff(x)
    for i in range(cross_idx - 1, 1, -1):
        if dx[i] < 0 and dx[i - 1] >= 0:
            return i

    return 0  # Fallback

def sync_signals_by_transition(signal_a, signal_b, threshold=1e-6):
    """
    Aligns signal_b to signal_a using the first zero-to-positive transition.
    
    Parameters:
        signal_a (np.ndarray): Reference signal.
        signal_b (np.ndarray): Signal to align.
        threshold (float): Small threshold to detect the zero crossing with tolerance.

    Returns:
        int: Number of samples signal_b needs to be shifted to align with signal_a.
    """
    signal_a = np.asarray(signal_a)
    signal_b = np.asarray(signal_b)

    # Find first zero-to-positive transition in signal_a
    idx_a = np.where((signal_a[:-1] <= threshold) & (signal_a[1:] > threshold))[0]
    idx_a = idx_a[0] + 1 if len(idx_a) > 0 else None

    # Find first zero-to-positive transition in signal_b
    idx_b = np.where((signal_b[:-1] <= threshold) & (signal_b[1:] > threshold))[0]
    idx_b = idx_b[0] + 1 if len(idx_b) > 0 else None

    if idx_a is None or idx_b is None:
        return None  # Handle missing transitions gracefully

    # Calculate shift
    shift = idx_a - idx_b
    return shift


def find_first_peak(data):
    """
    Function to find the first peak in a vector.
    A peak is defined as a point where the signal changes from rising to falling.

    Parameters:
        data (list or np.ndarray): Input vector.

    Returns:
        tuple: (peak_index, peak_value)
            - peak_index (int): Index of the first peak.
            - peak_value (float): Value of the first peak.
    """
    # Ensure the input is a NumPy array
    data = np.array(data)

    # Calculate the difference between consecutive elements
    diff_data = np.diff(data)

    # Find where the signal changes from rising to falling
    for i in range(len(diff_data) - 1):
        if diff_data[i] > 0 and diff_data[i + 1] < 0:
            peak_index = i + 1  # Peak occurs at the next point
            peak_value = data[peak_index]
            return peak_index, peak_value

    # If no peak is found, return None
    return None, None

def detect_first_foot_movement_auto(df, window_size=5, activity_multiplier=3):
    """
    Automatically detects which foot moves first based on sudden increase in x-coordinate activity.
    
    Parameters:
        df: DataFrame with columns 'RAnkle_x' and 'LAnkle_x'
        window_size: Rolling window size for std calculation
        activity_multiplier: How many times above the mean std counts as 'movement'
        
    Returns:
        first_mover: 'RAnkle', 'LAnkle', 'Both', or 'None'
        first_index: Index of the first detected movement
    """
    ran = df['RAnkle_x'].values
    lan = df['LAnkle_x'].values

    ran_series = pd.Series(ran)
    lan_series = pd.Series(lan)

    ran_std = ran_series.rolling(window=window_size).std().fillna(0)
    lan_std = lan_series.rolling(window=window_size).std().fillna(0)

    ran_activity = ran_std / ran_std.mean()
    lan_activity = lan_std / lan_std.mean()

    ran_move_idx = np.argmax(ran_activity > activity_multiplier)
    lan_move_idx = np.argmax(lan_activity > activity_multiplier)

    ran_detected = ran_activity[ran_move_idx] > activity_multiplier
    lan_detected = lan_activity[lan_move_idx] > activity_multiplier

    if not ran_detected and not lan_detected:
        return 'None', -1
    elif ran_detected and (not lan_detected or ran_move_idx < lan_move_idx):
        return 'RAnkle', ran_move_idx
    elif lan_detected and (not ran_detected or lan_move_idx < ran_move_idx):
        return 'LAnkle', lan_move_idx
    else:
        return 'Both', ran_move_idx

# Example usage:
# foot, index = detect_first_foot_movement_auto(df_landmarks_filtered_filt)
# print(f"{foot} moved first at index {index}")

def create_labeled_df(filled_frames, total_frames_video):
    # landmark_names = [
    #     "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    #     "left_shoulder", "left_elbow", "left_wrist", "mid_hip",
    #     "right_hip", "right_knee", "right_ankle",
    #     "left_hip", "left_knee", "left_ankle",
    #     "right_eye", "left_eye", "right_ear", "left_ear",
    #     "left_big_toe", "left_small_toe", "left_heel",
    #     "right_big_toe", "right_small_toe", "right_heel", "pelvis"
    # ]
    landmark_names = [
    "Nose",        # 0
    "LEye",        # 1
    "REye",        # 2
    "LEar",        # 3
    "REar",        # 4
    "LShoulder",   # 5
    "RShoulder",   # 6
    "LElbow",      # 7
    "RElbow",      # 8
    "LWrist",      # 9
    "RWrist",      #10
    "LHip",        #11
    "RHip",        #12
    "LKnee",       #13
    "RKnee",       #14
    "LAnkle",      #15
    "RAnkle",      #16
    "Head",        #17
    "Neck",        #18
    "Hip",         #19 (center/pelvis)
    "LBigToe",     #20
    "RBigToe",     #21
    "LSmallToe",   #22
    "RSmallToe",   #23
    "LHeel",       #24
    "RHeel"        #25
]

    # Build headers like nose_x, nose_y, neck_x, neck_y, ...
    headers = [f"{name}_{axis}" for name in landmark_names for axis in ("x", "y")]

    # Create the DataFrame
    df = pd.DataFrame(filled_frames, columns=headers)

    # Add frame numbers
    df.insert(0, "frame", range(total_frames_video))

    return df

def fill_missing_frames(flattened_frames_df, tracked_frame_indices, total_frames_video):
    # Convert DataFrame to NumPy array
    flattened_frames = flattened_frames_df.values

    # Create output array initialized with NaNs
    filled = np.full((total_frames_video, flattened_frames.shape[1]), np.nan)

    # Precompute mapping from tracked_frame_indices to rows
    index_map = dict(zip(tracked_frame_indices, range(len(tracked_frame_indices))))

    for i in range(total_frames_video):
        if i in index_map:
            filled[i] = flattened_frames[index_map[i]]
        # Else block is no longer needed as NaNs are already initialized

    return filled

# def fill_missing_frames(flattened_frames_df, tracked_frame_indices, total_frames_video):
#     # Convert DataFrame to NumPy array
#     flattened_frames = flattened_frames_df.values

#     # Create output array
#     filled = np.zeros((total_frames_video, flattened_frames.shape[1]))

#     # Precompute mapping from tracked_frame_indices to rows
#     index_map = dict(zip(tracked_frame_indices, range(len(tracked_frame_indices))))

#     for i in range(total_frames_video):
#         if i in index_map:
#             filled[i] = flattened_frames[index_map[i]]
#         #else:
#             # Find the nearest tracked frame
#             #nearest = min(tracked_frame_indices, key=lambda x: abs(x - i))
#             #filled[i] = flattened_frames[index_map[nearest]]

#     return filled

# def fill_missing_frames(flattened_frames, tracked_frame_indices, total_frames_video):
#     filled = np.zeros((total_frames_video, flattened_frames.shape[1]))
#     for i in range(total_frames_video):
#         nearest = min(tracked_frame_indices, key=lambda x: abs(x - i))
#         filled[i] = flattened_frames[tracked_frame_indices.index(nearest)]
#     return filled

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

# def calculate_joint_angles(df_landmarks_filtered):
#     def calculate_angle(a, b, c):
#         """
#         Calculate the angle between three points.
#         a, b, c are tuples representing the (x, y) coordinates of the points.
#         """
#         ba = np.array(a) - np.array(b)
#         bc = np.array(c) - np.array(b)
#         cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#         angle = np.arccos(cosine_angle)
#         return np.degrees(angle)

#     # Define the joints to calculate angles for
#     joint_pairs = [
#         ('left_shoulder', 'left_elbow', 'left_wrist'),
#         ('right_shoulder', 'right_elbow', 'right_wrist'),
#         ('left_hip', 'left_knee', 'left_ankle'),
#         ('right_hip', 'right_knee', 'right_ankle'),
#         ('left_shoulder', 'left_hip', 'left_knee'),
#         ('right_shoulder', 'right_hip', 'right_knee')
#     ]

#     # Initialize a list to store the joint angles
#     joint_angles_list = []

#     # Iterate over each frame
#     for index, row in df_landmarks_filtered.iterrows():
#         frame_angles = {'frame': row['frame']}
        
#         # Calculate the angles for each joint pair
#         for joint_pair in joint_pairs:
#             joint1, joint2, joint3 = joint_pair
#             point1 = (row[f'{joint1}_x'], row[f'{joint1}_y'])
#             point2 = (row[f'{joint2}_x'], row[f'{joint2}_y'])
#             point3 = (row[f'{joint3}_x'], row[f'{joint3}_y'])
#             angle = calculate_angle(point1, point2, point3)
#             frame_angles[f'{joint2}_angle'] = angle
        
#         joint_angles_list.append(frame_angles)

#     # Convert the list of joint angles to a DataFrame
#     df_joint_angles = pd.DataFrame(joint_angles_list)

#     return df_joint_angles

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