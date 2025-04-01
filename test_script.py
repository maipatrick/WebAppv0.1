# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Load data
df_pose = pd.read_excel(r'C:\Users\adpatrick\OneDrive - nih.no\Desktop\WebApp1080sync\WebAppv0.1\df_velocity_m.xlsx')
df_signal_a = pd.read_excel(r'C:\Users\adpatrick\OneDrive - nih.no\Desktop\WebApp1080sync\WebAppv0.1\signal_a.xlsx')

# Find the turning point
idx = find_backwards_turning_point(df_pose['Hip_x'], 2)
print(f"Turning point index: {idx}")

# Pad df_signal_a['Speed [m/s]'] with 1000 zeros at the start and end
appended = pd.concat([pd.Series([0] * 100000), df_signal_a['Speed [m/s]'], pd.Series([0] * 100000)], ignore_index=True)

# Set all values to 0 from the start of df_pose['Hip_x'] to the turning point index
if idx is not None:
    df_pose.loc[:idx, 'Hip_x'] = 0

# Synchronize the signals
shift = sync_signals_by_transition(appended, df_pose['Hip_x'])
print(f"Calculated shift: {shift}")




# Keep only the part of appended from shift onward
# Keep only the part of appended from shift onward and reset the index
if shift is not None and shift > 0:
    appended = appended[shift:].reset_index(drop=True)  # Delete everything before shift and reset index
else:
    appended = appended.reset_index(drop=True)  # Reset index if no slicing is needed

# Plot df_pose['Hip_x'] and the modified appended signal
plt.figure(figsize=(10, 5))
plt.plot(df_pose['Hip_x'], label='Hip_x (Original)')
plt.plot(appended[:len(df_pose['Hip_x'])], label='Appended (Truncated)')
plt.legend()
plt.title('df_pose[\'Hip_x\'] and Appended (From Shift to End)')
plt.show()