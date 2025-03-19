import pandas as pd
import numpy as np

def read_jump_excel(file_path):
    """
    Reads an Excel file into a DataFrame and calculates the sampling frequency.
    
    Parameters:
    file_path (str): The path to the Excel file.
    
    Returns:
    tuple: A tuple containing the DataFrame, the sampling frequency, the length of the Time vector, and the total time.
    """
    df = pd.read_excel(file_path)
    
    # Calculate the sampling frequency using the Time column
    if 'Time' in df.columns:
        time_diffs = df['Time'].diff().dropna()
        sampling_frequency = 1 / time_diffs.mean()
        time_length = len(df['Time'])
        total_time = df['Time'].iloc[-1]
    else:
        raise ValueError("The DataFrame does not contain a 'Time' column.")
    
    return df, sampling_frequency, time_length, total_time

def calculate_com_position(df, sampling_frequency):
    """
    Calculates the center of mass (COM) position from the vertical force plate data.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the force plate data.
    sampling_frequency (float): The sampling frequency of the data.
    
    Returns:
    pd.Series: The COM position over time.
    """
    # Calculate the body mass from the first frames of the force data (assuming the participant is standing still)
    body_mass = df['Force'].iloc[:int(sampling_frequency)].mean() / 9.81  # Convert N to kg
    
    # Calculate the acceleration (a = F/m)
    df['Acceleration (m/s^2)'] = df['Force'] / body_mass - 9.81  # Subtract gravity to get net acceleration
    
    # Integrate acceleration to get velocity
    df['Velocity (m/s)'] = df['Acceleration (m/s^2)'].cumsum() / sampling_frequency
    
    # Integrate velocity to get displacement (COM position)
    df['COM Position (m)'] = df['Velocity (m/s)'].cumsum() / sampling_frequency
    
    return df['COM Position (m)']

def calculate_jump_height(force_data, sampling_frequency):
    """
    Calculate jump height from force plate data using the most accurate method.

    Parameters:
    - force_data: numpy array of vertical ground reaction forces (N)
    - sampling_frequency: int, data sampling rate (Hz)

    Returns:
    - jump_height: float, jump height in meters
    """

    # Constants
    g = 9.81  # Gravity (m/s²)
    
    # Estimate body weight from the first 5 frames
    body_weight = np.mean(force_data[:500])  # in Newtons
    mass = body_weight / g  # Convert weight to mass (kg)

    # Compute net force (F_net = F - mg)
    net_force = force_data - body_weight

    # Compute acceleration (a = F_net / m)
    acceleration = net_force / mass

    # Compute velocity by integrating acceleration using the trapezoidal rule
    time_step = 1 / sampling_frequency
    velocity = np.cumsum(acceleration) * time_step

    # Find takeoff moment (first instance where force reaches zero)
    takeoff_index = np.where(force_data <= 1e-3)[0][0]  # Small threshold to avoid noise
    takeoff_velocity = velocity[takeoff_index]  # Velocity at takeoff

    # Compute jump height using kinematic equation: h = (v^2) / (2g)
    jump_height = (takeoff_velocity ** 2) / (2 * g)

    return jump_height

