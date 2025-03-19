import pandas as pd

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

# Example usage
file_path = r"C:\Users\adpatrick\OneDrive - nih.no\Desktop\WebApp1080sync\WebAppv0.1\Jump1.xlsx"
df, sampling_frequency, time_length, total_time = read_jump_excel(file_path)
print(df.head())
print(f"Sampling Frequency: {sampling_frequency} Hz")
print(f"Length of Time vector: {time_length}")
print(f"Total Time: {total_time} seconds")