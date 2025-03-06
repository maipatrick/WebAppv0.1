# TODO move to processing functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate



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