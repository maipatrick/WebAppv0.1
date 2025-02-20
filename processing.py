# processing.py

import cv2
import tempfile
#import mediapipe as mp
#from Sports2D import Sports2D
import subprocess #har jeg?
import toml  # pip install toml if needed
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from designidea import plot_hand_drawn_style

def process_excel(uploaded_excel, progress_callback=None):
    """
    Process the uploaded Excel file.

    :param uploaded_excel: The uploaded Excel file object.
    :param progress_callback: A callback function accepting a float (0.0 to 1.0) to update progress.
    :return: A tuple (df, fig) where df is the DataFrame from the Excel file and fig is a matplotlib figure
             generated using a hand-drawn style plot if "Speed [m/s]" is found, otherwise None.
    """
    # Read the Excel file into a DataFrame
    df = pd.read_excel(uploaded_excel, header=0)
    
    # Simulate processing progress
    if progress_callback:
        for i in range(100):
            time.sleep(0.0005)  # Simulated delay for processing
            progress_callback((i + 1) / 100)
    
    fig = None
    # # If the "Speed [m/s]" column exists, generate a plot
    # if "Speed [m/s]" in df.columns:
    #     y = df["Speed [m/s]"]
    #     x = np.arange(len(y))
    #     fig = plot_hand_drawn_style(x, y)
    
    return df, fig


def process_video(uploaded_video, height, progress_callback=None, frame_callback=None):
    """
    Process the uploaded video file using Sports2D pose estimation.
    This function creates a temporary configuration file based on the base TOML,
    overrides the video input with the user's uploaded file, and runs Sports2D
    as a separate subprocess.

    :param uploaded_video: The uploaded video file object.
    :param progress_callback: Not used (placeholder for compatibility).
    :param frame_callback: Not used (placeholder for compatibility).
    :return: A tuple (stdout, stderr) from the Sports2D process.
    """
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_video.read())
        video_path = tfile.name

    # Load the base configuration from the provided Sports2D TOML file
    base_config_file = "Sports2D Config Demo.toml"
    with open(base_config_file, "r") as f:
        config = toml.load(f)

    # Override the video input parameter with the temporary video file path
    config["project"]["video_input"] = video_path

    # Optionally override other parameters, for example:
    # config["project"]["output"] = "output.mp4"
    config["project"]["px_to_m_person_height"] = height
    config["process"]["multiperson"] = False

    # Write the updated configuration to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".toml") as tmp:
        toml.dump(config, tmp)
        tmp_config_path = tmp.name

    # Build the command to run Sports2D processing.
    # In this example, we're calling the script 'run_pose2d.py' from Sports2D.
    # If Sports2D is installed in a separate environment, you might need to provide the full path to its Python interpreter.
    command = ["python", "run_pose2d.py", tmp_config_path]

    # Run Sports2D as a subprocess; capture stdout and stderr
    result = subprocess.run(command, capture_output=True, text=True)

    # Return the output for further processing or debugging
    return result.stdout, result.stderr
