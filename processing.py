# processing.py

import cv2
import tempfile
import mediapipe as mp
#from Sports2D import Sports2D
import subprocess #har jeg?
import toml  # pip install toml if needed
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from designidea import plot_hand_drawn_style
import os
import json
import requests
import glob

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


def process_video(uploaded_video, height, show_pose=True, progress_callback=None, frame_callback=None):
    """
    Process the uploaded video file using Sports2D in a separate environment.
    
    :param uploaded_video: The uploaded video file object
    :param height: Height of the person in meters for scaling
    :param show_pose: If True, shows pose visualization. If False, disables pose visualization
    :param progress_callback: Optional callback function to report progress
    :param frame_callback: Optional callback function for frame processing
    :return: Tuple of (results, error_message)
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
        tmp_video.write(uploaded_video.read())
        tmp_video.flush()
        video_path = tmp_video.name

        # Get the directory where the video is saved and the Sports2D output directory
        output_dir = os.path.dirname(video_path)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        sports2d_dir = os.path.join(output_dir, f"{base_name}_Sports2D")

        script_content = '''
import subprocess
import sys
import os
import glob
import shutil

try:
    video_path = r'{}'
    height = {}
    output_dir = r'{}'
    sports2d_dir = r'{}'
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Run Sports2D using command line
    command = ["sports2d", 
              "--video_input", video_path,
              "--to_meters", "True",
              "--do_ik", "True",
              "--save_angles", "True",
              "--save_pose", "True",
              "--show_graphs", "False",
              "-r", output_dir,
              "--multiperson", "False",
              "--save_vid", "True",
              "--save_img", "False"]
    
    if not {}:
        command.extend(["--show_realtime_results", "False"])
    
    if height:
        command.extend(["--px_to_m_person_height", str(height)])
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Try to find and read all relevant output files
    results_data = {{}}  # Double braces to escape
    results_data["stdout"] = result.stdout
    results_data["stderr"] = result.stderr
    
    # Look for TRC files (pose data) in Sports2D directory
    trc_files = glob.glob(os.path.join(sports2d_dir, "*.trc"))
    for trc_file in trc_files:
        if os.path.exists(trc_file):
            with open(trc_file, 'r') as f:
                results_data["trc_" + os.path.basename(trc_file)] = f.read()
    
    # Look for MOT files (angles data) in Sports2D directory
    mot_files = glob.glob(os.path.join(sports2d_dir, "*.mot"))
    for mot_file in mot_files:
        if os.path.exists(mot_file):
            with open(mot_file, 'r') as f:
                results_data["mot_" + os.path.basename(mot_file)] = f.read()
    
    # Look for processed video in Sports2D directory
    video_files = glob.glob(os.path.join(sports2d_dir, "*.mp4"))
    for video_file in video_files:
        if os.path.exists(video_file):
            with open(video_file, 'rb') as f:
                results_data["video_" + os.path.basename(video_file)] = f.read().hex()
    
    # Look for processed images in Sports2D directory
    img_dir = os.path.join(sports2d_dir, base_name + "_Sports2D_img")
    if os.path.exists(img_dir):
        results_data["img_dir_exists"] = True
        # Optional: read some sample images
        img_files = glob.glob(os.path.join(img_dir, "*.png"))[:5]  # First 5 images
        for img_file in img_files:
            with open(img_file, 'rb') as f:
                results_data["img_" + os.path.basename(img_file)] = f.read().hex()
    
    print("RESULTS_START")
    print(str(results_data))
    print("RESULTS_END")
    
    print("ERROR_START")
    print(result.stderr)
    print("ERROR_END")
    
except Exception as e:
    print("ERROR_START")
    print(str(e))
    print("ERROR_END")
'''.format(video_path, height, output_dir, sports2d_dir, show_pose)

        try:
            # Create temporary script file
            with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as script_file:
                script_file.write(script_content)
                script_path = script_file.name

            # Use conda run to execute in sports2d environment
            result = subprocess.run(
                ["conda", "run", "-n", "sports2d", "python", script_path],
                capture_output=True,
                text=True
            )

            # Parse the output
            output = result.stdout
            try:
                results = output[output.index("RESULTS_START")+13:output.index("RESULTS_END")].strip()
                error = output[output.index("ERROR_START")+11:output.index("ERROR_END")].strip()
                
                if error and not results:
                    return None, error
                return results, error if error else None
            except ValueError:
                return None, f"Error parsing results: {output}"

        except Exception as e:
            return None, f"Error running Sports2D script: {str(e)}"
        finally:
            # Clean up temporary files
            try:
                os.unlink(video_path)
                os.unlink(script_path)
            except Exception:
                pass