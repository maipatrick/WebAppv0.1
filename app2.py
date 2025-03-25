import streamlit as st
from video_fun_pm import process_video, filter_landmarks, calculate_com, process_and_overlay_videoStreamlit,process_and_overlay_videoStreamlit_force, process_and_overlay_videoStreamlit_None, process_video_blured, process_video_blured_new, filter_landmarks_new, pose_estimation_rmt_pose
from d1080_fun_pm import read_1080, filter_1080_data
from default_processing_pm import pad_df, sync_signals, upsample_signal, sync_signals22, downsample_df, replace_non_finite_valuesDF, calculate_joint_angles, pad_sync_signal
from dforce_fun_pm import read_jump_excel, calculate_com_position, calculate_jump_height
import pandas as pd
import tempfile
import os
import logging
import requests
import time
import numpy as np

# Configure logging
logging.basicConfig(filename='user_activity.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def get_user_country():
    try:
        ip_info = requests.get('https://ipinfo.io').json()
        return ip_info.get('country', 'Unknown')
    except Exception as e:
        return 'Unknown'

def log_activity(activity):
    country = get_user_country()
    logging.info(f"{activity} - Country: {country}")

# Set page configuration
st.set_page_config(
    page_title="Athlete Metrics",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Create tabs
tab1, tab2 = st.tabs(["Main", "Help"])

with tab1:
    st.title("Athlete MetrIcs")
    st.image("https://i.ibb.co/zVStNZg7/logo.png", width=200)
    # Initialize session state
    if 'processing_done' not in st.session_state:
        st.session_state.processing_done = False

    # Upload video file
    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"], accept_multiple_files=False, key="video", help="Supported formats: mp4, mov, avi")
    if video_file:
        log_activity(f"Raw video file uploaded: {video_file.name}")
        # checkbox to blur faces
        blur_facesuser = st.checkbox("Blur faces", value=True)
        option = st.selectbox(
        "Source of the data",
        ("1080", "Force plate Jumps", "None", "TBD"), index=2)

        # Dropdown to select the source of the data

        if option == "None":
            st.write("This will apply pose estimation only!")
        elif option == "TBD":
            st.write("This option is still under development. Please select another option.")
        elif option == "1080":
            st.write("This option is for 1080 data processing.")
            type_of_1080 = st.selectbox(
                "Select the type of 1080 data",
                ("Linear Sprint", "CoD m505", "CoD m1005", "TBT"), index=0)
            
        elif option == "Force plate Jumps":
            st.write("This option is for force plate jumps data processing. Note that the excel sheet should contain a Time column and a Force column.")

        # if options force and 1080 are selected, upload the excel file
        if option == "Force plate Jumps" or option == "1080":
            # Upload Excel file
            excel_file = st.file_uploader("Upload an Excel file", type=["xlsx"], accept_multiple_files=False)
            if excel_file:
                log_activity(f"Raw excel file uploaded: {excel_file.name}")

    # Show the "Process" button only under certain conditions
    if (video_file and option == "None") or (video_file and option != "None" and excel_file):
        if st.button("Process"):
            # Create temporary files
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                temp_video.write(video_file.read())
                video_path = temp_video.name
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp_excel:
                    temp_excel.write(excel_file.read())
                    excel_path = temp_excel.name
            except Exception as e:
                pass

            try:
                start_time = time.time()  # Record the start time
                if option == "Force plate Jumps":
                    # FORCE PLATE JUMPS DATA PROCESSING
                    df_force_unfiltered, fps_force, capturing_length_force, total_frames_force = read_jump_excel(excel_path)
                    jump_height = calculate_jump_height(df_force_unfiltered['Force'], fps_force)
                    com_position = calculate_com_position(df_force_unfiltered, fps_force)
                    st.write(f"Jump height seems: {jump_height:.2f} m")

                elif option == "1080":
                # 1080 DATA PROCESSING
                    df_1080_unfiltered, capturing_length_1080, total_frames_1080, fps_10802, fps_1080 = read_1080(excel_path)
                    df_1080_filtered = filter_1080_data(df_1080_unfiltered, fps_1080, 10)
                    df_1080_filtered['Speed [m/s]'] = df_1080_filtered['Speed [m/s]'].abs()
                elif option == "None":
                    pass

                # VIDEO PROCESSING
                 
            
                # df_landmarks_raw, fps_video, capturing_length_video, total_frames_video =process_video(video_path, show_pose=1)
                #RMT POSE ESTIMATION|
                fps_video, total_number_of_frames, caputure_length ,video_path  = pose_estimation_rmt_pose(video_path)
                a=2
                #df_landmarks_raw, fps_video, capturing_length_video, total_frames_video, video_path= process_video_blured_new(video_path, show_pose=1, blur_faces=blur_facesuser)
                # user warning if the video frame rate is less than 30 fps may yields poor results
                if fps_video <=30:
                    st.warning("The video frame rate is less than 30 fps. The processing may not be accurate. Consider using a video with a higher frame rate.")
                # Replace inf and -inf with NaN
                #df_landmarks_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
                # Fill NaNs with the nearest valid value (forward then backward)
                #df_landmarks_raw = df_landmarks_raw.fillna(method='ffill').fillna(method='bfill')
                # filter the landmarks skipe for now TODO
                #df_landmarks_filtered = df_landmarks_raw
                df_landmarks_filtered = filter_landmarks_new(df_landmarks_raw)
                ####df_landmarks_filtered = filter_landmarks(df_landmarks_raw, fps_video, 5)
                # calculate the joint angles
                df_joint_angles = calculate_joint_angles(df_landmarks_filtered)
                #calculate the center of mass
                df_landmarks_filtered = calculate_com(df_landmarks_filtered)
                # calculate the velocity
                df_velocity = df_landmarks_filtered.diff() * fps_video
                # calculate the acceleration
                df_acceleration = df_velocity.diff() * fps_video
                # TODO is this a problem?
                df_velocity['com_x'] = df_velocity['com_x'].abs()
                
                if option == "Force plate Jumps":
                    com_position = com_position*-1
                    df_landmarks_filtered['com_y'] = df_landmarks_filtered['com_y']-df_landmarks_filtered['com_y'].iloc[0]
                    if fps_video > fps_force:
                        signal_a = upsample_signal(com_position, fps_force, fps_video)
                        signal_b = df_velocity['com_x']
                    else:
                        signal_a = com_position
                        signal_a =  downsample_df(signal_a, fps_force, fps_video )
                        force_down = downsample_df(df_force_unfiltered['Force'], fps_force, fps_video )
                        signal_b = df_landmarks_filtered['com_y']
                        factor = signal_a.max() / signal_b.max()
                        signal_b = signal_b * factor

                    factor = signal_a.max() / signal_b.max()
                    signal_b = signal_b * factor
                    sync_a, sync_b, lag, cut_index = sync_signals(signal_a, signal_b)
                    padded_force = pad_sync_signal(force_down, lag, cut_index, total_frames_video)
                    video_data = process_and_overlay_videoStreamlit_force(video_path, padded_force, capturing_length_video)

                    
                elif option == "1080":
                    if fps_video > fps_1080:
                        signal_a = upsample_signal(df_1080_filtered['Speed [m/s]'], fps_1080, fps_video)
                        signal_b = df_velocity['com_x']
                    else:
                        signal_a = upsample_signal(df_velocity['com_x'], fps_video, fps_1080)
                        signal_b = df_1080_filtered['Speed [m/s]']

                    df_velocity = replace_non_finite_valuesDF(df_velocity)
                    df_1080_filtered = replace_non_finite_valuesDF(df_1080_filtered)

                    signal_a = downsample_df(df_1080_filtered['Speed [m/s]'], fps_1080, fps_video )
                    df_distance = downsample_df(df_1080_filtered['Distance since start [m]'], fps_1080, fps_video )
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
                    
                    # Process video and get video data
                    video_data = process_and_overlay_videoStreamlit(video_path, df_pos_com, sync_a, lag, cut_index, total_time, padded_df_distance)

                elif option == "None":
                    video_data = process_and_overlay_videoStreamlit_None(video_path, df_landmarks_filtered)


                # Create download button
                st.download_button(
                    label="Download Processed Video",
                    data=video_data,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
                log_activity("Processed video downloaded!")
                
                end_time = time.time()  # Record the end time
                processing_time = end_time - start_time  # Calculate the processing duration
                log_activity(f"Processing time: {processing_time:.2f} seconds")
                
                # Mark processing as done
                st.session_state.processing_done = True
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                log_activity(f"Error occurred: {str(e)}")
            finally:
                # Clean up temporary files
                try:
                    os.unlink(video_path)
                    os.unlink(excel_path)
                except:
                    pass

with tab2:
    st.header("Help")
    st.write("""
    **How to use the app:**
    1. Upload a video file in mp4, mov, or avi format.
    2. Select the source of the data from the dropdown menu.
    3. If required, upload an Excel file in xlsx format.
    4. Click the "Process" button to process the files.
    5. Download the processed video using the download button.

    **Contact Us:**
    If you encounter any issues, please contact us at [patrickm@nih.no](mailto:patrickm@nih.no).

    **Developer Access:**
    Developers can download the user activity log by entering the developer password in the sidebar.
    """)


# Contact Us section
st.sidebar.header("Contact Us")
contact_email = "patrickm@nih.no"
subject = "WebApp"
body = "Please describe the error and attach the files so that we can debug 🐛."
mailto_link = f"mailto:{contact_email}?subject={subject}&body={body}"

if st.sidebar.button("Contact Us"):
    st.sidebar.markdown(f'<meta http-equiv="refresh" content="0; url={mailto_link}">', unsafe_allow_html=True)


# Developer access section in the sidebar
st.sidebar.header("Developer Access")
developer_password = st.sidebar.text_input("Enter developer password", type="password")

if developer_password == "Hawaii":  # Replace with your actual secure password
    if st.sidebar.button("Download User Activity Log"):
        with open('user_activity.log', 'r') as log_file:
            log_contents = log_file.read()
            st.sidebar.download_button(
                label="Download Log File",
                data=log_contents,
                file_name="user_activity.log",
                mime="text/plain"
            )
else:
    st.sidebar.warning("Incorrect password. Access denied.")