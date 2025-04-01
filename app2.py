import streamlit as st
from video_fun_pm import process_video, filter_landmarks, calculate_com, process_and_overlay_videoStreamlit,process_and_overlay_videoStreamlit_force, process_and_overlay_videoStreamlit_None, process_video_blured, process_video_blured_new, filter_landmarks_new, pose_estimation_rmt_pose, pose_estimation_rmt_pose_new, process_video_multi_person_rstLIB, new_video_for_linear, convert_pose_data_to_meters_multiple
from d1080_fun_pm import read_1080, filter_1080_data
from default_processing_pm import pad_df, sync_signals, upsample_signal, sync_signals22, downsample_df, replace_non_finite_valuesDF, calculate_joint_angles, pad_sync_signal, fill_missing_frames, create_labeled_df, detect_first_foot_movement_auto, find_first_peak, find_backwards_turning_point, sync_signals_by_transition, process_pose_data, interpolate_pose_data, calculate_joint_angles, filter_pose_data, process_scores, filter_landmarks_by_confidence, calculate_time_derivative, calculate_time_derivative2
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
    ##### Global variables
    cut_off_video =4 #in Hz
    #####
    st.title("Athlete MetrIcs")
    st.image("https://i.ibb.co/zVStNZg7/logo.png", width=200)
    # Initialize session state
    if 'processing_done' not in st.session_state:
        st.session_state.processing_done = False

    # Upload video file ✅
    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"], accept_multiple_files=False, key="video", help="Supported formats: mp4, mov, avi")
    if video_file:
        log_activity(f"Raw video file uploaded: {video_file.name}")
        # userinput for subjects body height in meter default is 1.95 ✅
        subject_height = st.number_input("Enter the subject's body height in meters", value=1.95, step=0.01)
        # checkbox to blur faces ❌ TODO
        blur_faces_user = st.checkbox("Blur faces", value=True)
        # Dropdown to select the source of the data ✅
        option = st.selectbox(
        "Source of the data",
        ("1080", "Force plate Jumps", "None", "TBD"), index=2)

        # Dropdown to select the source of the data for 1080 and some hints ✅
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

        # if options force or 1080 are selected, upload the excel file  ✅
        if option == "Force plate Jumps" or option == "1080":
            # Upload Excel file
            excel_file = st.file_uploader("Upload the corresponding Excel file", type=["xlsx"], accept_multiple_files=False)
            if excel_file:
                log_activity(f"Raw excel file uploaded: {excel_file.name}")

    # Show the "Process" button only under certain conditions ✅
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
                if option == "Force plate Jumps": #TODO ❌
                    # FORCE PLATE JUMPS DATA PROCESSING
                    df_force_unfiltered, fps_force, capturing_length_force, total_frames_force = read_jump_excel(excel_path)
                    jump_height = calculate_jump_height(df_force_unfiltered['Force'], fps_force)
                    com_position = calculate_com_position(df_force_unfiltered, fps_force)
                    st.write(f"Jump height seems: {jump_height:.2f} m")

                elif option == "1080": #TODO ❌ problem coloums named different depending on the type of 1080 data
                # 1080 DATA PROCESSING
                    df_1080_unfiltered, capturing_length_1080, total_frames_1080, fps_10802, fps_1080 = read_1080(excel_path)
                    df_1080_filtered = filter_1080_data(df_1080_unfiltered, fps_1080, 6)
                elif option == "None":
                    pass

                #### VIDEO PROCESSING ####
                # Process the video and get the pose estimation results ✅
                pose_data, fps_video, total_frames_video, capture_length_video = process_video_multi_person_rstLIB(video_path) 
                
                if fps_video <30: # simple warning if the video frame rate is less than 30 fps
                    st.warning("The video frame rate is less than 30 fps. The processing may not be accurate. Consider using a video with a higher frame rate.")
                # TODO ❌ if multiple persons are detected make a workflow to select the right person automatically? --> reference_person_id as string this person will only be processed!
                # convert from pixel to meter ✅
                pose_data_m = convert_pose_data_to_meters_multiple(pose_data, reference_height_m= subject_height)
                # process the pose data and bring it in a more handy format ✅
                pose_data_df = process_pose_data(total_frames_video, pose_data, pose_data_m)
                pose_data_df = process_scores(pose_data, pose_data_df, total_frames_video)
                # drop the ones that have a low confidence score ✅
                pose_data_df = filter_landmarks_by_confidence(pose_data_df, confidence_threshold=0.55)
                # fill the pose data with the closest known variable ✅ TODO check again
                pose_data_df = interpolate_pose_data(pose_data_df)
                # calculate the joint angles ✅
                pose_data_df = calculate_joint_angles(pose_data_df)
                pose_data_df = process_scores(pose_data, pose_data_df, total_frames_video)
                
                pose_data_df = filter_pose_data(pose_data_df, 'Angles', fps_video, cut_off_video)
                pose_data_df = calculate_time_derivative(pose_data_df, fps_video)
                pose_data_df = calculate_time_derivative2(pose_data_df, fps_video)
                # filter some data
                pose_data_df = filter_pose_data(pose_data_df, 'filled_vel', fps_video, cut_off_video)
                pose_data_df = filter_pose_data(pose_data_df, 'scaled_filled_vel', fps_video, cut_off_video)
                pose_data_df = filter_pose_data(pose_data_df, 'filled_acc', fps_video, cut_off_video)
                pose_data_df = filter_pose_data(pose_data_df, 'scaled_filled_acc', fps_video, cut_off_video)
                pose_data_df = filter_pose_data(pose_data_df, 'filled', fps_video, cut_off_video)
                pose_data_df = filter_pose_data(pose_data_df, 'scaled_filled', fps_video, cut_off_video)

                # determine person of interest 
                POI = '1' # TODO ❌
                # TODO replace ith with new variables from pose_data_df 
                
                
                # df_velocity = pose_data_df['1']['filled'].diff() * fps_video
                # # calculate the velocity in meters/S ✅
                # df_velocity_m = pose_data_df['1']['scaled_filled'].diff() * fps_video
                # # fill with the clostest know variable ❌ TODO not the best solution
                # df_velocity = df_velocity.fillna(method='ffill').fillna(method='bfill')
                # # fill with the clostest know variable in meters ❌ TODO not the best solution
                # df_velocity_m = df_velocity_m.fillna(method='ffill').fillna(method='bfill')
                # #filter the velocity with the butterworth filter ✅
                # df_velocity = filter_landmarks(df_velocity, fps_video, cut_off_video)
                # #filter the velocity with the butterworth filter in meters ✅
                # df_velocity_m = filter_landmarks(df_velocity_m, fps_video, cut_off_video)
                # # calculate the acceleration ✅
                # df_acceleration = df_velocity.diff() * fps_video
                # # calculate the acceleration in meters/S^2 ✅
                # df_acceleration_m = df_velocity_m.diff() * fps_video
                # # fill with the clostest know variable ✅ TODO not the best solution
                # df_acceleration = df_acceleration.fillna(method='ffill').fillna(method='bfill')
                # # fill with the clostest know variable in meters ✅ TODO not the best solution
                # df_acceleration_m = df_acceleration_m.fillna(method='ffill').fillna(method='bfill')
                

                # not working  ❌ TODO
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
                    # if fps_video > fps_1080:
                    #     signal_a = upsample_signal(df_1080_filtered['Speed [m/s]'], fps_1080, fps_video)
                    #     signal_b = df_velocity['com_x']
                    # else:
                    #     signal_a = upsample_signal(df_velocity['com_x'], fps_video, fps_1080)
                    #     signal_b = df_1080_filtered['Speed [m/s]']
                    # get signal_a and signal_b in a csv file
                    # TODO NEDDED ? ❌
                    #df_velocity = replace_non_finite_valuesDF(df_velocity)
                    #df_1080_filtered = replace_non_finite_valuesDF(df_1080_filtered)
                    #downsample the 1080 data to the video data ✅

                    signal_b = pose_data_df[POI]['scaled_filled_vel_filt']['Hip_x']# df_velocity_m['Hip_x']

                    if type_of_1080 == "Linear Sprint":
                        signal_a = downsample_df(df_1080_filtered['Speed [m/s]'], fps_1080, fps_video )
                        # TODO NEDDED ? ❌ 'Distance since start [m]' is different for linear sprint
                        signal_a_distance = downsample_df(df_1080_filtered['Distance [m]'], fps_1080, fps_video )
                        #TODO in only if movement is from left to right and video is in landscape mode ❌
                    elif type_of_1080 == "CoD m505":
                        signal_a = downsample_df(df_1080_filtered['Speed [m/s]'], fps_1080, fps_video )
                        # TODO NEDDED ? ❌ 'Distance since start [m]' is different for linear sprint
                        signal_a_distance = downsample_df(df_1080_filtered['Distance since start [m]'], fps_1080, fps_video )
                        signal_b = signal_b*-1
                    
                    #write signal_b to csv
                    #pd.DataFrame(signal_b).to_csv('signal_b.csv', index=False)
                    # prepare the data for the sync signal function ✅
                    idx = find_backwards_turning_point(signal_b, 2)
                    # Pad df_signal_a['Speed [m/s]'] with 1000 zeros at the start and end
                    appended = pd.concat([pd.Series([0] * 100000), signal_a, pd.Series([0] * 100000)], ignore_index=True)
                    appended_distance = pd.concat([pd.Series([0] * 100000), signal_a_distance, pd.Series([0] * 100000)], ignore_index=True)

                    if idx is not None:
                        signal_b.loc[:idx] = 0
                    shift = sync_signals_by_transition(appended, signal_b)
                    
                    # Keep only the part of appended from shift onward
                    # Keep only the part of appended from shift onward and reset the index
                    if shift is not None and shift > 0:
                        appended = appended[shift:].reset_index(drop=True)  # Delete everything before shift and reset index
                        appended_distance = appended_distance[shift:].reset_index(drop=True)  # Delete everything before shift and reset index
                    else:
                        appended = appended.reset_index(drop=True)  # Reset index if no slicing is needed
                        appended_distance = appended_distance.reset_index(drop=True)  # Reset index if no slicing is needed
                    velocity_1080_in_video = appended[:len(signal_b)] 
                    distance_1080_in_video = appended_distance[:len(signal_b)]
                    # detect when the first foot movement is detected ✅
                    # foot, index = detect_first_foot_movement_auto(df_landmarks_filtered_filt)
                    # peak_index_1080, peak_value_1080 = find_first_peak(signal_a)
                    # peak_index_com, peak_value_com = find_first_peak(signal_b[index:])
                    # peak_index_com = peak_index_com+index

                    # make a break here to see what I have
                    # write df_landmarks_filtered_filt_m to excel
                    #df_landmarks_filtered_filt_m.to_excel('df_landmarks_filtered_filt_m.xlsx', index=False)
                    # write df_landmarks_filtered_filt to excel
                    #df_landmarks_filtered_filt.to_excel('df_landmarks_filtered_filt.xlsx', index=False)
                    # write df_velocity to excel
                    #df_velocity.to_excel('df_velocity.xlsx', index=False)
                    # write df_velocity_m to excel
                    #df_velocity_m.to_excel('df_velocity_m.xlsx', index=False)
                    # write signal_a to excel
                    
                    # write signal_b to excel
                    #pd.DataFrame(signal_a).to_excel('signal_a.xlsx', index=False)
                    #  df_landmarks_filtered_filt_m df_velocity  df_velocity_m signal_a
                    
                    
                    # write landmarks_filled_filt
                    #factor = signal_a.max() / signal_b.max()
                    #signal_b = signal_b * factor

                    #sync_a, sync_b, lag, cut_index = sync_signals(signal_a, signal_b)
                    #sync_a_df = pd.DataFrame(sync_a)
                    #sync_b_df = pd.DataFrame(sync_b)
                    # write sync_a and sync_b as csv
                    #sync_a_df.to_csv('sync_a.csv', index=False)
                    #sync_b_df.to_csv('sync_b.csv', index=False)
                    #padded_df_distance = pad_df(df_distance, len(sync_a), lag, cut_index)

                    #df_pos_com = pd.DataFrame({
                    #    'com_x': df_landmarks_filtered['com_x'],
                    #    'com_y': df_landmarks_filtered['com_y']
                    #})

                    total_time = df_1080_unfiltered['Time since start [s]'].iloc[-1]

                    # Find the start_frame (first index where the signal turns from 0 to positive) ✅
                    start_frame = velocity_1080_in_video[(velocity_1080_in_video > 0) & (velocity_1080_in_video.shift(1) == 0)].index[0]
                    #start_frame = start_frame - 5 # some buffer frames
                    # Find the end_frame (first index where the signal turns from positive to 0) ✅
                    end_frame = velocity_1080_in_video[(velocity_1080_in_video == 0) & (velocity_1080_in_video.shift(1) > 0)].index[0]
                    #end_frame = end_frame +  5# some buffer frames
                    # write the final video ✅
                    #landmarks_draw = ["LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist", "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle", "Hip", "Neck", "RHeel", "LHeel", "RBigToe", "LBigToe"]
                    landmarks_draw = []
                    draw_skeleton_in_video = False
                    video_data = new_video_for_linear(video_path, velocity_1080_in_video, start_frame, end_frame, pose_data_df[POI]['filled'], landmarks_draw, distance_1080_in_video, pose_data_df, blur_faces_user, draw_skeleton_in_video)
                    # video_data=new_video_for_linear(video_path, velocity_1080_in_video, start_frame, end_frame, df_landmarks_filtered, landmarks_draw, distance_1080_in_video)
                    #video_data = process_and_overlay_videoStreamlit(video_path, df_pos_com, sync_a, lag, cut_index, total_time, padded_df_distance)

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