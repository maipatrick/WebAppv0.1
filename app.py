# Add this at the very top of your app.py, before any other Streamlit commands
import streamlit as st
st.set_page_config(
    layout="wide",  # Use wide mode by default
    initial_sidebar_state="collapsed"  # Optional: collapse sidebar by default
)

import streamlit.components.v1 as components
import pandas as pd
import tempfile
import cv2
import time
import processing  # Our custom module for backend processing
import os
import json
import matplotlib.pyplot as plt
import glob

# Display the logo and set up the main title
st.image("logo.PNG", width=200)
st.title("Sync AI")

# # Sidebar with an About button
# st.sidebar.title("Sidebar")
# if st.sidebar.button("About"):
#     js = """
#     <script type="text/javascript">
#         window.open('https://example.com', '_blank').focus();
#     </script>
#     """
#     components.html(js, height=0)

# Create three tabs for the UI
tab1, tab2, tab3, tab4 = st.tabs(["Sync AI", "Output (Dev)","Results", "How to use"])

# Initialize session state for results if not exists
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = None
if 'processing_error' not in st.session_state:
    st.session_state.processing_error = None
if 'show_error' not in st.session_state:
    st.session_state.show_error = False
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

with tab1:
    tab1.write('Check out "How to use" tab for more information.')
    st.header("Please upload your files")
    
    # Excel file uploader and processing (optional)
    uploaded_excel = st.file_uploader("(Optional) Choose an Excel file for force data", type=["xlsx", "xls"])
    if uploaded_excel:
        st.write(f"Excel file {uploaded_excel.name} uploaded successfully!")
        excel_progress = st.progress(0)
        
        # Callback function to update Excel processing progress
        def excel_progress_callback(progress):
            excel_progress.progress(progress)
        
        # Process the Excel file and get back the DataFrame and optional figure
        df, fig = processing.process_excel(uploaded_excel, progress_callback=excel_progress_callback)
        
        if fig:
            st.pyplot(fig)

    # Video file uploader and processing
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
    if uploaded_video:
        st.write(f"Video file {uploaded_video.name} uploaded successfully!")
        st.video(uploaded_video)
        
        # Show additional inputs for video processing
        bodyheight = st.number_input("Body height (m)", min_value=0.0, max_value=2.5, value=1.87, format="%.2f")
        bodymass = st.number_input("Body mass (kg)", min_value=20.0, max_value=200.0, value=80.0, format="%.2f")
        
        # Dropdown selections for system and movement type
        options_Systems = ["MRD", "TBT"]
        selected_System = st.selectbox("System", options_Systems)
        if selected_System == "MRD":
            options = ["5O5", "10O5", "TBT"]
        else:
            options = ["TBT", "TBT2", "TBT"]
        selected_Movement = st.selectbox("Type of movement", options)
        
        # Option to show/hide pose estimation window
        show_pose = st.checkbox("Show pose estimation window", value=False)

        # When the OK button is pressed, process the video
        if st.button("OK"):
            st.write("Processing video for pose estimation using Sports2D. Please keep the window open...")
            results, error = processing.process_video(uploaded_video, bodyheight, show_pose=show_pose)
            
            # Store results in session state
            st.session_state.processing_results = results
            st.session_state.processing_error = error
            st.session_state.processing_complete = True
        
        # Show results if processing is complete
        if st.session_state.processing_complete:
            st.write("Pose estimation completed!")
            
            if uploaded_excel:
                st.info("Force data synchronization will be implemented soon!")
            
            if st.session_state.processing_error:
                st.error("Errors during processing")
                if st.button("Press to see errors", key="error_button"):
                    st.session_state.show_error = not st.session_state.show_error
                if st.session_state.show_error:
                    st.markdown("**Error Details:**")
                    st.code(st.session_state.processing_error)

with tab2:
    st.header("Results")
    
    # Check if we have processed video data to display
    if st.session_state.processing_results:
        st.write("Processing Results:")
        
        # Try to parse results as dictionary
        try:
            results_dict = eval(st.session_state.processing_results)  # Safely convert string representation to dict
            
            # Create tabs for different result types
            result_tabs = st.tabs(["Pose Data (TRC)", "Angles (MOT)", "Processing Log"])
            
            # Tab for TRC files (pose data)
            with result_tabs[0]:
                trc_files = [k for k in results_dict.keys() if k.startswith('trc_')]
                if trc_files:
                    for trc_file in trc_files:
                        #st.write(f"### {trc_file}")
                        #st.text(results_dict[trc_file])
                        st.download_button(
                            label=f"Download {trc_file}",
                            data=results_dict[trc_file],
                            file_name=trc_file.replace('trc_', ''),
                            mime="text/plain"
                        )
                else:
                    st.write("No pose data (TRC) files available.")
            
            # Tab for MOT files (angles)
            with result_tabs[1]:
                mot_files = [k for k in results_dict.keys() if k.startswith('mot_')]
                if mot_files:
                    for mot_file in mot_files:
                        #st.write(f"### {mot_file}")
                        #st.text(results_dict[mot_file])
                        st.download_button(
                            label=f"Download {mot_file}",
                            data=results_dict[mot_file],
                            file_name=mot_file.replace('mot_', ''),
                            mime="text/plain"
                        )
                else:
                    st.write("No angle data (MOT) files available.")
            
            # Tab for processing log
            with result_tabs[2]:
                st.write("### Standard Output")
                if results_dict.get('stdout'):
                    st.text(results_dict['stdout'])            
                
                if results_dict.get('stderr'):
                    st.write("### Errors/Warnings")
                    st.error(results_dict['stderr'])
                
        except Exception as e:
            st.error(f"Error parsing results: {str(e)}")
            st.text(st.session_state.processing_results)  # Show raw results as fallback
            
            # Still provide download option for raw data
            st.download_button(
                label="Download Raw Results",
                data=st.session_state.processing_results,
                file_name="sports2d_results.txt",
                mime="text/plain"
            )
    else:
        st.write("No results available yet. Please process a video file first.")

with tab3:
    st.header("Visualization")
    
    if not st.session_state.processing_results:
        st.warning("No data available. Please process a video first in the Sync AI tab.")
    else:
        try:
            # # Use custom CSS to reduce padding
            # st.markdown("""
            #     <style>
            #         .block-container {
            #             padding-top: 1rem;
            #             padding-bottom: 0rem;
            #             padding-left: 1rem;
            #             padding-right: 1rem;
            #         }
            #     </style>
            # """, unsafe_allow_html=True)
            
            # Change column ratio to be more equal
            col1, col2 = st.columns([1, 1])  # Equal width columns
            
            with col1:
                st.subheader("Data Visualization")
                # Get the results dictionary
                results_dict = eval(st.session_state.processing_results)
                
                # Add dropdown for visualization type
                viz_type = st.selectbox(
                    "Select visualization type",
                    ["Joint Angles", "Segment Angles", "Force Data"],
                    key="viz_selector"
                )
                
                # Container for up to 3 plots
                plot_container = st.container()
                with plot_container:
                    if viz_type == "Joint Angles":
                        mot_files = [k for k in results_dict.keys() if k.startswith('mot_')]
                        if mot_files:
                            mot_data = results_dict[mot_files[0]]
                            df_angles = processing.parse_mot_file(mot_data)
                            available_joints = [col for col in df_angles.columns if col not in ['time', 'frame']]
                            
                            # Define defaults in lower case
                            default_joints_lower = {"left ankle", "left knee", "left hip"}
                            # Choose joints from available_joints if their lower-case version is in the default set
                            default_selection = [joint for joint in available_joints if joint.lower() in default_joints_lower]
                            
                            joints = st.multiselect(
                                "Select joints to visualize (max 3)",
                                available_joints,
                                default=default_selection,
                                max_selections=3
                            )

                            
                            if st.button("Generate Plot", key="joint_plot") and joints:
                                # Use container width for consistent sizing
                                for i, joint in enumerate(joints[:3]):
                                    fig, ax = plt.subplots(figsize=(12, 6))
                                    ax.plot(df_angles['time'], df_angles[joint], label=joint)
                                    ax.set_xlabel('Time (s)')
                                    ax.set_ylabel(f'{joint} (degrees)')
                                    ax.set_title(f'{joint} over time (s)')
                                    ax.legend()
                                    ax.grid(True)
                                    st.pyplot(fig, use_container_width=True)
                        else:
                            st.warning("No joint angle data available")
                    
                    elif viz_type == "Segment Angles":
                        # Similar structure for segment angles...
                        pass
                    
                    elif viz_type == "Force Data":
                        # Similar structure for force data...
                        pass
            
            with col2:
                st.subheader("Pose Visualization")
                video_files = uploaded_video
                st.video(uploaded_video)
                # # Look for processed video in results
                # # st.write(f"results dict {results_dict.keys()}")
                # # st.write(f"video prosessert {results_dict['video_processed']}")
                # video_files = [k for k in results_dict['video_processed']]
                # # st.write(f'(Debug) VIDEO FILES: {video_files}')
                # if video_files:
                #     # # Use the first video file found
                #     # video_file = video_files[0]
                #     # with open(video_file, 'rb') as f:
                #     #     video_bytes = f.read()
                #     st.video(video_files[0])
                # else:
                #     st.image("logo.png",
                #             caption="Processed video not found",
                #             use_container_width=True)
                
                # Add some spacing
                st.write(" ")
                
                # Video controls (can be implemented later with actual video frames)
                if video_files:
                    with st.expander("Video Controls (Coming Soon)"):
                        st.slider("Frame", 0, 100, 50, key="video_frame", disabled=True)
                        
                        # Center the playback controls
                        with st.container():
                            col2_1, col2_2, col2_3, col2_4, col2_5 = st.columns([1,1,1,1,1])
                            with col2_2:
                                st.button("⏮️", key="prev_frame", disabled=True)
                            with col2_3:
                                st.button("⏯️", key="play_pause", disabled=True)
                            with col2_4:
                                st.button("⏭️", key="next_frame", disabled=True)
                
                # Add video info
                with st.expander("Video Information"):
                    if video_files:
                        st.write("✅ Processed video available")
                        st.write("Note: Frame-by-frame controls coming soon")
                    else:
                        st.write("❌ No processed video available")
                        st.write("Try processing the video again with pose estimation enabled")

        except Exception as e:
            st.error(f"Error processing data for visualization: {str(e)}")

with tab4:
    st.header("How to use")
    st.write("1. Upload an Excel file with kinetic data. For now we accept force plates and motorized equipement")
    st.write("2. Upload a video file for pose estimation. Generally we recommend to film using a tripod, parallel from the motion plane.")
    st.write("3. Enter your body height and mass, select the system and movement type, and press OK.")
    st.write("4. We will handle your kinetic data and synchronize it with kinematic data.")
    st.write("5. Check the Results tab for visualizations and insights.")
    st.write("6. Enjoy the experience of Sync AI!")