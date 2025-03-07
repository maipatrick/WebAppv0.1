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
import io


# Display the logo and set up the main title
st.image("logo.PNG", width=200)
st.title("Sync AI")

# Create three tabs for the UI
tab1, tab2, tab3 = st.tabs(["Sync AI", "Results", "How to use"])

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
    st.header("Visualization")
    
    if not st.session_state.processing_results:
        st.warning("No data available. Please process a video first in the Sync AI tab.")
    else:
        try:
            # Change column ratio to be more equal
            col1, col2 = st.columns([1, 1])  # Equal width columns
            
            with col1:
                st.subheader("Data Visualization")
                
                # Add dropdown for visualization type
                viz_type = st.selectbox(
                    "Select visualization type",
                    ["Joint Angles", "Force Data"],
                    key="viz_selector"
                )
                
                # Container for up to 3 plots
                plot_container = st.container()
                with plot_container:
                    if viz_type == "Joint Angles":
                        # Read the dummy joint angle CSV file
                        dummy_csv_path = "/Users/emilcarlsen/Documents/Python/MOCA/WebAppv0.1/dummy_joint_angles.csv"
                        df_angles = processing.read_joint_angle_csv(dummy_csv_path)
                        available_joints = [col for col in df_angles.columns if col != 'time']
                        
                        # Define defaults in lower case
                        default_joints_lower = {"ankle_angle_l", "knee_angle_l", "hip_angle_l"}
                        # Choose joints from available_joints if their lower-case version is in the default set
                        default_selection = [joint for joint in available_joints if joint.lower() in default_joints_lower]
                        
                        joints = st.multiselect(
                            "Select joints to visualize (max 8)",
                            available_joints,
                            default=default_selection,
                            max_selections=8
                        )

                        if joints:
                            # Use container width for consistent sizing
                            for i, joint in enumerate(joints[:8]):
                                fig, ax = plt.subplots(figsize=(12, 6))
                                ax.plot(df_angles['time'], df_angles[joint], label=joint)
                                ax.set_xlabel('Time (s)')
                                ax.set_ylabel(f'{joint} (degrees)')
                                ax.set_title(f'{joint} over time (s)')
                                ax.legend()
                                ax.grid(True)
                                st.pyplot(fig, use_container_width=True)
                                
                                # Save the plot to a BytesIO object
                                buf = io.BytesIO()
                                fig.savefig(buf, format='png')
                                buf.seek(0)
                                
                                # Store the buffer in session state for download
                                st.session_state[f"{joint}_plot"] = buf
                        else:
                            st.warning("No joint angle data available")
                    
                    elif viz_type == "Force Data":
                        # Similar structure for force data...
                        pass
            
            with col2:
                st.subheader("Download Plots")
                for joint in joints:
                    if f"{joint}_plot" in st.session_state:
                        st.download_button(
                            label=f"Download {joint} plot",
                            data=st.session_state[f"{joint}_plot"],
                            file_name=f"{joint}_plot.png",
                            mime="image/png"
                        )
                st.write("More options to download your plots and files coming soon ...")
        except Exception as e:
            st.error(f"Error processing data for visualization: {str(e)}")

with tab3:
    st.header("How to use")
    st.write("1. Upload an Excel file with kinetic data. For now we accept force plates and motorized equipement")
    st.write("2. Upload a video file for pose estimation. Generally we recommend to film using a tripod, parallel from the motion plane.")
    st.write("3. Enter your body height and mass, select the system and movement type, and press OK.")
    st.write("4. We will handle your kinetic data and synchronize it with kinematic data.")
    st.write("5. Check the Results tab for visualizations and insights.")
    st.write("6. Enjoy the experience of Sync AI!")