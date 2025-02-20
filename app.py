import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import tempfile
import cv2
import time
import processing  # Our custom module for backend processing
import os
import json

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
    
    # Excel file uploader and processing
    uploaded_excel = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
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
        
        # Show additional inputs only if both files are uploaded (if needed)
        if uploaded_excel and uploaded_video:
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
                st.session_state.processing_complete = True  # Mark processing as complete
            
            # Show results if processing is complete
            if st.session_state.processing_complete:
                st.write("Pose estimation completed!")
                # st.write("Sync AI is now working on synchronizing your data... (coming soon)")
                # ## add function to sync the data.
                # st.write("There we go! Please check the Results tab for visualizations and insights.")
                
                if st.session_state.processing_error:
                    st.error("Errors during processing")
                    
                    # Toggle the error display
                    if st.button("Press to see errors", key="error_button"):
                        st.session_state.show_error = not st.session_state.show_error
                    
                    # Show error if toggled
                    if st.session_state.show_error:
                        st.markdown("**Error Details:**")
                        st.code(st.session_state.processing_error)
    
    #mer kode her?

with tab2:
    st.header("Results")
    
    # Check if we have processed video data to display
    if st.session_state.processing_results:
        st.write("Sports2D Processing Results:")
        
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
    st.header("How to use")
    st.write("1. Upload an Excel file with kinetic data. For now we accept force plates and motorized equipement")
    st.write("2. Upload a video file for pose estimation. Generally we recommend to film using a tripod, parallel from the motion plane.")
    st.write("3. Enter your body height and mass, select the system and movement type, and press OK.")
    st.write("4. We will handle your kinetic data and synchronize it with kinematic data.")
    st.write("5. Check the Results tab for visualizations and insights.")
    st.write("6. Enjoy the experience of Sync AI!")