import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import tempfile
import cv2
import time
import processing  # Our custom module for backend processing

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
        
        # Show additional inputs only if both files are uploaded
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
            
            # When the OK button is pressed, process the video
            if st.button("OK"):
                st.write("Processing video for pose estimation...")
                video_progress = st.progress(0)
                video_placeholder = st.empty()
                
                # Callback to update video processing progress
                def video_progress_callback(progress):
                    video_progress.progress(progress)
                
                # Callback to update the video frame display
                def frame_callback(frame):
                    video_placeholder.image(frame, channels="BGR")
                
                processing.process_video(
                    uploaded_video,
                    progress_callback=video_progress_callback,
                    frame_callback=frame_callback
                )
                st.write("Pose estimation completed!")
                st.write("Kinematic analysis in progress... (coming soon)")
                st.write("Processing kinetic data and synchronizing with kinematic data... (coming soon)")

with tab2:
    st.header("Results")
    st.write("This is the Results tab.")

with tab3:
    st.header("How to use")
    st.write("1. Upload an Excel file with kinetic data. For now we accept force plates and motorized equipement")
    st.write("2. Upload a video file for pose estimation. Generally we recommend to film using a tripod, parallel from the motion plane.")
    st.write("3. Enter your body height and mass, select the system and movement type, and press OK.")
    st.write("4. We will handle your kinetic data and synchronize it with kinematic data.")
    st.write("5. Check the Results tab for visualizations and insights.")
    st.write("6. Enjoy the experience of Sync AI!")