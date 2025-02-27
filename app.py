import streamlit as st
import streamlit.components.v1 as components


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from video_fun import process_video

# Display the logo
st.image("logo.PNG", width=200)

st.title("Sync AI")
st.header("How to use")

st.write("Nice animation here how it works?")

st.sidebar.title("Sidebar")
if st.sidebar.button("About"):
    js = """
    <script type="text/javascript">
        window.open('https://example.com', '_blank').focus();
    </script>
    """
    components.html(js, height=0)

# File uploader for Excel files
uploaded_excel = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

# File uploader for video files
uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if uploaded_excel:
    st.write(f"Excel file {uploaded_excel.name} uploaded successfully!")
    
    # Read the Excel file as a DataFrame
    df = pd.read_excel(uploaded_excel, header=0)
    
    # Display the DataFrame
    #st.write(df)
    
    # Simulate data processing with a progress bar
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.0005)  # Simulate processing time
        progress_bar.progress(i + 1)
    
    # # Access the data in the "Speed [m/s]" column
    # if "Speed [m/s]" in df.columns:
    #     y = df["Speed [m/s]"]
        
    #     # Generate the x array with the same length as y
    #     x = np.arange(len(y))
        
    #     # Create the plot with XKCD style
    #     fig = plot_hand_drawn_style(x, y)
        
    #     # Display the plot
    #     st.pyplot(fig)
    # else:
    #     st.write("Column 'Speed [m/s]' not found in the uploaded Excel file.")

if uploaded_video:
    st.write(f"Video file {uploaded_video.name} uploaded successfully!")
    st.video(uploaded_video)

    # Enable the "OK" button only if both files are uploaded
    if uploaded_excel and uploaded_video:
        bodyheight = st.number_input("Body height (m)", min_value=0.0, max_value=2.5 ,value=1.87,format="%.2f")
    bodymass = st.number_input("Body mass (kg)", min_value=20.0,max_value=200.0 , value=80.0, format="%.2f")
    # Add a dropdown list with various items
    options_Systems = ["MRD", "TBT"]
    selected_System = st.selectbox("System", options_Systems)
    # Conditional logic to change the options in the second dropdown list
    if selected_System == "MRD":
        options = ["5O5", "10O5", "TBT"]
    else:
        options = ["TBT", "TBT2", "TBT"]
    selected_Movement = st.selectbox("Type of movement", options)

    if st.button("OK"):
        st.write("Processing video for pose estimation...")
        process_video(uploaded_video, show_pose=0)




    
    st.write("Pose estimation completed!")