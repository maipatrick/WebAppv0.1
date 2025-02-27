import streamlit as st
import streamlit.components.v1 as components
import cv2
import tempfile
import mediapipe as mp
import numpy as np
import pandas as pd
from designidea import plot_hand_drawn_style
import matplotlib.pyplot as plt
import time
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize SQLite database
conn = sqlite3.connect('users.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
conn.commit()

# Function to check login credentials
def check_login(username, password):
    c.execute('SELECT password FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    if result and check_password_hash(result[0], password):
        return True
    return False

# Function to register a new user
def register_user(username, password):
    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
    try:
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

# Login and registration forms
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("Login or Register")

    # Tabs for login and registration
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            if check_login(username, password):
                st.session_state.logged_in = True
                st.success("Login successful!")
                st.experimental_set_query_params(logged_in="true")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")

    with tab2:
        st.subheader("Register")
        new_username = st.text_input("New Username", key="register_username")
        new_password = st.text_input("New Password", type="password", key="register_password")
        if st.button("Register"):
            if register_user(new_username, new_password):
                st.success("Registration successful! You can now log in.")
            else:
                st.error("Username already exists. Please choose a different username.")
else:
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

    # Log off button
    if st.sidebar.button("Log off"):
        st.session_state.logged_in = False
        st.experimental_set_query_params(logged_in="false")
        st.experimental_rerun()

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

            # Save the uploaded video to a temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())

            # Load the video file
            cap = cv2.VideoCapture(tfile.name)

            # Initialize MediaPipe Pose
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose()
            mp_drawing = mp.solutions.drawing_utils

            # Create a placeholder for the video frames
            frame_placeholder = st.empty()
            progress_bar = st.progress(0)

            # Get the total number of frames in the video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Process the video frame by frame
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert the frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Perform pose estimation
                results = pose.process(frame_rgb)

                # Draw the pose annotation on the frame
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Update the placeholder with the new frame
                frame_placeholder.image(frame, channels="BGR")

                # Update the progress bar
                frame_count += 1
                progress_bar.progress(frame_count / total_frames)

            cap.release()
            st.write("Pose estimation completed!")